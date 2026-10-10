#!/usr/bin/env python3
"""Overlay 补丁单元测试：结构化输出「缺 key」兜底 + 思维链剥离 + provider schema 净化。

不依赖 dify 运行时：用 stub 顶掉 dify/graphon 的 import，直接加载并测试 overlay 文件里的
纯逻辑（_parse_structured_output / _apply_schema_defaults / _strip_unsupported_keywords）。

运行：python3 docker/overlay/test_structured_output_overlay.py
退出码：0 = 全绿。
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "api/core/llm_generator/output_parser/structured_output.py"


def _install_stubs() -> None:
    """顶掉 dify/graphon 依赖，只保留被测文件需要的名字。"""

    class OutputParserError(ValueError):
        pass

    def _mk(name: str, **attrs: object) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    _mk("core")
    _mk("core.llm_generator")
    _mk("core.llm_generator.output_parser")
    _mk("core.llm_generator.output_parser.errors", OutputParserError=OutputParserError)
    _mk("core.llm_generator.prompts", STRUCTURED_OUTPUT_PROMPT="{{schema}}")
    _mk("core.model_manager", ModelInstance=object)

    for name in (
        "graphon",
        "graphon.model_runtime",
        "graphon.model_runtime.callbacks",
        "graphon.model_runtime.entities",
    ):
        _mk(name)
    _mk("graphon.model_runtime.callbacks.base_callback", Callback=object)
    _mk(
        "graphon.model_runtime.entities.llm_entities",
        LLMResult=object,
        LLMResultChunk=object,
        LLMResultChunkDelta=object,
        LLMResultChunkWithStructuredOutput=object,
        LLMResultWithStructuredOutput=object,
    )
    _mk(
        "graphon.model_runtime.entities.message_entities",
        AssistantPromptMessage=object,
        PromptMessage=object,
        PromptMessageTool=object,
        SystemPromptMessage=object,
        TextPromptMessageContent=object,
    )
    _mk("graphon.model_runtime.entities.model_entities", AIModelEntity=object, ParameterRule=object)


def _load_overlay():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("overlay_structured_output", TARGET)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


M = _load_overlay()

_failures: list[str] = []


def check(label: str, got, expected) -> None:
    if got != expected:
        _failures.append(f"{label}\n     got={got!r}\n     exp={expected!r}")
        print(f"  ✗ {label}\n      got={got!r}\n      exp={expected!r}")
    else:
        print(f"  ✓ {label}")


SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
}


print("1) 缺 key 无 default -> 类型零值 + __degraded__ 标注")
out = M._parse_structured_output("{}", SCHEMA)
check("verdict 补空串", out.get("verdict"), "")
check("issues 补空数组", out.get("issues"), [])
check("__degraded__ 记录路径", out.get("__degraded__"), ["verdict", "issues"])

print("2) 缺 key 有 default -> 用 default")
schema_d = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "default": "N"},
        "issues": {"type": "array", "items": {"type": "string"}, "default": []},
    },
}
out = M._parse_structured_output("{}", schema_d)
check("verdict 取 default", out.get("verdict"), "N")
check("issues 取 default", out.get("issues"), [])

print("3) 正常完整输出 -> 原样保留，__degraded__ 为空")
out = M._parse_structured_output('{"verdict":"Y","issues":["x"]}', SCHEMA)
check("verdict 原样", out.get("verdict"), "Y")
check("issues 原样", out.get("issues"), ["x"])
check("__degraded__ 空", out.get("__degraded__"), [])

print("4) 嵌套对象缺子字段 -> 递归补全")
schema_nested = {
    "type": "object",
    "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}}},
}
out = M._parse_structured_output('{"a":{}}', schema_nested)
check("a.b 补空串", out["a"]["b"], "")
check("__degraded__ 记 a.b", out.get("__degraded__"), ["a.b"])

print("5) 数组 of object 内缺字段 -> 递归补全")
schema_arr = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {"type": "object", "properties": {"t": {"type": "string"}}},
        }
    },
}
out = M._parse_structured_output('{"cards":[{}]}', schema_arr)
check("cards[0].t 补空串", out["cards"][0]["t"], "")
check("__degraded__ 记 cards[0].t", out.get("__degraded__"), ["cards[0].t"])

print("6) 非法 JSON（代码块包裹）-> json_repair 修复后仍补全")
out = M._parse_structured_output('```json\n{"verdict":"Y"}\n```', SCHEMA)
check("verdict 修复保留", out.get("verdict"), "Y")
check("issues 补空数组", out.get("issues"), [])
check("__degraded__ 记 issues", out.get("__degraded__"), ["issues"])

print("7) 回归：<think> 思维链剥离仍生效")
out = M._parse_structured_output(
    '<think>\n{"verdict":"Y"}\n</think>\n{"verdict":"Y","issues":[]}',
    SCHEMA,
)
check("剥离思维链后解析", (out.get("verdict"), out.get("issues")), ("Y", []))
check("无兜底", out.get("__degraded__"), [])

print("8) 无 schema / 空 properties -> 原样返回（不注入 __degraded__）")
raw = {"anything": 1}
check("空 schema 原样", M._parse_structured_output('{"anything":1}', {}), raw)
check("None schema 原样", M._parse_structured_output('{"anything":1}', None), raw)

print("9) _strip_unsupported_keywords：递归剥离 default（供 provider/提示词用）")
schema_s = {
    "type": "object",
    "properties": {"v": {"type": "string", "default": "N"}},
    "default": {"v": "x"},
}
M._strip_unsupported_keywords(schema_s)
check("顶层 default 移除", "default" in schema_s, False)
check("嵌套 default 移除", "default" in schema_s["properties"]["v"], False)
check("其它关键字保留", schema_s["properties"]["v"]["type"], "string")

print("10) 显式 null 视为缺失 -> 兜底")
out = M._parse_structured_output('{"verdict":null,"issues":null}', SCHEMA)
check("null verdict -> 空串", out.get("verdict"), "")
check("null issues -> 空数组", out.get("issues"), [])

print()
if _failures:
    print(f"结果：不通过（{len(_failures)} 个用例失败）")
    sys.exit(1)
print("结果：通过（全部用例绿）")
