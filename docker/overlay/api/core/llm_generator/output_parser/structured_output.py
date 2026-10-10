import json
import re
from collections.abc import Generator, Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from typing import Any, Literal, cast, overload

import json_repair
from pydantic import TypeAdapter, ValidationError

from core.llm_generator.output_parser.errors import OutputParserError
from core.llm_generator.prompts import STRUCTURED_OUTPUT_PROMPT
from core.model_manager import ModelInstance
from graphon.model_runtime.callbacks.base_callback import Callback
from graphon.model_runtime.entities.llm_entities import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMResultChunkWithStructuredOutput,
    LLMResultWithStructuredOutput,
)
from graphon.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
)
from graphon.model_runtime.entities.model_entities import AIModelEntity, ParameterRule


class ResponseFormat(StrEnum):
    """Constants for model response formats"""

    JSON_SCHEMA = "json_schema"  # model's structured output mode. some model like gemini, gpt-4o,  support this mode.
    JSON = "JSON"  # model's json mode. some model like claude support this mode.
    JSON_OBJECT = "json_object"  # json mode's another alias. some model like deepseek-chat, qwen use this alias.


class SpecialModelType(StrEnum):
    """Constants for identifying model types"""

    GEMINI = "gemini"
    OLLAMA = "ollama"


@overload
def invoke_llm_with_structured_output(
    *,
    provider: str,
    model_schema: AIModelEntity,
    model_instance: ModelInstance,
    prompt_messages: Sequence[PromptMessage],
    json_schema: Mapping[str, Any],
    model_parameters: Mapping | None = None,
    tools: Sequence[PromptMessageTool] | None = None,
    stop: list[str] | None = None,
    stream: Literal[True],
    callbacks: list[Callback] | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> Generator[LLMResultChunkWithStructuredOutput, None, None]: ...
@overload
def invoke_llm_with_structured_output(
    *,
    provider: str,
    model_schema: AIModelEntity,
    model_instance: ModelInstance,
    prompt_messages: Sequence[PromptMessage],
    json_schema: Mapping[str, Any],
    model_parameters: Mapping | None = None,
    tools: Sequence[PromptMessageTool] | None = None,
    stop: list[str] | None = None,
    stream: Literal[False],
    callbacks: list[Callback] | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> LLMResultWithStructuredOutput: ...
@overload
def invoke_llm_with_structured_output(
    *,
    provider: str,
    model_schema: AIModelEntity,
    model_instance: ModelInstance,
    prompt_messages: Sequence[PromptMessage],
    json_schema: Mapping[str, Any],
    model_parameters: Mapping | None = None,
    tools: Sequence[PromptMessageTool] | None = None,
    stop: list[str] | None = None,
    stream: bool = True,
    callbacks: list[Callback] | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> LLMResultWithStructuredOutput | Generator[LLMResultChunkWithStructuredOutput, None, None]: ...
def invoke_llm_with_structured_output(
    *,
    provider: str,
    model_schema: AIModelEntity,
    model_instance: ModelInstance,
    prompt_messages: Sequence[PromptMessage],
    json_schema: Mapping[str, Any],
    model_parameters: Mapping | None = None,
    tools: Sequence[PromptMessageTool] | None = None,
    stop: list[str] | None = None,
    stream: bool = True,
    callbacks: list[Callback] | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> LLMResultWithStructuredOutput | Generator[LLMResultChunkWithStructuredOutput, None, None]:
    """
    Invoke large language model with structured output
    1. This method invokes model_instance.invoke_llm with json_schema
    2. Try to parse the result as structured output

    :param prompt_messages: prompt messages
    :param json_schema: json schema
    :param model_parameters: model parameters
    :param tools: tools for tool calling
    :param stop: stop words
    :param stream: is stream response
    :param callbacks: callbacks
    :return: full response or stream response chunk generator result
    """

    # handle native json schema
    model_parameters_with_json_schema: dict[str, Any] = {
        **(model_parameters or {}),
    }

    if model_schema.support_structure_output:
        model_parameters = _handle_native_json_schema(
            provider, model_schema, json_schema, model_parameters_with_json_schema, model_schema.parameter_rules
        )
    else:
        # Set appropriate response format based on model capabilities
        _set_response_format(model_parameters_with_json_schema, model_schema.parameter_rules)

        # handle prompt based schema
        prompt_messages = _handle_prompt_based_schema(
            prompt_messages=prompt_messages,
            structured_output_schema=json_schema,
        )

    llm_result = model_instance.invoke_llm(
        prompt_messages=list(prompt_messages),
        model_parameters=model_parameters_with_json_schema,
        tools=tools,
        stop=stop,
        stream=stream,
        callbacks=callbacks,
        request_metadata=request_metadata,
    )

    if isinstance(llm_result, LLMResult):
        if not isinstance(llm_result.message.content, str):
            raise OutputParserError(
                f"Failed to parse structured output, LLM result is not a string: {llm_result.message.content}"
            )

        return LLMResultWithStructuredOutput(
            structured_output=_parse_structured_output(llm_result.message.content, json_schema),
            model=llm_result.model,
            message=llm_result.message,
            usage=llm_result.usage,
            system_fingerprint=llm_result.system_fingerprint,
            prompt_messages=llm_result.prompt_messages,
        )
    else:

        def generator() -> Generator[LLMResultChunkWithStructuredOutput, None, None]:
            result_text: str = ""
            prompt_messages: Sequence[PromptMessage] = []
            system_fingerprint: str | None = None
            for event in llm_result:
                if isinstance(event, LLMResultChunk):
                    prompt_messages = event.prompt_messages
                    system_fingerprint = event.system_fingerprint

                    match event.delta.message.content:
                        case str():
                            result_text += event.delta.message.content
                        case list():
                            for item in event.delta.message.content:
                                if isinstance(item, TextPromptMessageContent):
                                    result_text += item.data

                yield LLMResultChunkWithStructuredOutput(
                    model=model_schema.model,
                    prompt_messages=prompt_messages,
                    system_fingerprint=system_fingerprint,
                    delta=event.delta,
                )

            yield LLMResultChunkWithStructuredOutput(
                structured_output=_parse_structured_output(result_text, json_schema),
                model=model_schema.model,
                prompt_messages=prompt_messages,
                system_fingerprint=system_fingerprint,
                delta=LLMResultChunkDelta(
                    index=0,
                    message=AssistantPromptMessage(content=""),
                    usage=None,
                    finish_reason=None,
                ),
            )

        return generator()


def _handle_native_json_schema(
    provider: str,
    model_schema: AIModelEntity,
    structured_output_schema: Mapping,
    model_parameters: dict[str, Any],
    rules: list[ParameterRule],
) -> dict[str, Any]:
    """
    Handle structured output for models with native JSON schema support.

    :param model_parameters: Model parameters to update
    :param rules: Model parameter rules
    :return: Updated model parameters with JSON schema configuration
    """
    # Process schema according to model requirements
    schema_json = _prepare_schema_for_model(provider, model_schema, structured_output_schema)

    # Set JSON schema in parameters
    model_parameters["json_schema"] = json.dumps(schema_json, ensure_ascii=False)

    # Set appropriate response format if required by the model
    for rule in rules:
        if rule.name == "response_format" and ResponseFormat.JSON_SCHEMA in rule.options:
            model_parameters["response_format"] = ResponseFormat.JSON_SCHEMA

    return model_parameters


def _set_response_format(model_parameters: dict[str, Any], rules: list[ParameterRule]) -> None:
    """
    Set the appropriate response format parameter based on model rules.

    :param model_parameters: Model parameters to update
    :param rules: Model parameter rules
    """
    for rule in rules:
        if rule.name == "response_format":
            if ResponseFormat.JSON in rule.options:
                model_parameters["response_format"] = ResponseFormat.JSON
            elif ResponseFormat.JSON_OBJECT in rule.options:
                model_parameters["response_format"] = ResponseFormat.JSON_OBJECT


def _handle_prompt_based_schema(
    prompt_messages: Sequence[PromptMessage], structured_output_schema: Mapping
) -> list[PromptMessage]:
    """
    Handle structured output for models without native JSON schema support.
    This function modifies the prompt messages to include schema-based output requirements.

    Args:
        prompt_messages: Original sequence of prompt messages

    Returns:
        list[PromptMessage]: Updated prompt messages with structured output requirements
    """
    # Convert schema to string format
    # [overlay] 提示词路径剥离 default 等不支持关键字：否则模型会误以为该字段可省略
    prompt_schema = dict(deepcopy(structured_output_schema))
    _strip_unsupported_keywords(prompt_schema)
    schema_str = json.dumps(prompt_schema, ensure_ascii=False)

    # Find existing system prompt with schema placeholder
    system_prompt = next(
        (prompt for prompt in prompt_messages if isinstance(prompt, SystemPromptMessage)),
        None,
    )
    structured_output_prompt = STRUCTURED_OUTPUT_PROMPT.replace("{{schema}}", schema_str)
    # Prepare system prompt content
    system_prompt_content = (
        structured_output_prompt + "\n\n" + system_prompt.content
        if system_prompt and isinstance(system_prompt.content, str)
        else structured_output_prompt
    )
    system_prompt = SystemPromptMessage(content=system_prompt_content)

    # Extract content from the last user message

    filtered_prompts = [prompt for prompt in prompt_messages if not isinstance(prompt, SystemPromptMessage)]
    updated_prompt = [system_prompt] + filtered_prompts

    return updated_prompt


# ==== [overlay] 二次开发补丁：推理模型思维链污染结构化输出 ====
# 背景：reasoning 模型的思维链在「流式」路径里以内联 <think>...</think> 输出；
#       上游只在 LLM 节点 text 输出侧(graphon ThinkStreamFilter)剥离，未覆盖本函数，
#       导致 _parse_structured_output 收到含思维链的原始流 -> TypeAdapter 校验失败 ->
#       json_repair 把思维链里的 schema 片段修成 {"topics": []} 之类空壳 -> 静默丢数据。
# 方案：解析前先剥离 <think> 块。升级上游时若该行为已被官方修复，本补丁可移除。
# 兼容校验：docker/overlay/verify-overlay.py
_THINK_BLOCK_RE = re.compile(r"<think(?:\s[^<>]*)?>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think(?:\s[^<>]*)?>", re.IGNORECASE)


def _strip_reasoning_blocks(text: str) -> str:
    """[overlay] 剥离推理模型内联的 <think>...</think> 思维链。"""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # 生成被截断时可能残留未闭合 <think>，其后内容一律视为思维链丢弃
    trailing = _THINK_OPEN_RE.search(cleaned)
    if trailing:
        cleaned = cleaned[: trailing.start()]
    return cleaned.strip()


# ==== [overlay] 二次开发补丁：结构化输出「缺 key」兜底（schema 驱动） ====
# 背景：上游 _parse_structured_output 只保证「是 JSON 对象」，不按 schema 校验/补全。
#       模型漏吐字段时，下游强类型消费方（LLM 提示词 / 变量赋值器 / IF-ELSE）会
#       VariableNotFoundError 硬失败并 abort 整个 run。
# 方案：解析后按 schema 补全缺失字段——有 default 用 default，否则补类型零值；
#       并把被兜底的字段路径写进保留键 __degraded__，供下游区分「真值 vs 兜底值」。
#       仅当缺失时才兜底（模型正常输出的字段原样保留），不引入额外歧义。
# 注意：default 仅本地解析端使用，外发 provider/提示词前由 _strip_unsupported_keywords 剥离。
# 兼容校验：docker/overlay/verify-overlay.py
_DEGRADED_KEY = "__degraded__"
# provider 不支持或会误导模型的 JSON Schema 关键字（仅本地解析端消费）
_UNSUPPORTED_SCHEMA_KEYWORDS = ("default",)
_MISSING = object()


def _strip_unsupported_keywords(schema: Any) -> None:
    """[overlay] 就地移除 schema 里 provider/提示词不支持的关键字（如 default）。"""
    if not isinstance(schema, dict):
        return
    for keyword in _UNSUPPORTED_SCHEMA_KEYWORDS:
        schema.pop(keyword, None)
    for value in schema.values():
        if isinstance(value, dict):
            _strip_unsupported_keywords(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strip_unsupported_keywords(item)


def _zero_value(prop_schema: Mapping[str, Any]) -> Any:
    """[overlay] 按声明类型给出安全零值。"""
    ptype = prop_schema.get("type") if isinstance(prop_schema, Mapping) else None
    match ptype:
        case "array":
            return []
        case "object":
            return {}
        case "integer" | "number":
            return 0
        case "boolean":
            return False
        case _:
            return ""


def _complete_value(
    prop_schema: Mapping[str, Any],
    value: Any,
    path: str,
    degraded: list[str],
) -> Any:
    """[overlay] 按 schema 递归补全单个节点；缺失字段记入 degraded。"""
    schema = prop_schema if isinstance(prop_schema, Mapping) else {}
    props = schema.get("properties")
    is_object = schema.get("type") == "object" or isinstance(props, Mapping)
    stype = schema.get("type")

    # 缺失或显式 null：优先 default，其次按类型补零（对象递归补全其子字段）
    if value is _MISSING or value is None:
        if "default" in schema:
            degraded.append(path)
            return deepcopy(schema["default"])
        if is_object:
            obj: dict[str, Any] = {}
            for key, child_schema in (props or {}).items():
                child_path = f"{path}.{key}" if path else key
                obj[key] = _complete_value(child_schema, _MISSING, child_path, degraded)
            if not props:
                degraded.append(path)
            return obj
        degraded.append(path)
        return _zero_value(schema)

    # 对象已存在：逐字段补全，保留模型额外产出的键
    if is_object and isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child_schema in (props or {}).items():
            child_path = f"{path}.{key}" if path else key
            if key in value:
                out[key] = _complete_value(child_schema, value[key], child_path, degraded)
            else:
                out[key] = _complete_value(child_schema, _MISSING, child_path, degraded)
        for key, val in value.items():
            if key not in (props or {}):
                out[key] = val
        return out

    # 数组已存在：按 items schema 逐项补全
    if stype == "array" and isinstance(value, list):
        items_schema = schema.get("items") or {}
        return [
            _complete_value(items_schema, item, f"{path}[{idx}]", degraded)
            for idx, item in enumerate(value)
        ]

    return value


def _apply_schema_defaults(
    structured_output: Mapping[str, Any],
    json_schema: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """[overlay] 按 JSON Schema 补全结构化输出的缺失字段。

    - 有 ``default`` 用 ``default``，否则补类型零值（``""`` / ``[]`` / ``{}`` / ``0``）；
    - 对象/数组递归补全，保证下游任意层级取值都不会缺 key；
    - 被兜底的字段路径写入保留键 ``__degraded__``（无缺失时为 ``[]``）。
    """
    if not isinstance(json_schema, Mapping):
        return structured_output
    props = json_schema.get("properties")
    if not isinstance(props, Mapping) or not props:
        # 无 schema 或未声明字段：不兜底，保持上游行为
        return structured_output

    degraded: list[str] = []
    completed = _complete_value(json_schema, dict(structured_output), "", degraded)
    if not isinstance(completed, dict):
        return structured_output
    completed[_DEGRADED_KEY] = degraded
    return completed


def _parse_structured_output(
    result_text: str,
    json_schema: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    # [overlay] 解析前剥离思维链，避免 json_repair 误取思维链片段
    result_text = _strip_reasoning_blocks(result_text)
    structured_output: Mapping[str, Any] = {}
    parsed: Mapping[str, Any] = {}
    try:
        parsed = TypeAdapter(Mapping).validate_json(result_text)
        if not isinstance(parsed, dict):
            raise OutputParserError(f"Failed to parse structured output: {result_text}")
        structured_output = parsed
    except ValidationError:
        # if the result_text is not a valid json, try to repair it
        temp_parsed = json_repair.loads(result_text)
        match temp_parsed:
            case dict():
                pass
            case list():
                # handle reasoning model like deepseek-r1 got '<think>\n\n</think>\n' prefix
                temp_parsed = next((item for item in temp_parsed if isinstance(item, dict)), {})
            case _:
                raise OutputParserError(f"Failed to parse structured output: {result_text}")
        structured_output = cast(dict, temp_parsed)
    # [overlay] 按 schema 补全缺失字段 + 标注兜底字段
    return _apply_schema_defaults(structured_output, json_schema)


def _prepare_schema_for_model(provider: str, model_schema: AIModelEntity, schema: Mapping):
    """
    Prepare JSON schema based on model requirements.

    Different models have different requirements for JSON schema formatting.
    This function handles these differences.

    :param schema: The original JSON schema
    :return: Processed schema compatible with the current model
    """

    # Deep copy to avoid modifying the original schema
    processed_schema = dict(deepcopy(schema))

    # [overlay] 剥离 default 等 provider 不支持的关键字（仅本地解析端使用 default）
    _strip_unsupported_keywords(processed_schema)

    # Convert boolean types to string types (common requirement)
    convert_boolean_to_string(processed_schema)

    # Apply model-specific transformations
    if SpecialModelType.GEMINI in model_schema.model:
        remove_additional_properties(processed_schema)
        return processed_schema
    elif SpecialModelType.OLLAMA in provider:
        return processed_schema
    else:
        # Default format with name field
        return {"schema": processed_schema, "name": "llm_response"}


def remove_additional_properties(schema: dict[str, Any]) -> None:
    """
    Remove additionalProperties fields from JSON schema.
    Used for models like Gemini that don't support this property.

    :param schema: JSON schema to modify in-place
    """
    if not isinstance(schema, dict):
        return

    # Remove additionalProperties at current level
    schema.pop("additionalProperties", None)

    # Process nested structures recursively
    for value in schema.values():
        match value:
            case dict():
                remove_additional_properties(value)
            case list():
                for item in value:
                    if isinstance(item, dict):
                        remove_additional_properties(item)


def convert_boolean_to_string(schema: dict[str, Any]) -> None:
    """
    Convert boolean type specifications to string in JSON schema.

    :param schema: JSON schema to modify in-place
    """
    if not isinstance(schema, dict):
        return

    # Check for boolean type at current level
    if schema.get("type") == "boolean":
        schema["type"] = "string"

    # Process nested dictionaries and lists recursively
    for value in schema.values():
        match value:
            case dict():
                convert_boolean_to_string(value)
            case list():
                for item in value:
                    if isinstance(item, dict):
                        convert_boolean_to_string(item)
