#!/usr/bin/env python3
"""Overlay 兼容性校验（升级上游前必跑）。

作用：确认二次开发覆盖层（docker/overlay/*）在目标上游版本上仍然适用，
      挡住"升级后补丁静默失效 -> 线上事故"。

校验项（每个 overlay 文件）：
  1. 目标版本仍存在该文件；
  2. manifest 里声明的 anchors（补丁挂载点）在目标版本中仍存在 —— 缺失即 ERROR；
  3. 补丁标记 patch_marker 尚未出现在目标版本 —— 出现即 WARN（上游可能已官方修复，可移除该 overlay）；
  4. overlay 文件本身可 Python 编译（.py）。

用法：
  python3 docker/overlay/verify-overlay.py                    # 校验当前 base_version
  python3 docker/overlay/verify-overlay.py --target upstream/1.18.0
  python3 docker/overlay/verify-overlay.py --target 1.18.0

退出码：0 = 通过（可升级）；非 0 = 有 ERROR（禁止升级，先改 overlay）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "overlay.manifest.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=HERE, capture_output=True, text=True, check=False
    ).stdout


def _repo_root() -> Path:
    out = _git("rev-parse", "--show-toplevel").strip()
    if not out:
        print("✗ 不在 git 仓库内，无法校验", file=sys.stderr)
        sys.exit(2)
    return Path(out)


def _show(ref: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=HERE, capture_output=True, text=True, check=False
    )
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="目标上游 ref，如 upstream/1.18.0 或 1.18.0；默认 upstream/<base_version>")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    target = args.target or manifest["base_version"]

    def _resolve(ref: str) -> str | None:
        for cand in (ref, f"upstream/{ref}", f"origin/{ref}"):
            if _git("rev-parse", "--verify", f"{cand}^{{commit}}").strip():
                return cand
        return None

    resolved = _resolve(target)
    if resolved is None:
        print(f"✗ 目标 ref 不存在：{target}（先 git fetch upstream --tags）", file=sys.stderr)
        return 2
    target = resolved

    print(f"Overlay 兼容校验：base={manifest['base_version']}  target={target}\n")
    errors: list[str] = []
    warns: list[str] = []

    for f in manifest["files"]:
        path = f["path"]
        print(f"• {path}")
        local = HERE / path
        upstream = _show(target, path)

        if upstream is None:
            errors.append(f"{path}: 目标版本不存在该文件（路径可能已变）")
            print("    ✗ 目标版本不存在该文件")
            continue
        if not local.exists():
            errors.append(f"{path}: overlay 文件缺失")
            print("    ✗ overlay 文件缺失")
            continue

        # 决定性校验：overlay 是「整文件 COPY」覆盖，若目标版本该文件相对基线已变，
        # 覆盖会抹掉上游改动 -> 必须先 rebase 补丁再升级。
        base_sha = f.get("base_sha256")
        if base_sha:
            up_sha = hashlib.sha256(upstream.encode("utf-8")).hexdigest()
            if up_sha == base_sha:
                print("    ✓ 目标版本该文件与基线一致（可整文件覆盖）")
            else:
                errors.append(
                    f"{path}: 目标版本该文件已变更（base={base_sha[:12]}… target={up_sha[:12]}…），"
                    "overlay 整文件覆盖会抹掉上游改动，须先把补丁 rebase 到新版本再升级"
                )
                print(f"    ✗ 目标版本该文件已变更（{base_sha[:12]}… -> {up_sha[:12]}…）")

        for anchor in f.get("anchors", []):
            if anchor not in upstream:
                errors.append(f"{path}: 挂载点缺失 -> {anchor!r}")
                print(f"    ✗ 挂载点缺失: {anchor!r}")
            else:
                print(f"    ✓ 挂载点存在: {anchor[:60]!r}")

        marker = f.get("patch_marker")
        if marker and marker in upstream:
            warns.append(f"{path}: 目标版本已含补丁标记 {marker!r}（上游或已修复，考虑移除该 overlay）")
            print(f"    ⚠ 目标版本已含补丁标记 {marker!r} —— 上游可能已修复")

        if path.endswith(".py"):
            try:
                py_compile.compile(str(local), doraise=True)
                print("    ✓ overlay 文件可编译")
            except py_compile.PyCompileError as e:
                errors.append(f"{path}: overlay 文件编译失败 {e}")
                print(f"    ✗ 编译失败: {e}")

    print()
    for w in warns:
        print(f"⚠ WARN  {w}")
    for e in errors:
        print(f"✗ ERROR {e}")
    if errors:
        print(f"\n结果：不通过（{len(errors)} 个 ERROR）—— 禁止升级，先修正 overlay。")
        return 1
    print(f"\n结果：通过（{len(warns)} 个 WARN）—— 可继续构建镜像并升级。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
