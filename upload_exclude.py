
#!/usr/bin/env python3
"""Upload project files smaller than 100 MB to a Git remote."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# 单文件必须严格小于 100 MB
MAX_FILE_BYTES = 10_000_000

# 每批最多约 900 MB
BATCH_BYTES = 90_000_000


def git(repo_dir, *args, input_bytes=None, check=True):
    cmd = ["git"]
    if repo_dir is not None:
        cmd += ["-C", str(repo_dir)]
    cmd += list(args)

    result = subprocess.run(
        cmd,
        input=input_bytes,
        capture_output=True
    )

    if check and result.returncode != 0:
        detail = result.stderr.decode(
            "utf-8", "replace"
        ).strip()
        raise RuntimeError(
            f"Git 命令失败：{detail or result.returncode}"
        )
    return result


def is_excluded(relative, excludes):
    """按项目根目录的相对路径排除文件或整个目录。"""
    name = relative.as_posix()
    return any(
        name == excluded or name.startswith(excluded + "/")
        for excluded in excludes
    )


def normalize_excludes(values):
    """规范化 --exclude，防止把绝对路径或父目录当作排除项。"""
    normalized = []
    for raw in values:
        name = raw.strip().replace("\\", "/")
        while name.startswith("./"):
            name = name[2:]
        if name.startswith("/") or ".." in name.split("/"):
            raise ValueError(
                f"--exclude 必须是项目目录内的相对路径：{raw}"
            )
        name = name.rstrip("/")
        if not name or name == ".":
            raise ValueError(f"无效的 --exclude 路径：{raw}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def scan_project(root, excludes=()):
    selected, oversized = [], []
    symlink_count = 0

    for current, folders, filenames in os.walk(
        root, followlinks=False
    ):
        kept = []

        for name in folders:
            folder = Path(current) / name

            if name == ".git":
                continue
            if is_excluded(folder.relative_to(root), excludes):
                continue
            if folder.is_symlink():
                symlink_count += 1
                continue
            kept.append(name)

        folders[:] = kept

        for name in filenames:
            if name == ".git":
                continue

            file = Path(current) / name

            if is_excluded(file.relative_to(root), excludes):
                continue
            if file.is_symlink():
                symlink_count += 1
                continue
            if not file.is_file():
                continue

            entry = (
                file.relative_to(root),
                file.stat().st_size
            )

            if entry[1] < MAX_FILE_BYTES:
                selected.append(entry)
            else:
                oversized.append(entry)

    selected.sort(key=lambda x: x[0].as_posix())
    oversized.sort(key=lambda x: x[0].as_posix())

    return selected, oversized, symlink_count


def batches(entries):
    batch, size = [], 0

    for item in entries:
        if batch and size + item[1] > BATCH_BYTES:
            yield batch
            batch, size = [], 0

        batch.append(item)
        size += item[1]

    if batch:
        yield batch


def choose_branch(checkout, requested):
    if requested:
        git(checkout, "check-ref-format",
            "--branch", requested)

        found = git(
            checkout,
            "show-ref", "--verify", "--quiet",
            f"refs/remotes/origin/{requested}",
            check=False
        )

        if found.returncode == 0:
            git(
                checkout, "checkout", "-q", "-B",
                requested, f"origin/{requested}"
            )
        elif git(
            checkout, "rev-parse", "--verify",
            "HEAD", check=False
        ).returncode == 0:
            git(
                checkout, "checkout", "-q",
                "-b", requested
            )
        else:
            git(
                checkout, "symbolic-ref",
                "HEAD", f"refs/heads/{requested}"
            )

        return requested, found.returncode != 0

    result = git(
        checkout, "symbolic-ref",
        "--short", "HEAD", check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            "无法识别默认分支，请指定 --branch main"
        )

    branch = result.stdout.decode(
        "utf-8", "replace"
    ).strip()

    return branch, False


def safe_destination(checkout, relative):
    dest = checkout

    for part in relative.parts:
        dest = dest / part
        if dest.is_symlink():
            raise RuntimeError(
                f"目标仓库存在同名符号链接：{relative}"
            )

    return dest


def upload(root, remote, entries, branch_name, message):
    with tempfile.TemporaryDirectory(
        prefix="git_small_upload_"
    ) as temporary:

        checkout = Path(temporary) / "repository"

        print("正在克隆目标仓库……", flush=True)

        git(
            None, "clone", "--quiet", "--",
            remote, str(checkout)
        )

        branch, new_remote_branch = choose_branch(
            checkout, branch_name
        )

        committed = 0

        for group in batches(entries):
            paths = []

            for relative, _ in group:
                original = root / relative

                if original.is_symlink() or not original.is_file():
                    raise RuntimeError(
                        f"源文件已经变化：{relative}"
                    )

                if original.stat().st_size >= MAX_FILE_BYTES:
                    raise RuntimeError(
                        f"扫描后文件变大：{relative}"
                    )

                dest = safe_destination(
                    checkout, relative
                )

                dest.parent.mkdir(
                    parents=True, exist_ok=True
                )

                shutil.copy2(original, dest)

                if dest.stat().st_size >= MAX_FILE_BYTES:
                    raise RuntimeError(
                        f"复制过程中文件变大：{relative}"
                    )

                paths.append(relative)

            # 兼容空格、中文、换行及通配符文件名
            # payload = (
            #     b"\0".join(
            #         os.fsencode(str(path))
            #         for path in paths
            #     ) + b"\0"
            # )

            # git(
            #     checkout,
            #     "--literal-pathspecs",
            #     "add", "-f",
            #     "--pathspec-from-file=-",
            #     "--pathspec-file-nul",
            #     input_bytes=payload
            # )
            
            git(checkout, "add", "-f", "-A", "--", ".")

            diff = git(
                checkout,
                "diff", "--cached", "--quiet",
                check=False
            )

            if diff.returncode == 0:
                print(
                    f"本批 {len(group)} 个文件无变化，跳过。"
                )
                continue

            if diff.returncode != 1:
                raise RuntimeError("检查暂存区失败")

            git(
                checkout,
                "commit", "--quiet",
                "-m", message
            )

            git(
                checkout,
                "push", "--quiet",
                "origin",
                f"HEAD:refs/heads/{branch}"
            )

            committed += 1

            print(
                f"已推送第 {committed} 批"
                f"（本批 {len(group)} 个文件）。",
                flush=True
            )

        # 新分支与默认分支内容相同时仍要创建分支
        if new_remote_branch and committed == 0:
            result = git(
                checkout, "rev-parse",
                "--verify", "HEAD", check=False
            )

            if result.returncode == 0:
                git(
                    checkout,
                    "push", "--quiet",
                    "origin",
                    f"HEAD:refs/heads/{branch}"
                )
                print(f"已创建远端分支：{branch}")

        print(
            f"完成！推送 {committed} 次文件更新，"
            f"处理 {len(entries)} 个符合条件的文件。"
        )


def main():
    parser = argparse.ArgumentParser(
        description="上传所有小于 100 MB 的普通文件"
    )

    parser.add_argument(
        "project_dir",
        type=Path,
        help="本地项目目录"
    )
    parser.add_argument(
        "remote",
        help="Git 仓库地址"
    )
    parser.add_argument(
        "--branch",
        help="目标分支，默认使用远端默认分支"
    )
    parser.add_argument(
        "--message",
        default="Upload project files under 100 MB",
        help="Git 提交信息"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="RELATIVE_PATH",
        help="跳过指定文件或整个目录（相对项目根目录，可多次指定）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不执行上传"
    )

    args = parser.parse_args()
    root = args.project_dir.expanduser().resolve()

    if not root.is_dir():
        parser.error(f"项目目录不存在：{root}")

    try:
        excludes = normalize_excludes(args.exclude)
    except ValueError as exc:
        parser.error(str(exc))

    if excludes:
        print("排除以下文件或目录（相对于项目根目录）：")
        for excluded in excludes:
            print(f"  [排除] {excluded}")

    selected, oversized, skipped_links = scan_project(root, excludes)

    print(f"符合条件：{len(selected)} 个")
    print(f"文件过大：{len(oversized)} 个")
    print(f"跳过符号链接：{skipped_links} 个")

    for path, size in oversized:
        print(f"[跳过] {path} ({size} 字节)")

    if args.dry_run:
        for path, size in selected:
            print(f"[将上传] {path} ({size} 字节)")
        return

    if not selected:
        print("没有符合条件的文件。")
        return

    print(
        "提醒：包括被 .gitignore 忽略的文件，"
        "请确认没有 .env、私钥等敏感内容。"
    )

    upload(
        root,
        args.remote,
        selected,
        args.branch,
        args.message
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)