from pathlib import Path

root_dir = Path("/data3/user25/dataset-make/gait/OpenGait/data/sustech1k/SUSTech1K-Released-pkl")

# 使用 set 自动去重
second_level_names = set()

# 第一层目录，例如 /data3/user25/lrgait/000
for first_level in root_dir.iterdir():
    if not first_level.is_dir():
        continue

    # 第二层目录，例如 /data3/user25/lrgait/000/00-nm
    for second_level in first_level.iterdir():
        if second_level.is_dir():
            second_level_names.add(second_level.name)

# 排序后输出
second_level_names = sorted(second_level_names)

print(f"二级子目录名称数量（去重后）：{len(second_level_names)}")
for name in second_level_names:
    print(name)
