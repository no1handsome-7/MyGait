import os
import yaml

seq_path = "/data3/user25/dataset-make/gait/OpenGait/data/sustech1k/SUSTech1K-Released-pkl/0000/00-nm/000"
cfg_path = "configs/gaitoracle/gaitoracle_sustech1k_rgbpc.yaml"

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

data_in_use = cfg["data_cfg"]["data_in_use"]

files = sorted(os.listdir(seq_path))

print("===== raw sorted files =====")
for i, name in enumerate(files):
    use = data_in_use[i] if i < len(data_in_use) else None
    print(f"{i:02d} | use={use} | {name}")

print("\n===== selected files -> ipts index =====")
selected = []
for raw_i, name in enumerate(files):
    if raw_i < len(data_in_use) and data_in_use[raw_i]:
        selected.append((raw_i, name))

for ipt_i, (raw_i, name) in enumerate(selected):
    print(f"ipts[{ipt_i}] <- raw[{raw_i}] {name}")
