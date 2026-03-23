import json, sys
path = sys.argv[1]
with open(path) as f:
    meta = json.load(f)
keys = list(meta["state_dict"].keys())
print(f"Total tensors: {len(keys)}")
marlin_keys = [k for k in keys if "marlin" in k.lower()]
print(f"Marlin keys: {len(marlin_keys)}")
for k in keys:
    if "gate_proj.weight_packed" in k:
        info = meta["state_dict"][k]
        print(f"  {k}: shape={info['shape']}, dtype={info['dtype']}, bytes={info['byte_size']}")
        break
for k in keys:
    if "gate_proj.weight_scale" in k:
        info = meta["state_dict"][k]
        print(f"  {k}: shape={info['shape']}, dtype={info['dtype']}, bytes={info['byte_size']}")
        break
