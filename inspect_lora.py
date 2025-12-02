import sys
import torch
from safetensors.torch import load_file

path = sys.argv[1]
print(f"Loading {path}")

if path.endswith(".safetensors"):
    sd = load_file(path)
else:
    sd = torch.load(path, map_location="cpu")

print(f"Total keys: {len(sd)}")
for k in list(sd.keys())[:50]:
    print(k)
