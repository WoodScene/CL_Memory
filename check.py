#!/usr/bin/env python3
"""
精确检查 Qwen3-0.6B 的 attention / mlp 模块名称
确定可用的 lora_target_modules
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_qwen_modules(base_model_path="/workspace/Qwen3-0.6B"):
    print(f"[INFO] Loading model: {base_model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print("\n================ MODEL MODULE STRUCTURE ================\n")

    attention_modules = set()
    mlp_modules = set()

    for name, module in model.named_modules():

        # Qwen3 的注意力层一般命名为 self_attn
        if "self_attn" in name:
            # 记录最末尾的模块名
            parts = name.split(".")
            mod = parts[-1]
            attention_modules.add(mod)

        # Qwen3 MLP 层一般叫 mlp
        if "mlp" in name:
            parts = name.split(".")
            mod = parts[-1]
            mlp_modules.add(mod)

    print("🔹 Attention 下的所有模块名：")
    for m in sorted(attention_modules):
        print("   -", m)

    print("\n🔹 MLP 下的模块：")
    for m in sorted(mlp_modules):
        print("   -", m)

    # 检查第一层
    print("\n================ LAYER 0 MODULES ================\n")
    for name, _ in model.named_parameters():
        if "layers.0" in name:
            print("   ", name)

    print("\n================ 推断可用的 LoRA target_modules ================\n")

    # Qwen3 常见 attention 模块（正确 LoRA target）
    qwen_candidates = ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]

    found = [m for m in attention_modules if m in qwen_candidates]

    if found:
        print("✅ 推荐使用的 LoRA target_modules：")
        print("   ", found)
    else:
        print("⚠ 未找到 Qwen3 常见模块，请人工检查 attention modules：")
        print("   ", attention_modules)

    return attention_modules, mlp_modules


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Qwen3-0.6B"
    check_qwen_modules(base)
