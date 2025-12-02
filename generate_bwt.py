# 生成计算 BWT 指标所需要的预测结果
# 这里只需要计算第 i 次训练完，在 service i 上的效果就行。另一部分的结果已经计算了。

import os
import sys
import json
import fire
import torch
import transformers
from peft import PeftModel
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.dataset_order import get_dataset_order

# 设备选择
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def main(
    base_model: str = "",
    lora_weights: str = "",
    prompt_template: str = "", 
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
    testfile_name: str = "",
    testfile_idx: str = "",
    output_file: str = "",
    dataset_id: int = 1,
    service_begin_id: int = 0,
    method_name: str = "",
    model_type: str = "",
    output_root: str = "./output",     # ⭐⭐⭐ 新增自定义输出路径
    lora_checkpoint_path: str = "/workspace/CL_checkpoints_exp_1",  # ⭐ 新增：checkpoint 根目录路径
):
    print(f"base_model: {base_model}")
    print(f"dataset_id: {dataset_id}")
    print(f"service_begin_id: {service_begin_id}")
    print(f"method_name: {method_name}")
    print(f"output_root: {output_root}")
    print(f"lora_checkpoint_path: {lora_checkpoint_path}")

    dataset_order = get_dataset_order(dataset_id)
    service_name = dataset_order[service_begin_id]
    model_name = base_model.split("/")[-1] + "lora" + str(model_type)
    print(f"model_name: {model_name}")

    # 对应第 i 次训练完成后的 LoRA 权重： i-taskName
    # lora路径修改
    lora_weights = os.path.join(
        lora_checkpoint_path,
        model_name + "_" + method_name + "_dataset_id_" + str(dataset_id),
        f"{service_begin_id}-{service_name}",
    )

    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)

    # ========= BWT 专用输出结构 =========
    output_dir = os.path.join(
        output_root,
        model_name + "_" + method_name + "_dataset_id_" + str(dataset_id) + "_bwt",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"lora_weights: {lora_weights}")
    print(f"output_dir: {output_dir}")

    # ===== 3. 构建 tokenizer / model =====
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # ---- 打印并确认 tokenizer 的特殊 token 配置 ----
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print(f"[Tokenizer] bos={bos}, eos={eos}, pad={pad}")

    # ---- 兜底处理：如果没有 pad_token，才 fallback 到 eos ----
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Qwen/LLaMA 系列强烈推荐 left padding（训练和推理一致）----
    tokenizer.padding_side = "left"

    # ==================================================
    #  加载模型（注意：不要覆盖 Qwen 的 bos/eos 配置）
    #  只同步 pad 即可！
    # ==================================================

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    
    # ---- 只同步 pad_token_id，保持 Qwen 原生 bos/eos ----
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.02,  
        top_p=0,
        top_k=1,
        num_beams=1,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # ✅ 不再使用 GenerationConfig，直接在 generate 里写参数
        # ✅ 这里选择确定性解码，适合做 FWT/BWT 评估
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(
            s,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return prompter.get_response(output)


    # ===== 只在当前 service 上做推理 =====
    service_id = service_begin_id
    print(f"current service name: {dataset_order[service_id]}... begin generating!")

    output_file = os.path.join(
        output_dir,
        f"{service_id}-{dataset_order[service_id]}_result.txt",
    )
    print(f"output filename: {output_file}")

    # 测试集路径
    testfile_name = "./data/test/" + dataset_order[service_id] + ".json"
    print(f"test filename: {testfile_name}")

    if not os.path.isfile(testfile_name):
        print(f"test file {testfile_name} not found!")
        sys.exit(1)

    # ===== 断点续跑 =====
    if not os.path.isfile(output_file):
        result_out = open(output_file, "w", encoding="utf-8")
        begin_id = 0
        print("———————— 从头开始写入 ————————")
    else:
        with open(output_file, "r", encoding="utf-8") as f:
            begin_id = len(f.readlines())
        print(f"———————— 从第 {begin_id} 行开始写入 ————————")
        result_out = open(output_file, "a", encoding="utf-8")

    data = json.load(open(testfile_name, "r", encoding="utf-8"))
    
    for idx_ in range(begin_id, len(data)):
        sample = data[idx_]

        Response_list = []
        Response = evaluate(
            instruction=sample["instruction"], input=sample["input"]
        )
        Response_list.append(Response)

        print("idx:", idx_)
        print("Response list:", Response_list)
        print("Ground truth:", sample["output"])
        print()

        # ⭐ 核心：把 gold 变成单行并合并多空白
        gold = " ".join(sample["output"].split())

        result_out.write(
            sample["id"] + "|||" + gold + "|||" + str(Response_list)
        )
        result_out.write("\n")


    result_out.close()
    print(f"current service name: {service_name}... Generate End!")


if __name__ == "__main__":
    fire.Fire(main)
