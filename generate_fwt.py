# 生成计算 FWT 指标所需要的预测结果
# 这里只需要计算每一个 task 单独微调后的性能就可以了，
# where a0,t refers to the performance of training task t individually

import os
import sys
import json
import fire
import torch
import transformers
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from copy import deepcopy

from utils.dataset_order import get_dataset_order
from utils.prompter import Prompter
from utils.callbacks import Iteratorize, Stream  # 现在没用到，保留也无妨

# ========= 设备选择 =========
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    base_model: str = "",
    lora_weights: str = "",                     # 可选：显式指定权重路径，否则按 FWT 结构推断
    prompt_template: str = "",                  # The prompt template to use, will default to alpaca.
    dataset_id: int = 1,                        # 1 - 5  共 5 次实验
    max_new_tokens: int = 128,
    service_begin_id: int = 0,
    data_dir: str = "./data",                   # 根目录，真实数据在 data_dir/test/*.json
    output_root: str = "./output",              # 输出根目录，下面会拼 FWT 子目录
    method_name: str = "",
    lora_checkpoint_path: str = "/workspace/CL_checkpoints_exp_1",  # ⭐ 新增：checkpoint 根目录路径
):
    print(f"base_model: {base_model}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"dataset_id: {dataset_id}")
    print(f"service_begin_id: {service_begin_id}")
    print(f"data_dir: {data_dir}")
    print(f"output_root: {output_root}")
    print(f"method_name: {method_name}")
    print(f"lora_checkpoint_path: {lora_checkpoint_path}")

    dataset_order = get_dataset_order(dataset_id)
    service_name = dataset_order[service_begin_id]
    model_name = base_model.split("/")[-1] + "lora"   # 对 Qwen3-0.6B 是 Qwen3-0.6Blora

    print(f"model_name: {model_name}")

    # ========= FWT 专用 checkpoint 结构 =========
    # 若未显式传入 lora_weights，则按：
    # /workspace/CL_checkpoints_exp_1/{model_name}_fwtß_dataset_id_{dataset_id}/{service_id}-{service_name}
    if not lora_weights:
        lora_weights = os.path.join(
            lora_checkpoint_path, 
            model_name + "_" + method_name + "_dataset_id_" + str(dataset_id),
            f"{service_begin_id}-{service_name}",
        )

    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)

    # ========= FWT 专用输出结构 =========
    # {output_root}/{model_name}_fwt_dataset_id_{dataset_id}_fwt/{service_name}_result.txt
    output_dir = os.path.join(
        output_root,
        f"{model_name}_fwt_dataset_id_{dataset_id}_fwt",
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
        # 如果这里报错，可以直接注释掉这一行
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


    # ===== 推理当前 service =====
    service_id = service_begin_id
    print(f"current service name: {dataset_order[service_id]}... begin generating!")

    output_file = os.path.join(
        output_dir,
        f"{service_id}-{dataset_order[service_id]}_result.txt",
    )
    print(f"output filename: {output_file}")

    # 测试集路径：data_dir/test/{task}.json
    testfile_name = os.path.join(
        data_dir,
        "test",
        dataset_order[service_id] + ".json",
    )
    print(f"test filename: {testfile_name}")

    if not os.path.isfile(testfile_name):
        print(f"test file {testfile_name} not found!")
        sys.exit(1)

    # 断点续跑逻辑
    if not os.path.isfile(output_file):
        # 如果文件不存在，就新建文件，从头开始写入
        result_out = open(output_file, "w", encoding="utf-8")
        begin_id = 0
        print("———————— 从头开始写入 ————————")
    else:
        # 如果文件已经存在了，看看已经写了多少行，需要从哪里继续
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            begin_id = len(lines)
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
