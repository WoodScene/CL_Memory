# 生成计算 average JGA 指标所需要的预测结果

import os
import sys
import json
import fire
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.dataset_order import get_dataset_order
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

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
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    testfile_name: str = "",
    output_file: str = "",
    dataset_id: int = 1,  # 1 - 5  5次实验
    method_name: str = "",
    model_type: str = "",
    hyper_para: int = 0,  # 超参数的组合
    use_latest_checkpoint: bool = False,  # 如果 True，使用最新的 checkpoint 而不是最后一个任务
    max_task_id: int = None,             # 如果指定，使用该 task_id 的 checkpoint
    output_root: str = "./output",       # ⭐ 自定义输出根目录
    lora_checkpoint_path: str = "/workspace/CL_checkpoints_exp_1",  # ⭐ checkpoint 根目录路径
):
    print(f"base_model: {base_model}")
    print(f"dataset_id: {dataset_id}")
    print(f"method_name: {method_name}")
    print(f"hyper_para: {hyper_para}")
    print(f"output_root: {output_root}")
    print(f"lora_checkpoint_path: {lora_checkpoint_path}")

    # ===== 1. 确定使用哪个 checkpoint =====
    dataset_order = get_dataset_order(dataset_id)
    model_name = base_model.split("/")[-1] + "lora" + str(model_type)

    if max_task_id is not None:
        # 使用指定的 task_id
        task_id_to_use = max_task_id
        service_name_to_use = dataset_order[task_id_to_use]
    elif use_latest_checkpoint:
        # 自动查找最新的 checkpoint（task_id 最大的那个）
        checkpoint_base = os.path.join(
            "./checkpoint_files",
            model_name + "_" + method_name + "_dataset_id_" + str(dataset_id),
        )
        if os.path.exists(checkpoint_base):
            max_task_id_found = -1
            for item in os.listdir(checkpoint_base):
                item_path = os.path.join(checkpoint_base, item)
                # 目录名形如 "2-task1290"
                if os.path.isdir(item_path) and item[0].isdigit():
                    try:
                        task_id_tmp = int(item.split("-")[0])
                        if task_id_tmp > max_task_id_found:
                            max_task_id_found = task_id_tmp
                    except Exception:
                        pass
            if max_task_id_found >= 0:
                task_id_to_use = max_task_id_found
                service_name_to_use = dataset_order[task_id_to_use]
                print(
                    f"自动找到最新的 checkpoint: task_id={task_id_to_use}, "
                    f"service={service_name_to_use}"
                )
            else:
                task_id_to_use = len(dataset_order) - 1
                service_name_to_use = dataset_order[-1]
        else:
            task_id_to_use = len(dataset_order) - 1
            service_name_to_use = dataset_order[-1]
    else:
        # 默认使用最后一个任务的 checkpoint
        task_id_to_use = len(dataset_order) - 1
        service_name_to_use = dataset_order[-1]

    # ===== 2. 组装 LoRA checkpoint 路径 =====
    lora_weights = os.path.join(
        lora_checkpoint_path,
        model_name + "_" + method_name + "_dataset_id_" + str(dataset_id),
        f"{task_id_to_use}-{service_name_to_use}",
    )

    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)

    # ========= avgPerf 专用输出结构 =========
    output_dir = os.path.join(
        output_root,
        model_name + "_" + method_name + "_dataset_id_" + str(dataset_id) + "_avgPerf",
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
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
            trust_remote_code=True,
        )
        # 让 PEFT 自动读取 adapter_config.json 里的 target_modules
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
            trust_remote_code=True,
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
            trust_remote_code=True,
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
        try:
            model = torch.compile(model)
        except Exception:
            # 有些 peft + compile 组合不稳定，失败就忽略
            pass

    # ===== 4. 定义推理函数 =====
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

    # ===== 5. 遍历每个 service，生成预测结果 =====
    # 使用到 task_id_to_use 为止（例如只训练到第 2 个任务，就只评 0,1,2）
    max_service_id = (
        task_id_to_use + 1
        if (use_latest_checkpoint or max_task_id is not None)
        else len(dataset_order)
    )

    for service_id in range(0, max_service_id):
        print(
            f"current service name: {dataset_order[service_id]}... begin generating!"
        )

        output_file = os.path.join(
            output_dir, f"{service_id}-{dataset_order[service_id]}_result.txt"
        )
        print(f"output filename: {output_file}")

        # 这里统一用 ./data/test/{task_name}.json
        testfile_name = "./data/test/" + dataset_order[service_id] + ".json"
        print(f"test filename: {testfile_name}")

        if not os.path.isfile(testfile_name):
            print(f"test file {testfile_name} not found!")
            sys.exit(1)

        # 判断是否断点续生成
        if not os.path.isfile(output_file):
            # 文件不存在，从头开始写
            result_out = open(output_file, "w", encoding="utf-8")
            begin_id = 0
            print("———— 从头开始写入 ————")
        else:
            # 文件存在，接着之前的行数继续生成
            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                begin_id = len(lines)
            print(f"———— 从第 {begin_id} 行开始写入 ————")
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
        print(f"current service name: {dataset_order[service_id]}... Generate End!")


if __name__ == "__main__":
    fire.Fire(main)
