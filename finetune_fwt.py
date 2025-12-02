# 单独用每个task 训练一下模型之后查看效果，需要用来计算FWT指标的结果

import os
import sys
from typing import List

import fire
import gc
import time
import shutil
import torch
import transformers
import numpy as np
import pandas as pd

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from utils.prompter import Prompter
from utils.dataset_order import get_dataset_order
from utils.load_data import load_current_task_data, load_validation_set

set_seed(42)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_dir: str = "./data",
    output_path: str = "./checkpoint_files",
    cache_dir: str = "/workspace/CL_Memory/cache",
    # training hyperparams
    batch_size: int = 64,          # 64
    micro_batch_size: int = 16,    # 16
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    max_input_length: int = 1024,  # 这里作为 prompt 的 cutoff_len 用
    val_set_size: int = 20,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # Qwen / LLaMA 风格的 target modules
    lora_target_modules: List[str] = (
        ["q_proj", "v_proj"]  # 适配 Qwen3-0.6B
    ),
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # CL hyperparams
    dataset_id: int = 1,  # 1 - 5  5次实验
    task_id: int = 0,     # 这个表示从哪个service开始训练，默认从头开始训练
):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"\n"
            f"Training Qwen3 FWT LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"max_input_length: {max_input_length}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"dataset_id: {dataset_id}\n"
            f"task_id: {task_id}\n"
            f"\n"
        )

    assert base_model, "Please specify a --base_model, e.g. --base_model='/workspace/Qwen3-0.6B'"

    dataset_order = get_dataset_order(dataset_id)

    print(f"current service name: {dataset_order[task_id]}... begin fine tuning!")
    model_name = base_model.split("/")[-1] + "lora"
    output_dir = os.path.join(
        output_path,
        model_name + "_fwt_dataset_id_" + str(dataset_id),
        str(task_id) + "-" + dataset_order[task_id],
    )
    print(f"output_dir: {output_dir}")

    # FWT：每个 task 单独从 base model 训练，不用 memory buffer
    current_data = load_current_task_data(dataset_id, task_id, data_dir, cache_dir)
    data = current_data

    # FWT：每个 task 都从 scratch LoRA 开始
    lora_weights = ""
    print(f"lora_weights: {lora_weights}\n")

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # wandb 开关逻辑（和之前保持一致）
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # ====== Qwen3-0.6B: Causal LM + fp16 ======
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print(f"Qwen tokenizer config: bos={bos}, eos={eos}, pad={pad}")
    # Qwen3 已经内置了 pad_token_id=151643，无需手动修改
    # 只设置 padding_side 即可
    tokenizer.padding_side = "left"

    cutoff_len = max_input_length

    def tokenize(prompt: str, add_eos_token: bool = True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  
        return tokenized_full_prompt

    # ====== LoRA 配置: Causal LM ======
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print("FWT: fine tune Qwen LoRA from scratch for this task!")

    # resume_from_checkpoint: 保留原逻辑，如果你以后想对同一 task 断点续训可以用
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    # ====== 划分 train / val 并做 tokenization ======
    # if val_set_size > 0:
    #     val_set_len = int(len(data) * 10 / 100)  # 10% 做验证
    #     train_val = data.train_test_split(
    #         test_size=val_set_len, shuffle=True, seed=42
    #     )
    #     print(f"训练数据总量：{len(train_val['train'])}")
    #     print(f"验证数据总量：{len(train_val['test'])}")

    #     train_data = (
    #         train_val["train"]
    #         .shuffle()
    #         .map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"]
    #         .shuffle()
    #         .map(generate_and_tokenize_prompt)
    #     )
    # else:
    #     train_data = data.shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    if val_set_size > 0:

        val_data = load_validation_set(data_dir, dataset_id, task_id, cache_dir, model_name)


        val_data = (
            val_data.shuffle(seed=42).map(generate_and_tokenize_prompt)
        )

        train_data = (
            data.shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,   # ✅ Qwen + fp16
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=40 if val_set_size > 0 else None,
            save_steps=40,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # PeftModel.save_pretrained 只会保存 LoRA adapter
    model.save_pretrained(output_dir, safe_serialization=False)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
