# 生成计算averageJGA指标所需要的预测结果

import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.dataset_order import get_dataset_order
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

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
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    testfile_name: str = "",
    testfile_idx: str = "",
    output_file: str = "",
    lora_type: str = "TaSL",
    dataset_id: int = 1, # 1 - 5  5次实验
    method_name: str = "",
    model_type: str = "",
    hyper_para: int = 0, # 超参数的组合
):
    print(f"dataset_id: {dataset_id}")
    print(f"method_name: {method_name}")
    print(f"lora_type: {lora_type}")
    print(f"hyper_para: {hyper_para}")
    # 已知了dataset id，我们就知道最后一个service是什么，然后用他的weight来进行测试
    dataset_order = get_dataset_order(dataset_id)
    last_service_name = dataset_order[-1]
    model_name = base_model.split("/")[-1] + "lora" + str(model_type)

    lora_weights = os.path.join("./checkpoint_files", model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id), str(len(dataset_order)-1)+"-"+last_service_name)


    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)   
    assert (
        lora_weights
    ), "Please specify a --lora_weights, e.g. --lora_weights='xxx'"

    output_dir = os.path.join("./output", model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id)+"_avgPerf")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"lora_weights: {lora_weights}")
    print(f"output_dir: {output_dir}")
    
    
    
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    #print(torch.cuda.is_available())
    #sys.exit(1)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    #print(model.config.use_cache)
    #sys.exit(1)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

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
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        #print(generation_output)
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
        #return tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #return output.split("### Response:")[1].strip()


    # 从这开始遍历15个service，每一个service的结果存在一个文件夹中
    for service_id in range(0, len(dataset_order)):
        print(f"current service name: {dataset_order[service_id]}... begin generating!")
        
        output_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")
        print(f"output filename: {output_file}")
        
        testfile_name = "./data_longsequence_llama/test/" + dataset_order[service_id] + ".json"
        
        print(f"test filename: {testfile_name}")
        

        if not os.path.isfile(output_file): # 如果文件不存在，就新建文件，从头开始写入
            result_out = open(output_file, "w", encoding='utf-8')
            begin_id = 0 # 相当于从头开始获取结果
            print("——————————————————————————————从头开始写入——————————————————————————————")
        else: # 如果文件已经存在了，我们看看已经写了多少行了，需要继续从哪里开始获取结果
            with open(output_file, "r") as f:
                lines = f.readlines()
                begin_id = len(lines)
                f.close()
            print(f"——————————————————————————————从第{begin_id}行开始写入——————————————————————————————")
            result_out = open(output_file, "a", encoding='utf-8')
        

        data = json.load(open(testfile_name)) 
        for idx_ in range(begin_id, len(data)):
            sample = data[idx_]

            Response_list = []

            Response = evaluate(instruction = sample['instruction'], input = sample['input'])
            Response_list.append(Response)

            #print("Input:", input2)
            print("Response list:", Response_list)
            print("Ground truth:", sample['output'])
            print()
            # if "NONE" not in Response:
            #     break
            # if sample['output'] != "NONE":
            #     break
            result_out.write(sample['id'] + "|||" + sample['output']+ "|||" + str(Response_list))
            result_out.write("\n")

            #break
        result_out.close()
        print(f"current service name: {dataset_order[service_id]}... Generate End!")

if __name__ == "__main__":
    fire.Fire(main)
