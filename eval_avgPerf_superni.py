# 计算 average JGA / avgPerf 指标

import os
import sys
import json
from glob import glob
import argparse
import pandas as pd
from utils.dataset_order import get_dataset_order
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
import re

# 哪些任务是分类任务，用 accuracy；其他用 ROUGE-L
acc_task_list = ['task363', 'task875', 'task1687']


def compute_rouge_l_multiple(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)['rougeL'].fmeasure
        for ref, pred in zip(references, predictions)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def cal_rouge_score(ground_truth_list, predictions_list):
    assert len(ground_truth_list) == len(predictions_list)
    rouge_l_score = compute_rouge_l_multiple(ground_truth_list, predictions_list)
    return rouge_l_score


def cal_accuracy(ground_truth_list, predictions_list):
    assert len(ground_truth_list) == len(predictions_list)
    accuracy = accuracy_score(ground_truth_list, predictions_list)
    return accuracy


def extract_class_label(prediction: str, task_id: str = None) -> str:
    """
    从预测字符串中抽取分类标签，支持多种分类任务：
    - task363: P/N -> 返回 P 或 N
    - task1687: p/n -> 返回 p 或 n
    - task875: joy/love/anger/fear/surprise -> 返回完整单词
    """
    text = prediction.lower().strip()
    
    # task875: 只匹配完整的情感单词
    if task_id == "task875":
        emotion_words = ['joy', 'love', 'anger', 'fear', 'surprise']
        for emotion in emotion_words:
            if re.search(r'\b' + emotion + r'\b', text):
                return emotion
        # 如果都没找到，返回原文本（会被标记为错误）
        return text
    
    # task363 和 task1687: 只匹配单字符 P/N/p/n
    single_char_match = re.search(r'\b([PNpn])\b', text)
    if single_char_match:
        char = single_char_match.group(1)
        # task363 需要返回大写，task1687 需要返回小写
        if task_id == "task363":
            return char.upper()
        elif task_id == "task1687":
            return char.lower()
        else:
            return char.lower()
    
    # 如果都没找到，返回原文本（会被标记为错误）
    return text


def main(args):
    dataset_order = get_dataset_order(args.dataset_id)

    # 结果目录，例如：./output/Qwen3-0.6Blora_vanilla_dataset_id_7_avgPerf
    output_dir = os.path.join("./output", args.test_data_name)
    if not os.path.exists(output_dir):
        print(f"results dir {output_dir} not find!")
        sys.exit(1)

    # 如果没有显式指定 service_end_id，就默认用所有任务
    if args.service_end_id is None:
        service_end_id = len(dataset_order) - 1
    else:
        service_end_id = min(args.service_end_id, len(dataset_order) - 1)

    avgPerf_list = []
    acc_task_num = 0

    print(f"evaluate services from 0 to {service_end_id}")
    for service_id in range(0, service_end_id + 1):
        task_id = dataset_order[service_id]
        print(f"task name: {task_id}")

        result_file = os.path.join(
            output_dir, f"{service_id}-{task_id}_result.txt"
        )
        if not os.path.exists(result_file):
            print(f"result file {result_file} not found, stop here.")
            break

        with open(result_file, "r", encoding="utf-8") as f:
            model_results = f.readlines()

        # ✅ 测试集路径改成 ./data/test/{task}.json
        testfile_name = "./data/test/" + task_id + ".json"
        if not os.path.exists(testfile_name):
            print(f"test file {testfile_name} not found!")
            sys.exit(1)

        with open(testfile_name, "r", encoding="utf-8") as f:
            test_lines = json.load(f)

        ground_truth_list = []
        predictions_list = []

        # 保守一些，防止长度不一致导致越界
        num_samples = min(len(test_lines), len(model_results))

        for idx_ in range(num_samples):
            ground_truth = test_lines[idx_]['output']
            result_line = model_results[idx_].rstrip("\n")

            print(idx_)

            # 取 "id|||gt|||['pred']" 里的最后一段，得到 "['pred']"
            raw_pred_str = result_line.split("|||")[-1]

            # 解析 "['xxx']" 变成 "xxx"
            try:
                prediction = eval(raw_pred_str)[0]
            except Exception:
                prediction = raw_pred_str

            prediction = prediction.strip()
            sample_id = result_line.split("|||")[0]

            if "</s>" in prediction.lower():
                prediction = prediction.replace("</s>", "")

            # 对所有分类任务统一做 label 抽取
            if task_id in acc_task_list:
                prediction = extract_class_label(prediction, task_id=task_id)

            if test_lines[idx_]['id'].lower() != sample_id.lower():
                print("行没对齐！")
                print(sample_id, test_lines[idx_]['id'])
                sys.exit(1)

            ground_truth_list.append(ground_truth.lower())
            predictions_list.append(prediction.lower())

        # 分类任务：accuracy；其他：ROUGE-L
        if task_id in acc_task_list:
            acc_task_num += 1
            joint_accuracy = cal_accuracy(ground_truth_list, predictions_list)
            print(f"分类问题的结果是：{joint_accuracy}")
        else:
            joint_accuracy = cal_rouge_score(ground_truth_list, predictions_list)

        avgPerf_list.append(round(joint_accuracy, 4))

    print(f"average Performance is {sum(avgPerf_list) / len(avgPerf_list)}")
    print()
    print(f"acc task number is {acc_task_num}, it should be 3.")
    average_JGA = sum(avgPerf_list) / len(avgPerf_list)
    print(avgPerf_list)

    return average_JGA


if __name__ == '__main__':
    mean_list = []
    for data_id in range(1):
        parser = argparse.ArgumentParser()
        # 所有可能用到的参数
        parser.add_argument("--dataset_id", default=data_id + 7, type=int)

        # ✅ 改成你现在的结果目录名
        parser.add_argument(
            "--test_data_name",
            type=str,
            default="Qwen3-0.6Blora_vanilla_dataset_id_7_avgPerf",
            help="-averaging",
        )

        # ✅ 可选的 service_end_id，只评到第几个 task（比如 0,1,2）
        parser.add_argument(
            "--service_end_id",
            type=int,
            default=2,   # 你现在只生成了 0,1,2 三个任务的结果
        )

        args = parser.parse_args()
        average_JGA = main(args)
        mean_list.append(average_JGA)

    import numpy as np
    print(np.mean(mean_list))
