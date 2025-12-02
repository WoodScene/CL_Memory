# fwt的计算公式是：遍历所有task，at,t - a0,t， 求平均
# at,t 在计算bwt的时候已经有了，这里重新读一遍；a0,t 是单 task 微调的结果

import os
import sys
import json
from glob import glob
import argparse
import re

from utils.dataset_order import get_dataset_order
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer

# 对 dataset_id=7，分类任务列表（对齐 eval_bwt_superni.py）
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
    return compute_rouge_l_multiple(ground_truth_list, predictions_list)


def cal_accuracy(ground_truth_list, predictions_list):
    assert len(ground_truth_list) == len(predictions_list)
    return accuracy_score(ground_truth_list, predictions_list)


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


def get_jga_scores(output_dir, dataset_order, name_flag=False):
    """
    计算一个目录下所有 task 的 JGA（分类用 acc，其它用 ROUGE-L）
    - name_flag=False: 结果文件名为 {service_id}-{task_id}_result.txt（连续训练 at,t）
    - name_flag=True:  结果文件名为 {task_id}_result.txt（单 task 微调 a0,t）
    """
    JGA_list = []
    acc_task_num = 0

    for service_id in range(len(dataset_order)):
        task_id = dataset_order[service_id]

        if name_flag:
            # 单 task 微调的结果：task_id_result.txt
            result_file = os.path.join(output_dir, f"{service_id}-{task_id}_result.txt")
        else:
            # 连续训练的结果：serviceId-taskId_result.txt
            result_file = os.path.join(output_dir, f"{service_id}-{task_id}_result.txt")

        if not os.path.exists(result_file):
            print(f"[ERROR] result file {result_file} not found!")
            sys.exit(1)

        with open(result_file, "r", encoding="utf-8") as f:
            model_results = f.readlines()

        # 测试集路径与 BWT 一致
        testfile_name = f"./data/test/{task_id}.json"
        if not os.path.exists(testfile_name):
            print(f"[ERROR] test file {testfile_name} not found!")
            sys.exit(1)

        test_lines = json.load(open(testfile_name, "r", encoding="utf-8"))

        print(f"scoring {result_file}")
        ground_truth_list = []
        predictions_list = []

        for idx_ in range(len(test_lines)):
            ground_truth = test_lines[idx_]['output']
            result_line = model_results[idx_].strip().lower()

            # 解析 "['xxx']" 这种字符串
            prediction = result_line.split("|||")[-1]
            prediction = eval(prediction)[0]

            if "</s>" in prediction:
                prediction = prediction.replace("</s>", "")

            # id 对齐检查
            sample_id = result_line.split("|||")[0]
            if test_lines[idx_]['id'].lower() != sample_id:
                print("行没对齐！")
                print(sample_id, test_lines[idx_]['id'])
                sys.exit(1)

            # 分类任务 label 处理
            if task_id in acc_task_list:
                prediction = extract_class_label(prediction, task_id=task_id)

            ground_truth_list.append(ground_truth.lower())
            predictions_list.append(prediction.lower())

        # 分类任务：acc；其他：ROUGE-L
        if task_id in acc_task_list:
            acc_task_num += 1
            joint_score = cal_accuracy(ground_truth_list, predictions_list)
            print(f"[ACC] task {task_id} accuracy = {joint_score:.4f}")
        else:
            joint_score = cal_rouge_score(ground_truth_list, predictions_list)
            print(f"[ROUGE-L] task {task_id} rougeL = {joint_score:.4f}")

        JGA_list.append(joint_score)

    print(f"acc task number is {acc_task_num}")
    return JGA_list


def main(args):
    dataset_order = get_dataset_order(args.dataset_id)
    print(f"dataset_order: {dataset_order}")

    # 公式中的第一部分：at,t（连续训练的结果）
    output_dir = os.path.join("./output", args.test_data_name)
    if not os.path.exists(output_dir):
        print(f"[ERROR] results dir {output_dir} not found!")
        sys.exit(1)

    # 公式中的第二部分：a0,t（单 task 微调的结果）
    output_dir2 = os.path.join("./output", args.test_data_name2)
    if not os.path.exists(output_dir2):
        print(f"[ERROR] results dir2 {output_dir2} not found!")
        sys.exit(1)

    # at,t
    avgPerf_list1 = get_jga_scores(output_dir, dataset_order, name_flag=False)
    print(f"avgPerf_list1 (at,t): {avgPerf_list1}")

    # a0,t
    avgPerf_list2 = get_jga_scores(output_dir2, dataset_order, name_flag=True)
    print(f"avgPerf_list2 (a0,t): {avgPerf_list2}")

    # FWT: at,t - a0,t
    avgPerf_diff = [
        avgPerf_list1[i] - avgPerf_list2[i] for i in range(len(avgPerf_list1))
    ]
    print(f"FWT diff list (at,t - a0,t): {avgPerf_diff}")

    fwt_value = sum(avgPerf_diff) / len(avgPerf_diff)
    print(f"\nAverage FWT is {fwt_value}")
    return fwt_value


if __name__ == '__main__':
    # 默认值和 BWT 脚本对齐，你也可以直接用命令行覆盖
    dataset_id = 7
    model_name = "Qwen3-0.6Blora"
    method_name = "fwt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default=dataset_id, type=int)

    # 连续训练（bwt side）的结果目录名
    parser.add_argument(
        "--test_data_name",
        type=str,
        default=f"{model_name}_{method_name}_dataset_id_{dataset_id}_bwt",
        help="-averaging (at,t: sequential training side)",
    )

    # 单 task 微调（fwt side）的结果目录名
    parser.add_argument(
        "--test_data_name2",
        type=str,
        default=f"{model_name}_{method_name}_dataset_id_{dataset_id}_fwt",
        help="-averaging (a0,t: single-task fine-tuning side)",
    )

    args = parser.parse_args()
    average_JGA = main(args)
    print("\nFinal FWT:", average_JGA)
