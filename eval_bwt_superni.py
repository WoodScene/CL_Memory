import os
import sys
import json
from glob import glob
import argparse
from utils.dataset_order import get_dataset_order
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
import re 

# 对 dataset_id=7，只 task363 是分类任务
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


def get_jga_scores_for_dir(output_dir, dataset_order, service_id_list):
    """
    在给定的 output_dir 下，按 service_id_list 计算每个 task 的 JGA/Acc。
    返回一个按 service_id_list 顺序的分数列表。
    """
    JGA_list = []
    acc_task_num = 0

    for service_id in service_id_list:
        task_id = dataset_order[service_id]
        result_file = os.path.join(
            output_dir, f"{service_id}-{task_id}_result.txt"
        )
        if not os.path.exists(result_file):
            print(f"[ERROR] result file {result_file} not found!")
            sys.exit(1)

        model_results = open(result_file, "r", encoding="utf-8").readlines()

        # 测试集路径：./data/test/{task}.json
        testfile_name = f"./data/test/{task_id}.json"
        if not os.path.exists(testfile_name):
            print(f"[ERROR] test file {testfile_name} not found!")
            sys.exit(1)

        test_lines = json.load(open(testfile_name, "r", encoding="utf-8"))

        print(f"scoring {result_file}")
        ground_truth_list = []
        predictions_list = []

        # 按 test 集长度读取
        for idx_ in range(len(test_lines)):
            ground_truth = test_lines[idx_]['output']
            result_line = model_results[idx_].strip().lower()

            # 解析 "['xxx']" 这种字符串
            prediction = result_line.split("|||")[-1]
            prediction = eval(prediction)[0]  # 变成真正的字符串

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

    print(f"acc task number in this dir = {acc_task_num}")
    return JGA_list


def main(args):
    dataset_order = get_dataset_order(args.dataset_id)
    print(f"dataset_order: {dataset_order}")

    output_dir1 = os.path.join("./output", args.test_data_name)
    output_dir2 = os.path.join("./output", args.test_data_name2)

    if not os.path.exists(output_dir1):
        print(f"[ERROR] results dir {output_dir1} not found!")
        sys.exit(1)
    if not os.path.exists(output_dir2):
        print(f"[ERROR] results dir2 {output_dir2} not found!")
        sys.exit(1)

    # 根据 BWT 定义：只到 T-1（最后一个 task 不参与）
    max_bwt_service_id = len(dataset_order) - 2  # 0 ... T-2

    # 找出两个目录中都存在 result.txt 的 service_id，并且不超过 max_bwt_service_id
    common_service_ids = []
    for service_id in range(len(dataset_order)):
        if service_id > max_bwt_service_id:
            break
        task_id = dataset_order[service_id]
        f1 = os.path.join(output_dir1, f"{service_id}-{task_id}_result.txt")
        f2 = os.path.join(output_dir2, f"{service_id}-{task_id}_result.txt")
        if os.path.exists(f1) and os.path.exists(f2):
            common_service_ids.append(service_id)

    if not common_service_ids:
        print("[ERROR] no common result files between two dirs!")
        sys.exit(1)

    print(f"common service ids used for BWT: {common_service_ids}")

    # 分别在两个目录下算 JGA
    avgPerf_list1 = get_jga_scores_for_dir(
        output_dir1, dataset_order, common_service_ids
    )
    print(f"avgPerf_list1 (bwt side): {avgPerf_list1}")

    avgPerf_list2 = get_jga_scores_for_dir(
        output_dir2, dataset_order, common_service_ids
    )
    print(f"avgPerf_list2 (avgPerf side): {avgPerf_list2}")

    # BWT: 后一份结果减前一份结果
    avgPerf_diff = [
        avgPerf_list2[i] - avgPerf_list1[i] for i in range(len(avgPerf_list1))
    ]
    print(f"BWT diff list: {avgPerf_diff}")
    bwt_value = sum(avgPerf_diff) / len(avgPerf_diff)
    print(f"\nBWT is {bwt_value}")

    return bwt_value


if __name__ == '__main__':

    dataset_id = 7
    model_name = "Qwen3-0.6Blora"
    method_name = "vanilla"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default=dataset_id, type=int)
    parser.add_argument(
        "--test_data_name",
        type=str,
        default=f"{model_name}_{method_name}_dataset_id_{dataset_id}_bwt",
        help="-averaging (bwt side)",
    )
    parser.add_argument(
        "--test_data_name2",
        type=str,
        default=f"{model_name}_{method_name}_dataset_id_{dataset_id}_avgPerf",
        help="-averaging (avgPerf side)",
    )

    args = parser.parse_args()
    average_JGA = main(args)
    print("\nFinal BWT:", average_JGA)
