import os
import json
import re
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams

# --- 工具函数 ---
def get_ref_letter(index):
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    return mapping.get(index, "N/A")

def extract_answer(text):
    """多级容错提取"""
    clean_text = text.split("</think>")[-1] if "</think>" in text else text
    res1 = re.search(r"ANSWER:\s*([A-D])", clean_text, re.IGNORECASE)
    if res1: return res1.group(1).upper()
    res2 = re.search(r"correct\s*answer\s*is\s*([A-D])", clean_text, re.IGNORECASE)
    if res2: return res2.group(1).upper()
    res3 = re.findall(r"\b([A-D])\b", clean_text[-50:])
    if res3: return res3[-1].upper()
    return "ERR"

# --- MedMCQA 评测逻辑 (保留本地离线推理与双日志记录) ---
def run_medmcqa_eval(args):
    print(f"📦 正在加载模型: {args.model_path}")
    llm = LLM(model=args.model_path, trust_remote_code=True, gpu_memory_utilization=0.90)
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=1024, 
        stop=["<|im_end|>", "Question:", "User:"]
    )

    raw_data, prompts = [], []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            raw_data.append(item)
            c_str = "\n".join([f"{['A','B','C','D'][i]}. {item['choices'][i]}" for i in range(len(item['choices']))])
            prompt = (
                f"Question: {item.get('question')}\nChoices:\n{c_str}\n\n"
                f"Assistant: <think>\nAnalyzing the medical evidence... Finally, provide the answer in 'ANSWER: X' format."
            )
            prompts.append(prompt)

    if args.sample_num:
        prompts = prompts[:args.sample_num]
        raw_data = raw_data[:args.sample_num]

    print(f"🚀 开始 MedMCQA 测试推理 (样本数: {len(prompts)})...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_duration = time.time() - start_time

    correct_count = 0
    results_to_save, debug_logs = [], []

    for i, output in enumerate(outputs):
        full_content = output.outputs[0].text.strip()
        ref_letter = get_ref_letter(raw_data[i].get('answer_index'))
        pred_letter = extract_answer(full_content)
        
        is_correct = (pred_letter == ref_letter)
        if is_correct: correct_count += 1
        
        results_to_save.append({
            "id": i + 1, "is_correct": is_correct, "ref": ref_letter,
            "pred": pred_letter, "out_len": len(output.outputs[0].token_ids)
        })
        debug_logs.append({
            "id": i + 1, "question": raw_data[i].get('question'),
            "model_output": full_content, "ref": ref_letter, "pred": pred_letter
        })

    # 保存双日志
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for res in results_to_save: f.write(json.dumps(res, ensure_ascii=False) + "\n")
    with open(args.debug_log_path, 'w', encoding='utf-8') as f:
        for log in debug_logs: f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"\n" + "="*25 + " 最终评估报告 " + "="*25)
    print(f"📊 准确率: {correct_count / len(prompts):.2%}")
    print(f"⏲️ 平均速度: {len(prompts) / total_duration:.2f} samples/s")
    print(f"📝 完整 QA 已存入: {args.debug_log_path}")
    print("="*64)

# --- GSM8K 评测逻辑 (保留 API 模式与前 10 例实时监控) ---
def run_gsm8k_eval(args):
    client_vllm = OpenAI(api_key="EMPTY", base_url=args.vllm_url)
    client_judge = OpenAI(api_key=args.judge_key, base_url=args.judge_url)

    results = []
    correct_count, total_tps = 0, 0

    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:args.sample_num]

    print(f"🚀 开始 GSM8K 评测 (样本数: {len(lines)})")

    for i, line in enumerate(tqdm(lines)):
        data = json.loads(line)
        q = data.get('query') or data.get('question')
        ref = data.get('response') or data.get('answer')

        start_t = time.time()
        try:
            completion = client_vllm.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": f"Instruction: Solve this math problem... Question: {q}\nAnswer: Let's calculate."}],
                temperature=0.0, max_tokens=256, stop=["Question:", "<|im_end|>"]
            )
            pred = completion.choices[0].message.content.strip()
            duration = time.time() - start_t
            tps = (len(pred) / 4) / duration if duration > 0 else 0
        except Exception as e:
            pred, tps, duration = f"ERROR: {e}", 0, 0

        judge_res = "错误"
        try:
            j_comp = client_judge.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": f"判断正确/错误: [问题]{q} [标准]{ref} [学生]{pred}"}]
            )
            judge_res = j_comp.choices[0].message.content.strip()
        except: pass

        is_correct = "正确" in judge_res
        if is_correct: correct_count += 1
        total_tps += tps

        if i < 10:
            print(f"\n{'='*10} Case {i+1} {'='*10}\nQ: {q}\nPred: {pred}\nJudge: {judge_res}")

        results.append({"id": i, "is_correct": is_correct, "tps": tps, "prediction": pred})

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n📊 Accuracy: {correct_count/len(lines):.2%} | Avg TPS: {total_tps/len(lines):.2f}")

# --- 参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "medmcqa"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=None)
    
    # MedMCQA 专用
    parser.add_argument("--model_path", type=str, help="本地模型路径")
    parser.add_argument("--debug_log_path", type=str, default="debug.jsonl", help="调试日志路径")
    
    # GSM8K 专用
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--judge_key", type=str)
    parser.add_argument("--judge_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--judge_model", type=str, default="qwen2.5-72b-instruct")

    args = parser.parse_args()

    if args.task == "gsm8k":
        run_gsm8k_eval(args)
    elif args.task == "medmcqa":
        run_medmcqa_eval(args)