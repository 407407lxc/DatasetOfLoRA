import os
import json
import re
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams

# --- 工具函数 ---
def get_ref_letter(item):
    """
    通用参考答案提取
    传入的是整个字典对象 item
    """
    # 如果是 MedMCQA，item 里面会有这个 key
    if isinstance(item, dict) and 'answer_index' in item:
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        return mapping.get(item['answer_index'], "N/A")
    
    # 针对 MathQA，item 里面会有 answer 字段
    if isinstance(item, dict):
        return str(item.get('answer', '')).strip().upper()
    
    return "N/A"

def extract_answer(text):
    """多级容错提取：升级正则表达式支持 [A-E]"""
    # 1. 过滤掉思考过程
    clean_text = text.split("</think>")[-1] if "</think>" in text else text
    
    # 2. 强约束格式：支持 A-E
    res1 = re.search(r"ANSWER:\s*([A-E])", clean_text, re.IGNORECASE)
    if res1: return res1.group(1).upper()
    
    # 3. 结论句式
    res2 = re.search(r"correct\s*answer\s*is\s*([A-E])", clean_text, re.IGNORECASE)
    if res2: return res2.group(1).upper()
    
    # 4. 尾部提取 (针对生成较长且不规范的情况)
    res3 = re.findall(r"\b([A-E])\b", clean_text[-50:])
    if res3: return res3[-1].upper()
    
    return "ERR"


# --- MathQA 评测逻辑 ---
def run_mathqa_eval(args):
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
            # MathQA 通常有 A-E 5个选项
            labels = ['A', 'B', 'C', 'D', 'E']
            choices = item.get('choices', [])
            c_str = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices)) if i < len(labels)])
            
            prompt = (
                f"Question: {item.get('question')}\nChoices:\n{c_str}\n\n"
                f"Assistant: <think>\nStep-by-step mathematical reasoning... Finally, provide the answer in 'ANSWER: X' format."
            )
            prompts.append(prompt)

    if args.sample_num:
        prompts = prompts[:args.sample_num]
        raw_data = raw_data[:args.sample_num]

    print(f"🚀 开始 MathQA 测试推理 (样本数: {len(prompts)})...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_duration = time.time() - start_time

    correct_count = 0
    results_to_save, debug_logs = [], []

    for i, output in enumerate(outputs):
        full_content = output.outputs[0].text.strip()
        ref_letter = get_ref_letter(raw_data[i]) # 使用通用提取
        pred_letter = extract_answer(full_content) # 使用通用提取
        
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

    print(f"\n" + "="*25 + " MathQA 评估报告 " + "="*25)
    print(f"📊 准确率: {correct_count / len(prompts):.2%}")
    print(f"⏲️ 平均速度: {len(prompts) / total_duration:.2f} samples/s")
    print(f"📝 完整 QA 已存入: {args.debug_log_path}")
    print("="*64)

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
        ref_letter = get_ref_letter(raw_data[i])
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
import random
import concurrent.futures
from tqdm import tqdm
import json
import time

def run_gsm8k_eval(args):
    # 1. 加载模型
    print(f"📦 正在本地加载模型进行 GSM8K 测试: {args.model_path}")
    llm = LLM(model=args.model_path, trust_remote_code=True, gpu_memory_utilization=0.90)
    
    # 【优化点 1】增加停止符，防止无限复读
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=256, 
        # 增加多种停止符：除了 im_end，如果输出两个换行或检测到重复输出特征也停止
        stop=["Question:", "<|im_end|>", "\n\n\n", "#### 3\n#### 3"] 
    )
    
    # 2. 准备数据
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:args.sample_num]
    
    questions, refs, prompts = [], [], []
    for line in lines:
        data = json.loads(line)
        q = data.get('query') or data.get('question')
        r = data.get('response') or data.get('answer')
        questions.append(q)
        refs.append(r)
        prompts.append(
            f"Instruction: Solve this math problem concisely within 256 tokens. "
            f"Directly provide steps and the final answer with ####.\n"
            f"Question: {q}\nAnswer: Let's calculate."
        )

    # 3. 批量推理
    print(f"🚀 开始批量推理 (样本数: {len(prompts)})...")
    start_inference = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_duration = time.time() - start_inference
    
    predictions = [output.outputs[0].text.strip() for output in outputs]
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    avg_tps = total_tokens / total_duration if total_duration > 0 else 0

    # 4. 健壮的并行打分逻辑
    print(f"⚖️ 正在并行打分 (含 429 退避机制与内容清洗)...")
    client_judge = OpenAI(api_key=args.judge_key, base_url=args.judge_url)

    def get_judge_score(idx):
        q_item, ref_item, raw_pred = questions[idx], refs[idx], predictions[idx]
        
        # 【优化点 2】清洗 Prediction：截断复读内容
        # 如果模型输出了多个 ####，只保留第一个及其后的数字部分，忽略后面的复读
        clean_pred = raw_pred
        if "####" in raw_pred:
            parts = raw_pred.split("####")
            # 组合：[计算过程] + #### + [第一个数值]
            clean_pred = parts[0] + "#### " + parts[1].split("\n")[0].strip()

        max_retries = 5
        base_delay = 5 
        
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(0.5, 1.5)) 
                # 使用清洗后的 clean_pred 发送给裁判
                j_prompt = f"你是一名严格的数学老师。\n[问题]: {q_item}\n[标准答案]: {ref_item}\n[学生回答]: {clean_pred}\n\n要求：\n1. 学生必须给出最终数字结果。\n2. 如果回答在中途断掉（如没写完 #### 后的数字），一律判为“错误”。\n3. 仅当数值结果一致时判定为“正确”。\n\n只输出正确/错误"
                
                j_comp = client_judge.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": j_prompt}],
                    temperature=0.0
                )
                return j_comp.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                return f"ERROR_JUDGE: {e}"

    # 5. 执行打分
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        judge_results = list(tqdm(executor.map(get_judge_score, range(len(predictions))), total=len(predictions)))

    # 6. 汇总与输出 (逻辑保持不变)
    results = []
    correct_count = 0
    error_api_count = 0

    for i in range(len(lines)):
        res_text = judge_results[i]
        is_correct = "正确" in res_text
        if is_correct: correct_count += 1
        if "ERROR_JUDGE" in res_text: error_api_count += 1
        
        if i < 10:
            print(f"\n{'='*20} Case {i+1} {'='*20}")
            print(f"Q: {questions[i]}\nRef: {refs[i]}\nPred: {predictions[i]}\nJudge: {res_text}")

        results.append({
            "id": i, "question": questions[i], "reference": refs[i], 
            "prediction": predictions[i], "judge_full": res_text, "is_correct": is_correct
        })

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    actual_evaluated = len(lines) - error_api_count
    print(f"\n" + "="*25 + " 评估结果 " + "="*25)
    print(f"📊 总样本数: {len(lines)}")
    print(f"✅ 正确数量: {correct_count}")
    print(f"❌ API 失败: {error_api_count}")
    acc = correct_count/actual_evaluated if actual_evaluated > 0 else 0
    print(f"📈 有效准确率: {acc:.2%}")
    print(f"🚀 推理速度: {avg_tps:.2f} tokens/s")
    print("="*60)


# --- 参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1. 在 choices 中增加 mathqa
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "medmcqa", "mathqa"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=None)
    
    # 本地推理通用参数 (MedMCQA & MathQA 共用)
    parser.add_argument("--model_path", type=str, help="本地模型路径")
    parser.add_argument("--debug_log_path", type=str, default="debug.jsonl", help="调试日志路径")
    
    # GSM8K 专用
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--judge_key", type=str)
    parser.add_argument("--judge_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--judge_model", type=str, default="qwen2.5-72b-instruct")

    args = parser.parse_args()

    # 2. 判定逻辑分支
    if args.task == "gsm8k":
        run_gsm8k_eval(args)
    elif args.task == "medmcqa":
        run_medmcqa_eval(args)
    elif args.task == "mathqa":
        # 新增 mathqa 调用入口
        run_mathqa_eval(args)