import pandas as pd
import json
import os
import argparse
import numpy as np

def process_gsm8k(df):
    """处理 GSM8K 数据集"""
    alpaca_data = []
    for _, row in df.iterrows():
        alpaca_data.append({
            "instruction": "Solve this grade school math problem.",
            "input": row['question'],
            "output": row['answer']
        })
    return alpaca_data

def process_mathqa(df):
    """处理 MathQA 数据集"""
    alpaca_data = []
    for _, row in df.iterrows():
        # MathQA 的选项通常是一个长字符串，例如 "a ) 12 , b ) 15 , c ) 18 ..."
        # 我们直接提取题目和选项
        problem = row.get('Problem', '')
        options = row.get('options', '')
        rationale = row.get('rationale', '')
        correct_answer = row.get('correct', '')

        input_field = f"Question: {problem}\nOptions: {options}"
        
        # 将解题逻辑和最终答案组合在 output 中
        output_field = f"Rationale: {rationale}\nThe correct answer is {correct_answer}."

        alpaca_data.append({
            "instruction": "Solve the following math problem with steps.",
            "input": input_field,
            "output": output_field
        })
    return alpaca_data

def process_mathqa(df):
    """针对用户提供的特定 MathQA 格式进行处理"""
    alpaca_data = []
    for _, row in df.iterrows():
        # 1. 提取原始字段
        question = row.get('question', '')
        reasoning = row.get('reasoning', '')
        choices = row.get('choices', [])
        answer = row.get('answer', '')

        # 2. 清洗 reasoning 中的特殊乱码字符 (可选)
        # 例如将 'â ˆ ’' 替换为 '-'，'ã —' 替换为 '*'
        clean_reasoning = reasoning.replace('â ˆ ’', '-').replace('ã —', '*').replace('â € “', '-')
        
        # 3. 格式化选项 (将列表转换为 A: xxx B: xxx 格式)
        label_list = ["A", "B", "C", "D", "E"]
        options_text = ""
        if isinstance(choices, (list, np.ndarray)):
            options_text = ", ".join([f"{label_list[i]}: {str(c)}" for i, c in enumerate(choices) if i < len(label_list)])
        else:
            options_text = str(choices)

        # 4. 严格按照 Alpaca 结构组合
        input_field = f"Question: {question}\nOptions: {options_text}"
        output_field = f"Rationale: {clean_reasoning}\nThe correct answer is {answer}."

        alpaca_data.append({
            "instruction": "Solve the following math problem with steps.",
            "input": input_field,
            "output": output_field
        })
    return alpaca_data

def main():
    parser = argparse.ArgumentParser(description="Dataset Converter: Parquet -> JSON -> Alpaca")
    
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input parquet file")
    parser.add_argument("--json_raw_path", type=str, required=True, help="Path to save the raw JSON file")
    parser.add_argument("--alpaca_path", type=str, required=True, help="Path to save the Alpaca formatted JSON file")
    # 添加 mathqa 选项
    parser.add_argument("--dataset_name", type=str, choices=["gsm8k", "medmcqa", "mathqa"], required=True, help="Name of the dataset")

    args = parser.parse_args()

    print(f"Loading {args.input_path}...")
    df = pd.read_parquet(args.input_path)

    print(f"Saving raw JSON to {args.json_raw_path}...")
    df.to_json(args.json_raw_path, orient="records", force_ascii=False, indent=4)

    print(f"Converting to Alpaca format for {args.dataset_name}...")
    if args.dataset_name == "gsm8k":
        alpaca_list = process_gsm8k(df)
    elif args.dataset_name == "medmcqa":
        alpaca_list = process_medmcqa(df)
    elif args.dataset_name == "mathqa":
        alpaca_list = process_mathqa(df)

    with open(args.alpaca_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_list, f, ensure_ascii=False, indent=4)
    
    print("Successfully completed!")

if __name__ == "__main__":
    main()