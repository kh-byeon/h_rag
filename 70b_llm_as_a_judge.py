import json
from vllm import LLM, SamplingParams

# Llama-3 등 70B 모델은 영문 Instruction을 훨씬 더 정확하게 따릅니다.
JUDGE_PROMPT_TEMPLATE = """You are an expert judge evaluating the performance of a legal contract (CUAD) QA system from a senior lawyer's perspective.
[Question]: {query}
[Query Aspects (Evaluation Criteria)]:
{aspects}

[Answer A (L2: Pure Distance-based)]:
{answer_l2}

[Answer B (MMR: Diverse Functional Roles)]:
{answer_mmr}

Evaluate the two answers based on the 5 criteria below on a scale of 1 to 5, and determine the winner. Give significant weight to criterion 5.
1. Comprehensiveness (Is it detailed and information-dense?)
2. Conciseness (Is it direct and free of redundancy?)
3. Faithfulness (Is it strictly grounded in the legal facts?)
4. Aspect Coverage (Does it accurately cover all specific scopes/obligations requested?)
5. Legal Advisory Utility (Does it identify potential risks or hidden conditions like assignment, termination, or survival clauses from the supplemental context to assist a lawyer's review?)

You MUST respond strictly in the following JSON format without any markdown blocks or extra text:
{"winner": "A|B|TIE", "reason": "...", "scores": {"Answer_A": {"comprehensiveness": 0, "conciseness": 0, "faithfulness": 0, "aspect_coverage": 0, "legal_utility": 0}, "Answer_B": {"comprehensiveness": 0, "conciseness": 0, "faithfulness": 0, "aspect_coverage": 0, "legal_utility": 0}}}
"""

def main():
    input_file = "generation_eval_details.jsonl"
    output_file = "scored_results_70B.jsonl"
    
    # 1. JSONL 데이터 로드
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    print(f"Loaded {len(records)} queries for evaluation.")

    # 2. 프롬프트 생성
    prompts = []
    for record in records:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=record["query"],
            aspects=record["aspect_queries_formatted"],
            answer_l2=record["answer_l2"],
            answer_mmr=record["answer_mmr"]
        )
        prompts.append(prompt)

    # 3. vLLM 모델 로드 (GPU 환경에 맞게 tensor_parallel_size 조절 필요)
    # 예: A100/RTX4090 2장 이상 사용 시 tensor_parallel_size=2
    print("Loading 70B model via vLLM...")
    llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct", tensor_parallel_size=2, trust_remote_code=True)
    
    # JSON 출력을 강제하기 위한 온도 낮춤
    sampling_params = SamplingParams(temperature=0.1, max_tokens=500)

    # 4. 병렬 추론 실행
    print("Generating judgments...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. 결과 파싱 및 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for record, output in zip(records, outputs):
            raw_response = output.outputs[0].text.strip()
            
            try:
                # LLM이 혹시라도 마크다운(```json)을 붙였을 경우를 대비한 안전장치
                clean_json = raw_response.replace("```json", "").replace("```", "").strip()
                record["judge_result_70B"] = json.loads(clean_json)
            except json.JSONDecodeError:
                print(f"[Warning] Failed to parse JSON for idx {record.get('idx')}")
                record["judge_result_70B"] = {"error": "JSON parse failed", "raw": raw_response}
                
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Evaluation complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()