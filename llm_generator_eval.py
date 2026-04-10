"""
검색된 Top-K 문서를 활용한 LLM 생성(Generation) 및 양자화(INT8/INT4) 수준별 성능 평가.
`online_accelerator`가 내보낸 JSON(b1_contexts / b2_contexts / mmr_contexts)을 읽어 로컬 LLM으로 답변 생성 후,
Semantic Similarity(정답에 `[SEP]`이 있으면 조각별 max)·고유 단어 비율·Answer Coverage(Token Recall)·선택적 LLM-as-a-Judge.
생성 프롬프트는 계약 발췌 우선·질문 말미 배치(Lost in the Middle 완화).

`sweep_eval.py`가 저장한 `run_manifest.json` 또는 `retrieval_*.jsonl`을 읽어 파라미터 조합별로
동일한 디렉터리 구조로 저장할 수 있다:
  - llm_generation_detail_<run_id>.jsonl (쿼리별 생성·프롬프트·점수)
  - combos/llm_combo_<idx>.json (조합별 평균·샘플 리스트)
  - llm_eval_summary.csv (조합별 집계)
  - llm_run_manifest.json

필요 패키지:
  pip install transformers accelerate bitsandbytes sentence-transformers scipy
  Judge API: pip install openai  및/또는  pip install google-generativeai
  스윕(모델×양자화): python llm_generator_eval.py --sweep [--eval_data ...] → 조합별 CSV + sweep_summary.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from string import Template
from typing import Any, Dict, List, Optional, Tuple

def _calc_unique_token_ratio(text: str) -> float:
    """생성 텍스트의 유니그램 기준 고유 단어 비율 |V|/N (다양성 휴리스틱)."""
    if not text:
        return 0.0
    tokens = [w.lower() for w in re.findall(r"\w+", text)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# Lost in the Middle 완화: 긴 계약 발췌를 앞에 두고 질문·지시는 끝에 둔다.
PROMPT_TEMPLATE = """You are an expert legal assistant. Read the following contract excerpts carefully.

[Contract Excerpts]
{context}

[Instruction]
Based strictly on the excerpts above, extract the exact clauses or provide a direct answer to the following question. Do not add outside information.

Question: {query}
Answer:"""


def build_generation_prompt(query: str, contexts: List[str]) -> str:
    """LLM에 전달되는 전체 사용자 프롬프트(컨텍스트 포함). 로깅·JSONL 저장용."""
    context_block = "\n\n".join(contexts) if contexts else "(No context)"
    return (
        PROMPT_TEMPLATE.replace("{context}", context_block).replace(
            "{query}", str(query or "")
        )
    )


# ----- 심판 LLM용 프롬프트 ($query / $answer_b1 / $answer_b2 / $answer_mmr 는 Template 치환) -----
JUDGE_PROMPT_TEMPLATE = Template(
    """당신은 법률 계약서(CUAD) QA 시스템의 성능을 평가하는 심판입니다.
[Question]: $query
[Answer A (B1: Single-query Pure L2)]: $answer_b1
[Answer B (B2: Sequential Multi-query Pure L2)]: $answer_b2
[Answer C (MMR: Sequential Multi-query Smart Acceleration)]: $answer_mmr

아래 3가지 항목을 1~5점으로 평가하고 승자를 가려주세요.
1. Comprehensiveness (포괄성 및 정보 밀도)
2. Conciseness (간결성 및 중복 없음)
3. Faithfulness (근거 충실도)

반드시 아래 JSON 형식으로만 응답하세요:
{"winner": "A|B|C|TIE", "reason": "...", "scores": {"Answer_A": {"comprehensiveness": 0, "conciseness": 0, "faithfulness": 0}, "Answer_B": {"comprehensiveness": 0, "conciseness": 0, "faithfulness": 0}, "Answer_C": {"comprehensiveness": 0, "conciseness": 0, "faithfulness": 0}}}
"""
)


def _template_escape(s: str) -> str:
    """string.Template에서 $ 리터럴을 쓰기 위해 $$ 로 이스케이프."""
    return (s or "").replace("$", "$$")


def build_judge_prompt_text(
    query: str, answer_b1: str, answer_b2: str, answer_mmr: str
) -> str:
    """JUDGE_PROMPT_TEMPLATE 을 채운 단일 사용자 프롬프트 문자열."""
    return JUDGE_PROMPT_TEMPLATE.substitute(
        query=_template_escape(query),
        answer_b1=_template_escape(answer_b1),
        answer_b2=_template_escape(answer_b2),
        answer_mmr=_template_escape(answer_mmr),
    )


def parse_judge_json(raw: str) -> Dict[str, Any]:
    """심판 LLM 응답에서 JSON 객체를 추출해 파싱한다."""
    text = (raw or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if m:
            text = m.group(0).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": raw[:2000]}


def evaluate_with_llm_judge(
    query: str,
    answer_b1: str,
    answer_b2: str,
    answer_mmr: str,
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    OpenAI 또는 Google Gemini API로 LLM-as-a-Judge 평가.
    API 키: OPENAI_API_KEY / GOOGLE_API_KEY 또는 GEMINI_API_KEY (os.getenv).
    """
    prompt = build_judge_prompt_text(query, answer_b1, answer_b2, answer_mmr)
    raw = ""
    provider = (provider or "openai").lower().strip()

    try:
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                return {
                    "winner": "TIE",
                    "reason": "OPENAI_API_KEY 미설정",
                    "scores": {},
                    "_parse_error": True,
                    "_api_error": "missing OPENAI_API_KEY",
                }
            from openai import OpenAI

            client = OpenAI(api_key=key)
            mdl = model or os.getenv("OPENAI_JUDGE_MODEL") or "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=mdl,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            raw = (resp.choices[0].message.content or "").strip()

        elif provider in ("gemini", "google"):
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                return {
                    "winner": "TIE",
                    "reason": "GOOGLE_API_KEY 또는 GEMINI_API_KEY 미설정",
                    "scores": {},
                    "_parse_error": True,
                    "_api_error": "missing GOOGLE_API_KEY/GEMINI_API_KEY",
                }
            import google.generativeai as genai

            genai.configure(api_key=key)
            mdl = model or os.getenv("GEMINI_JUDGE_MODEL") or "gemini-1.5-flash"
            gmodel = genai.GenerativeModel(mdl)
            resp = gmodel.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            try:
                raw = (resp.text or "").strip()
            except Exception:
                raw = ""
                if getattr(resp, "candidates", None):
                    parts = resp.candidates[0].content.parts
                    raw = "".join(getattr(p, "text", "") for p in parts).strip()

        else:
            return {
                "winner": "TIE",
                "reason": f"지원하지 않는 judge provider: {provider}",
                "scores": {},
                "_parse_error": True,
                "_api_error": "bad provider",
            }

    except Exception as e:
        return {
            "winner": "TIE",
            "reason": f"API 오류: {e}",
            "scores": {},
            "_parse_error": True,
            "_api_error": str(e),
            "_raw_response": raw,
        }

    parsed = parse_judge_json(raw)
    ok = (
        isinstance(parsed, dict)
        and parsed.get("winner") is not None
        and not parsed.get("_parse_error")
    )
    if ok:
        parsed["_raw_response"] = raw
        parsed["_judge_provider"] = provider
        return parsed
    return {
        "winner": "TIE",
        "reason": "JSON 파싱 실패 또는 형식 불일치",
        "scores": {},
        "_parse_error": True,
        "_raw_response": raw,
        "_judge_provider": provider,
    }


def run_llm_as_judge(
    model: Any,
    tokenizer: Any,
    query: str,
    answer_a: str,
    answer_b: str,
    answer_c: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """로컬 transformers 모델로 심판 (Answer A=B1, B=B2, C=MMR)."""
    prompt = build_judge_prompt_text(query, answer_a, answer_b, answer_c)
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt_tok = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_tok = prompt
    else:
        prompt_tok = prompt

    inputs = tokenizer(prompt_tok, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    parsed = parse_judge_json(raw)
    ok = (
        isinstance(parsed, dict)
        and parsed.get("winner") is not None
        and not parsed.get("_parse_error")
    )
    if ok:
        parsed["_raw_response"] = raw
        parsed["_judge_provider"] = "local"
        return parsed
    return {
        "winner": "TIE",
        "reason": "JSON 파싱 실패 또는 형식 불일치",
        "scores": {},
        "_parse_error": True,
        "_raw_response": raw,
        "_judge_provider": "local",
    }


def aggregate_judge_stats(judge_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """winner 집계(A=B1, B=B2, C=MMR) 및 평균 점수 요약."""
    n = len(judge_results)
    if n == 0:
        return {}

    wins_b1 = wins_b2 = wins_mmr = ties = 0
    sum_a = {"comprehensiveness": 0.0, "conciseness": 0.0, "faithfulness": 0.0}
    sum_b = {"comprehensiveness": 0.0, "conciseness": 0.0, "faithfulness": 0.0}
    sum_c = {"comprehensiveness": 0.0, "conciseness": 0.0, "faithfulness": 0.0}
    count_scores = 0

    for jr in judge_results:
        w = str(jr.get("winner") or "TIE").strip().upper()
        if "|" in w:
            w = w.split("|")[0].strip()
        if w == "A":
            wins_b1 += 1
        elif w == "B":
            wins_b2 += 1
        elif w == "C":
            wins_mmr += 1
        else:
            ties += 1
        scores = jr.get("scores") or {}
        sa = scores.get("Answer_A") or scores.get("answer_a") or {}
        sb = scores.get("Answer_B") or scores.get("answer_b") or {}
        sc = scores.get("Answer_C") or scores.get("answer_c") or {}
        if (
            isinstance(sa, dict)
            and isinstance(sb, dict)
            and isinstance(sc, dict)
            and sa
            and sb
            and sc
        ):
            for k in sum_a:
                try:
                    sum_a[k] += float(sa.get(k, 0) or 0)
                    sum_b[k] += float(sb.get(k, 0) or 0)
                    sum_c[k] += float(sc.get(k, 0) or 0)
                except (TypeError, ValueError):
                    pass
            count_scores += 1

    out: Dict[str, Any] = {
        "judge_n": n,
        "judge_wins_b1": wins_b1,
        "judge_wins_b2": wins_b2,
        "judge_wins_mmr": wins_mmr,
        "judge_ties": ties,
        "judge_win_rate_b1": wins_b1 / n,
        "judge_win_rate_b2": wins_b2 / n,
        "judge_win_rate_mmr": wins_mmr / n,
        "judge_tie_rate": ties / n,
    }
    if count_scores:
        for k in sum_a:
            out[f"judge_avg_A_{k}"] = sum_a[k] / count_scores
            out[f"judge_avg_B_{k}"] = sum_b[k] / count_scores
            out[f"judge_avg_C_{k}"] = sum_c[k] / count_scores
    return out


def load_eval_data_json(path: str) -> List[Dict[str, Any]]:
    """JSON 파일을 eval_data 리스트로 로드 (리스트 또는 {"items": [...]} 형식 지원)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("items", "data", "queries", "eval_data", "results"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return [data]
    return []


def load_sweep_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_retrieval_jsonl_records(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_artifact_root_and_jsonl(
    sweep_manifest: Optional[str],
    retrieval_jsonl: Optional[str],
) -> Tuple[str, str, str]:
    """
    sweep_manifest 또는 retrieval_jsonl로부터 (artifact_root, jsonl_path, run_id)를 구한다.
    """
    if retrieval_jsonl:
        jsonl_path = os.path.abspath(retrieval_jsonl)
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(jsonl_path)
        artifact_root = os.path.dirname(jsonl_path)
        base = os.path.basename(jsonl_path)
        m = re.match(r"retrieval_(.+)\.jsonl$", base, re.I)
        run_id = m.group(1) if m else "unknown"
        if run_id == "unknown":
            recs = load_retrieval_jsonl_records(jsonl_path)
            if recs and recs[0].get("run_id"):
                run_id = str(recs[0]["run_id"])
        return artifact_root, jsonl_path, run_id

    if sweep_manifest:
        man_path = os.path.abspath(sweep_manifest)
        if not os.path.isfile(man_path):
            raise FileNotFoundError(man_path)
        manifest = load_sweep_manifest(man_path)
        artifact_root = os.path.abspath(
            manifest.get("artifact_root") or os.path.dirname(man_path)
        )
        jsonl_path = manifest.get("jsonl_path")
        run_id = str(manifest.get("run_id") or "")
        if not jsonl_path:
            jsonl_path = os.path.join(artifact_root, f"retrieval_{run_id}.jsonl")
        else:
            jsonl_path = os.path.abspath(jsonl_path)
        if not run_id:
            base = os.path.basename(jsonl_path)
            m2 = re.match(r"retrieval_(.+)\.jsonl$", base, re.I)
            run_id = m2.group(1) if m2 else "unknown"
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(jsonl_path)
        return artifact_root, jsonl_path, run_id

    raise ValueError("retrieval_jsonl 또는 sweep_manifest 중 하나가 필요합니다.")


def group_retrieval_by_sweep_idx(
    records: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    g: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        sid = int(r.get("sweep_idx", -1))
        if sid < 0:
            continue
        g.setdefault(sid, []).append(r)
    for sid in g:
        g[sid].sort(key=lambda x: int(x.get("query_idx", 0)))
    return dict(sorted(g.items()))


def retrieval_record_to_eval_item(r: Dict[str, Any]) -> Dict[str, Any]:
    """JSONL 원본 행을 그대로 넘겨 normalize_eval_item이 대체 키(question 등)를 처리하도록 한다."""
    return r.copy()


def _as_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, list):
        out: List[str] = []
        for x in val:
            if isinstance(x, str) and x.strip():
                out.append(x)
            elif isinstance(x, dict):
                t = (
                    x.get("text")
                    or x.get("content")
                    or x.get("chunk")
                    or x.get("context")
                    or ""
                )
                if isinstance(t, str) and t.strip():
                    out.append(t)
            elif x is not None:
                out.append(str(x))
        return out
    return []


def normalize_eval_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    online_accelerator export 또는 변형 키를 통일.
    - query / question / prompt
    - b1_contexts / context_b1 / b1_context (하위 호환: l2_contexts 계열)
    - b2_contexts / context_b2 / b2_context
    - mmr_contexts / context_mmr / mmr_context
    """
    q = item.get("query") or item.get("question") or item.get("prompt") or ""
    gt = (
        item.get("ground_truth")
        or item.get("ground_truth_answer")
        or item.get("answer")
        or ""
    )
    if not isinstance(gt, str):
        gt = str(gt) if gt is not None else ""

    b1 = (
        item.get("b1_contexts")
        or item.get("context_b1")
        or item.get("b1_context")
        or item.get("l2_contexts")
        or item.get("context_l2")
        or item.get("l2_context")
    )
    b2 = item.get("b2_contexts") or item.get("context_b2") or item.get("b2_context")
    mmr = item.get("mmr_contexts") or item.get("context_mmr") or item.get("mmr_context")

    b1_list = _as_str_list(b1)
    b2_list = _as_str_list(b2)
    mmr_list = _as_str_list(mmr)

    return {
        "query": str(q).strip(),
        "ground_truth": gt.strip(),
        "b1_contexts": b1_list,
        "b2_contexts": b2_list,
        "mmr_contexts": mmr_list,
    }


# ========== 가상 입력 (JSON 미지정 시) ==========
EVAL_DATA: List[Dict[str, Any]] = [
    {
        "query": "What is the main obligation of the borrower?",
        "b1_contexts": [
            "The borrower shall repay the principal amount within 24 months.",
            "Interest rate is fixed at 5% per annum.",
        ],
        "b2_contexts": [
            "The borrower shall repay the principal amount within 24 months.",
            "Payment obligations survive early termination of this agreement.",
        ],
        "mmr_contexts": [
            "The borrower shall repay the principal amount within 24 months.",
            "Default occurs if payment is delayed by more than 30 days.",
        ],
        "ground_truth": "The borrower must repay the principal within 24 months.",
    },
    {
        "query": "When does the contract expire?",
        "b1_contexts": [
            "This agreement is effective from January 1, 2024.",
            "The contract shall expire on December 31, 2025.",
        ],
        "b2_contexts": [
            "The contract starts on January 1, 2024 and lasts for two years.",
            "Any renewal requires written notice 60 days before expiry.",
        ],
        "mmr_contexts": [
            "The contract shall expire on December 31, 2025.",
            "Renewal requires written notice 60 days prior to expiry.",
        ],
        "ground_truth": "December 31, 2025.",
    },
]


def load_llm(model_name: str, quantization: str):
    """양자화 수준에 따라 LLM 및 토크나이저 로드."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quantization == "fp16":
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif quantization == "int8":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif quantization == "int4":
        from transformers import BitsAndBytesConfig
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"지원하지 않는 양자화: {quantization}. fp16, int8, int4 중 선택하세요.")
    return model, tokenizer


def generate_answer(
    model: Any,
    tokenizer: Any,
    query: str,
    contexts: List[str],
    max_new_tokens: int = 256,
) -> str:
    """계약 발췌를 앞에 두고(PROMPT_TEMPLATE), 질문·답 유도는 끝에 둔 형태로 답변 생성."""
    prompt = build_generation_prompt(query, contexts)
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return answer


def semantic_similarity(text_a: str, text_b: str, encoder: Any) -> float:
    try:
        import numpy as np

        a = encoder.encode(text_a or " ", convert_to_numpy=True)
        b = encoder.encode(text_b or " ", convert_to_numpy=True)
        a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    except Exception:
        return 0.0


def split_ground_truth_sep(gt: str) -> List[str]:
    """Ground Truth를 [SEP]로 분리한 조각 리스트(공백만인 조각은 제외)."""
    raw = str(gt or "")
    pieces = [p.strip() for p in raw.split("[SEP]")]
    pieces = [p for p in pieces if p]
    if not pieces:
        t = raw.strip()
        return [t] if t else [""]
    return pieces


def max_semantic_vs_gt_pieces(
    gt_pieces: List[str],
    hypothesis: str,
    encoder: Any,
) -> float:
    """각 정답 조각과의 Semantic 유사도 중 최댓값."""
    best = 0.0
    for p in gt_pieces:
        best = max(best, semantic_similarity(p, hypothesis, encoder))
    return best


def calc_answer_coverage(gt_pieces: List[str], answer: str) -> float:
    """
    각 정답 조각(GT piece)의 단어들이 생성된 답변(answer)에 얼마나 포함되었는지
    Token Recall(포함된 단어 수 / GT 단어 수)을 계산하고, 그중 최댓값을 반환합니다.
    """
    if not gt_pieces or not answer:
        return 0.0
    ans_tokens = set(re.findall(r"\w+", str(answer).lower()))
    if not ans_tokens:
        return 0.0

    best_cov = 0.0
    for p in gt_pieces:
        gt_toks = re.findall(r"\w+", str(p).lower())
        if not gt_toks:
            continue
        matched = sum(1 for w in gt_toks if w in ans_tokens)
        cov = matched / len(gt_toks)
        if cov > best_cov:
            best_cov = cov
    return best_cov


def _flatten_judge_scores(jr: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "judge_winner": jr.get("winner", ""),
        "judge_reason": (jr.get("reason") or "").replace("\n", " ").strip(),
        "judge_parse_error": bool(jr.get("_parse_error")),
        "judge_provider": jr.get("_judge_provider", ""),
    }
    sc = jr.get("scores") or {}
    sa = sc.get("Answer_A") or sc.get("answer_a") or {}
    sb = sc.get("Answer_B") or sc.get("answer_b") or {}
    sc3 = sc.get("Answer_C") or sc.get("answer_c") or {}
    for k in ("comprehensiveness", "conciseness", "faithfulness"):
        out[f"A_{k}"] = sa.get(k, "") if isinstance(sa, dict) else ""
        out[f"B_{k}"] = sb.get(k, "") if isinstance(sb, dict) else ""
        out[f"C_{k}"] = sc3.get(k, "") if isinstance(sc3, dict) else ""
    return out


def write_generation_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sweep_csv_basename(model_id: str, quantization: str) -> str:
    """스윕용 CSV 파일명 (예: eval_Llama-3.2-1B-Instruct_int8.csv)."""
    short = (model_id or "").split("/")[-1]
    safe = re.sub(r"[^\w.\-]+", "_", short).strip("_") or "model"
    return f"eval_{safe}_{quantization}.csv"


SWEEP_CONFIGS: List[Dict[str, str]] = [
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "quant": "fp16"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "quant": "int4"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "quant": "int8"},
    {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "quant": "fp16"},
    {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "quant": "int4"},
    {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "quant": "int8"},
]


def extract_sweep_summary_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    """sweep_summary.json에 넣을 핵심 지표."""
    return {
        "b1_semantic_avg": float(report.get("b1_semantic_avg", 0.0)),
        "b2_semantic_avg": float(report.get("b2_semantic_avg", 0.0)),
        "mmr_semantic_avg": float(report.get("mmr_semantic_avg", 0.0)),
        "b1_unique_ratio_avg": float(report.get("b1_unique_ratio_avg", 0.0)),
        "b2_unique_ratio_avg": float(report.get("b2_unique_ratio_avg", 0.0)),
        "mmr_unique_ratio_avg": float(report.get("mmr_unique_ratio_avg", 0.0)),
        "b1_answer_coverage_avg": float(report.get("b1_coverage_avg", 0.0)),
        "b2_answer_coverage_avg": float(report.get("b2_coverage_avg", 0.0)),
        "mmr_answer_coverage_avg": float(report.get("mmr_coverage_avg", 0.0)),
    }


def cleanup_generation_gpu_memory(
    model: Any,
    tokenizer: Any,
    semantic_encoder: Any,
) -> None:
    """스윕 반복마다 모델·토크나이저·임베더를 해제하고 CUDA 캐시를 비운다."""
    for obj in (model, tokenizer, semantic_encoder):
        if obj is not None:
            del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


PreloadedModels = Tuple[Any, Any, Any]


def run_eval(
    eval_data: List[Dict[str, Any]],
    model_name: str,
    quantization: str,
    max_new_tokens: int = 256,
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    run_judge: bool = False,
    judge_backend: str = "openai",
    judge_model: Optional[str] = None,
    judge_max_new_tokens: int = 512,
    judge_temperature: float = 0.0,
    return_model_for_cleanup: bool = False,
    preloaded: Optional[PreloadedModels] = None,
    include_prompts: bool = False,
) -> Any:
    """
    eval_data 순회: b1_contexts/b2_contexts/mmr_contexts 각각으로 답변 생성.
    Ground Truth에 [SEP]가 있으면 조각별 Semantic 유사도의 max를 쿼리 점수로 사용.
    Semantic / Unique Token Ratio / Answer Coverage 유지. run_judge 시 API 또는 local 심판.
    preloaded=(model, tokenizer, semantic_encoder)이면 로드 생략(스윕 검색 조합 연속 평가용).
    include_prompts=True이면 per_query_details에 전체 프롬프트 문자열 포함.
    return_model_for_cleanup=True 이면
    (report, csv_rows, per_query_details, model, tokenizer, semantic_encoder) 반환.
    그렇지 않으면 (report, csv_rows, per_query_details) 반환.
    """
    if preloaded is not None:
        model, tokenizer, semantic_encoder = preloaded
    else:
        model, tokenizer = load_llm(model_name, quantization)

        from sentence_transformers import SentenceTransformer

        semantic_encoder = SentenceTransformer(semantic_model)

    b1_semantic: List[float] = []
    b2_semantic: List[float] = []
    mmr_semantic: List[float] = []
    b1_unique_list: List[float] = []
    b2_unique_list: List[float] = []
    mmr_unique_list: List[float] = []
    b1_coverage_list: List[float] = []
    b2_coverage_list: List[float] = []
    mmr_coverage_list: List[float] = []
    judge_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    per_query_details: List[Dict[str, Any]] = []

    jb = (judge_backend or "openai").lower().strip()

    for i, raw_item in enumerate(eval_data):
        item = normalize_eval_item(raw_item)
        query = item["query"]
        gt = item["ground_truth"]
        b1_ctx = item["b1_contexts"]
        b2_ctx = item["b2_contexts"]
        mmr_ctx = item["mmr_contexts"]

        answer_b1 = generate_answer(
            model, tokenizer, query, b1_ctx, max_new_tokens=max_new_tokens
        )
        answer_b2 = generate_answer(
            model, tokenizer, query, b2_ctx, max_new_tokens=max_new_tokens
        )
        answer_mmr = generate_answer(
            model, tokenizer, query, mmr_ctx, max_new_tokens=max_new_tokens
        )

        gt_pieces = split_ground_truth_sep(gt)
        s_b1 = max_semantic_vs_gt_pieces(gt_pieces, answer_b1, semantic_encoder)
        s_b2 = max_semantic_vs_gt_pieces(gt_pieces, answer_b2, semantic_encoder)
        s_mmr = max_semantic_vs_gt_pieces(gt_pieces, answer_mmr, semantic_encoder)
        b1_unique_ratio = _calc_unique_token_ratio(answer_b1)
        b2_unique_ratio = _calc_unique_token_ratio(answer_b2)
        mmr_unique_ratio = _calc_unique_token_ratio(answer_mmr)

        cov_b1 = calc_answer_coverage(gt_pieces, answer_b1)
        cov_b2 = calc_answer_coverage(gt_pieces, answer_b2)
        cov_mmr = calc_answer_coverage(gt_pieces, answer_mmr)

        b1_semantic.append(s_b1)
        b2_semantic.append(s_b2)
        mmr_semantic.append(s_mmr)
        b1_unique_list.append(b1_unique_ratio)
        b2_unique_list.append(b2_unique_ratio)
        mmr_unique_list.append(mmr_unique_ratio)
        b1_coverage_list.append(cov_b1)
        b2_coverage_list.append(cov_b2)
        mmr_coverage_list.append(cov_mmr)

        raw_item["answer_b1"] = answer_b1
        raw_item["answer_b2"] = answer_b2
        raw_item["answer_mmr"] = answer_mmr
        raw_item["b1_answer"] = answer_b1
        raw_item["b2_answer"] = answer_b2
        raw_item["mmr_answer"] = answer_mmr
        raw_item["b1_semantic"] = s_b1
        raw_item["b2_semantic"] = s_b2
        raw_item["mmr_semantic"] = s_mmr
        raw_item["b1_unique_ratio"] = b1_unique_ratio
        raw_item["b2_unique_ratio"] = b2_unique_ratio
        raw_item["mmr_unique_ratio"] = mmr_unique_ratio
        raw_item["b1_answer_coverage"] = cov_b1
        raw_item["b2_answer_coverage"] = cov_b2
        raw_item["mmr_answer_coverage"] = cov_mmr
        raw_item["ground_truth_pieces"] = gt_pieces

        jr: Dict[str, Any] = {}
        if run_judge:
            if jb == "local":
                jr = run_llm_as_judge(
                    model,
                    tokenizer,
                    query,
                    answer_b1,
                    answer_b2,
                    answer_mmr,
                    max_new_tokens=judge_max_new_tokens,
                )
            else:
                jr = evaluate_with_llm_judge(
                    query,
                    answer_b1,
                    answer_b2,
                    answer_mmr,
                    provider=jb,
                    model=judge_model,
                    temperature=judge_temperature,
                )
            raw_item["judge_result"] = jr
            judge_rows.append(jr)

        flat = {
            "idx": i,
            "query": query,
            "ground_truth": gt,
            "answer_b1": answer_b1,
            "answer_b2": answer_b2,
            "answer_mmr": answer_mmr,
            "b1_semantic": f"{s_b1:.6f}",
            "b2_semantic": f"{s_b2:.6f}",
            "mmr_semantic": f"{s_mmr:.6f}",
            "b1_unique_ratio": f"{b1_unique_ratio:.6f}",
            "b2_unique_ratio": f"{b2_unique_ratio:.6f}",
            "mmr_unique_ratio": f"{mmr_unique_ratio:.6f}",
            "b1_answer_coverage": f"{cov_b1:.6f}",
            "b2_answer_coverage": f"{cov_b2:.6f}",
            "mmr_answer_coverage": f"{cov_mmr:.6f}",
            "num_gt_pieces": str(len(gt_pieces)),
        }
        if run_judge:
            flat.update(_flatten_judge_scores(jr))
        csv_rows.append(flat)

        detail: Dict[str, Any] = {
            "query_idx": i,
            "query": query,
            "ground_truth": gt,
            "generation": {
                "answer_b1": answer_b1,
                "answer_b2": answer_b2,
                "answer_mmr": answer_mmr,
            },
            "metrics": {
                "b1_semantic_similarity": s_b1,
                "b2_semantic_similarity": s_b2,
                "mmr_semantic_similarity": s_mmr,
                "b1_unique_token_ratio": b1_unique_ratio,
                "b2_unique_token_ratio": b2_unique_ratio,
                "mmr_unique_token_ratio": mmr_unique_ratio,
                "b1_answer_coverage": cov_b1,
                "b2_answer_coverage": cov_b2,
                "mmr_answer_coverage": cov_mmr,
            },
        }
        if include_prompts:
            detail["prompt"] = {
                "b1": build_generation_prompt(query, b1_ctx),
                "b2": build_generation_prompt(query, b2_ctx),
                "mmr": build_generation_prompt(query, mmr_ctx),
            }
        if run_judge:
            detail["judge_result"] = jr
        per_query_details.append(detail)

    n = len(eval_data)
    report: Dict[str, Any] = {
        "model_name": model_name,
        "quantization": quantization.upper(),
        "num_samples": n,
        "b1_semantic_avg": sum(b1_semantic) / n if n else 0.0,
        "b2_semantic_avg": sum(b2_semantic) / n if n else 0.0,
        "mmr_semantic_avg": sum(mmr_semantic) / n if n else 0.0,
        "b1_unique_ratio_avg": sum(b1_unique_list) / n if n else 0.0,
        "b2_unique_ratio_avg": sum(b2_unique_list) / n if n else 0.0,
        "mmr_unique_ratio_avg": sum(mmr_unique_list) / n if n else 0.0,
        "b1_coverage_avg": sum(b1_coverage_list) / n if n else 0.0,
        "b2_coverage_avg": sum(b2_coverage_list) / n if n else 0.0,
        "mmr_coverage_avg": sum(mmr_coverage_list) / n if n else 0.0,
    }
    if run_judge and judge_rows:
        report.update(aggregate_judge_stats(judge_rows))
    if return_model_for_cleanup:
        return (
            report,
            csv_rows,
            per_query_details,
            model,
            tokenizer,
            semantic_encoder,
        )
    return report, csv_rows, per_query_details


def print_report(report: Dict[str, Any]) -> None:
    q = report.get("quantization", "?")
    n = report.get("num_samples", 0)
    mn = report.get("model_name")
    print()
    print("=" * 60)
    if mn:
        print(f"  [모델: {mn}]")
    print(f"  [현재 양자화 수준: {q}]  (샘플 수: {n})")
    print("=" * 60)
    print("  Semantic Similarity (평균, GT에 [SEP] 시 조각별 max):")
    print(f"    B1  Top-K 컨텍스트: {report.get('b1_semantic_avg', 0):.4f}")
    print(f"    B2  Top-K 컨텍스트: {report.get('b2_semantic_avg', 0):.4f}")
    print(f"    MMR Top-K 컨텍스트: {report.get('mmr_semantic_avg', 0):.4f}")
    print("  Unique Token Ratio (고유 단어 비율, 평균):")
    print(f"    B1  생성 답변: {report.get('b1_unique_ratio_avg', 0):.4f}")
    print(f"    B2  생성 답변: {report.get('b2_unique_ratio_avg', 0):.4f}")
    print(f"    MMR 생성 답변: {report.get('mmr_unique_ratio_avg', 0):.4f}")
    print("  Answer Coverage (GT Token Recall, 평균):")
    print(f"    B1  생성 답변: {report.get('b1_coverage_avg', 0):.4f}")
    print(f"    B2  생성 답변: {report.get('b2_coverage_avg', 0):.4f}")
    print(f"    MMR 생성 답변: {report.get('mmr_coverage_avg', 0):.4f}")
    if report.get("judge_n"):
        print("  LLM-as-a-Judge (A=B1, B=B2, C=MMR):")
        print(
            f"    승: B1={report.get('judge_wins_b1', 0)}, "
            f"B2={report.get('judge_wins_b2', 0)}, "
            f"MMR={report.get('judge_wins_mmr', 0)}, "
            f"무승부={report.get('judge_ties', 0)} "
            f"(비율 B1/B2/MMR/TIE: "
            f"{report.get('judge_win_rate_b1', 0):.2f} / "
            f"{report.get('judge_win_rate_b2', 0):.2f} / "
            f"{report.get('judge_win_rate_mmr', 0):.2f} / "
            f"{report.get('judge_tie_rate', 0):.2f})"
        )
        for k in ("comprehensiveness", "conciseness", "faithfulness"):
            ka = report.get(f"judge_avg_A_{k}")
            kb = report.get(f"judge_avg_B_{k}")
            kc = report.get(f"judge_avg_C_{k}")
            if ka is not None and kb is not None and kc is not None:
                print(f"    평균 점수 {k}: A(B1)={ka:.2f}, B(B2)={kb:.2f}, C(MMR)={kc:.2f}")
    print("=" * 60)


def run_llm_pipeline_from_retrieval(
    artifact_root: str,
    jsonl_path: str,
    run_id: str,
    *,
    llm_output_root: Optional[str],
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    semantic_model: str,
    run_judge: bool,
    judge_backend: str,
    judge_model: Optional[str],
    judge_max_new_tokens: int,
    judge_temperature: float,
    target_indices: Optional[List[int]] = None,
) -> None:
    """
    sweep_eval이 저장한 retrieval_*.jsonl을 sweep_idx별로 묶어 LLM 생성·평가 후
    sweep_exports와 동일한 패턴으로 JSONL / combos / CSV / manifest 저장.
    """
    records = load_retrieval_jsonl_records(jsonl_path)
    groups = group_retrieval_by_sweep_idx(records)
    if not groups:
        print("[오류] jsonl에 유효한 sweep_idx가 없습니다.", file=sys.stderr)
        sys.exit(1)

    if target_indices is not None:
        want = {int(x) for x in target_indices}
        present = set(groups.keys())
        groups = {k: v for k, v in groups.items() if k in want}
        missing = sorted(want - present)
        if missing:
            print(
                f"[경고] --target_indices 중 jsonl에 없는 sweep_idx: {missing}",
                file=sys.stderr,
                flush=True,
            )
        if not groups:
            print(
                "[오류] 필터 후 실행할 sweep_idx가 없습니다.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"[LLM 파이프라인] target_indices 필터 적용 → {len(groups)}개 조합 실행 "
            f"(요청 {len(want)}개)",
            flush=True,
        )

    out_root = os.path.abspath(llm_output_root or artifact_root)
    os.makedirs(os.path.join(out_root, "combos"), exist_ok=True)

    detail_path = os.path.join(out_root, f"llm_generation_detail_{run_id}.jsonl")
    summary_csv = os.path.join(out_root, "llm_eval_summary.csv")

    print(
        f"[LLM 파이프라인] 검색 레코드 {len(records)}줄, sweep 조합 {len(groups)}개",
        flush=True,
    )
    print(f"[LLM 파이프라인] 출력 루트: {out_root}", flush=True)

    model, tokenizer = load_llm(model_name, quantization)
    from sentence_transformers import SentenceTransformer

    semantic_encoder = SentenceTransformer(semantic_model)
    preloaded: PreloadedModels = (model, tokenizer, semantic_encoder)

    summary_rows: List[Dict[str, Any]] = []

    preferred_csv_columns = [
        "mmr_lambda",
        "tau_dup",
        "gamma",
        "top_k",
        "window_size",
        "sweep_idx",
        "domain",
        "num_queries",
        "model_name",
        "quantization",
        "avg_b1_semantic_similarity",
        "avg_b2_semantic_similarity",
        "avg_mmr_semantic_similarity",
        "avg_b1_unique_token_ratio",
        "avg_b2_unique_token_ratio",
        "avg_mmr_unique_token_ratio",
        "avg_b1_answer_coverage",
        "avg_b2_answer_coverage",
        "avg_mmr_answer_coverage",
        "judge_n",
        "judge_win_rate_b1",
        "judge_win_rate_b2",
        "judge_win_rate_mmr",
        "judge_tie_rate",
        "avg_llm_judge_B1_comprehensiveness",
        "avg_llm_judge_B1_conciseness",
        "avg_llm_judge_B1_faithfulness",
        "avg_llm_judge_B2_comprehensiveness",
        "avg_llm_judge_B2_conciseness",
        "avg_llm_judge_B2_faithfulness",
        "avg_llm_judge_MMR_comprehensiveness",
        "avg_llm_judge_MMR_conciseness",
        "avg_llm_judge_MMR_faithfulness",
        "llm_combo_artifact",
    ]

    with open(detail_path, "w", encoding="utf-8") as detail_f:
        for sweep_idx, group_rows in groups.items():
            raw_eval = [retrieval_record_to_eval_item(r) for r in group_rows]

            report, _csv_rows, per_query_details = run_eval(
                raw_eval,
                model_name=model_name,
                quantization=quantization,
                max_new_tokens=max_new_tokens,
                semantic_model=semantic_model,
                run_judge=run_judge,
                judge_backend=judge_backend,
                judge_model=judge_model,
                judge_max_new_tokens=judge_max_new_tokens,
                judge_temperature=judge_temperature,
                return_model_for_cleanup=False,
                preloaded=preloaded,
                include_prompts=True,
            )

            first = group_rows[0]
            params = {
                "mmr_lambda": first.get("mmr_lambda"),
                "tau_dup": first.get("tau_dup"),
                "gamma": first.get("gamma"),
                "top_k": first.get("top_k"),
                "window_size": first.get("window_size"),
            }

            combo_path = os.path.join(out_root, "combos", f"llm_combo_{sweep_idx:05d}.json")
            combo_payload: Dict[str, Any] = {
                "run_id": run_id,
                "sweep_idx": sweep_idx,
                "domain": first.get("domain"),
                "parameters": params,
                "metrics_avg": dict(report),
                "per_query_results": [],
            }

            for r, det in zip(group_rows, per_query_details):
                line = {
                    "run_id": run_id,
                    "sweep_idx": sweep_idx,
                    "query_idx": int(r.get("query_idx", 0)),
                    "domain": r.get("domain"),
                    **params,
                    "query": r.get("query"),
                    "ground_truth": r.get("ground_truth"),
                    "prompt": det.get("prompt"),
                    "generation": det.get("generation"),
                    "metrics": det.get("metrics"),
                }
                if det.get("judge_result") is not None:
                    line["judge_result"] = det["judge_result"]
                detail_f.write(json.dumps(line, ensure_ascii=False) + "\n")

                combo_payload["per_query_results"].append(
                    {
                        "query_idx": int(r.get("query_idx", 0)),
                        "query": r.get("query"),
                        "ground_truth": r.get("ground_truth"),
                        "prompt": det.get("prompt"),
                        "generation": det.get("generation"),
                        "metrics": det.get("metrics"),
                        "judge_result": det.get("judge_result"),
                    }
                )

            print_report(report)
            print(
                f"[조합 sweep_idx={sweep_idx}] 쿼리 {len(group_rows)}건 → {combo_path}",
                flush=True,
            )

            with open(combo_path, "w", encoding="utf-8") as jf:
                json.dump(combo_payload, jf, ensure_ascii=False, indent=2)

            try:
                combo_rel = os.path.relpath(combo_path, out_root)
            except ValueError:
                combo_rel = combo_path

            summary_row: Dict[str, Any] = {
                "mmr_lambda": params["mmr_lambda"],
                "tau_dup": params["tau_dup"],
                "gamma": params["gamma"],
                "top_k": params["top_k"],
                "window_size": params["window_size"],
                "sweep_idx": sweep_idx,
                "domain": first.get("domain"),
                "num_queries": len(group_rows),
                "llm_combo_artifact": combo_rel,
                "avg_b1_semantic_similarity": report.get("b1_semantic_avg"),
                "avg_b2_semantic_similarity": report.get("b2_semantic_avg"),
                "avg_mmr_semantic_similarity": report.get("mmr_semantic_avg"),
                "avg_b1_unique_token_ratio": report.get("b1_unique_ratio_avg"),
                "avg_b2_unique_token_ratio": report.get("b2_unique_ratio_avg"),
                "avg_mmr_unique_token_ratio": report.get("mmr_unique_ratio_avg"),
                "avg_b1_answer_coverage": report.get("b1_coverage_avg"),
                "avg_b2_answer_coverage": report.get("b2_coverage_avg"),
                "avg_mmr_answer_coverage": report.get("mmr_coverage_avg"),
                "model_name": model_name,
                "quantization": quantization.upper(),
            }
            if report.get("judge_n"):
                summary_row["judge_n"] = report.get("judge_n")
                summary_row["judge_win_rate_b1"] = report.get("judge_win_rate_b1")
                summary_row["judge_win_rate_b2"] = report.get("judge_win_rate_b2")
                summary_row["judge_win_rate_mmr"] = report.get("judge_win_rate_mmr")
                summary_row["judge_tie_rate"] = report.get("judge_tie_rate")
                for k in ("comprehensiveness", "conciseness", "faithfulness"):
                    ka = report.get(f"judge_avg_A_{k}")
                    kb = report.get(f"judge_avg_B_{k}")
                    kc = report.get(f"judge_avg_C_{k}")
                    if ka is not None:
                        summary_row[f"avg_llm_judge_B1_{k}"] = ka
                    if kb is not None:
                        summary_row[f"avg_llm_judge_B2_{k}"] = kb
                    if kc is not None:
                        summary_row[f"avg_llm_judge_MMR_{k}"] = kc
            summary_rows.append(summary_row)

    cleanup_generation_gpu_memory(model, tokenizer, semantic_encoder)

    union_keys: set = set()
    for sr in summary_rows:
        union_keys.update(sr.keys())
    ordered_cols: List[str] = []
    for p in preferred_csv_columns:
        if p in union_keys:
            ordered_cols.append(p)
            union_keys.discard(p)
    for k in sorted(union_keys):
        ordered_cols.append(k)

    # 스윕 등 연속 호출 시 이전 양자화/모델 결과가 지워지지 않도록 추가 모드로 기록
    summary_exists = os.path.isfile(summary_csv) and os.path.getsize(summary_csv) > 0
    existing_header: Optional[List[str]] = None
    if summary_exists:
        with open(summary_csv, "r", encoding="utf-8", newline="") as rf:
            r0 = next(csv.reader(rf), None)
            if r0:
                existing_header = r0
    if summary_exists and existing_header:
        fieldnames = existing_header
        csv_mode = "a"
        write_header = False
    else:
        fieldnames = ordered_cols
        csv_mode = "w"
        write_header = True

    with open(summary_csv, csv_mode, encoding="utf-8", newline="") as sf:
        w = csv.DictWriter(sf, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for sr in summary_rows:
            w.writerow({c: sr.get(c, "") for c in fieldnames})

    manifest_llm = {
        "run_id": run_id,
        "source_retrieval_jsonl": os.path.abspath(jsonl_path),
        "artifact_root_search": os.path.abspath(artifact_root),
        "llm_output_root": out_root,
        "llm_generation_detail_jsonl": os.path.abspath(detail_path),
        "llm_eval_summary_csv": os.path.abspath(summary_csv),
        "model_name": model_name,
        "quantization": quantization,
        "num_parameter_combos": len(summary_rows),
    }
    man_path = os.path.join(out_root, "llm_run_manifest.json")
    with open(man_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_llm, mf, ensure_ascii=False, indent=2)

    print(f"\n[완료] 상세 JSONL: {detail_path}", flush=True)
    print(f"[완료] 요약 CSV: {summary_csv}", flush=True)
    print(f"[완료] {man_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUAD RAG JSON → 로컬 LLM 생성 → Semantic/Unique/Answer Coverage + 선택적 LLM Judge → CSV/JSON"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["fp16", "int8", "int4"],
        help="모델 양자화: fp16, int8, int4",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="로컬 생성용 LLM",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--semantic_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="online_accelerator export JSON (query, b1_contexts, b2_contexts, mmr_contexts, …)",
    )
    parser.add_argument(
        "--output_results",
        type=str,
        default=None,
        help="전체 결과(생성·Judge·메트릭) JSON 저장 경로",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="generation_eval_results.csv",
        help="쿼리별 요약 CSV (기본: generation_eval_results.csv)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="LLM-as-a-Judge 실행 (--judge_backend 로 API 또는 local 선택)",
    )
    parser.add_argument(
        "--judge_backend",
        type=str,
        default="openai",
        choices=["openai", "gemini", "local"],
        help="openai / gemini: 환경변수 API 키. local: 동일 로컬 LLM으로 심판",
    )
    parser.add_argument("--judge_model", type=str, default=None, help="Judge 전용 모델명 (선택)")
    parser.add_argument("--judge_max_new_tokens", type=int, default=512)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="모델×양자화 SWEEP_CONFIGS 를 순회하며 평가 (VRAM 16GB 기준 6조합: 1B·8B × fp16·int4·int8)",
    )
    parser.add_argument(
        "--sweep_summary",
        type=str,
        default="sweep_summary.json",
        help="--sweep 시 지표 요약 JSON 경로 (기본: sweep_summary.json)",
    )
    parser.add_argument(
        "--sweep_manifest",
        type=str,
        default=None,
        help="sweep_eval 산출 run_manifest.json 경로 (retrieval jsonl 위치·run_id 해석)",
    )
    parser.add_argument(
        "--retrieval_jsonl",
        type=str,
        default=None,
        help="sweep_eval 산출 retrieval_<run_id>.jsonl 직접 지정 (--sweep_manifest와 택일 가능, 동시 지정 시 본 인자 우선)",
    )
    parser.add_argument(
        "--llm_output_root",
        type=str,
        default=None,
        help="LLM 산출물 저장 디렉터리 (기본: manifest/jsonl의 artifact_root와 동일)",
    )
    parser.add_argument(
        "--target_indices",
        type=int,
        nargs="+",
        default=None,
        help="특정 sweep_idx만 선택해서 실행 (예: --target_indices 15 80)",
    )
    args = parser.parse_args()

    if args.retrieval_jsonl or args.sweep_manifest:
        try:
            if args.retrieval_jsonl:
                artifact_root, jsonl_path, run_id = resolve_artifact_root_and_jsonl(
                    None, args.retrieval_jsonl
                )
            else:
                artifact_root, jsonl_path, run_id = resolve_artifact_root_and_jsonl(
                    args.sweep_manifest, None
                )
        except (FileNotFoundError, ValueError) as e:
            print(f"[오류] 검색 산출물 경로 해석 실패: {e}", file=sys.stderr)
            sys.exit(1)

        def _call_retrieval_pipeline(mname: str, quant: str) -> None:
            run_llm_pipeline_from_retrieval(
                artifact_root,
                jsonl_path,
                run_id,
                llm_output_root=args.llm_output_root,
                model_name=mname,
                quantization=quant,
                max_new_tokens=args.max_new_tokens,
                semantic_model=args.semantic_model,
                run_judge=args.judge,
                judge_backend=args.judge_backend,
                judge_model=args.judge_model,
                judge_max_new_tokens=args.judge_max_new_tokens,
                judge_temperature=args.judge_temperature,
                target_indices=args.target_indices,
            )

        if args.sweep:
            for si, cfg in enumerate(SWEEP_CONFIGS, start=1):
                mname, quant = cfg["model"], cfg["quant"]
                print(
                    f"\n{'=' * 60}\n"
                    f"[retrieval sweep {si}/{len(SWEEP_CONFIGS)}] model={mname}  quant={quant}\n"
                    f"{'=' * 60}",
                    flush=True,
                )
                _call_retrieval_pipeline(mname, quant)
        else:
            _call_retrieval_pipeline(args.model, args.quantization)
        return

    if args.eval_data:
        if not os.path.isfile(args.eval_data):
            print(f"[오류] 파일 없음: {args.eval_data}", file=sys.stderr)
            sys.exit(1)
        eval_data = load_eval_data_json(args.eval_data)
        print(f"[로드] {args.eval_data} → {len(eval_data)}개 항목")
    else:
        eval_data = EVAL_DATA

    if not eval_data:
        print("평가 데이터가 비어 있습니다.")
        return

    if args.sweep:
        csv_dir = os.path.dirname(os.path.abspath(args.output_csv))
        if not csv_dir:
            csv_dir = os.getcwd()
        os.makedirs(csv_dir, exist_ok=True)

        summary_runs: List[Dict[str, Any]] = []
        for si, cfg in enumerate(SWEEP_CONFIGS, start=1):
            mname, quant = cfg["model"], cfg["quant"]
            print(f"\n{'=' * 60}\n[sweep {si}/{len(SWEEP_CONFIGS)}] model={mname}  quant={quant}\n{'=' * 60}")
            report, csv_rows, _per_q, model, tokenizer, semantic_encoder = run_eval(
                eval_data,
                model_name=mname,
                quantization=quant,
                max_new_tokens=args.max_new_tokens,
                semantic_model=args.semantic_model,
                run_judge=args.judge,
                judge_backend=args.judge_backend,
                judge_model=args.judge_model,
                judge_max_new_tokens=args.judge_max_new_tokens,
                judge_temperature=args.judge_temperature,
                return_model_for_cleanup=True,
            )
            print_report(report)

            csv_name = sweep_csv_basename(mname, quant)
            out_csv = os.path.join(csv_dir, csv_name)
            write_generation_csv(out_csv, csv_rows)
            print(f"[CSV] 저장: {out_csv} ({len(csv_rows)} rows)")

            if args.output_results:
                base, ext = os.path.splitext(args.output_results)
                ext = ext or ".json"
                outp = os.path.abspath(f"{base}_{csv_name.replace('.csv', '')}{ext}")
                with open(outp, "w", encoding="utf-8") as f:
                    json.dump(
                        {"report": report, "samples": eval_data},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[JSON] 저장: {outp}")

            run_entry: Dict[str, Any] = {
                "model": mname,
                "quantization": quant,
                "csv_path": os.path.abspath(out_csv),
            }
            run_entry.update(extract_sweep_summary_metrics(report))
            summary_runs.append(run_entry)

            cleanup_generation_gpu_memory(model, tokenizer, semantic_encoder)

        summary_path = os.path.abspath(args.sweep_summary)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"runs": summary_runs}, f, ensure_ascii=False, indent=2)
        print(f"\n[sweep] 통합 요약 저장: {summary_path} ({len(summary_runs)} runs)")
        return

    report, csv_rows, per_query_details = run_eval(
        eval_data,
        model_name=args.model,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        semantic_model=args.semantic_model,
        run_judge=args.judge,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_temperature=args.judge_temperature,
        include_prompts=True,
    )
    print_report(report)

    out_csv = os.path.abspath(args.output_csv)
    write_generation_csv(out_csv, csv_rows)
    print(f"[CSV] 저장: {out_csv} ({len(csv_rows)} rows)")

    if args.output_results:
        outp = os.path.abspath(args.output_results)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "report": report,
                    "samples": eval_data,
                    "per_query_details": per_query_details,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[JSON] 저장: {outp}")


if __name__ == "__main__":
    main()
