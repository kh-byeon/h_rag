
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# Regex: strip CUAD "Highlight the parts..." lawyer-review preamble before the substantive ask.
_HIGHLIGHT_LAWYER_PREAMBLE_RE = re.compile(
    r"(?is)^\s*Highlight\s+the\s+parts\s*(?:\(if\s+any\))?\s+of\s+this\s+contract\s+related\s+to\s+"
    r"[\s\S]+?\s+that\s+should\s+be\s+reviewed\s+by\s+a\s+lawyer\.?\s*"
)


def extract_decision_question_from_cuad_query(query: str) -> str:
    """
    Heuristic core question for generation: prefer text after 'Details:', drop highlight preamble,
    normalize whitespace. Retrieval is unchanged; this is generation-only.
    """
    s = str(query or "").strip()
    if not s:
        return ""
    m_details = re.search(r"(?is)Details:\s*", s)
    if m_details:
        s = s[m_details.end() :].strip()
    s = _HIGHLIGHT_LAWYER_PREAMBLE_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rewrite_query_for_generation(query: str) -> str:
    """
    Normalize CUAD-style wording for the generation user message only (retrieval unchanged).
    Does not impose Yes/No or other binary answer formats.
    """
    core = extract_decision_question_from_cuad_query(query)
    if not core:
        core = re.sub(r"\s+", " ", str(query or "").strip()).strip()
    if not core:
        core = "Analyze the question using only the TARGET CONTRACT passages below."
    return core.rstrip(".,; ")


# 1. 시스템 프롬프트는 가볍게 역할만 부여하도록 단순화
GENERATION_SYSTEM_INSTRUCTION_MAIN = """You are an expert legal AI assistant. Your task is to analyze contract excerpts strictly according to the user's step-by-step instructions."""

# HotpotQA (multi-hop factoid) system instruction — kept separate from the CUAD path.
GENERATION_SYSTEM_INSTRUCTION_HOTPOT = """You are a precise question-answering assistant. Answer multi-hop factual questions using only the provided passages. Reason step by step across the passages, then give a short, exact final answer."""

def filter_target_contract_contexts(contexts: List[str]) -> List[str]:
    """
    Keep only TARGET CONTRACT contexts.
    If tagged TARGET chunks exist, keep only those.
    Otherwise, drop explicit REFERENCE-tagged chunks and keep the remaining contexts.
    """
    tagged_target: List[str] = []
    fallback: List[str] = []
    for c in contexts or []:
        s = str(c or "").strip()
        if not s:
            continue
        u = s.upper()
        if "TARGET CONTRACT" in u:
            tagged_target.append(s)
        elif "REFERENCE CONTRACT" not in u:
            fallback.append(s)
    return tagged_target if tagged_target else fallback


def _format_tagged_document_section(
    heading: str,
    subheading: str,
    blocks: List[str],
    empty_note: str,
) -> str:
    """
    Build one document-tier section: clear heading plus numbered passages (tag-assigned chunks only).
    """
    lines: List[str] = [heading, subheading, ""]
    if not blocks:
        lines.append(empty_note)
        return "\n".join(lines)
    for i, block in enumerate(blocks, start=1):
        lines.append(f"--- Chunk {i} ---")
        lines.append(str(block).strip())
        lines.append("")
    return "\n".join(lines).rstrip()


# 2. 유저 프롬프트 하단(질문 바로 위)에 앵커링할 강력한 지시사항 배치
GENERATION_USER_GUIDANCE_MAIN = """[Final Evaluation Instructions]
You are acting as a senior paralegal preparing a briefing for a lawyer.
Structure your response into these four professional sections:

[Step 1: Core Finding]
- Analyze the "=== Primary Evidence (Direct Answer Source) ===" (Top-3 Chunks).
- Provide a direct, factual answer to the question.
- Refer to passages as "Chunk 1", "Chunk 2", etc.

[Step 2: Legal Context & Potential Risks]
- Review the "=== Supplemental Context (Risk & Context Analysis) ===" (MMR-based Diverse Chunks).
- Identify related legal issues that a lawyer should review (e.g., assignment restrictions, termination rights, or survival clauses) even if they don't directly change the Step 1 answer.
- Highlight any "hidden" conditions or cross-referenced clauses found here.
- Refer to passages as "Chunk N" as in Step 1.

[Step 3: Executive Summary for Lawyer]
- Provide a 1-2 sentence final summary.
- Clearly state if the evidence is sufficient for a definitive legal conclusion
  or if specific risks found in Step 2 require closer inspection.
- If the license term is unclear, explicitly state: check whether the license
  clause is listed in the survival clause index to determine if it survives termination.
  
[Step 4: Clause Reference Index]
Map every Chunk cited in Steps 1 and 2 to its actual clause or section number.
Follow these two rules strictly:
- Rule A: Only include Chunks that were explicitly mentioned in Step 1 or Step 2. Do not introduce any new Chunks here.
- Rule B: For each Chunk, look inside its passage text for a clause or section number (e.g., "Section 18.4", "Article 3", "Clause 12(b)"). If no clause number appears in the text, copy a short identifying phrase (10 words or fewer) from that passage instead. Do not invent clause numbers.

Format as a numbered list:
- Chunk 1 → [clause number found in text, e.g. "Section 18.4"] : [one-line description of what this clause covers]
- Chunk 2 → [short identifying phrase if no clause number] : [one-line description]
- (repeat for each cited Chunk only)"""

# HotpotQA user guidance — multi-hop reasoning + short parseable final answer (no legal framing).
GENERATION_USER_GUIDANCE_HOTPOT = """[Answer Instructions]
The passages come from different Wikipedia articles. The question usually requires combining facts from TWO different passages (multi-hop).
1. Briefly identify the relevant fact from each passage needed to answer (1 sentence each; refer to passages as "Passage 1", "Passage 2", ...).
2. Then output the final answer on its own last line, prefixed exactly with "Answer: ".
Keep the final answer as short as possible — a name, entity, date, yes/no, or short phrase. Do not add explanation after the Answer line."""

# Chat + plain fallback: max input tokens for retrieval-heavy prompts.
GENERATION_MAX_INPUT_LENGTH = 8192


def build_generation_user_content_main(query: str, contexts: List[str]) -> str:
    """User content: question + structured TARGET passages."""
    target_ctx = filter_target_contract_contexts(contexts)
    q = str(query or "").strip()

    # Split TARGET contexts: Strictly top 3 unique chunks go to L1 (Primary), the rest to L2 (Supplemental).
    primary_ctx: List[str] = []
    supplemental_ctx: List[str] = []
    seen_texts = set()

    for block in target_ctx:
        text = str(block or "").strip()
        if not text:
            continue

        # 중복 제거 로직 (앞 100글자 기준)
        text_key = text[:100]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        # 핵심: 상위 3개까지만 Primary Evidence에 할당하여 모델의 SNR(신호 대 잡음비)을 최적화
        if len(primary_ctx) < 3:
            primary_ctx.append(text)
        else:
            supplemental_ctx.append(text)

    primary_block = _format_tagged_document_section(
        "=== Primary Evidence (Direct Answer Source) ===",
        "Top-3 chunks: factual basis for the core finding.",
        primary_ctx,
        "(No target passages retrieved.)",
    )
    supplemental_block = ""
    if supplemental_ctx:
        supplemental_block = _format_tagged_document_section(
            "=== Supplemental Context (Risk & Context Analysis) ===",
            "Diverse clauses for risk screening, cross-references, and related legal context.",
            supplemental_ctx,
            "(No supplemental passages retrieved.)",
        )

    parts: List[str] = [
        primary_block,
    ]
    if supplemental_block:
        parts.extend([
            "",
            supplemental_block,
        ])
    parts.extend([
        "",
        GENERATION_USER_GUIDANCE_MAIN,
        "",
        "Question:",
        q if q else "(No question)",
    ])
    return "\n".join(parts)


# Retrieval-provenance header emitted by the retrieval writer, e.g.
#   --- [Source: L2 | Section: ] ---
#   --- [Source: L2_Anchor | Section: ] ---
#   --- [Source: MMR | Section: ] ---
# The tag differs BY ARM, so leaving it in leaks the arm identity into the
# prompt and confounds the L2-vs-MMR answer comparison. For hotpot it also
# carries no information (Section is always empty). Strip it so both arms see
# byte-identical formatting and only the retrieved content differs.
_SOURCE_HEADER_RE = re.compile(r"^[ \t]*---[ \t]*\[Source:[^\]]*\][ \t]*---[ \t]*$", re.M)


def strip_source_header(block: str) -> str:
    """Remove '--- [Source: ... ] ---' provenance lines from a context block."""
    return _SOURCE_HEADER_RE.sub("", str(block or "")).strip()


def build_generation_user_content_hotpot(query: str, contexts: List[str]) -> str:
    """
    HotpotQA user content: single "=== Passages ===" section + multi-hop guidance + question.

    No TARGET/REFERENCE filtering and no Primary/Supplemental split (that is CUAD-only).
    Arm-identifying '[Source: ...]' headers are stripped (see _SOURCE_HEADER_RE) so
    the L2 and MMR prompts differ only in retrieved content. Stripping happens
    before the first-100-chars dedup key is taken, so identical passages now
    dedup correctly even when they carried different source tags.
    """
    q = str(query or "").strip()

    passages: List[str] = []
    seen_texts = set()
    for block in contexts or []:
        text = strip_source_header(block)
        if not text:
            continue
        text_key = text[:100]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        passages.append(text)

    lines: List[str] = ["=== Passages ==="]
    if not passages:
        lines.append("(No passages retrieved.)")
    else:
        for i, text in enumerate(passages, start=1):
            lines.append(f"--- Passage {i} ---")
            lines.append(text)
            lines.append("")
    passages_block = "\n".join(lines).rstrip()

    parts: List[str] = [
        passages_block,
        "",
        GENERATION_USER_GUIDANCE_HOTPOT,
        "",
        "Question:",
        q if q else "(No question)",
    ]
    return "\n".join(parts)


def select_generation_prompts(domain: Any):
    """
    Domain-aware prompt dispatch for the main generation stage.

    Returns (system_instruction, user_content_builder). Defaults to the CUAD path
    so existing behavior is unchanged unless domain == "hotpot".
    """
    if str(domain or "").lower() == "hotpot":
        return GENERATION_SYSTEM_INSTRUCTION_HOTPOT, build_generation_user_content_hotpot
    return GENERATION_SYSTEM_INSTRUCTION_MAIN, build_generation_user_content_main


def build_generation_messages(system_instruction: str, user_content: str) -> List[Dict[str, str]]:
    """Zero-shot chat messages for one generation stage."""
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def build_generation_prompt_fallback(system_instruction: str, user_content: str) -> str:
    """Plain-text fallback: system block + user block (no chat template)."""
    return (
        "[System]\n"
        + system_instruction
        + "\n\n[User]\n"
        + user_content
    )


def tokenizer_supports_chat_template(tokenizer: Any) -> bool:
    """True if tokenizer exposes a non-empty chat_template and apply_chat_template."""
    if tokenizer is None:
        return False
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    try:
        tpl = getattr(tokenizer, "chat_template", None)
    except Exception:
        return False
    if tpl is None:
        return False
    if isinstance(tpl, str) and not tpl.strip():
        return False
    return True


def build_generation_prompt(query: str, contexts: List[str]) -> str:
    """
    Backward-compatible helper: Stage 1 plain prompt (TARGET-only main answer).
    """
    q = rewrite_query_for_generation(str(query or "").strip())
    user_content = build_generation_user_content_main(q, contexts)
    return build_generation_prompt_fallback(GENERATION_SYSTEM_INSTRUCTION_MAIN, user_content)


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
    - l2_contexts / context_l2 / l2_context (하위 호환: b1_contexts 계열)
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

    retrieval = item.get("retrieval") or {}
    l2 = (
        item.get("l2_contexts")
        or item.get("context_l2")
        or item.get("l2_context")
        or item.get("b1_contexts")
        or item.get("context_b1")
        or item.get("b1_context")
        or (retrieval.get("l2") or {}).get("contexts")   # ← 중첩 구조 fallback
    )
    mmr = (
        item.get("mmr_contexts")
        or item.get("context_mmr")
        or item.get("mmr_context")
        or (retrieval.get("mmr") or {}).get("contexts")  # ← 중첩 구조 fallback
    )
    aspects_raw = item.get("aspect_queries") or item.get("query_aspects") or []
    if isinstance(aspects_raw, str):
        aspects_raw = [aspects_raw]
    chunk_lut = item.get("chunk_lut") or {
        "l2":  (retrieval.get("l2") or {}).get("chunk_lut") or [],
        "mmr": (retrieval.get("mmr") or {}).get("chunk_lut") or [],
    }
    l2_list = _as_str_list(l2)
    mmr_list = _as_str_list(mmr)
    aspect_queries = _as_str_list(aspects_raw)

    return {
        "query": str(q).strip(),
        "ground_truth": gt.strip(),
        "l2_contexts": l2_list,
        "mmr_contexts": mmr_list,
        "aspect_queries": aspect_queries,
        "chunk_lut": chunk_lut,
    }


def normalize_to_canonical_eval_record(
    item: Dict[str, Any],
    query_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize retrieval record into canonical evaluation schema."""
    n = normalize_eval_item(item)
    # Reuse normalize_eval_item's chunk_lut (top-level or retrieval.l2 / retrieval.mmr nested).
    chunk_lut = n.get("chunk_lut") or {}
    l2_lut = chunk_lut.get("l2") or []
    mmr_lut = chunk_lut.get("mmr") or []

    mode_upper = str(item.get("mode", "")).upper().strip()
    if mode_upper == "L2":
        l2_contexts = n["l2_contexts"] or _as_str_list(item.get("contexts"))
        mmr_contexts = n["mmr_contexts"]
        l2_lut = l2_lut or []
    elif mode_upper == "MMR":
        l2_contexts = n["l2_contexts"]
        mmr_contexts = n["mmr_contexts"] or _as_str_list(item.get("contexts"))
        mmr_lut = mmr_lut or []
    else:
        l2_contexts = n["l2_contexts"]
        mmr_contexts = n["mmr_contexts"]

    # Multi-hop supporting-passage recall (HotpotQA): carry gold_support and the
    # ranked parent doc_ids per arm. Absent/None for CUAD → downstream stays empty.
    gold_support = item.get("gold_support")
    rdi = item.get("retrieved_doc_ids") or {}
    l2_doc_ids = rdi.get("l2") if isinstance(rdi, dict) else None
    mmr_doc_ids = rdi.get("mmr") if isinstance(rdi, dict) else None

    canonical = {
        "query_idx": int(item.get("query_idx", query_idx if query_idx is not None else 0)),
        "query": n["query"],
        "rewritten_query": item.get("rewritten_query"),
        "ground_truth": n["ground_truth"],
        "aspect_queries": n["aspect_queries"],
        "gold_support": [int(g) for g in gold_support] if gold_support else None,
        "retrieval": {
            "l2": {
                "contexts": l2_contexts,
                "chunk_lut": l2_lut if isinstance(l2_lut, list) else [],
                "retrieved_doc_ids": [int(x) for x in l2_doc_ids] if l2_doc_ids else None,
            },
            "mmr": {
                "contexts": mmr_contexts,
                "chunk_lut": mmr_lut if isinstance(mmr_lut, list) else [],
                "retrieved_doc_ids": [int(x) for x in mmr_doc_ids] if mmr_doc_ids else None,
            },
        },
        "generation": {
            "l2": {"answer": None},
            "mmr": {"answer": None},
        },
        "metadata": {
            "model_name": item.get("model_name"),
            "quantization": item.get("quantization"),
            "run_id": item.get("run_id"),
            "sweep_idx": item.get("sweep_idx"),
            "domain": item.get("domain"),
            "top_k": item.get("top_k"),
            "window_size": item.get("window_size"),
            "mmr_lambda": item.get("mmr_lambda"),
            "tau_dup": item.get("tau_dup"),
            "gamma": item.get("gamma"),
            "retrieval_mode": item.get("retrieval_mode") or item.get("mode"),
        },
    }
    return canonical


def attach_generation_to_canonical_record(
    canonical_record: Dict[str, Any],
    *,
    rewritten_query: Optional[str],
    answer_l2: Optional[str],
    answer_mmr: Optional[str],
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach generated answers to canonical schema record."""
    out = dict(canonical_record)
    out["rewritten_query"] = rewritten_query
    out["generation"] = {
        "l2": {"answer": answer_l2},
        "mmr": {"answer": answer_mmr},
    }
    meta = dict(out.get("metadata") or {})
    if metadata_overrides:
        for k, v in metadata_overrides.items():
            if v is not None:
                meta[k] = v
    out["metadata"] = meta
    return out


def save_normalized_generation_export(path: str, records: List[Dict[str, Any]]) -> None:
    """Save canonical evaluation-ready generation records as JSON list."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ========== 가상 입력 (JSON 미지정 시) ==========
EVAL_DATA: List[Dict[str, Any]] = [
    {
        "query": "What is the main obligation of the borrower?",
        "l2_contexts": [
            "The borrower shall repay the principal amount within 24 months.",
            "Interest rate is fixed at 5% per annum.",
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
        "l2_contexts": [
            "This agreement is effective from January 1, 2024.",
            "The contract shall expire on December 31, 2025.",
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
    max_new_tokens: int = 512,
    *,
    domain: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Single-stage generation. CUAD path is unchanged; domain=="hotpot" uses the
    multi-hop passage prompt (no TARGET filtering / no query rewrite).
    """
    import torch

    original_query = str(query or "").strip()
    system_instruction, user_content_builder = select_generation_prompts(domain)
    is_hotpot = str(domain or "").lower() == "hotpot"
    if is_hotpot:
        # Hotpot queries are already natural questions; use raw query + contexts as-is.
        generation_query = original_query
        main_user = user_content_builder(generation_query, contexts)
    else:
        generation_query = rewrite_query_for_generation(original_query)
        target_ctx = filter_target_contract_contexts(contexts)
        main_user = user_content_builder(generation_query, target_ctx)
    meta: Dict[str, Any] = {
        "mode": "single_stage_hotpot" if is_hotpot else "single_stage_target_only",
        "original_query": original_query,
        "rewritten_query": generation_query,
        "system_prompt_main": system_instruction,
        "user_prompt_main": main_user,
        "stage_prompts": {},
    }

    def _run_generate_from_ids(input_ids: Any, attention_mask: Any) -> str:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        terminators = [tokenizer.eos_token_id]
        try:
            terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        except Exception:
            pass
        terminators = list(dict.fromkeys(t for t in terminators if t is not None))
        gen_kw: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.1,
            "repetition_penalty": 1.05,
            "pad_token_id": pad_id,
        }
        if terminators:
            gen_kw["eos_token_id"] = terminators
        with torch.no_grad():
            out = model.generate(**gen_kw)
        in_len = int(input_ids.shape[1])
        gen = out[0][in_len:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _run_single_stage(system_instruction: str, user_content: str, stage_key: str) -> str:
        stage_meta: Dict[str, Any] = {
            "mode": "plain_prompt",
            "system_prompt": system_instruction,
            "user_prompt": user_content,
            "messages": None,
            "fallback_prompt": None,
            "rendered_prompt_text": None,
            "decoded_input_text": None,
        }
        use_chat = tokenizer_supports_chat_template(tokenizer)
        if use_chat:
            try:
                messages = build_generation_messages(system_instruction, user_content)
                stage_meta["messages"] = messages
                stage_meta["mode"] = "chat_template"
                prompt_tok = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                stage_meta["rendered_prompt_text"] = prompt_tok
                try:
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=GENERATION_MAX_INPUT_LENGTH,
                    )
                except TypeError:
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                if torch.is_tensor(encoded):
                    input_ids = encoded
                    attn = None
                else:
                    input_ids = encoded["input_ids"]
                    attn = encoded.get("attention_mask")
                if attn is None:
                    attn = torch.ones_like(input_ids)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    attn = attn.unsqueeze(0) if attn.dim() == 1 else attn
                max_len = GENERATION_MAX_INPUT_LENGTH
                if int(input_ids.shape[1]) > max_len:
                    input_ids = input_ids[:, :max_len]
                    attn = attn[:, :max_len]
                input_ids = input_ids.to(model.device)
                attn = attn.to(model.device)
                stage_meta["decoded_input_text"] = tokenizer.decode(
                    input_ids[0], skip_special_tokens=False
                )
                answer = _run_generate_from_ids(input_ids, attn)
                meta["stage_prompts"][stage_key] = stage_meta
                return answer
            except Exception:
                stage_meta["mode"] = "plain_prompt"

        fallback = build_generation_prompt_fallback(system_instruction, user_content)
        stage_meta["fallback_prompt"] = fallback
        stage_meta["rendered_prompt_text"] = fallback
        enc = tokenizer(
            fallback,
            return_tensors="pt",
            truncation=True,
            max_length=GENERATION_MAX_INPUT_LENGTH,
        )
        input_ids = enc["input_ids"].to(model.device)
        stage_meta["decoded_input_text"] = tokenizer.decode(
            input_ids[0], skip_special_tokens=False
        )
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(model.device)
        else:
            attn = torch.ones_like(input_ids)
        answer = _run_generate_from_ids(input_ids, attn)
        meta["stage_prompts"][stage_key] = stage_meta
        return answer

    # Optional lossless memoization (env LLM_GEN_CACHE=<sqlite path>; unset = off).
    # Decoding is greedy (do_sample=False), so identical prompt => identical output.
    # A hit skips the forward pass entirely, which is what makes the 108-combo
    # sweep affordable (the L2 arm is invariant to lambda/tau/gamma).
    _cache_key = None
    try:
        from scripts import gen_cache as _gen_cache
    except ImportError:
        try:
            import gen_cache as _gen_cache  # type: ignore
        except ImportError:
            _gen_cache = None  # type: ignore
    if _gen_cache is not None and _gen_cache.cache_path():
        _cache_key = _gen_cache.make_key(system_instruction, main_user, max_new_tokens)
        _hit = _gen_cache.get(_cache_key)
        if _hit is not None:
            meta["cache_hit"] = True
            return _hit, meta

    answer = _run_single_stage(
        system_instruction,
        main_user,
        stage_key="main",
    )
    if _cache_key is not None and _gen_cache is not None:
        _gen_cache.put(_cache_key, answer)
        meta["cache_hit"] = False
    return answer, meta


def persist_generation_prompt_debug_files(
    directory: str,
    query_idx: int,
    variant: str,
    meta: Dict[str, Any],
) -> None:
    """
    Write rendered template / decoded input_ids to separate UTF-8 text files for debugging.
    Replaces inline `rendered_prompt_text` and `decoded_input_text` in `meta` with absolute paths.
    """
    os.makedirs(directory, exist_ok=True)
    stage_prompts = meta.get("stage_prompts")
    if isinstance(stage_prompts, dict) and stage_prompts:
        for stage_key, stage_meta in stage_prompts.items():
            if not isinstance(stage_meta, dict):
                continue
            persist_generation_prompt_debug_files(
                directory,
                query_idx,
                f"{variant}_{stage_key}",
                stage_meta,
            )
        meta["stage_prompt_paths"] = {
            str(k): {
                "rendered_prompt_path": (v or {}).get("rendered_prompt_path")
                if isinstance(v, dict)
                else None,
                "decoded_input_path": (v or {}).get("decoded_input_path")
                if isinstance(v, dict)
                else None,
            }
            for k, v in stage_prompts.items()
        }
        return
    base = os.path.join(directory, f"q{query_idx:05d}_{variant}")
    rendered_path = base + "_rendered.txt"
    decoded_path = base + "_decoded.txt"
    rendered = meta.get("rendered_prompt_text")
    decoded = meta.get("decoded_input_text")
    if rendered is not None:
        with open(rendered_path, "w", encoding="utf-8") as wf:
            wf.write(rendered if isinstance(rendered, str) else str(rendered))
        meta["rendered_prompt_path"] = os.path.abspath(rendered_path)
    else:
        meta["rendered_prompt_path"] = None
    if decoded is not None:
        with open(decoded_path, "w", encoding="utf-8") as wf:
            wf.write(decoded if isinstance(decoded, str) else str(decoded))
        meta["decoded_input_path"] = os.path.abspath(decoded_path)
    else:
        meta["decoded_input_path"] = None
    meta.pop("rendered_prompt_text", None)
    meta.pop("decoded_input_text", None)


def run_generation(
    eval_data: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 512,
    *,
    include_prompts: bool = False,
    prompt_debug_dir: Optional[str] = None,
    default_domain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    eval_data 순회: l2_contexts / mmr_contexts 각각으로 답변 생성.
    평가 메트릭 없이 생성 결과만 반환한다.
    반환: per_query_details 리스트

    도메인은 레코드의 domain 필드를 우선하고, 없으면 default_domain(CLI --domain)을 사용한다.
    """
    per_query_details: List[Dict[str, Any]] = []

    for i, raw_item in enumerate(eval_data):
        item = normalize_eval_item(raw_item)
        record_domain = raw_item.get("domain") or default_domain
        chunk_lut = item.get("chunk_lut")
        canonical_record = normalize_to_canonical_eval_record(raw_item, query_idx=i)
        query = item["query"]
        l2_ctx = item["l2_contexts"]
        mmr_ctx = item["mmr_contexts"]
        aspects_raw_list = [str(x).strip() for x in item.get("aspect_queries", []) if str(x).strip()]
        aspects_list = list(aspects_raw_list)
        if aspects_list and "[SEP]" in aspects_list[0]:
            sep_parts = [p.strip() for p in aspects_list[0].split("[SEP]")]
            kept = [p for p in sep_parts[1:] if p]
            aspects_list = kept + aspects_list[1:]
        formatted_lines = [f"{idx}. {txt}" for idx, txt in enumerate(aspects_list, start=1)]
        formatted_aspects = "\n".join(formatted_lines).strip()

        answer_l2, prompt_meta_l2 = generate_answer(
            model, tokenizer, query, l2_ctx, max_new_tokens=max_new_tokens, domain=record_domain
        )
        answer_mmr, prompt_meta_mmr = generate_answer(
            model, tokenizer, query, mmr_ctx, max_new_tokens=max_new_tokens, domain=record_domain
        )

        if prompt_debug_dir:
            pdir = os.path.abspath(prompt_debug_dir)
            for variant, pm in (("l2", prompt_meta_l2), ("mmr", prompt_meta_mmr)):
                persist_generation_prompt_debug_files(pdir, i, variant, pm)

        rewritten_query = prompt_meta_l2.get(
            "rewritten_query", rewrite_query_for_generation(query)
        )

        detail: Dict[str, Any] = {
            "query_idx": i,
            "query": query,
            "rewritten_query": rewritten_query,
            "ground_truth": item["ground_truth"],
            "aspect_queries_raw": aspects_list,
            "aspect_queries": aspects_list,
            "aspect_queries_formatted": formatted_aspects,
            "answer_l2": answer_l2,
            "answer_mmr": answer_mmr,
            "generation": {
                "answer_l2": answer_l2,
                "answer_mmr": answer_mmr,
            },
            "chunk_lut": chunk_lut,
            "canonical_record": attach_generation_to_canonical_record(
                canonical_record,
                rewritten_query=rewritten_query,
                answer_l2=answer_l2,
                answer_mmr=answer_mmr,
            ),
        }
        if include_prompts:
            detail["prompt"] = {
                "l2": prompt_meta_l2,
                "mmr": prompt_meta_mmr,
            }
        per_query_details.append(detail)

    return per_query_details


def run_llm_pipeline_from_retrieval(
    artifact_root: str,
    jsonl_path: str,
    run_id: str,
    *,
    llm_output_root: Optional[str],
    output_normalized_json: Optional[str],
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    target_indices: Optional[List[int]] = None,
    default_domain: Optional[str] = None,
) -> None:
    """
    sweep_eval이 저장한 retrieval_*.jsonl을 sweep_idx별로 묶어 LLM 생성 후
    sweep_exports와 유사한 패턴으로 JSONL / combos / manifest 저장.
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

    print(
        f"[LLM 파이프라인] 검색 레코드 {len(records)}줄, sweep 조합 {len(groups)}개",
        flush=True,
    )
    print(f"[LLM 파이프라인] 출력 루트: {out_root}", flush=True)

    model, tokenizer = load_llm(model_name, quantization)
    with open(detail_path, "w", encoding="utf-8") as detail_f:
        normalized_records: List[Dict[str, Any]] = []
        for sweep_idx, group_rows in groups.items():
            raw_eval = [retrieval_record_to_eval_item(r) for r in group_rows]

            combo_prompt_dir = os.path.join(
                out_root, "generation_prompt_logs", f"combo_{sweep_idx:05d}"
            )
            per_query_details = run_generation(
                raw_eval,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                include_prompts=True,
                prompt_debug_dir=combo_prompt_dir,
                default_domain=default_domain,
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
                "per_query_results": [],
            }

            for r, det in zip(group_rows, per_query_details):
                canonical = attach_generation_to_canonical_record(
                    normalize_to_canonical_eval_record(r, query_idx=int(r.get("query_idx", 0))),
                    rewritten_query=det.get("rewritten_query"),
                    answer_l2=det.get("answer_l2"),
                    answer_mmr=det.get("answer_mmr"),
                    metadata_overrides={
                        "model_name": model_name,
                        "quantization": quantization,
                        "run_id": run_id,
                        "sweep_idx": sweep_idx,
                        "domain": r.get("domain"),
                        "top_k": first.get("top_k"),
                        "window_size": first.get("window_size"),
                        "mmr_lambda": first.get("mmr_lambda"),
                        "tau_dup": first.get("tau_dup"),
                        "gamma": first.get("gamma"),
                    },
                )
                normalized_records.append(canonical)
                line = {
                    "run_id": run_id,
                    "sweep_idx": sweep_idx,
                    "query_idx": int(r.get("query_idx", 0)),
                    "domain": r.get("domain"),
                    **params,
                    "query": det.get("rewritten_query"),
                    "aspect_queries": det.get("aspect_queries", []),
                    "aspect_queries_raw": det.get("aspect_queries_raw", []),
                    "aspect_queries_formatted": det.get("aspect_queries_formatted", ""),
                    "answer_l2": det.get("answer_l2"),
                    "answer_mmr": det.get("answer_mmr"),
                    "rewritten_query": det.get("rewritten_query"),
                    "ground_truth": r.get("ground_truth"),
                    "prompt": det.get("prompt"),
                    "generation": det.get("generation"),
                    "chunk_lut": det.get("chunk_lut"),
                }
                detail_f.write(json.dumps(line, ensure_ascii=False) + "\n")

                combo_payload["per_query_results"].append(
                    {
                        "query_idx": int(r.get("query_idx", 0)),
                        "query": det.get("rewritten_query"),
                        "aspect_queries": det.get("aspect_queries", []),
                        "aspect_queries_raw": det.get("aspect_queries_raw", []),
                        "aspect_queries_formatted": det.get("aspect_queries_formatted", ""),
                        "answer_l2": det.get("answer_l2"),
                        "answer_mmr": det.get("answer_mmr"),
                        "rewritten_query": det.get("rewritten_query"),
                        "ground_truth": r.get("ground_truth"),
                        "prompt": det.get("prompt"),
                        "generation": det.get("generation"),
                        "chunk_lut": det.get("chunk_lut"),
                    }
                )
            print(
                f"[조합 sweep_idx={sweep_idx}] 쿼리 {len(group_rows)}건 → {combo_path}",
                flush=True,
            )

            with open(combo_path, "w", encoding="utf-8") as jf:
                json.dump(combo_payload, jf, ensure_ascii=False, indent=2)

    if output_normalized_json:
        save_normalized_generation_export(output_normalized_json, normalized_records)
        print(f"[완료] 정규화 생성 JSON: {output_normalized_json}", flush=True)

    manifest_llm = {
        "run_id": run_id,
        "source_retrieval_jsonl": os.path.abspath(jsonl_path),
        "artifact_root_search": os.path.abspath(artifact_root),
        "llm_output_root": out_root,
        "llm_generation_detail_jsonl": os.path.abspath(detail_path),
        "model_name": model_name,
        "quantization": quantization,
        "num_parameter_combos": len(groups),
    }
    man_path = os.path.join(out_root, "llm_run_manifest.json")
    with open(man_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_llm, mf, ensure_ascii=False, indent=2)

    print(f"\n[완료] 상세 JSONL: {detail_path}", flush=True)
    print(f"[완료] {man_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUAD RAG JSON → 로컬 LLM 생성 → JSONL 저장"
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
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="로컬 생성용 LLM",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--normalized_retrieval_json",
        type=str,
        default=None,
        help="online_accelerator 정규화 검색 결과 JSON 경로",
    )
    parser.add_argument(
        "--output_normalized_json",
        type=str,
        default="generation_normalized.json",
        help="정규화된 평가용 생성 결과(JSON list) 저장 경로",
    )
    parser.add_argument(
        "--prompt_debug_dir",
        type=str,
        default=None,
        help=(
            "Directory for rendered chat prompt and decoded input_ids (.txt) per query/variant. "
            "Default: generation_prompt_logs under current working directory."
        ),
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
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["cuad", "hotpot"],
        help=(
            "생성 프롬프트 도메인 fallback. 레코드의 domain 필드가 있으면 그것을 우선하고, "
            "없을 때만 이 값을 사용한다 (기본: CUAD 동작)."
        ),
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
                output_normalized_json=args.output_normalized_json,
                model_name=mname,
                quantization=quant,
                max_new_tokens=args.max_new_tokens,
                target_indices=args.target_indices,
                default_domain=args.domain,
            )

        _call_retrieval_pipeline(args.model, args.quantization)
        return

    if args.normalized_retrieval_json:
        if not os.path.isfile(args.normalized_retrieval_json):
            print(f"[오류] 파일 없음: {args.normalized_retrieval_json}", file=sys.stderr)
            sys.exit(1)
        eval_data = load_eval_data_json(args.normalized_retrieval_json)
        print(f"[로드] {args.normalized_retrieval_json} → {len(eval_data)}개 항목")
    else:
        eval_data = EVAL_DATA

    if not eval_data:
        print("생성 데이터가 비어 있습니다.")
        return

    model, tokenizer = load_llm(args.model, args.quantization)

    prompt_dbg = (
        os.path.abspath(args.prompt_debug_dir)
        if args.prompt_debug_dir
        else os.path.join(os.getcwd(), "generation_prompt_logs")
    )
    os.makedirs(prompt_dbg, exist_ok=True)
    print(f"[prompt debug] rendered/decoded .txt → {prompt_dbg}", flush=True)

    per_query_details = run_generation(
        eval_data,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        include_prompts=True,
        prompt_debug_dir=prompt_dbg,
        default_domain=args.domain,
    )

    normalized_records: List[Dict[str, Any]] = []
    for detail in per_query_details:
        canonical = detail.get("canonical_record")
        if isinstance(canonical, dict):
            canonical = attach_generation_to_canonical_record(
                canonical,
                rewritten_query=detail.get("rewritten_query"),
                answer_l2=detail.get("answer_l2"),
                answer_mmr=detail.get("answer_mmr"),
                metadata_overrides={
                    "model_name": args.model,
                    "quantization": args.quantization,
                },
            )
            normalized_records.append(canonical)
    out_normalized_json = os.path.abspath(args.output_normalized_json)
    save_normalized_generation_export(out_normalized_json, normalized_records)
    print(f"[JSON] 정규화 생성 결과 저장: {out_normalized_json}")


if __name__ == "__main__":
    main()
