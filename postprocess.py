"""
postprocess.py — chunk_lut 정규식 매핑 + 로컬 답변 정제.

외부 API / 네트워크 불필요. air-gapped edge device에서 동작 가능.

단계 1: chunk_lut + 정규식으로 조항 번호를 매핑 → clause_index_l2 / clause_index_mmr
단계 2: Step 헤더 제거 + Chunk N → 실제 조항 번호 치환 → refined_answer_l2 / refined_answer_mmr

입력:  llm_generator가 출력한 JSONL (chunk_lut 키 포함)
출력:  clause_index / refined_answer 가 추가된 JSONL

사용 예:
    python postprocess.py \
        --input  results_1q.jsonl \
        --output results_1q_final.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

from evaluation.metrics.answer_em_f1 import extract_final_answer

# 조항 번호 탐색 패턴 (우선순위 순서)
_CLAUSE_PATTERNS = [
    # 1. 명시적 키워드가 있는 경우 (가장 확실함)
    re.compile(r"\b(?:Section|Clause|Article|Appendix|Schedule|Exhibit)\s+[\d]+(?:\.\d+)*(?:\([a-z]\))?", re.IGNORECASE),
    re.compile(r"\§\s*[\d]+(?:\.\d+)*"),
    
    # 2. 키워드 없는 '숫자.숫자' 형태 (예: 16.2, 1.1.1)
    re.compile(r"\b\d+\.\d+(?:\.\d+)*\b"),
    
    # 3. 키워드 없는 대분류 조항 '숫자.' 형태 (예: " 14. ")
    re.compile(r"(?<=\s)\d{1,2}\.(?=\s+[A-Z])")
]
# answer에서 인용된 "Chunk N" 번호 추출
_CHUNK_REF_RE = re.compile(r"\bChunks?\s+(\d+)\b", re.IGNORECASE)

# RAG 파이프라인 내부/메타 표현 제거 패턴 (직후에 오는 소문자 캡처)
_META_PHRASES_RE = re.compile(
    r"(?i)^(?:\s*)(?:based on the primary evidence|reviewing the supplemental context|in summary|in conclusion|based on the provided text|according to the context)[,\:]?\s+([a-z])",
    re.MULTILINE
)
# Step 헤더 패턴: [Step N: Title] 또는 **Step N:** 또는 **Step N.**
_STEP_HEADER_RE = re.compile(
    r"(?:\*{1,2})?"
    r"\[?Step\s+\d+\s*[:\.]?\s*[^\]\n]*\]?"
    r"(?:\*{1,2})?"
    r"\s*\n?",
    re.IGNORECASE,
)

# Step 4 블록 전체 제거 패턴 (헤더 + 본문, 항상 마지막 섹션)
_STEP4_BLOCK_RE = re.compile(
    r"(?:\*{1,2})?\[?Step\s+4\b[^\n]*\]?(?:\*{1,2})?\n[\s\S]*$",
    re.IGNORECASE,
)

_OUTPUT_FIELDS = [
    "refined_answer_l2",
    "refined_answer_mmr",
]
def _extract_clause_from_text(text: str) -> Optional[str]:
    """청크 텍스트에서 조항 번호를 추출하고, 숫자만 있는 경우 적절한 접두어를 붙인다."""
    if not text:
        return None
        
    for pat in _CLAUSE_PATTERNS:
        m = pat.search(text)
        if m:
            extracted = m.group(0).strip()
            
            # 추출된 문자열에 알파벳(Section, Article 등)이 없는 순수 숫자 형태인 경우
            if not re.search(r'[A-Za-z]', extracted):
                # 청크 텍스트의 문맥을 파악하여 가장 적절한 명칭 추론
                text_lower = text.lower()
                if "article" in text_lower:
                    prefix = "Article"
                elif "clause" in text_lower:
                    prefix = "Clause"
                else:
                    # 영미권 계약서에서 "숫자.숫자" 형태의 기본 표준은 Section
                    prefix = "Section" 
                
                return f"{prefix} {extracted}"
            
            # 명시적 키워드가 이미 포함된 경우 원본 그대로 반환
            return extracted
            
    return None

def _fallback_phrase(text: str, max_words: int = 8) -> str:
    """Return a short readable fallback phrase when no clause number is found."""
    raw = " ".join((text or "").split())
    if not raw:
        return "the relevant provision"

    lower = raw.lower()

    # Heuristic label mapping for more natural prose.
    if "export" in lower or "import" in lower or "control regulations" in lower:
        return "the export/import control provision"
    if "transition plan" in lower:
        return "the transition plan provision"
    if "assignment" in lower or "assigned" in lower:
        return "the assignment provision"
    if "terminate" in lower or "termination" in lower:
        return "the termination provision"
    if "survive" in lower or "survival" in lower:
        return "the survival provision"
    if "confidential" in lower:
        return "the confidentiality provision"
    if "warrant" in lower or "warranty" in lower:
        return "the warranty provision"
    if "price" in lower or "payment" in lower:
        return "the price and payment provision"
    if "manufacture" in lower or "manufacturing" in lower:
        return "the manufacturing provision"

    words = raw.split()
    phrase = " ".join(words[:max_words])
    if len(words) > max_words:
        phrase += "…"
    return f"the provision stating \"{phrase}\""
def _render_chunk_reference(entry: Dict[str, Any]) -> str:
    """
    Render the replacement text for one cited chunk.
    Prefer a formal clause reference when available; otherwise use a readable fallback phrase.
    """
    clause_ref = str(entry.get("clause_ref", "") or "").strip()
    if clause_ref:
        if not entry.get("fallback"):
            return clause_ref
        return clause_ref

    return "the relevant provision"
def build_clause_index(
    answer: str,
    lut: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    answer에서 언급된 Chunk 번호만 대상으로 조항 번호를 매핑한다.

    반환 형식:
    [
        {"chunk_no": 1, "doc_id": "contract_A",
         "clause_ref": "Section 18.4", "fallback": False},
        {"chunk_no": 2, "doc_id": "contract_A",
         "clause_ref": "Philips may suspend its obligations…", "fallback": True},
    ]
    """
    if not answer or not lut:
        return []

    # 묶음 표현(Chunks 1, 2, and 6 등)을 펼친 뒤 Chunk 번호를 수집한다 (refine_answer와 동일 규칙).
    expanded = _expand_grouped_chunk_refs(answer)
    cited_nos = list(dict.fromkeys(
        int(m) for m in _CHUNK_REF_RE.findall(expanded)
    ))

    result: List[Dict[str, Any]] = []
    lut_by_no = {entry["chunk_no"]: entry for entry in lut}

    for no in cited_nos:
        entry = lut_by_no.get(no)
        if not entry:
            continue  # answer가 존재하지 않는 chunk 번호를 언급한 경우 무시
        chunk_text = entry.get("chunk_text", "")
        clause = _extract_clause_from_text(chunk_text)
        if clause:
            result.append({
                "chunk_no": no,
                "doc_id": entry.get("doc_id", ""),
                "clause_ref": clause,
                "fallback": False,
            })
        else:
            result.append({
                "chunk_no": no,
                "doc_id": entry.get("doc_id", ""),
                "clause_ref": _fallback_phrase(chunk_text),
                "fallback": True,
            })

    return result

def _expand_grouped_chunk_refs(text: str) -> str:
    """
    Expand grouped references like 'Chunks 1, 2, and 6' into
    'Chunk 1, Chunk 2, and Chunk 6' so the normal replacement regex can handle them.
    """
    def repl(m: re.Match) -> str:
        nums = re.findall(r"\d+", m.group(0))
        if not nums:
            return m.group(0)
        if len(nums) == 1:
            return f"Chunk {nums[0]}"
        if len(nums) == 2:
            return f"Chunk {nums[0]} and Chunk {nums[1]}"
        return ", ".join(f"Chunk {n}" for n in nums[:-1]) + f", and Chunk {nums[-1]}"

    pattern = re.compile(
        r"\bChunks?\s+\d+(?:\s*,\s*\d+)*(?:\s*,?\s*(?:and|or)\s+\d+)?\b",
        re.IGNORECASE,
    )
    return pattern.sub(repl, text)

def _replace_chunk_refs(answer: str, clause_index: List[Dict[str, Any]]) -> str:
    """
    Replace 'Chunk N' references with actual clause references or readable fallback phrases.
    Unlike the previous version, fallback=True entries are also replaced.
    """
    ref_map = {
        str(entry["chunk_no"]): _render_chunk_reference(entry)
        for entry in clause_index
    }

    def _replacer(m: re.Match) -> str:
        no = m.group(1)
        return ref_map.get(no, m.group(0))

    text = _CHUNK_REF_RE.sub(_replacer, answer)

    # Normalize plural marker if individual replacements already happened.
    text = re.sub(r"\bChunks\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

def refine_answer(answer: str, clause_index: List[Dict[str, Any]]) -> str:
    """
    answer를 로컬 정규식으로 정제한다. API 호출 없음.

    1. Step 4 블록 전체 제거
    2. Step 1~3 헤더 제거
    3. RAG 내부 메타 표현 제거 및 첫 글자 대문자화 (신규)
    4. 묶음 Chunk 표현(예: Chunks 1, 2, and 6)을 Chunk N 단위로 펼침
    5. 'Chunk N' → clause_index 기준으로 실제 조항 등으로 치환
    6. 연속된 빈 줄을 하나로 압축
    """
    if not answer:
        return answer

    # 1. Step 4 블록 전체 제거
    text = _STEP4_BLOCK_RE.sub("", answer)

    # 2. Step 1~3 헤더 제거
    text = _STEP_HEADER_RE.sub("", text)

    # 3. RAG 메타 표현 제거 (캡처된 첫 소문자를 대문자로 변환)
    text = _META_PHRASES_RE.sub(lambda m: m.group(1).upper(), text)

    # 4. 묶음 참조 → 개별 "Chunk N" (이후 치환 정규식과 호환)
    text = _expand_grouped_chunk_refs(text)

    # 5. Chunk N → 실제 조항 번호 등으로 치환
    text = _replace_chunk_refs(text, clause_index)

    # 6. 연속 빈 줄 압축
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _detect_domain(record: Dict[str, Any], default_domain: Optional[str] = None) -> Optional[str]:
    """Domain of a record: metadata.domain > top-level domain > default_domain (CLI)."""
    meta = record.get("metadata")
    if isinstance(meta, dict) and meta.get("domain") is not None:
        return str(meta.get("domain")).strip()
    if record.get("domain") is not None:
        return str(record.get("domain")).strip()
    return default_domain


def process_record(
    record: Dict[str, Any],
    default_domain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    JSONL 레코드 1건을 처리한다.

    - HotpotQA(domain=="hotpot"): 조항 로직을 전부 건너뛰고, 각 arm(L2/MMR)의
      refined_answer 를 extract_final_answer(raw_answer) 로 채운다 (짧은 최종 답).
    - CUAD(기타): 기존 경로 그대로.
      1. clause_index_l2 / clause_index_mmr 생성 (정규식 조항 번호 매핑)
      2. refined_answer_l2 / refined_answer_mmr 생성 (Step 헤더 제거 + Chunk → 조항 번호 치환)
    """
    if (_detect_domain(record, default_domain) or "").lower() == "hotpot":
        record["refined_answer_l2"] = extract_final_answer(record.get("answer_l2", ""))
        record["refined_answer_mmr"] = extract_final_answer(record.get("answer_mmr", ""))
        return record

    chunk_lut = record.get("chunk_lut") or {}
    l2_lut = chunk_lut.get("l2") or []
    mmr_lut = chunk_lut.get("mmr") or []

    record["clause_index_l2"] = build_clause_index(
        record.get("answer_l2", ""), l2_lut
    )
    record["clause_index_mmr"] = build_clause_index(
        record.get("answer_mmr", ""), mmr_lut
    )
    record["refined_answer_l2"] = refine_answer(
        record.get("answer_l2", ""), record["clause_index_l2"]
    )
    record["refined_answer_mmr"] = refine_answer(
        record.get("answer_mmr", ""), record["clause_index_mmr"]
    )

    return record


def _load_input_records(path: str) -> List[Dict[str, Any]]:
    """
    입력 파일에서 레코드를 로드한다.
    - JSONL: 줄 단위 JSON object
    - JSON:  object 1건 또는 object list
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if not raw_text.strip():
        return []

    # 1) JSON 전체 파싱 시도 (list/dict 지원)
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # 2) JSONL 파싱 fallback
    records: List[Dict[str, Any]] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[경고] JSON 파싱 실패: {e}", file=sys.stderr)
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _ensure_flat_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both the legacy flat record and the canonical nested record.

    process_record expects top-level answer_l2 / answer_mmr / chunk_lut{l2,mmr}.
    The current llm_generator emits canonical records where answers live under
    generation.{l2,mmr}.answer and the LUT under retrieval.{l2,mmr}.chunk_lut.
    Without this bridge, canonical input yields empty refined_answer_* fields.
    """
    gen = r.get("generation") or {}
    retrieval = r.get("retrieval") or {}

    def _lut_list(v: Any) -> List[Any]:
        return v if isinstance(v, list) else []

    top_lut = r.get("chunk_lut") or {}
    l2_lut = _lut_list(top_lut.get("l2"))
    mmr_lut = _lut_list(top_lut.get("mmr"))
    if not l2_lut and not mmr_lut:
        l2_lut = _lut_list((retrieval.get("l2") or {}).get("chunk_lut"))
        mmr_lut = _lut_list((retrieval.get("mmr") or {}).get("chunk_lut"))

    ans_l2 = r.get("answer_l2") or (gen.get("l2") or {}).get("answer") or ""
    ans_mmr = r.get("answer_mmr") or (gen.get("mmr") or {}).get("answer") or ""

    out = dict(r)
    out["chunk_lut"] = {"l2": l2_lut, "mmr": mmr_lut}
    out["answer_l2"] = ans_l2 if isinstance(ans_l2, str) else (str(ans_l2) if ans_l2 is not None else "")
    out["answer_mmr"] = ans_mmr if isinstance(ans_mmr, str) else (str(ans_mmr) if ans_mmr is not None else "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="chunk_lut → 정규식 조항 번호 매핑 후처리"
    )
    parser.add_argument("--input",  required=True, help="입력 JSONL 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["cuad", "hotpot"],
        help=(
            "도메인 fallback. 레코드의 metadata.domain(또는 top-level domain)이 있으면 "
            "그것을 우선하고, 없을 때만 이 값을 사용한다 (기본: CUAD 조항 후처리)."
        ),
    )
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output

    records = _load_input_records(in_path)
    if not records:
        print(
            "[오류] 입력에서 유효한 JSON 레코드를 찾지 못했습니다. "
            "출력 파일을 생성하지 않습니다.",
            file=sys.stderr,
        )
        sys.exit(1)

    processed = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for record in records:
            record = process_record(_ensure_flat_record(record), default_domain=args.domain)
            slim = {k: record.get(k) for k in _OUTPUT_FIELDS}
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")
            processed += 1

    print(f"[완료] {processed}건 처리 → {out_path}")


if __name__ == "__main__":
    main()
