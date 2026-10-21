"""
postprocess.py — chunk_lut 정규식 매핑 + 로컬 답변 정제.

외부 API / 네트워크 불필요. air-gapped edge device에서 동작 가능.

단계 1: chunk_lut + 정규식으로 조항 번호를 매핑 → clause_index_l2 / clause_index_mmr
단계 2: Step 헤더 제거 + Chunk N → 실제 조항 번호 치환 → refined_answer_l2 / refined_answer_mmr

Clause resolution (step ⑤ mapping):
  - Prefer the MOST SPECIFIC token in a chunk
    (Section X.Y.Z > Section X.Y > §X.Y > ARTICLE N), not first-match.
  - Headerless windows inherit from the nearest preceding same-doc chunk
    (suffix→prefix overlap); recorded as resolved_by="inherited".
  - Keyword fallbacks score the operative topic; never emit quoted fragments
    (neutral: "an unnumbered provision").

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
from typing import Any, Dict, List, Optional, Tuple

# Clause-token finders. Specificity (not first-match) decides the winner:
#   Section/Clause X.Y.Z > Section/Clause X.Y > §X.Y > ARTICLE N
_SECTION_CLAUSE_RE = re.compile(
    r"\b(?P<kw>Section|Clause)\s+(?P<num>\d+(?:\.\d+)*(?:\([a-z]\))?)",
    re.IGNORECASE,
)
_ARTICLE_LIKE_RE = re.compile(
    r"\b(?P<kw>Article|Appendix|Schedule|Exhibit)\s+(?P<num>\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_SECTION_SYMBOL_RE = re.compile(r"§\s*(?P<num>\d+(?:\.\d+)*)")
_BARE_DOTTED_RE = re.compile(r"\b(?P<num>\d+\.\d+(?:\.\d+)*)\b")
_BARE_MAJOR_RE = re.compile(r"(?<=\s)(?P<num>\d{1,2})\.(?=\s+[A-Z])")

# answer에서 인용된 "Chunk N" 번호 추출
_CHUNK_REF_RE = re.compile(r"\bChunks?\s+(\d+)\b", re.IGNORECASE)

# Operative-verb boosts for topic fallback (whole-chunk scoring).
_OPERATIVE_VERB_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmay\s+be\s+terminated\b",
        r"\bshall\s+(?:be\s+)?terminat(?:e|ed|ion)\b",
        r"\bupon\s+(?:the\s+)?termination\b",
        r"\bshall\s+refund\b",
        r"\bshall\s+(?:return|destroy)\b",
        r"\bmay\s+suspend\b",
        r"\bshall\s+(?:indemnify|defend)\b",
        r"\bshall\s+not\s+be\s+liable\b",
        r"\btotal\s+liability\b",
        r"\blimit\s+of\s+liability\b",
        r"\bexclusive\s+remed(?:y|ies)\b",
    )
]

# Topic lexicon: (label_phrase, keyword_patterns, operative_extra_patterns)
# Scores sum keyword hits + operative bonuses; list-item "assignment for the
# benefit of creditors" alone must not beat a termination-framed chunk.
_TOPIC_LEXICON: List[Tuple[str, List[str], List[str]]] = [
    (
        "the termination provision",
        ["terminat", "expir"],
        [r"\bpetition\s+in\s+bankruptcy\b", r"\bfails?\s+to\s+cure\b", r"\bdefault\b"],
    ),
    (
        "the refund provision",
        ["refund", "returned to", "have been destroyed", "return all copies"],
        [r"\bshall\s+refund\b", r"\bsole\s+remedy\b"],
    ),
    (
        "the liability provision",
        ["limit of liability", "total liability", "consequential damages", "liable"],
        [r"\bshall\s+not\s+be\s+liable\b", r"\blimit(?:ed|ation)?\s+of\s+liability\b"],
    ),
    (
        "the indemnification provision",
        ["indemnif", "hold them harmless", "hold harmless"],
        [r"\bshall\s+(?:indemnify|defend)\b"],
    ),
    (
        "the warranty provision",
        ["warrant"],
        [r"\bwarranty\s+described\b", r"\bdisclaims?\s+.*warrant"],
    ),
    (
        "the assignment provision",
        ["assignment of this agreement", "may not assign", "shall not assign"],
        [r"\bassign(?:s|ment)?\s+(?:of\s+)?this\s+agreement\b"],
    ),
    (
        "the confidentiality provision",
        ["confidential"],
        [r"\bconfidential\s+information\b"],
    ),
    (
        "the export/import control provision",
        ["export", "import", "control regulations"],
        [],
    ),
    (
        "the survival provision",
        ["surviv"],
        [r"\bshall\s+survive\b"],
    ),
    (
        "the payment provision",
        ["payment", "fees due", "amount due"],
        [r"\bshall\s+pay\b"],
    ),
]

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
    "query_idx",
    "query",
    "ground_truth",
    "answer_l2",
    "answer_mmr",
    "clause_index_l2",
    "clause_index_mmr",
    "refined_answer_l2",
    "refined_answer_mmr",
]

_NEUTRAL_FALLBACK = "an unnumbered provision"
_MIN_INHERIT_OVERLAP = 40


def _depth_of_num(num: str) -> int:
    """Dotted depth: '13.1.2(a)' → 3 (parenthetical suffix ignored)."""
    core = re.sub(r"\([a-z]\)$", "", num.strip(), flags=re.IGNORECASE)
    parts = [p for p in core.split(".") if p]
    return len(parts)


def _specificity_key(kind: str, num: str) -> Tuple[int, int]:
    """Higher tuple sorts as more specific.

    Precedence: Section/Clause X.Y.Z > Section/Clause X.Y > §X.Y > ARTICLE N.
    Bare dotted numbers are treated as Section-class. Bare major 'N.' headers
    are weaker than an explicit Article keyword (page numbers often look like
    '15 ARTICLE 14'), but still section-class labels when they win alone.
    """
    depth = _depth_of_num(num)
    if kind in ("section", "clause", "bare_section"):
        # depth>=3 → top band; depth==2 → next; depth==1 still beats §/article
        band = 400 + depth
    elif kind == "section_symbol":
        band = 300 + depth
    elif kind == "article":
        band = 100 + depth
    elif kind == "bare_major":
        band = 80 + depth
    else:
        band = depth
    return (band, depth)


def _collect_clause_candidates(text: str) -> List[Tuple[Tuple[int, int], str]]:
    """Return (specificity, display_label) for every clause token in text."""
    if not text:
        return []
    cands: List[Tuple[Tuple[int, int], str]] = []

    for m in _SECTION_CLAUSE_RE.finditer(text):
        kw = m.group("kw")
        num = m.group("num")
        kind = "section" if kw.lower() == "section" else "clause"
        # Preserve original keyword casing style lightly: Title-case keyword.
        label = f"{kw.capitalize()} {num}"
        cands.append((_specificity_key(kind, num), label))

    for m in _SECTION_SYMBOL_RE.finditer(text):
        num = m.group("num")
        cands.append((_specificity_key("section_symbol", num), f"§{num}"))

    for m in _ARTICLE_LIKE_RE.finditer(text):
        kw = m.group("kw")
        num = m.group("num")
        # Keep ARTICLE-style labels upper for Article; others Title-case.
        if kw.lower() == "article":
            label = f"ARTICLE {num}"
        else:
            label = f"{kw.capitalize()} {num}"
        cands.append((_specificity_key("article", num), label))

    for m in _BARE_DOTTED_RE.finditer(text):
        num = m.group("num")
        # Skip if this span is already covered by a keyworded match.
        span = m.span()
        covered = False
        for pat in (_SECTION_CLAUSE_RE, _SECTION_SYMBOL_RE, _ARTICLE_LIKE_RE):
            for km in pat.finditer(text):
                if km.start() <= span[0] and span[1] <= km.end():
                    covered = True
                    break
            if covered:
                break
        if covered:
            continue
        cands.append((_specificity_key("bare_section", num), f"Section {num}"))

    for m in _BARE_MAJOR_RE.finditer(text):
        num = m.group("num")
        # " 14. DEFINITIONS" / page-ish "15 ARTICLE 14": label as Section but
        # weaker than an explicit Article keyword (see _specificity_key).
        cands.append((_specificity_key("bare_major", num), f"Section {num}"))

    return cands


def _extract_clause_from_text(text: str) -> Optional[str]:
    """Pick the MOST SPECIFIC clause token in the chunk (not first-match)."""
    cands = _collect_clause_candidates(text)
    if not cands:
        return None
    # Article-level only when no section-level token exists anywhere.
    has_section_level = any(k[0] >= 300 for k, _ in cands)
    if has_section_level:
        cands = [(k, lab) for k, lab in cands if k[0] >= 300]
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _norm_ws(text: str) -> str:
    return " ".join((text or "").split())


def _suffix_prefix_overlap(prev: str, curr: str) -> int:
    """Longest k such that prev[-k:] == curr[:k] (whitespace-normalized)."""
    a = _norm_ws(prev)
    b = _norm_ws(curr)
    if not a or not b:
        return 0
    max_k = min(len(a), len(b), 240)
    for k in range(max_k, _MIN_INHERIT_OVERLAP - 1, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def _label_specificity(label: str) -> Tuple[int, int]:
    """Specificity key for an already-rendered clause label."""
    s = (label or "").strip()
    m = re.search(r"(\d+(?:\.\d+)*(?:\([a-z]\))?)", s, re.IGNORECASE)
    num = m.group(1) if m else "0"
    if s.startswith("§"):
        return _specificity_key("section_symbol", num)
    if re.search(r"(?i)\barticle\b", s):
        return _specificity_key("article", num)
    if re.search(r"(?i)\bclause\b", s):
        return _specificity_key("clause", num)
    return _specificity_key("section", num)


def _inherit_clause_from_lut(
    entry: Dict[str, Any],
    lut: List[Dict[str, Any]],
    extracted_by_no: Dict[int, str],
) -> Optional[str]:
    """Inherit clause label from the nearest preceding same-doc chunk window.

    Precedence uses suffix→prefix overlap (sliding-window continuation). Among
    overlapping predecessors that carry an extracted clause, pick the one with
    the largest overlap; ties prefer the more specific label.
    """
    doc_id = str(entry.get("doc_id", ""))
    curr_text = entry.get("chunk_text", "") or ""
    curr_no = entry.get("chunk_no")
    best: Optional[Tuple[int, Tuple[int, int], str]] = None  # overlap, spec, label

    for other in lut:
        if other is entry:
            continue
        if str(other.get("doc_id", "")) != doc_id:
            continue
        other_no = other.get("chunk_no")
        if other_no == curr_no:
            continue
        try:
            label = extracted_by_no.get(int(other_no))
        except (TypeError, ValueError):
            label = None
        if not label:
            continue
        ov = _suffix_prefix_overlap(other.get("chunk_text", "") or "", curr_text)
        if ov < _MIN_INHERIT_OVERLAP:
            continue
        cand = (ov, _label_specificity(label), label)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand

    return best[2] if best else None


def _topic_score(text: str, keywords: List[str], operative_extra: List[str]) -> float:
    lower = text.lower()
    score = 0.0
    for kw in keywords:
        if kw.lower() in lower:
            score += 1.0
    for pat in operative_extra:
        if re.search(pat, text, re.IGNORECASE):
            score += 2.0
    # Global operative-verb presence overlapping this topic's keywords area.
    for opre in _OPERATIVE_VERB_RES:
        m = opre.search(text)
        if not m:
            continue
        window = text[max(0, m.start() - 40) : m.end() + 40].lower()
        if any(kw.lower() in window for kw in keywords):
            score += 1.5
    # Down-weight "assignment for the benefit of creditors" list-item noise
    # unless assignment is clearly the operative subject of the clause.
    if "assignment for the benefit of creditors" in lower:
        if not re.search(
            r"(?i)\b(?:may\s+not|shall\s+not|cannot)\s+assign\b|\bassignment\s+of\s+this\s+agreement\b",
            text,
        ):
            # soft penalty applied by caller for assignment topic only
            pass
    return score


def _fallback_phrase(text: str) -> str:
    """Operative-topic fallback; never emit a quoted mid-sentence fragment."""
    raw = _norm_ws(text or "")
    if not raw:
        return _NEUTRAL_FALLBACK

    lower = raw.lower()
    scored: List[Tuple[float, str]] = []
    for label, keywords, operative_extra in _TOPIC_LEXICON:
        s = _topic_score(raw, keywords, operative_extra)
        if label == "the assignment provision" and "assignment for the benefit of creditors" in lower:
            # Enumerated insolvency trigger, not an assignment clause.
            s -= 3.0
        if s > 0:
            scored.append((s, label))

    if not scored:
        return _NEUTRAL_FALLBACK

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_label = scored[0]
    # Require a minimum confidence; otherwise stay neutral.
    if best_score < 1.0:
        return _NEUTRAL_FALLBACK
    return best_label


def _resolve_chunk_entry(
    entry: Dict[str, Any],
    lut: List[Dict[str, Any]],
    extracted_by_no: Dict[int, str],
) -> Dict[str, Any]:
    """Resolve one LUT chunk to clause_ref + resolved_by (+ fallback flag)."""
    no = int(entry["chunk_no"])
    doc_id = entry.get("doc_id", "")
    text = entry.get("chunk_text", "") or ""

    direct = extracted_by_no.get(no) or _extract_clause_from_text(text)
    if direct:
        return {
            "chunk_no": no,
            "doc_id": doc_id,
            "clause_ref": direct,
            "fallback": False,
            "resolved_by": "extracted",
        }

    inherited = _inherit_clause_from_lut(entry, lut, extracted_by_no)
    if inherited:
        return {
            "chunk_no": no,
            "doc_id": doc_id,
            "clause_ref": inherited,
            "fallback": False,
            "resolved_by": "inherited",
        }

    return {
        "chunk_no": no,
        "doc_id": doc_id,
        "clause_ref": _fallback_phrase(text),
        "fallback": True,
        "resolved_by": "fallback",
    }


def _render_chunk_reference(entry: Dict[str, Any]) -> str:
    """
    Render the replacement text for one cited chunk.
    Prefer a formal clause reference when available; otherwise use a readable fallback phrase.
    """
    clause_ref = str(entry.get("clause_ref", "") or "").strip()
    if clause_ref:
        return clause_ref
    return _NEUTRAL_FALLBACK


def build_clause_index(
    answer: str,
    lut: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    answer에서 언급된 Chunk 번호만 대상으로 조항 번호를 매핑한다.

    반환 형식:
    [
        {"chunk_no": 1, "doc_id": "contract_A",
         "clause_ref": "Section 18.4", "fallback": False,
         "resolved_by": "extracted"},
        {"chunk_no": 2, "doc_id": "contract_A",
         "clause_ref": "the termination provision", "fallback": True,
         "resolved_by": "fallback"},
    ]
    """
    if not answer or not lut:
        return []

    # Pre-extract direct clause tokens for every LUT row (inheritance needs them).
    extracted_by_no: Dict[int, str] = {}
    for entry in lut:
        try:
            no = int(entry["chunk_no"])
        except (KeyError, TypeError, ValueError):
            continue
        clause = _extract_clause_from_text(entry.get("chunk_text", "") or "")
        if clause:
            extracted_by_no[no] = clause

    # 묶음 표현(Chunks 1, 2, and 6 등)을 펼친 뒤 Chunk 번호를 수집한다.
    expanded = _expand_grouped_chunk_refs(answer)
    cited_nos = list(dict.fromkeys(
        int(m) for m in _CHUNK_REF_RE.findall(expanded)
    ))

    result: List[Dict[str, Any]] = []
    lut_by_no = {int(entry["chunk_no"]): entry for entry in lut if "chunk_no" in entry}

    for no in cited_nos:
        entry = lut_by_no.get(no)
        if not entry:
            continue
        result.append(_resolve_chunk_entry(entry, lut, extracted_by_no))

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


def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSONL 레코드 1건을 처리한다.
    1. clause_index_l2 / clause_index_mmr 생성 (정규식 조항 번호 매핑)
    2. refined_answer_l2 / refined_answer_mmr 생성 (Step 헤더 제거 + Chunk → 조항 번호 치환)
    """
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
            record = process_record(_ensure_flat_record(record))
            slim = {k: record.get(k) for k in _OUTPUT_FIELDS}
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")
            processed += 1

    print(f"[완료] {processed}건 처리 → {out_path}")


if __name__ == "__main__":
    main()
