"""
CUAD_v1.json과 full_contract_txt/*.txt를 사용해 RAG 평가용 코퍼스·쿼리를 생성한다.

- corpus: 각 계약서 전체 원문(파일 단위). 인덱스 = doc_id.
- queries: SQuAD paragraphs의 QA만 사용하며 paragraph `context`는 사용하지 않는다.
  질문·정답은 해당 계약의 doc_id(ref_doc_id)에 매핑한다.

원본 .txt는 읽은 뒤 EDGAR식 페이지 마커 제거·가로 공백 압축 후,
\\r\\n 통일 및 과다 개행(\\n{3,})만 \\n\\n으로 축소한다.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CUAD_v1 루트: 환경변수 > 스크립트 기준 ../CUAD_v1 또는 ./CUAD_v1
_env_root = os.environ.get("CUAD_V1_ROOT")
if _env_root:
    CUAD_V1_DIR = os.path.abspath(_env_root)
else:
    _cand = os.path.join(_SCRIPT_DIR, "CUAD_v1")
    CUAD_V1_DIR = _cand if os.path.isdir(_cand) else os.path.join(os.getcwd(), "CUAD_v1")

CUAD_JSON_PATH = os.path.join(CUAD_V1_DIR, "CUAD_v1.json")
FULL_CONTRACT_DIR = os.path.join(CUAD_V1_DIR, "full_contract_txt")

OUTPUT_DIR = "processed_cuad"
CORPUS_PATH = os.path.join(OUTPUT_DIR, "cuad_corpus.json")
QUERIES_PATH = os.path.join(OUTPUT_DIR, "cuad_queries.json")
CATEGORY_MAP_PATH = os.path.join(OUTPUT_DIR, "cuad_category_map.json")
QUERY_MAP_PATH = os.path.join(OUTPUT_DIR, "cuad_query_map.json")

ASPECT_NAMES = (
    "scope_subject",
    "obligation_right",
    "exclusion_limitation",
    "procedure_condition",
)


def _remove_edgar_page_markers(text: str) -> str:
    """
    본문 중간의 Page -N- / Page N of M 등 페이지 넘버링을 제거하고 한 칸 공백으로 이어 붙임.
    (줄바꿈으로 둘러싼 형태까지 한 덩어리로 제거)
    """
    t = text
    # Page -2- , Page - 12 - 등
    t = re.sub(
        r"\n*\s*(?:Page|PAGE)\s+-\s*\d+\s*-\s*\n*",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    # Page 2 of 10
    t = re.sub(
        r"\n*\s*(?:Page|PAGE)\s+\d+\s+(?:of|OF)\s+\d+\s*\n*",
        " ",
        t,
    )
    # [Page 3] (인라인)
    t = re.sub(
        r"\n*\s*\[\s*(?:Page|PAGE)\s+\d+\s*\]\s*\n*",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    return t


def _collapse_horizontal_whitespace(text: str) -> str:
    """스페이스·탭 연속 구간만 단일 공백으로. 줄바꿈(\\n)은 변경하지 않음."""
    return re.sub(r"[ \t]{2,}", " ", text)


def minimal_clean_contract_text(text: str) -> str:
    """
    파일 원문 직후 파이프라인:
    1) \\r\\n → \\n
    2) EDGAR 페이지 마커 제거(앞뒤 문장은 공백으로 연결)
    3) 가로 공백/탭 연속 압축 ([ \\t]{2,} → 공백 하나, \\n 유지)
    4) 3개 이상 연속 개행만 \\n\\n으로 축소
    """
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _remove_edgar_page_markers(t)
    t = _collapse_horizontal_whitespace(t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def minimal_clean_answer(text: str) -> str:
    """정답 스팬: 코퍼스와 동일 규칙 후 앞뒤 공백만 제거."""
    return minimal_clean_contract_text(text).strip()


def contract_txt_path_for_title(title: str) -> str:
    """title은 JSON의 article['title']이며 파일명과 동일(확장자 .txt)."""
    safe = (title or "").strip()
    return os.path.join(FULL_CONTRACT_DIR, f"{safe}.txt")


def extract_category_name(description: str) -> str:
    """
    CUAD 공식 질문 문구에서 카테고리명을 추출한다.
    예: ... related to "Anti-Assignment" ...
    """
    text = (description or "").strip()
    if not text:
        return "Unknown"
    m = re.search(r'related to\s+"([^"]+)"', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip() or "Unknown"
    return text


def load_cuad_categories(
    data: List[Dict[str, Any]],
    cache_path: str = CATEGORY_MAP_PATH,
) -> Dict[str, str]:
    """
    CUAD 41개 카테고리(카테고리명 -> 공식 description)를 로드한다.
    - cache_path가 있으면 캐시 우선 사용
    - 없으면 CUAD_v1.json의 qas.question에서 추출 후 캐시 저장
    """
    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, dict) and cached:
            return {str(k): str(v) for k, v in cached.items()}

    category_map: Dict[str, str] = {}
    for article in data:
        for para in article.get("paragraphs") or []:
            for qa in para.get("qas") or []:
                question = (qa.get("question") or "").strip()
                if not question:
                    continue
                category = extract_category_name(question)
                if category not in category_map:
                    category_map[category] = question

    # JSON은 키 순서를 고정 저장해 재실행 시 diff를 줄인다.
    category_map = dict(sorted(category_map.items(), key=lambda x: x[0].lower()))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)
    return category_map


def decompose_legal_query(description: str) -> List[str]:
    """
    공식 category description을 법률적 4요소 템플릿으로 분해한다.
    반환 순서:
      1) Scope/Subject
      2) Obligation/Right
      3) Exclusion/Limitation
      4) Procedure/Condition
    """
    desc = " ".join((description or "").replace("\xa0", " ").split())
    category = extract_category_name(desc)
    details = ""
    if "Details:" in desc:
        details = desc.split("Details:", 1)[1].strip()

    detail_hint = f" Focus: {details}" if details else ""
    return [
        (
            f"[Scope/Subject] Identify clauses that define scope, subject matter, "
            f"covered parties/assets/timeframe for '{category}'.{detail_hint}"
        ),
        (
            f"[Obligation/Right] Identify core rights, duties, permissions, or restrictions "
            f"imposed on parties regarding '{category}'.{detail_hint}"
        ),
        (
            f"[Exclusion/Limitation] Identify carve-outs, exceptions, caps, disclaimers, "
            f"or liability limits related to '{category}'.{detail_hint}"
        ),
        (
            f"[Procedure/Condition] Identify triggers, prerequisites, notice requirements, "
            f"timelines, and execution procedures for '{category}'.{detail_hint}"
        ),
    ]


def build_decomposed_query_map(category_map: Dict[str, str]) -> Dict[str, List[str]]:
    """
    {category: [sub_query_1..4]} 형태의 고정 맵 생성.
    """
    return {category: decompose_legal_query(desc) for category, desc in category_map.items()}


def load_corpus_and_mapping(data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[int, int], List[str]]:
    """
    full_contract_txt에서 전체 계약 텍스트를 읽어 corpus와 article_idx -> doc_id 매핑을 만든다.
    파일이 없으면 스킵하고 경고 출력.
    """
    corpus: List[str] = []
    article_to_doc_id: Dict[int, int] = {}
    skipped_titles: List[str] = []

    for article_idx, article in enumerate(data):
        title = article.get("title")
        if not title or not isinstance(title, str):
            print(
                f"[경고] article_idx={article_idx}: title 없음 — 스킵",
                file=sys.stderr,
            )
            skipped_titles.append(f"(idx={article_idx}, title=<empty>)")
            continue

        path = contract_txt_path_for_title(title)
        if not os.path.isfile(path):
            print(
                f"[경고] full_contract_txt에 파일 없음 — 스킵: {title!r} → {path}",
                file=sys.stderr,
            )
            skipped_titles.append(title)
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        except OSError as e:
            print(
                f"[경고] 파일 읽기 실패 — 스킵: {path} ({e})",
                file=sys.stderr,
            )
            skipped_titles.append(title)
            continue

        cleaned = minimal_clean_contract_text(raw)
        if not cleaned.strip():
            print(
                f"[경고] 빈 문서 — 스킵: {title!r}",
                file=sys.stderr,
            )
            skipped_titles.append(title)
            continue

        doc_id = len(corpus)
        article_to_doc_id[article_idx] = doc_id
        corpus.append(cleaned)

    return corpus, article_to_doc_id, skipped_titles


def extract_queries(
    data: List[Dict[str, Any]],
    article_to_doc_id: Dict[int, int],
    decomposed_query_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    paragraphs[].context는 무시하고 qas만 수집한다.
    """
    queries: List[Dict[str, Any]] = []

    for article_idx, article in enumerate(data):
        if article_idx not in article_to_doc_id:
            continue
        ref_doc_id = article_to_doc_id[article_idx]
        paragraphs = article.get("paragraphs") or []
        for para in paragraphs:
            qas = para.get("qas") or []
            for qa in qas:
                if qa.get("is_impossible") is True:
                    continue
                question = (qa.get("question") or "").strip()
                if not question:
                    continue
                answers = qa.get("answers") or []
                answer_texts: List[str] = []
                for a in answers:
                    t = minimal_clean_answer(a.get("text") or "")
                    if t:
                        answer_texts.append(t)
                if not answer_texts:
                    continue
                answers_joined = " [SEP] ".join(answer_texts)
                category = extract_category_name(question)
                aspect_queries = decomposed_query_map.get(category) or decompose_legal_query(question)
                aspects = dict(zip(ASPECT_NAMES, aspect_queries))
                query_concat = " [SEP] ".join([question] + aspect_queries)
                queries.append({
                    "category": category,
                    "query": question,
                    "query_aspects": aspect_queries,
                    "query_concat": query_concat,
                    "aspects": aspects,
                    "ref_doc_id": ref_doc_id,
                    "answers": answers_joined,
                })

    return queries


def save_preprocessed_data(
    corpus: List[str],
    queries: List[Dict[str, Any]],
    category_map: Dict[str, str],
    decomposed_query_map: Dict[str, List[str]],
) -> None:
    """
    전처리 결과 저장:
    - cuad_corpus.json
    - cuad_queries.json (query + 4-aspect sub-queries 포함)
    - cuad_category_map.json ({category: description})
    - cuad_query_map.json ({category: [sub_query_1..4]})
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    with open(CATEGORY_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)
    with open(QUERY_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(decomposed_query_map, f, ensure_ascii=False, indent=2)


def main() -> None:
    if not os.path.isfile(CUAD_JSON_PATH):
        raise FileNotFoundError(f"CUAD JSON 없음: {CUAD_JSON_PATH}")
    if not os.path.isdir(FULL_CONTRACT_DIR):
        raise FileNotFoundError(f"full_contract_txt 폴더 없음: {FULL_CONTRACT_DIR}")

    with open(CUAD_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("data")
    if not data:
        raise ValueError("CUAD JSON에 'data' 키가 없거나 비어 있습니다.")

    corpus, article_to_doc_id, skipped = load_corpus_and_mapping(data)
    category_map = load_cuad_categories(data)
    decomposed_query_map = build_decomposed_query_map(category_map)
    queries = extract_queries(data, article_to_doc_id, decomposed_query_map)

    total_paragraph_blocks = sum(
        len(a.get("paragraphs") or []) for a in data
    )

    save_preprocessed_data(corpus, queries, category_map, decomposed_query_map)

    print(f"1) full_contract_txt 기반 계약서 문서 수: {len(corpus)}")
    print(f"   - JSON article 수: {len(data)} | 로드 실패·스킵: {len(skipped)}")
    print(f"   - (참고) 원본 paragraphs 블록 수 합계: {total_paragraph_blocks}")
    print(f"2) 유효 Query 수: {len(queries)}")
    print(f"3) 카테고리 수: {len(category_map)}")
    print(f"   - Corpus: {os.path.abspath(CORPUS_PATH)}")
    print(f"   - Queries: {os.path.abspath(QUERIES_PATH)}")
    print(f"   - Category Map: {os.path.abspath(CATEGORY_MAP_PATH)}")
    print(f"   - Decomposed Query Map: {os.path.abspath(QUERY_MAP_PATH)}")


if __name__ == "__main__":
    main()
