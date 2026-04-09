from typing import List, Dict, Any, Optional, Tuple, Union
import json
import os
import pickle
import re

import numpy as np

# langchain RecursiveCharacterTextSplitter를 사용할 수 있으면 legal 문맥 분할 품질을 높이기 위해 활용
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    RecursiveCharacterTextSplitter = None  # type: ignore

# 도메인별 HuggingFace 임베딩 모델 (online_accelerator DOMAIN_EMBEDDING_MODEL_MAP과 동기화)
# 벡터 공간·L2 검색용. legal/cuad는 SentenceTransformer로 오프라인 인덱싱과 온라인 검색 일관 유지.
DOMAIN_MODES = ("general", "legal", "medical", "cuad")
DOMAIN_MODEL_MAP = {
    "general": "sentence-transformers/all-mpnet-base-v2",
    "legal": "sentence-transformers/all-mpnet-base-v2",
    "medical": "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # 벡터 공간·비대칭 검색
    "cuad": "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # QA 전용 비대칭 검색
}

# 조사된 벤치마크 데이터셋 (Parquet 기반, 로드 실패 시 즉시 예외)
# legal: cuad 대신 flowaicom/legalbench_contracts_qa_subset (계약서 QA, Parquet)
# cuad: 로컬 processed_cuad/cuad_corpus.json 사용 (preprocess_cuad.py 선행 실행)
DOMAIN_DATASET_MAP = {
    "general": "squad",
    "legal": "flowaicom/legalbench_contracts_qa_subset",
    "medical": "pubmed_qa",
    "cuad": "cuad",
}
# medical: pubmed_qa는 subset pqa_labeled 사용
DOMAIN_DATASET_CONFIG = {
    "general": None,
    "legal": None,
    "medical": "pqa_labeled",
    "cuad": None,
}


def load_dataset_for_domain(
    domain: str,
    split: str = "train",
    max_contexts: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    도메인별 벤치마크 데이터셋 로드. 실패 시 즉시 예외 발생 (무음 폴백 없음).
    반환: (contexts, eval_pairs) — eval_pairs = [(question, ref_doc_id), ...]
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("datasets 라이브러리가 필요합니다: pip install datasets")

    domain = domain.lower()
    if domain not in DOMAIN_MODES:
        raise ValueError(f"domain은 {DOMAIN_MODES} 중 하나여야 합니다. 입력: {domain!r}")

    if domain == "cuad":
        corpus_path = os.path.join(".", "processed_cuad", "cuad_corpus.json")
        if not os.path.isfile(corpus_path):
            raise FileNotFoundError(f"CUAD 코퍼스가 없습니다: {corpus_path}. preprocess_cuad.py를 먼저 실행하세요.")
        with open(corpus_path, "r", encoding="utf-8") as f:
            contexts = json.load(f)
        queries_path = os.path.join(".", "processed_cuad", "cuad_queries.json")
        eval_pairs: List[Tuple[str, int]] = []
        if os.path.isfile(queries_path):
            with open(queries_path, "r", encoding="utf-8") as f:
                queries_raw = json.load(f)
            eval_pairs = [(item["query"], int(item["ref_doc_id"])) for item in queries_raw]
        if max_contexts is not None and max_contexts > 0:
            contexts = contexts[:max_contexts]
            eval_pairs = [(q, ref_id) for q, ref_id in eval_pairs if ref_id < len(contexts)]
        return contexts, eval_pairs

    name = DOMAIN_DATASET_MAP[domain]
    config = DOMAIN_DATASET_CONFIG.get(domain)

    try:
        if config:
            ds = hf_load_dataset(name, config, split=split)
        else:
            ds = hf_load_dataset(name, split=split)
    except Exception as e:
        print(f"[load_dataset_for_domain] 데이터셋 로드 실패: domain={domain}, name={name}, config={config}, split={split}")
        print(f"  에러: {e}")
        raise

    contexts: List[str] = []
    eval_pairs: List[Tuple[str, int]] = []

    def _ctx(row: dict) -> str:
        for k in ("context", "text", "document", "article", "facts", "long_answer"):
            v = row.get(k)
            if v is None:
                continue
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                return " ".join(str(x) for x in v) if v else ""
            if isinstance(v, dict) and isinstance(v.get("text"), str):
                return v["text"]
        return ""

    def _q(row: dict) -> str:
        for k in ("question", "query", "prompt"):
            v = row.get(k)
            if v is not None and isinstance(v, str):
                return v
        return ""

    if name == "squad":
        for i, row in enumerate(ds):
            ctx = row.get("context", "")
            q = row.get("question", "")
            if ctx and isinstance(ctx, str):
                contexts.append(ctx)
                eval_pairs.append((q, len(contexts) - 1))
            if max_contexts and len(contexts) >= max_contexts:
                break
    elif "legalbench" in name:
        # flowaicom/legalbench_contracts_qa_subset: 필드 context, question
        for i, row in enumerate(ds):
            ctx = row.get("context", "") or _ctx(row)
            q = row.get("question", "") or _q(row)
            if ctx and isinstance(ctx, str) and q:
                contexts.append(ctx)
                eval_pairs.append((q, len(contexts) - 1))
            if max_contexts and len(contexts) >= max_contexts:
                break
    elif "pubmed_qa" in name:
        for i, row in enumerate(ds):
            ctx = row.get("long_answer", "") or row.get("context", "")
            if isinstance(ctx, dict):
                ctx = ctx.get("text", "") if isinstance(ctx.get("text"), str) else ""
            if isinstance(ctx, list):
                ctx = " ".join(str(x) for x in ctx) if ctx else ""
            q = row.get("question", "") or row.get("query", "")
            if ctx and isinstance(ctx, str):
                contexts.append(ctx)
                eval_pairs.append((q, len(contexts) - 1))
            if max_contexts and len(contexts) >= max_contexts:
                break
    else:
        for i, row in enumerate(ds):
            ctx = _ctx(row)
            q = _q(row)
            if ctx and q:
                contexts.append(ctx)
                eval_pairs.append((q, len(contexts) - 1))
            if max_contexts and len(contexts) >= max_contexts:
                break

    return contexts, eval_pairs


try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

# L2 가속기 시뮬레이션: 768차원 벡터를 효율적으로 처리하는 스켈레톤
EMBEDDING_DIM = 768
# 본문(context) 최소 길이: 이보다 짧으면 껍데기/메타만 있을 수 있음
MIN_CONTEXT_LEN = 50


class L2AcceleratorSimulator:
    """
    L2 가속기가 768차원 벡터를 효율적으로 처리한다고 가정한 시뮬레이션 스켈레톤.
    - centroid/LUT 데이터는 연속 메모리(float32)로 패킹.
    - 배치 L2 거리 연산 인터페이스를 제공 (실제 HW에서는 AIMDC로 병렬 연산).
    """
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    @staticmethod
    def pack_vectors_for_hw(vectors: np.ndarray) -> np.ndarray:
        """가속기 적재용: 2D 배열을 C-contiguous float32 1D로 패킹."""
        if vectors.size == 0:
            return np.empty(0, dtype=np.float32)
        return np.ascontiguousarray(vectors.astype(np.float32)).flatten()

    @staticmethod
    def l2_distance_batch(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """시뮬레이션: query (D,) vs keys (N, D) 간 L2 거리 (N,) 반환. 가속기에서 병렬 연산된다고 가정."""
        if keys.size == 0:
            return np.empty(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        k = np.asarray(keys, dtype=np.float32)
        diff = k - q[None, :]
        return np.sqrt(np.sum(diff * diff, axis=1))


class L2RAGOfflineIndexer:
    """
    AIMDC 하드웨어 가속기를 위한 오프라인 데이터 준비 및 인덱싱 파이프라인.
    도메인별 모델(General/Legal/Medical)로 768차원 벡터를 생성해 L2 가속기 시뮬레이션 성능 비교 가능.
    """

    def __init__(
        self,
        domain_mode: str = "general",
        num_centroids: int = 1024,
        num_subspaces: int = 8,
        output_base_dir: str = ".",
    ):
        domain_mode = domain_mode.lower().strip()
        if domain_mode not in DOMAIN_MODES:
            raise ValueError(f"domain_mode는 {DOMAIN_MODES} 중 하나여야 합니다. 입력: {domain_mode!r}")

        self.domain_mode = domain_mode
        self.model_name = DOMAIN_MODEL_MAP[domain_mode]
        self.embedding_dim = EMBEDDING_DIM  # 모든 도메인 모델 768차원
        self.num_centroids = num_centroids
        self.num_subspaces = num_subspaces
        self.output_base_dir = output_base_dir

        if self.embedding_dim % self.num_subspaces != 0:
            raise ValueError("embedding_dim 은 num_subspaces 로 나누어 떨어져야 합니다.")

        # 가속기에 적재될 오프라인 산출물
        self.centroids: np.ndarray | None = None
        self.pq_codebooks: np.ndarray | None = None
        self.posting_lists: Dict[int, List[Dict[str, Any]]] = {}
        self.metadata_db: Dict[str, Dict[str, Any]] = {}

        # OPQ 회전 행렬 (옵션)
        self._opq_rotation: np.ndarray | None = None
        # 도메인별 인코더 (lazy load)
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._l2_sim = L2AcceleratorSimulator(dim=self.embedding_dim)

    # -----------------------------
    # Step 1: 데이터 수집 & 정제
    # -----------------------------
    def step1_collect_and_clean_data(
        self,
        raw_documents: List[str],
        raw_doc_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[List[int]], List[str]]:
        """
        1. 데이터 수집 & 정제
        - 문서/텍스트 수집, 중복 제거 및 정제(cleaning) 수행.
        - raw_doc_ids: 각 문서의 제목(Title) 또는 고유 ID. 없으면 인덱스 기반 "doc_0", "doc_1" 사용.
        - 반환: (clean_docs, original_doc_ids_per_doc, doc_id_per_clean_doc)
          doc_id_per_clean_doc[i] = clean_docs[i]에 대응하는 문서 고유 ID(문자열).
        """

        def _clean_text(text: str) -> str:
            text = text.strip()
            text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text

        seen: Dict[str, int] = {}
        clean_docs: List[str] = []
        original_doc_ids_per_doc: List[List[int]] = []
        doc_id_per_clean_doc: List[str] = []

        for row_idx, doc in enumerate(raw_documents):
            cleaned = _clean_text(doc)
            if not cleaned:
                continue
            doc_id = (raw_doc_ids[row_idx] if raw_doc_ids and row_idx < len(raw_doc_ids) else None) or f"doc_{row_idx}"
            if cleaned in seen:
                idx = seen[cleaned]
                original_doc_ids_per_doc[idx].append(row_idx)
            else:
                seen[cleaned] = len(clean_docs)
                clean_docs.append(cleaned)
                original_doc_ids_per_doc.append([row_idx])
                doc_id_per_clean_doc.append(doc_id)

        return clean_docs, original_doc_ids_per_doc, doc_id_per_clean_doc

    # -----------------------------
    # Legal / CUAD: 조항(Clause) 단위 청킹 보조
    # -----------------------------
    _LEGAL_CLAUSE_BOUNDARY_RE = re.compile(
        r"(?:\n\n+)"
        r"|(?:\n(?=Article\b))"
        r"|(?:\n(?=SECTION\b))"
        r"|(?:\n(?=Section\b))"
        r"|(?:\n(?=\d{1,3}\.\d{1,3}(?:\.\d+)?\s))"
    )

    def _legal_clause_primary_split(self, text: str) -> List[str]:
        """
        1단계: 조항·문단 경계 정규식으로 분할 (글자 수 슬라이딩 없음).
        구분: \\n\\n, \\nArticle, \\nSection(/SECTION), \\n숫자.숫자 공백 (예: 10.8 ).
        """
        if not isinstance(text, str) or not text.strip():
            return []
        t = text.replace("\r\n", "\n").strip()
        boundaries = [0]
        for m in self._LEGAL_CLAUSE_BOUNDARY_RE.finditer(t):
            if m.start() > 0:
                boundaries.append(m.start())
        boundaries.append(len(t))
        boundaries = sorted(set(boundaries))
        segments: List[str] = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            seg = t[a:b].strip()
            if seg:
                segments.append(seg)
        return segments

    def _extract_legal_heading(self, text: str) -> str:
        """
        청크 앞부분(첫 1~2줄)에서 조항 헤딩을 추출한다.
        매칭 예: "Article 3 ...", "Section 10.2 ...", "10.2 ..."
        """
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not t:
            return ""
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if not lines:
            return ""
        head_scope = "\n".join(lines[:2])
        patterns = [
            r"(?im)^(Article\s+[A-Za-z0-9.\-()]+(?:\s*[:.\-]\s*|\s+).*)$",
            r"(?im)^(Section\s+[A-Za-z0-9.\-()]+(?:\s*[:.\-]\s*|\s+).*)$",
            r"(?im)^(\d{1,3}(?:\.\d{1,3}){1,3}(?:\s*[:.\-]\s*|\s+).*)$",
        ]
        for pat in patterns:
            m = re.search(pat, head_scope)
            if m:
                return m.group(1).strip()
        return ""

    def _legal_fallback_split_oversized(
        self,
        text: str,
        max_len: int,
        overlap: int,
    ) -> List[str]:
        """
        2단계: 조항이 chunk_size 초과일 때만 `. ` → `\\n` → 공백 순으로 끊고,
        연속 조각 사이에만 overlap(100~200자 권장) 적용.
        """
        if not text or not text.strip():
            return []
        t = text.strip()
        if len(t) <= max_len:
            return [t]

        half = max_len // 2
        ov = max(0, min(overlap, half - 1)) if half > 1 else 0
        pieces: List[str] = []
        i = 0
        n = len(t)
        prev_i = -1
        min_break_ahead = max(1, half)

        while i < n:
            if i == prev_i:
                i = min(i + max(1, max_len // 4), n)
                if i >= n:
                    break
            prev_i = i

            if n - i <= max_len:
                tail = t[i:n].strip()
                if tail:
                    pieces.append(tail)
                break

            chunk_end = min(i + max_len, n)
            search_lo = min(i + min_break_ahead, chunk_end - 1)
            if search_lo <= i:
                search_lo = i + 1
            if search_lo >= chunk_end and chunk_end > i + 1:
                search_lo = i + 1
            best_break = -1

            dot = t.rfind(". ", search_lo, chunk_end)
            if dot > i:
                best_break = dot + 2

            if best_break < 0:
                nl = t.rfind("\n", search_lo, chunk_end)
                if nl > i:
                    best_break = nl + 1

            if best_break < 0:
                sp = t.rfind(" ", i + 1, chunk_end)
                if sp > i:
                    best_break = sp + 1

            if best_break <= i:
                best_break = chunk_end

            piece = t[i:best_break].strip()
            if piece:
                pieces.append(piece)

            next_i = best_break - ov
            if next_i <= i:
                next_i = best_break
            i = next_i

        return pieces

    def _chunking_legal_clause_based(
        self,
        doc: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        legal/cuad: 조항 단위 1차 분할 → 초과 시에만 폴백 분할 + 소량 overlap.
        짧은 조항에는 overlap 없음(0).
        """
        clauses = self._legal_clause_primary_split(doc)
        if not clauses:
            return []

        fb_overlap = max(100, min(200, chunk_size // 10))
        if chunk_overlap > 0:
            fb_overlap = max(fb_overlap, min(200, chunk_overlap // 2))

        out: List[str] = []
        for clause in clauses:
            if len(clause) <= chunk_size:
                out.append(clause)
            else:
                out.extend(
                    self._legal_fallback_split_oversized(
                        clause,
                        max_len=chunk_size,
                        overlap=fb_overlap,
                    )
                )
        return out

    # -----------------------------
    # Step 2: Chunking
    # -----------------------------
    def step2_chunking(
        self,
        clean_docs: List[str],
        original_doc_ids_per_doc: List[List[int]],
        doc_id_per_clean_doc: Optional[List[str]] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        2. Chunking [cite: 36, 57]
        - general/medical 등: 고정 길이(chunk_size, chunk_overlap) + RecursiveCharacterTextSplitter 또는 슬라이딩 분할.
        - legal / cuad: 조항(Clause) 단위 1차 정규식 분할 후, chunk_size 초과 조항만
          마침표·줄바꿈 기준 2차 분할 + 소량 overlap(100~200자대).
        - doc_id_per_clean_doc[i]: clean_docs[i]의 문서 고유 ID(제목 등). 없으면 "doc_0", "doc_1" ...
        - 반환: [{"text", "metadata": {"doc_id", "chunk_idx"}, "doc_index", "original_doc_id"}, ...]
          청크 텍스트와 doc_id가 메타데이터로 1:1 매칭되며, 임베딩 배열 인덱스와 일치.
        """
        if len(original_doc_ids_per_doc) != len(clean_docs):
            raise ValueError("original_doc_ids_per_doc 길이는 clean_docs와 같아야 합니다.")
        if doc_id_per_clean_doc is not None and len(doc_id_per_clean_doc) != len(clean_docs):
            raise ValueError("doc_id_per_clean_doc 길이는 clean_docs와 같아야 합니다.")

        doc_ids = doc_id_per_clean_doc if doc_id_per_clean_doc is not None else [f"doc_{i}" for i in range(len(clean_docs))]
        chunks: List[Dict[str, Any]] = []
        use_legal_clause_chunking = self.domain_mode in ("cuad", "legal")

        def _sliding_char_split(text: str, cs: int, co: int) -> List[str]:
            """langchain 미설치 환경에서도 chunk_size/overlap을 만족하는 슬라이딩 분할."""
            if not isinstance(text, str) or not text:
                return []
            chunks: List[str] = []
            start = 0
            n = len(text)
            while start < n:
                end = min(n, start + cs)
                if end <= start:
                    break
                # 끝 경계 보정을 위해 중간 이후부터 whitespace/punct 위치를 찾는다.
                min_end_search = max(start + int(cs * 0.7), start + 1)
                # 단어/문장 경계에서 끊기도록 end를 조정
                # 우선순위: '\n' -> '. ' -> ', ' -> ' ' -> 마지막 문장부호
                candidates = [
                    text.rfind("\n", min_end_search, end),
                    text.rfind(". ", min_end_search, end),
                    text.rfind(", ", min_end_search, end),
                    text.rfind(" ", min_end_search, end),
                    max(
                        text.rfind(";", min_end_search, end),
                        text.rfind(":", min_end_search, end),
                        text.rfind("?", min_end_search, end),
                        text.rfind("!", min_end_search, end),
                    ),
                ]
                best = max([c for c in candidates if isinstance(c, int) and c > start], default=-1)
                if best != -1 and best > start + 10:
                    end = best + 1
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= n:
                    break
                # overlap 적용: 다음 start는 end - co
                next_start = end - co
                if next_start <= start:
                    next_start = start + max(1, int(cs * 0.25))
                start = next_start
            return chunks

        # splitter는 비(legal/cuad) 도메인에서만 사용
        splitter = None
        if not use_legal_clause_chunking and RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "; ", " ", ""],
            )

        for doc_index, doc in enumerate(clean_docs):
            orig_ids: Union[List[int], int] = original_doc_ids_per_doc[doc_index]
            if isinstance(orig_ids, list) and len(orig_ids) == 1:
                orig_ids = orig_ids[0]
            doc_id_str = doc_ids[doc_index]

            if use_legal_clause_chunking:
                piece_texts = self._chunking_legal_clause_based(doc, chunk_size, chunk_overlap)
            elif splitter is not None:
                piece_texts = splitter.split_text(doc)
            else:
                piece_texts = _sliding_char_split(doc, chunk_size, chunk_overlap)

            if doc_index == 0 and piece_texts:
                mode_tag = "clause-level (legal/cuad)" if use_legal_clause_chunking else "semantic"
                print(f"[Chunking Debug] First document - first 3 chunks ({mode_tag})")
                for dbg_i, dbg_t in enumerate(piece_texts[:3], start=1):
                    dbg_flat = dbg_t.replace("\n", " ")
                    preview = (dbg_flat[:220] + "...") if len(dbg_flat) > 220 else dbg_flat
                    print(f"  [doc0 chunk {dbg_i}] len={len(dbg_t)} | {preview}")

            for chunk_idx, piece in enumerate(piece_texts):
                piece = piece.strip()
                if not piece:
                    continue
                heading = self._extract_legal_heading(piece) if use_legal_clause_chunking else ""
                chunks.append({
                    "text": piece,
                    "metadata": {"doc_id": doc_id_str, "chunk_idx": chunk_idx},
                    "doc_index": doc_index,
                    "doc_id": doc_id_str,
                    "chunk_idx": chunk_idx,
                    "original_doc_id": orig_ids,
                    "section_heading": heading,
                })

        return chunks

    # -----------------------------
    # Step 3: Embedding 추출 (도메인별 768차원)
    # -----------------------------
    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder, self._tokenizer

        if self.domain_mode in ("general", "legal", "cuad", "medical"):
            if not _SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
                raise ImportError(
                    "general/legal/cuad/medical 모드는 sentence-transformers가 필요합니다: pip install sentence-transformers"
                )
            self._encoder = SentenceTransformer(self.model_name)
            self._tokenizer = None
            return self._encoder, self._tokenizer

        # 기타 도메인: HuggingFace BERT + mean pooling
        if not _TRANSFORMERS_AVAILABLE or AutoModel is None or AutoTokenizer is None or torch is None:
            raise ImportError(
                "해당 도메인은 transformers와 torch가 필요합니다: pip install transformers torch"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._encoder = AutoModel.from_pretrained(self.model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = self._encoder.to(device)
        self._encoder.eval()
        return self._encoder, self._tokenizer

    def _encode_general(self, texts: List[str], batch_size: int) -> np.ndarray:
        encoder, _ = self._get_encoder()
        emb = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def _encode_bert_mean_pool(self, texts: List[str], batch_size: int) -> np.ndarray:
        encoder, tokenizer = self._get_encoder()
        device = next(encoder.parameters()).device
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inp = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                out = encoder(**inp)
            # (B, L, 768) -> mean over L with attention_mask
            mask = inp["attention_mask"]
            last = out.last_hidden_state
            pooled = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1e-9).unsqueeze(-1)
            # L2 정규화
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
            all_emb.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(all_emb)

    @staticmethod
    def _l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """모든 벡터를 단위 길이(1)로 L2 정규화. 가속기 L2 거리 = 코사인 유사도와 수학적 일치."""
        if embeddings.size == 0:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)  # 0벡터는 그대로 두기 위해 1로 둠 → 0/1=0
        return (embeddings / norms).astype(np.float32)

    def step3_extract_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        """
        3. Embedding 추출 [cite: 37, 38, 57]
        - 도메인 모델(General/Legal/Medical)로 각 chunk의 768차원 임베딩 계산.
        - 추출 직후 L2 정규화하여 단위 벡터로 만듦 (IVF/PQ 입력 및 가속기 L2 거리 일치).
        """
        if not chunks:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        texts = [chunk.get("text", "") for chunk in chunks]
        # 텍스트 추출 검증: 진짜 본문(context)인지 방어 로직 (너무 짧으면 경고)
        short_idx = [i for i, t in enumerate(texts) if not isinstance(t, str) or len(t.strip()) < MIN_CONTEXT_LEN]
        if short_idx:
            sample = short_idx[:5]
            lens = [len((texts[i] or "").strip()) for i in sample]
            print(f"[경고] 본문이 너무 짧은 청크가 있습니다 (총 {len(short_idx)}개). context가 맞는지 확인하세요. 예: 인덱스={sample}, 길이={lens}")

        if self.domain_mode in ("general", "legal", "cuad", "medical"):
            embeddings = self._encode_general(texts, batch_size)
        else:
            embeddings = self._encode_bert_mean_pool(texts, batch_size)

        if embeddings.shape[1] != self.embedding_dim:
            raise RuntimeError(f"도메인 모델은 768차원을 반환해야 합니다. 얻은 차원: {embeddings.shape[1]}")
        # [필수] encoder.encode 직후 모든 문서 벡터를 L2 정규화(길이 1). 가속기 L2 거리와 코사인 유사도 일치.
        embeddings = self._l2_normalize_embeddings(embeddings)
        return embeddings

    # -----------------------------
    # Step 4: OPQ (옵션)
    # -----------------------------
    def step4_optional_opq(self, embeddings: np.ndarray) -> np.ndarray:
        """
        4. OPQ (옵션) 
        - Orthogonal Product Quantization으로 벡터 분산 정규화하여 PQ 효율성 향상.
        - 여기서는 간단히 정규 직교 행렬을 임의로 생성해 회전을 적용하는 방식으로 근사.
        """
        if embeddings.size == 0:
            return embeddings

        n, d = embeddings.shape
        if d != self.embedding_dim:
            raise ValueError("임베딩 차원이 embedding_dim 과 일치해야 합니다.")

        if self._opq_rotation is None:
            # 재현 가능한 난수 시드를 위해 고정값 사용
            rng = np.random.default_rng(seed=42)
            # 랜덤 행렬에 대해 QR 분해로 직교 행렬 생성
            random_matrix = rng.standard_normal(size=(d, d))
            q, _ = np.linalg.qr(random_matrix)
            self._opq_rotation = q.astype(np.float32)

        rotated_embeddings = embeddings @ self._opq_rotation
        return rotated_embeddings

    # -----------------------------
    # 공통: 간단한 k-means 유틸
    # -----------------------------
    def _kmeans(self, x: np.ndarray, k: int, num_iters: int = 100) -> np.ndarray:
        """
        x: (N, D), k: 클러스터 개수
        반환: (k, D) centroids
        """
        n, d = x.shape
        if n == 0:
            return np.zeros((k, d), dtype=np.float32)

        if k > n:
            k = n

        rng = np.random.default_rng(seed=0)
        # K-Means++ 초기화: 첫 centroid 랜덤 선택 후, 거리^2 비례 확률로 나머지 centroid 선택
        first_idx = int(rng.integers(0, n))
        centroids_init = [x[first_idx].astype(np.float32)]

        closest_dist_sq = np.sum((x - centroids_init[0][None, :]) ** 2, axis=1).astype(np.float64)
        for _ in range(1, k):
            total = float(np.sum(closest_dist_sq))
            if total <= 1e-12:
                # 모든 점이 동일/거의 동일하면 중복 없는 임의 샘플로 보강
                remain = np.setdiff1d(np.arange(n), np.array([first_idx]), assume_unique=False)
                if remain.size == 0:
                    next_idx = int(rng.integers(0, n))
                else:
                    next_idx = int(rng.choice(remain))
            else:
                probs = closest_dist_sq / total
                next_idx = int(rng.choice(n, p=probs))
            next_c = x[next_idx].astype(np.float32)
            centroids_init.append(next_c)
            dist_sq_new = np.sum((x - next_c[None, :]) ** 2, axis=1).astype(np.float64)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_new)

        centroids = np.stack(centroids_init, axis=0).astype(np.float32)

        for _ in range(num_iters):
            # 거리 계산 (N, k)
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            x_norm = np.sum(x * x, axis=1, keepdims=True)  # (N, 1)
            c_norm = np.sum(centroids * centroids, axis=1, keepdims=True).T  # (1, k)
            distances = x_norm + c_norm - 2.0 * (x @ centroids.T)

            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int32)
            for i in range(n):
                lbl = labels[i]
                new_centroids[lbl] += x[i]
                counts[lbl] += 1

            for j in range(k):
                if counts[j] > 0:
                    new_centroids[j] /= float(counts[j])
                else:
                    # 비어 있는 클러스터는 임의 재초기화
                    new_centroids[j] = x[rng.integers(0, n)]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        return centroids

    # -----------------------------
    # Step 5: IVF 훈련
    # -----------------------------
    def step5_train_ivf_centroids(self, embeddings: np.ndarray) -> np.ndarray:
        """
        5. IVF 훈련 (centroid clustering) [cite: 41, 57]
        - 정규화된 임베딩으로 centroids 학습. 저장 직전 centroids도 L2 정규화하여 Unit Vector로 저장.
        """
        if embeddings.size == 0:
            self.centroids = np.empty((0, self.embedding_dim), dtype=np.float32)
            return self.centroids

        k = min(self.num_centroids, embeddings.shape[0])
        centroids = self._kmeans(embeddings.astype(np.float32), k=k)

        # [가장 중요] np.save 하기 직전 센트로이드 L2 재정규화. K-Means 평균으로 줄어든 길이를 1.0으로 복구.
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(c_norms, 1e-10)
        centroids = centroids.astype(np.float32)

        # num_centroids 가 더 크면 남는 슬롯은 0으로 패딩
        if k < self.num_centroids:
            padded = np.zeros((self.num_centroids, self.embedding_dim), dtype=np.float32)
            padded[:k] = centroids
            centroids = padded
            # 패딩이 아닌 행만 이미 단위벡터; 패딩(0벡터)은 그대로 유지

        self.centroids = centroids
        return centroids

    # -----------------------------
    # Step 6: PQ codebook 훈련
    # -----------------------------
    def step6_train_pq_codebook(self, embeddings: np.ndarray, codebook_size: int = 256) -> np.ndarray:
        """
        6. PQ codebook 훈련 [cite: 42, 57]
        - 각 벡터를 M개의 sub-vector로 나누고, subspace마다 k-means codebook 학습.
        """
        if embeddings.size == 0:
            self.pq_codebooks = np.empty(
                (self.num_subspaces, codebook_size, self.embedding_dim // self.num_subspaces),
                dtype=np.float32,
            )
            return self.pq_codebooks

        sub_dim = self.embedding_dim // self.num_subspaces
        pq_codebooks = np.zeros(
            (self.num_subspaces, codebook_size, sub_dim),
            dtype=np.float32,
        )

        for m in range(self.num_subspaces):
            start = m * sub_dim
            end = start + sub_dim
            sub_vectors = embeddings[:, start:end]

            # 학습 데이터가 codebook_size 보다 적으면 k를 줄임
            k = min(codebook_size, sub_vectors.shape[0])
            codebook = self._kmeans(sub_vectors.astype(np.float32), k=k)

            # 남는 슬롯은 0 패딩
            if k < codebook_size:
                tmp = np.zeros((codebook_size, sub_dim), dtype=np.float32)
                tmp[:k] = codebook
                codebook = tmp

            pq_codebooks[m] = codebook

        self.pq_codebooks = pq_codebooks
        return pq_codebooks

    # -----------------------------
    # Step 7: IVF-PQ 인덱싱
    # -----------------------------
    def _assign_ivf(self, embeddings: np.ndarray) -> np.ndarray:
        if self.centroids is None or self.centroids.size == 0:
            raise RuntimeError("IVF centroids 가 학습되지 않았습니다. step5 를 먼저 호출하세요.")

        valid_centroids = self.centroids
        # 전부 0인 패딩 centroid 는 제외
        norms = np.linalg.norm(valid_centroids, axis=1)
        non_zero_mask = norms > 0
        if not np.any(non_zero_mask):
            raise RuntimeError("유효한 IVF centroid 가 없습니다.")

        valid_centroids = valid_centroids[non_zero_mask]

        x = embeddings.astype(np.float32)
        x_norm = np.sum(x * x, axis=1, keepdims=True)
        c_norm = np.sum(valid_centroids * valid_centroids, axis=1, keepdims=True).T
        distances = x_norm + c_norm - 2.0 * (x @ valid_centroids.T)
        assigned = np.argmin(distances, axis=1)

        # 원래 centroid 인덱스로 매핑
        original_indices = np.where(non_zero_mask)[0]
        return original_indices[assigned]

    def _pq_encode(self, embeddings: np.ndarray) -> np.ndarray:
        if self.pq_codebooks is None or self.pq_codebooks.size == 0:
            raise RuntimeError("PQ codebook 이 학습되지 않았습니다. step6 을 먼저 호출하세요.")

        sub_dim = self.embedding_dim // self.num_subspaces
        codes = np.zeros((embeddings.shape[0], self.num_subspaces), dtype=np.uint8)

        for i, vec in enumerate(embeddings):
            for m in range(self.num_subspaces):
                start = m * sub_dim
                end = start + sub_dim
                sub_vec = vec[start:end][None, :]  # (1, sub_dim)
                codebook = self.pq_codebooks[m]  # (K, sub_dim)

                # L2 거리 최소 인덱스
                diff = codebook - sub_vec
                dist = np.sum(diff * diff, axis=1)
                code = int(np.argmin(dist))
                codes[i, m] = code

        return codes

    def step7_ivf_pq_indexing(self, embeddings: np.ndarray, chunks: List[Dict]) -> Dict:
        """
        7. IVF-PQ 인덱싱 [cite: 39, 40, 57]
        - 각 chunk embedding에 대해 centroid를 할당하고 PQ encoding 수행.
        - Posting list에 저장.
        구조 예시: {centroid_id: [{'chunk_id': id, 'pq_code': code, 'vector_id': idx}, ...]}
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings 개수와 chunks 개수가 일치해야 합니다.")

        if embeddings.size == 0:
            posting_lists: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(self.num_centroids)}
            self.posting_lists = posting_lists
            return posting_lists

        assigned_centroids = self._assign_ivf(embeddings)
        pq_codes = self._pq_encode(embeddings)

        posting_lists: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(self.num_centroids)}

        for idx, (chunk, centroid_id) in enumerate(zip(chunks, assigned_centroids)):
            chunk_id = f"{chunk.get('doc_index', chunk.get('doc_id'))}_{chunk['chunk_idx']}"
            entry = {
                "chunk_id": chunk_id,
                "pq_code": pq_codes[idx].copy(),
                "vector_id": idx,
            }
            posting_lists[int(centroid_id)].append(entry)

        self.posting_lists = posting_lists
        return posting_lists

    # -----------------------------
    # Step 8: Metadata 부착
    # -----------------------------
    def step8_attach_metadata(self, chunks: List[Dict], posting_lists: Dict) -> Dict:
        """
        8. Metadata 부착 
        - 각 chunk에 대해 doc_id, chunk_idx, facet label 등 메타데이터 정리. [cite: 51, 52, 53, 54]
        - embedding_l2_normalized 플래그로 디버깅 시 정규화 여부 확인 가능.
        """
        metadata_db: Dict[str, Dict[str, Any]] = {}

        # chunk_id 외에 doc_id(문서 고유 ID), original_doc_id(행 번호) 저장. 임베딩 배열과 청크 인덱스 1:1 유지.
        for chunk in chunks:
            chunk_id = f"{chunk.get('doc_index', chunk.get('doc_id'))}_{chunk['chunk_idx']}"
            metadata_db[chunk_id] = {
                "doc_id": chunk.get("doc_id", chunk.get("doc_index", "")),
                "original_doc_id": chunk.get("original_doc_id", chunk.get("doc_index")),
                "chunk_idx": chunk["chunk_idx"],
                "text": chunk.get("text", ""),
                "section_heading": chunk.get("section_heading", ""),
                "facet_label": chunk.get("facet_label", ""),
                "embedding_l2_normalized": True,
            }

        # posting_lists 를 순회하며 centroid 와 vector_id 정보 추가
        for centroid_id, entries in posting_lists.items():
            for entry in entries:
                chunk_id = entry["chunk_id"]
                meta = metadata_db.setdefault(chunk_id, {})
                meta["centroid_id"] = centroid_id
                meta["vector_id"] = entry.get("vector_id")

        self.metadata_db = metadata_db
        return metadata_db

    # -----------------------------
    # Step 9: 저장소 최적화
    # -----------------------------
    def step9_storage_optimization(self):
        """
        9. 저장소 최적화 
        - posting list 정렬/압축, PQ code packing, doc_id delta-encode.
        """
        # 간단한 구현:
        # - centroid 별로 chunk_id 기준 정렬
        # - pq_code 리스트를 고정 길이 numpy 배열로 변환 (uint8)
        optimized_posting_lists: Dict[int, List[Dict[str, Any]]] = {}

        for centroid_id, entries in self.posting_lists.items():
            if not entries:
                optimized_posting_lists[centroid_id] = []
                continue

            entries_sorted = sorted(entries, key=lambda e: e["chunk_id"])

            # pq_code 배열은 그대로 유지하되, 정렬만 수행
            optimized_posting_lists[centroid_id] = entries_sorted

        self.posting_lists = optimized_posting_lists

    # -----------------------------
    # Step 10: Accelerator 배치 (768차원 L2 가속기 시뮬레이션 호환)
    # -----------------------------
    def step10_deploy_to_accelerator(self) -> Dict[str, Any]:
        """
        10. Accelerator 배치 
        - centroids, PQ codebooks, LUTs를 L2 가속기/SLM에 pre-load 하기 위한 상태로 포맷팅.
        - 모든 도메인 모델이 768차원을 쓰므로 L2 가속기_1/2 데이터 구조는 동일 유지.
        """
        sim = self._l2_sim
        centroids_arr = self.centroids
        lut_arr = self.pq_codebooks

        centroids_packed = sim.pack_vectors_for_hw(centroids_arr) if centroids_arr is not None and centroids_arr.size > 0 else None
        lut_packed = sim.pack_vectors_for_hw(lut_arr) if lut_arr is not None and lut_arr.size > 0 else None

        hw_ready_state = {
            "l2_accelerator_1_centroids": self.centroids,
            "l2_accelerator_2_pq_codes": self.posting_lists,
            "lut_memory": self.pq_codebooks,
            "metadata_sram": self.metadata_db,
            "l2_accelerator_1_centroids_packed": centroids_packed,
            "lut_memory_packed": lut_packed,
            "embedding_dim": self.embedding_dim,
            "domain_mode": self.domain_mode,
            "l2_simulator": sim,
        }
        return hw_ready_state

    def _save_artifacts(self, hw_ready_state: Dict[str, Any], output_dir: str) -> None:
        """
        하드웨어 매핑 데이터를 offline_data_{domain} 폴더에 저장.
        - centroids.npy: L2 가속기_1용 센트로이드 벡터
        - pq_codebooks.npy: 가속기 내 LUT용 PQ 코드북
        - posting_lists.pkl: 각 센트로이드별 청크 ID·PQ 코드 리스트
        - metadata_db.pkl: doc_id, chunk_idx, facet_label(MMR 검증용) 등 메타데이터
        - config.pkl: 임베딩 모델명·설정(온라인 파이프라인과 동일 모델 로드용)
        """
        os.makedirs(output_dir, exist_ok=True)

        if hw_ready_state.get("l2_accelerator_1_centroids") is not None:
            np.save(os.path.join(output_dir, "centroids.npy"), hw_ready_state["l2_accelerator_1_centroids"])
        if hw_ready_state.get("lut_memory") is not None:
            np.save(os.path.join(output_dir, "pq_codebooks.npy"), hw_ready_state["lut_memory"])
        if hw_ready_state.get("l2_accelerator_1_centroids_packed") is not None:
            np.save(os.path.join(output_dir, "centroids_packed.npy"), hw_ready_state["l2_accelerator_1_centroids_packed"])
        if hw_ready_state.get("lut_memory_packed") is not None:
            np.save(os.path.join(output_dir, "lut_packed.npy"), hw_ready_state["lut_memory_packed"])

        with open(os.path.join(output_dir, "posting_lists.pkl"), "wb") as f:
            pickle.dump(hw_ready_state["l2_accelerator_2_pq_codes"], f)
        with open(os.path.join(output_dir, "metadata_db.pkl"), "wb") as f:
            pickle.dump(hw_ready_state["metadata_sram"], f)

        config = {
            "embedding_dim": hw_ready_state.get("embedding_dim", self.embedding_dim),
            "domain_mode": hw_ready_state.get("domain_mode", self.domain_mode),
            "num_centroids": self.num_centroids,
            "num_subspaces": self.num_subspaces,
            "model_name": self.model_name,
            "embedding_model_name": self.model_name,
            "chunk_size_chars": getattr(self, "_last_chunk_size", None),
            "chunk_overlap_chars": getattr(self, "_last_chunk_overlap", None),
            "embeddings_l2_normalized": True,  # 디버깅: 오프라인 임베딩/센트로이드 L2 정규화 적용됨
            "embedding_config": {
                "normalize": True,
                "max_length": 512,
            },
        }
        with open(os.path.join(output_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

    def _compute_and_save_df_stats(self, docs: List[str], output_dir: str) -> None:
        """
        전체 문서에서 단어별 Document Frequency(DF) 계산 후 df_stats.json 저장.
        단어: 소문자화, \\w+ 토큰만 사용.
        """
        df_dict: Dict[str, int] = {}
        for doc in docs:
            if not doc or not isinstance(doc, str):
                continue
            words = set(re.findall(r"\w+", doc.lower()))
            for w in words:
                if w:
                    df_dict[w] = df_dict.get(w, 0) + 1
        total_docs = len([d for d in docs if d and isinstance(d, str)])
        payload = {"total_docs": total_docs, "df_dict": df_dict}
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "df_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return None

    # -----------------------------
    # 파이프라인 실행
    # -----------------------------
    def run_pipeline(
        self,
        raw_documents: List[str],
        doc_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
    ) -> Dict[str, Any]:
        """
        오프라인 파이프라인 전체 실행.
        doc_ids: 각 문서의 제목/고유 ID. 없으면 "doc_0", "doc_1" ... 사용.
        지정된 output_dir(또는 offline_data_{domain_mode})에 모든 산출물 자동 저장.
        """
        clean_docs, original_doc_ids_per_doc, doc_id_per_clean_doc = self.step1_collect_and_clean_data(
            raw_documents, raw_doc_ids=doc_ids
        )
        self._last_chunk_size = chunk_size
        self._last_chunk_overlap = chunk_overlap
        chunks = self.step2_chunking(
            clean_docs,
            original_doc_ids_per_doc,
            doc_id_per_clean_doc=doc_id_per_clean_doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # step3: general/legal/medical 모두 임베딩 추출 직후 L2 정규화(norm=1) 적용 → IVF/PQ 입력
        embs = self.step3_extract_embeddings(chunks, batch_size=batch_size)
        embs = self.step4_optional_opq(embs)

        self.step5_train_ivf_centroids(embs)  # centroid 저장 직전에도 L2 정규화 적용됨
        self.step6_train_pq_codebook(embs)
        self.step7_ivf_pq_indexing(embs, chunks)

        save_dir = output_dir
        if save_dir is None:
            save_dir = os.path.join(self.output_base_dir, f"offline_data_{self.domain_mode}")

        self.step8_attach_metadata(chunks, self.posting_lists)
        self.step9_storage_optimization()

        hw_ready_state = self.step10_deploy_to_accelerator()

        self._compute_and_save_df_stats(clean_docs, save_dir)
        self._save_artifacts(hw_ready_state, save_dir)

        return hw_ready_state

    def run_pipeline_from_dataset(
        self,
        split: str = "train",
        max_docs: Optional[int] = None,
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
    ) -> Dict[str, Any]:
        """
        도메인별 벤치마크 데이터셋을 로드한 뒤 오프라인 파이프라인 실행.
        offline_data_{domain} 폴더에 centroids.npy, pq_codebooks.npy, posting_lists.pkl, metadata_db.pkl, config.pkl 저장.
        """
        contexts, _ = load_dataset_for_domain(self.domain_mode, split=split, max_contexts=max_docs)
        if not contexts:
            raise RuntimeError(f"도메인 {self.domain_mode}에서 로드된 문맥이 없습니다.")
        doc_ids = [str(i) for i in range(len(contexts))]
        return self.run_pipeline(
            contexts,
            doc_ids=doc_ids,
            output_dir=output_dir,
            batch_size=batch_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="오프라인 인덱싱 (도메인별 HF 모델 + 벤치마크 데이터셋)")
    parser.add_argument("--domain", type=str, default=None, choices=["general", "legal", "medical", "cuad"], help="단일 도메인만 실행 시 지정")
    parser.add_argument("--from_dataset", action="store_true", help="벤치마크 데이터셋(squad/cuad/pubmed_qa)으로부터 로드")
    parser.add_argument("--max_docs", type=int, default=500, help="데이터셋 사용 시 최대 문서 수")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1500,
        help="청크 크기(문자 기준). cuad/legal 긴 조항을 잘리지 않게 크게 설정",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=150,
        help="청크 오버랩 크기(문자 기준). Ground Truth 커버리지 향상 목적",
    )
    args = parser.parse_args()

    domains = [args.domain] if args.domain else ["general", "legal", "medical"]
    BASE_OUTPUT_DIR = args.output_dir

    for domain_mode in domains:
        print(f"[{domain_mode.upper()}] 오프라인 인덱싱 (모델: {DOMAIN_MODEL_MAP[domain_mode]}, 데이터셋: {DOMAIN_DATASET_MAP.get(domain_mode, 'N/A')})")
        try:
            indexer = L2RAGOfflineIndexer(domain_mode=domain_mode, output_base_dir=BASE_OUTPUT_DIR)
            if args.from_dataset:
                hw_state = indexer.run_pipeline_from_dataset(
                    split="train",
                    max_docs=args.max_docs,
                    output_dir=None,
                    batch_size=16,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            else:
                SAMPLE_DOCS = [
                    "This is a sample document for testing the offline indexer pipeline.",
                    "Legal contracts may contain clauses regarding liability and indemnification.",
                    "Medical abstracts often describe clinical trials and treatment outcomes.",
                ]
                hw_state = indexer.run_pipeline(
                    SAMPLE_DOCS,
                    output_dir=None,
                    batch_size=8,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            save_dir = os.path.join(BASE_OUTPUT_DIR, f"offline_data_{domain_mode}")
            print(f"  -> 저장 완료: {save_dir}")
            print(f"  -> embedding_dim={hw_state['embedding_dim']}, centroids shape={indexer.centroids.shape if indexer.centroids is not None else None}")
        except Exception as e:
            print(f"  -> 오류: {e}")
        print()