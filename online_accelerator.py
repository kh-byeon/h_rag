from typing import List, Dict, Any, Tuple, Optional
import json
import os
import pickle
import re

import numpy as np

# CUAD 전용: 로컬 processed_cuad와 오프라인 인덱스 경로 일치
DOMAIN_DATASET_MAP = {"cuad": "cuad"}
DOMAIN_DATASET_CONFIG = {"cuad": None}

# [H-AIMDC] CUAD 쿼리 임베딩 (오프라인 인덱서와 동일 모델)
DOMAIN_EMBEDDING_MODEL_MAP = {
    "cuad": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
}

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

# 하드웨어 타이밍 시뮬레이션용 상수 (가상, ms 단위)
T_COARSE_MS = 0.5   # T_coarse: centroid 탐색
T_CANDIDATES_MS = 1.0  # T_candidates: posting list 로드
T_L2_MS = 2.0       # T_L2: L2 거리 + MMR 연산 (후보 1개당 비례 가정)


class L2RAGOnlinePipeline:
    """
    H-AIMDC 하드웨어 가속기 기반 실시간 L2 검색 및 On-the-fly MMR 파이프라인 (CUAD 전용).
    - 임베딩 모델(_encoder): SentenceTransformer로 쿼리/문서 벡터 생성·L2 거리 검색 (lazy load).
    """
    def __init__(
        self,
        hw_ready_state: Dict[str, Any],
        domain_mode: Optional[str] = None,
        nprobe: int = 8,
        top_r: int = 128,
        top_k: int = 10,
        tau_dup: float = 0.5,
        mmr_lambda: float = 0.5,
        gamma: float = 0.7,
        section_penalty_weight: float = 1.5,
        anchor_lambda: float = 1.0,
        anchor_tau_dup: float = 0.0,
        log_facets: bool = True,
        verbose: bool = True,
    ):
        # 오프라인에서 로드된 하드웨어 상태 (Pre-loaded data)
        self.centroids = hw_ready_state.get("l2_accelerator_1_centroids")
        self.pq_codes = hw_ready_state.get("l2_accelerator_2_pq_codes")
        self.lut_memory = hw_ready_state.get("lut_memory")
        self.metadata_sram = hw_ready_state.get("metadata_sram")

        self.domain_mode = (domain_mode or hw_ready_state.get("domain_mode") or "cuad").lower()
        if self.domain_mode != "cuad":
            self.domain_mode = "cuad"
        self.log_facets = log_facets
        self.verbose = verbose

        # DF 기반 동적 필터링 (df_stats.json)
        self.df_dict: Dict[str, int] = hw_ready_state.get("df_dict") or {}
        self.total_docs: int = int(hw_ready_state.get("total_docs") or 0)
        if (not self.df_dict or self.total_docs <= 0) and self.verbose:
            print("[경고] df_stats.json 없음 또는 비어 있음 → DF 필터 비활성화(엔티티 필터만 적용)")

        # 검색 하이퍼파라미터
        self.nprobe = nprobe
        self.top_r = top_r
        self.top_k = top_k
        self.tau_dup = tau_dup
        self.mmr_lambda = mmr_lambda
        # 하이브리드(intra-doc vs inter-doc) Top-K 슬롯 비율: local ≈ gamma * k, global ≈ (1-gamma) * k
        self.gamma = float(gamma)
        # 동일 section_heading 후보는 더 가깝게(중복으로) 보이도록 거리 축소 가중치
        self.section_penalty_weight = max(float(section_penalty_weight), 1e-9)
        # PROPOSED 모드 1라운드(Anchor) step5 오버라이드 (CLI·평가에서 주입)
        self.anchor_lambda = float(anchor_lambda)
        self.anchor_tau_dup = float(anchor_tau_dup)

        # 임베딩 차원 추론 (centroids 또는 LUT 기반)
        self.embedding_dim = 0
        if isinstance(self.centroids, np.ndarray) and self.centroids.size > 0:
            self.embedding_dim = self.centroids.shape[1]
        elif isinstance(self.lut_memory, np.ndarray) and self.lut_memory.size > 0:
            m, _, sub_dim = self.lut_memory.shape
            self.embedding_dim = m * sub_dim
        else:
            self.embedding_dim = 768

        # 도메인별 768차원 쿼리 임베딩 모델 (오프라인 인덱싱과 동일 — lazy load)
        self._encoder: Any = None
        self._tokenizer: Any = None
        # 오프라인 산출물에서 로드한 OPQ 회전 행렬(opq_rotation.npy)
        self.opq_rotation: Optional[np.ndarray] = hw_ready_state.get("opq_rotation")

        # 검색 품질 분석용 (마지막 쿼리 결과)
        self._last_summary: Optional[Dict[str, Any]] = None
        self._last_latency_ms: Optional[Dict[str, float]] = None

        # doc_id -> [(centroid_id, pq_entry)] 역인덱스 (target_doc_id 하이브리드 시 Local O(1) 직접 로드)
        self.doc_to_pq_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        if isinstance(self.pq_codes, dict) and isinstance(self.metadata_sram, dict):
            for cid_pl, posting_list in self.pq_codes.items():
                try:
                    cid_int = int(cid_pl)
                except (TypeError, ValueError):
                    continue
                if not isinstance(posting_list, list):
                    continue
                for entry in posting_list:
                    if not isinstance(entry, dict):
                        continue
                    ch = entry.get("chunk_id")
                    if ch is None:
                        continue
                    meta = self.metadata_sram.get(ch)
                    if not isinstance(meta, dict):
                        continue
                    did = meta.get("doc_id")
                    if did is None:
                        continue
                    ds = str(did).strip()
                    if not ds:
                        continue
                    self.doc_to_pq_entries.setdefault(ds, []).append((cid_int, entry))

    # -----------------------------
    # 쿼리 임베딩 (도메인별 HF 모델, 768차원)
    # -----------------------------
    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder, self._tokenizer
        model_name = DOMAIN_EMBEDDING_MODEL_MAP["cuad"]
        if not _SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            raise ImportError(
                "CUAD 쿼리 임베딩을 위해 sentence-transformers가 필요합니다: pip install sentence-transformers"
            )
        self._encoder = SentenceTransformer(model_name)
        self._tokenizer = None
        return self._encoder, self._tokenizer

    def _encode_query_general(self, text: str) -> np.ndarray:
        encoder, _ = self._get_encoder()
        emb = encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32).flatten()

    def step1_query_in_and_encode(self, user_query: str) -> np.ndarray:
        """
        1. Query In: 사용자 쿼리 수신 및 Query encoder 수행 (시작 시각 t0).
        """
        query_emb = self._encode_query_general(user_query)
        # 오프라인에서 저장한 OPQ 회전행렬이 있으면 쿼리에도 동일 회전 적용
        query_emb = np.asarray(query_emb, dtype=np.float32).flatten()
        if self.opq_rotation is not None:
            rot = np.asarray(self.opq_rotation, dtype=np.float32)
            if rot.ndim == 2 and rot.shape[0] == query_emb.shape[0]:
                query_emb = query_emb @ rot
        return np.asarray(query_emb, dtype=np.float32)

    @staticmethod
    def _parse_cuad_query_streams(user_query: str) -> Tuple[Optional[str], str]:
        """
        Dual-stream: keyword = 첫 번째 "..." 내용, details = Details: 이후 본문.
        반환: (keyword 또는 None, 임베딩용 details 문자열)
        """
        keyword: Optional[str] = None
        m_kw = re.search(r'"([^"]*)"', user_query)
        if m_kw:
            keyword = (m_kw.group(1) or "").strip() or None
        m_det = re.search(r"Details:\s*(.*)$", user_query, re.IGNORECASE | re.DOTALL)
        if m_det:
            details = (m_det.group(1) or "").strip()
        else:
            details = user_query.strip()
        return keyword, details

    def _prepare_cuad_search_text(self, user_query: str) -> Tuple[str, Optional[str]]:
        """Keyword-Context Fusion용 search_text 구성(인코딩 없음)."""
        cuad_keyword, details = self._parse_cuad_query_streams(user_query)
        details = re.sub(
            r'^Highlight the parts \(if any\) of this contract related to ".*?" that should be reviewed by a lawyer\.\s*Details:\s*',
            "",
            details or "",
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        search_text = f"{cuad_keyword}. {details}" if cuad_keyword else details
        if not (search_text or "").strip():
            search_text = user_query
        return search_text, cuad_keyword

    def _build_query_embedding_streams(
        self, user_query: str
    ) -> Tuple[np.ndarray, List[str], Optional[str]]:
        """
        CUAD: Keyword-Context Fusion(search_text = keyword + Details).
        반환: (query_emb, weighting_entities, cuad_keyword 또는 None)
        """
        search_text, cuad_keyword = self._prepare_cuad_search_text(user_query)
        weighting_entities = self.extract_weighting_entities(search_text)
        if self.verbose:
            print(
                f"  [디버그-인코더 입력] Keyword-Context Fusion 임베딩: '{search_text[:200]}{'…' if len(search_text) > 200 else ''}'"
            )
        if cuad_keyword and self.verbose:
            print(f"  [CUAD] keyword (quoted): {cuad_keyword!r}")
        query_emb = self.step1_query_in_and_encode(search_text)
        return query_emb, weighting_entities, cuad_keyword

    def extract_weighting_entities(self, user_query: str) -> List[str]:
        """
        CUAD 전용. NER 미사용 — 기존 파이프라인과 동일하게 가중 엔티티 리스트는 비어 있음.
        """
        min_len = 2
        entities_out: List[str] = []
        if self.verbose:
            print(f"[엔티티-추출] Domain: cuad, MinLen: {min_len}, 결과: {entities_out}")
        return entities_out

    def step3_coarse_search(self, query_emb: np.ndarray, facet: str) -> List[int]:
        """
        3. Retrieval Start (T_coarse): Facet별 쿼리에 대해 nprobe 개의 최근접 centroids 탐색.
        - 오프라인에서 미리 L2 가속기_1에 로드된 centroids 활용.
        """
        if not isinstance(self.centroids, np.ndarray) or self.centroids.size == 0:
            return []

        centroids = self.centroids
        # 전부 0인 패딩 centroid 제거
        norms = np.linalg.norm(centroids, axis=1)
        non_zero_mask = norms > 0
        if not np.any(non_zero_mask):
            return []

        valid_centroids = centroids[non_zero_mask]

        q = query_emb.astype(np.float32)[None, :]  # (1, D)
        x_norm = np.sum(q * q, axis=1, keepdims=True)            # (1, 1)
        c_norm = np.sum(valid_centroids * valid_centroids, axis=1, keepdims=True).T  # (1, C)
        distances = x_norm + c_norm - 2.0 * (q @ valid_centroids.T)  # (1, C)
        distances = distances[0]

        nprobe = min(self.nprobe, valid_centroids.shape[0])
        nearest_idx = np.argpartition(distances, nprobe - 1)[:nprobe]
        
        # [신규 디버그] 전체 중 최소 거리 및 선택된 nprobe개 거리 확인
        min_dist_val = np.min(distances)
        min_dist_idx = np.argmin(distances)
        #print(f"[디버그] 전체 {len(distances)}개 중 최소 L2 거리: {min_dist_val:.4f} (Centroid ID: {min_dist_idx})")
        #print(f"[디버그] 가속기가 선택한 {nprobe}개 센트로이드 거리: {distances[nearest_idx]}")

        original_indices = np.where(non_zero_mask)[0]
        selected_centroid_ids = [int(original_indices[i]) for i in nearest_idx]
        
        # [신규 디버그] 최종 선택된 원래 센트로이드 ID 출력
        #print(f"[디버그] 최종 매핑된 센트로이드 IDs: {selected_centroid_ids}")
        
        return selected_centroid_ids

    def step4_candidate_load(
        self,
        centroid_ids: List[int],
        target_doc_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        4. Candidate Load (T_candidates).
        - target_doc_id 없음: IVF nprobe centroid들에서 top_r만큼만 적재(기존).
        - target_doc_id 있음(하이브리드): Local은 doc_to_pq_entries로 해당 문서 PQ entry 전부(top_r 제한 없음),
          Global은 centroid_ids의 posting에서 동일 문서가 아닌 청크만 top_r까지 적재.
        """
        if self.pq_codes is None or self.lut_memory is None:
            return []

        lut = self.lut_memory
        m, k_size, sub_dim = lut.shape

        def _entry_to_candidate(cid: int, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            pq_code = entry.get("pq_code")
            if pq_code is None:
                return None
            approx_vec_parts = []
            for subspace_idx in range(m):
                code = int(pq_code[subspace_idx])
                if code < 0 or code >= k_size:
                    continue
                approx_vec_parts.append(lut[subspace_idx, code])
            if len(approx_vec_parts) != m:
                return None
            approx_vec = np.concatenate(approx_vec_parts, axis=0).astype(np.float32)
            chunk_id = entry.get("chunk_id")
            meta = self.metadata_sram.get(chunk_id, {}) if isinstance(self.metadata_sram, dict) else {}
            return {
                "chunk_id": chunk_id,
                "embedding": approx_vec,
                "centroid_id": cid,
                "vector_id": entry.get("vector_id"),
                "metadata": meta,
            }

        def _entry_target_doc_id(entry: Dict[str, Any]) -> Optional[str]:
            ch = entry.get("chunk_id")
            meta = self.metadata_sram.get(ch, {}) if isinstance(self.metadata_sram, dict) else {}
            if not isinstance(meta, dict):
                return None
            did = meta.get("doc_id")
            if did is None:
                return None
            s = str(did).strip()
            return s if s else None

        seen_chunk_ids: set = set()
        want = str(target_doc_id).strip() if target_doc_id and str(target_doc_id).strip() else None

        local_cands: List[Dict[str, Any]] = []
        standard: List[Dict[str, Any]] = []

        if want:
            for cid, entry in self.doc_to_pq_entries.get(want, []):
                ch = entry.get("chunk_id")
                if ch is None or ch in seen_chunk_ids:
                    continue
                cand = _entry_to_candidate(int(cid), entry)
                if cand:
                    local_cands.append(cand)
                    seen_chunk_ids.add(ch)

            for cid in centroid_ids:
                posting_list = self.pq_codes.get(cid, [])
                if not posting_list:
                    continue
                for entry in posting_list[: self.top_r]:
                    ch = entry.get("chunk_id")
                    if ch is None or ch in seen_chunk_ids:
                        continue
                    td = _entry_target_doc_id(entry)
                    if td == want:
                        continue
                    cand = _entry_to_candidate(cid, entry)
                    if cand:
                        standard.append(cand)
                        seen_chunk_ids.add(ch)

            candidates = local_cands + standard
            if self.verbose:
                print(
                    f"  [H-AIMDC Step4] Hybrid — Local (doc={want!r}, direct index): {len(local_cands)} | "
                    f"Global (IVF, top_r={self.top_r}): {len(standard)} | Total: {len(candidates)}"
                )
            return candidates

        for cid in centroid_ids:
            posting_list = self.pq_codes.get(cid, [])
            if not posting_list:
                continue
            for entry in posting_list[: self.top_r]:
                ch = entry.get("chunk_id")
                if ch in seen_chunk_ids:
                    continue
                cand = _entry_to_candidate(cid, entry)
                if cand:
                    standard.append(cand)
                    seen_chunk_ids.add(ch)

        if self.verbose:
            print(f"  [H-AIMDC Step4] Standard candidates: {len(standard)}")
        return standard

    def _hybrid_local_global_quotas(self, k: int) -> Tuple[int, int]:
        """Intra-doc(local) vs Inter-doc(global) 슬롯: local ≈ gamma*k, global = k - local."""
        if k <= 0:
            return 0, 0
        g = max(0.0, min(1.0, float(self.gamma)))
        local_want = max(0, int(round(g * k)))
        global_want = k - local_want
        return local_want, global_want

    def _split_candidates_local_global(
        self,
        candidates: List[Dict],
        target_doc_id: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        want = str(target_doc_id).strip()
        local_c: List[Dict] = []
        global_c: List[Dict] = []
        for c in candidates:
            doc_id = (c.get("metadata") or {}).get("doc_id")
            if doc_id is not None and str(doc_id).strip() == want:
                local_c.append(c)
            else:
                global_c.append(c)
        return local_c, global_c

    @staticmethod
    def _chunk_key(item: Dict) -> Any:
        return item.get("chunk_id") if item.get("chunk_id") is not None else item.get("vector_id")

    def _merge_hybrid_ranked_lists_for_doc(
        self,
        local_ranked: List[Dict],
        global_ranked: List[Dict],
        k_final: int,
        local_quota: int,
        global_quota: int,
        target_doc_id: str,
    ) -> Tuple[List[Dict], int, int]:
        """
        점수(또는 L2 거리) 순위 리스트 두 개에서 Local/ Global 할당량을 먼저 채우고,
        부족하면 나머지 풀에서 순서대로 채운다.
        """
        if k_final <= 0:
            return [], 0, 0

        def key_of(x: Dict) -> Any:
            k = self._chunk_key(x)
            return k if k is not None else id(x)

        out: List[Dict] = []
        seen: set = set()
        li, gi = 0, 0

        # 1) Local 슬롯 (최대 local_quota)
        while len(out) < k_final and len(out) < local_quota and li < len(local_ranked):
            while li < len(local_ranked) and key_of(local_ranked[li]) in seen:
                li += 1
            if li >= len(local_ranked):
                break
            seen.add(key_of(local_ranked[li]))
            out.append(local_ranked[li])
            li += 1

        # 2) Global 슬롯 (최대 global_quota)
        glob_added = 0
        while (
            glob_added < global_quota
            and len(out) < k_final
            and gi < len(global_ranked)
        ):
            while gi < len(global_ranked) and key_of(global_ranked[gi]) in seen:
                gi += 1
            if gi >= len(global_ranked):
                break
            seen.add(key_of(global_ranked[gi]))
            out.append(global_ranked[gi])
            gi += 1
            glob_added += 1

        # 3) 백필: local 잔여 → global 잔여
        while len(out) < k_final:
            progressed = False
            while li < len(local_ranked) and len(out) < k_final:
                if key_of(local_ranked[li]) not in seen:
                    seen.add(key_of(local_ranked[li]))
                    out.append(local_ranked[li])
                    progressed = True
                li += 1
            while gi < len(global_ranked) and len(out) < k_final:
                if key_of(global_ranked[gi]) not in seen:
                    seen.add(key_of(global_ranked[gi]))
                    out.append(global_ranked[gi])
                    progressed = True
                gi += 1
            if not progressed:
                break

        want = str(target_doc_id).strip()
        n_local = 0
        n_global = 0
        for item in out[:k_final]:
            doc_id = (item.get("metadata") or {}).get("doc_id")
            if doc_id is not None and str(doc_id).strip() == want:
                n_local += 1
            else:
                n_global += 1
        return out[:k_final], n_local, n_global

    def _estimate_generation_input_tokens(self, s_buffer: List[Dict]) -> int:
        """청크 metadata.text 기준 대략적 토큰 수(문자/4 휴리스틱)."""
        total = 0
        for item in s_buffer:
            text = (item.get("metadata") or {}).get("text")
            if not isinstance(text, str) or not text:
                continue
            total += max(1, len(text) // 4)
        return int(total)

    def _sticky_buffer_try_insert(
        self,
        buffer: List[Dict],
        entry: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        단순 Top-K 삽입.
        - 비어 있으면 추가
        - 꽉 차 있으면 buffer 내 최저 점수(score)보다 높을 때만 교체
        반환: (삽입/교체 성공 여부, 이벤트 태그).
        """
        new_score = float(entry["score"])
        if top_k is None:
            top_k = self.top_k
        if top_k <= 0:
            return False, "buffer_top_k_zero"

        if len(buffer) < top_k:
            buffer.append(entry)
            return True, "appended_not_full"

        assert len(buffer) == top_k

        min_i = min(range(top_k), key=lambda i: float(buffer[i]["score"]))
        if new_score > float(buffer[min_i]["score"]):
            buffer[min_i] = entry
            return True, "replaced_lowest"
        return False, "rejected_low_score"

    def step5_on_the_fly_mmr(
        self,
        query_emb: np.ndarray,
        candidates: List[Dict],
        s_buffer: List[Dict],
        buffer_top_k: Optional[int] = None,
        anchor_buffer: Optional[List[Dict]] = None,
        tau_dup_override: Optional[float] = None,
        mmr_lambda_override: Optional[float] = None,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        5. On-the-fly Re-rank (T_L2): MMR 재정렬.
        - score(c) = -λ·d(q,c) + (1-λ)·min_{s∈S∪A} d(c,s)  (S=가변 MMR 버퍼, A=anchor_buffer)
        - 최초에 S와 A가 모두 비어 있으면 relevance만: -λ·d(q,c)
        - λ는 기본 self.mmr_lambda; mmr_lambda_override가 있으면 그 값을 사용 (예: λ=1 순수 L2).
        - Early Dedup: min d(c,·) < tau_dup 이면 스코어 계산 전 스킵 (앵커·버퍼 모두 기준).
        - tau_dup_override: 지정 시 self.tau_dup 대신 이 값을 조기 탈락 임계로 사용 (Anchor Phase 등).
        - 후보 순서는 사전 정렬하지 않으며, 가속기(Posting List)가 넘겨준 순서 그대로 스트리밍 처리한다.
        buffer_top_k: MMR로 채울 슬롯 수(기본 self.top_k). anchor_buffer가 있으면 **글로벌 후보만** 이 크기만큼 채운 뒤,
          앵커 문서와 합쳐 반환한다(앵커는 점수 비교로 밀려나지 않음).
        anchor_buffer: Core(순수 L2로 고정된 청크). 다양성 페널티(min d) 계산에만 포함되며, 반환 시 앞쪽에 그대로 붙인다.
        """
        buf_k = self.top_k if buffer_top_k is None else buffer_top_k
        anchors: List[Dict] = [dict(x) for x in (anchor_buffer or [])]

        if buf_k <= 0:
            if anchors:
                out = anchors[: self.top_k]
                return out, {
                    "early_dedup_count": 0,
                    "min_dists_accepted": [float("inf")] * len(out),
                }
            return [], {
                "early_dedup_count": 0,
                "min_dists_accepted": [],
            }

        mmr_lambda_eff = (
            float(mmr_lambda_override)
            if mmr_lambda_override is not None
            else float(self.mmr_lambda)
        )
        tau_thresh = float(self.tau_dup) if tau_dup_override is None else float(tau_dup_override)

        updated_s_buffer: List[Dict] = [dict(x) for x in s_buffer]
        early_dedup_count = 0
        skipped_for_dedup: List[Tuple[Dict, float]] = []

        def _section_heading_of(item: Dict[str, Any]) -> str:
            meta = item.get("metadata") if isinstance(item, dict) else None
            if isinstance(meta, dict):
                return str(meta.get("section_heading", "") or "").strip()
            return str(item.get("section_heading", "") or "").strip() if isinstance(item, dict) else ""

        def _min_dist_to_sets(cand: Dict[str, Any]) -> float:
            cand_emb = np.asarray(cand["embedding"], dtype=np.float32)
            cand_heading = _section_heading_of(cand)
            min_d = float("inf")
            for s in updated_s_buffer:
                s_emb = np.asarray(s["embedding"], dtype=np.float32)
                d = float(np.linalg.norm(cand_emb - s_emb))
                s_heading = _section_heading_of(s)
                if cand_heading and s_heading and cand_heading == s_heading:
                    d = d / self.section_penalty_weight
                if np.isfinite(d) and d < min_d:
                    min_d = d
            for a in anchors:
                a_emb = np.asarray(a["embedding"], dtype=np.float32)
                d = float(np.linalg.norm(cand_emb - a_emb))
                a_heading = _section_heading_of(a)
                if cand_heading and a_heading and cand_heading == a_heading:
                    d = d / self.section_penalty_weight
                if np.isfinite(d) and d < min_d:
                    min_d = d
            return min_d

        def _make_entry(
            cand: Dict,
            dist_q_c: float,
            min_dist_c_s: Optional[float],
            mmr_score: float,
        ) -> Dict[str, Any]:
            md = (
                float(min_dist_c_s)
                if min_dist_c_s is not None and np.isfinite(min_dist_c_s)
                else float("inf")
            )
            return {
                **cand,
                "score": mmr_score,
                "_mmr_min_dist_s": md,
            }

        query_emb = np.asarray(query_emb, dtype=np.float32)

        # 사전 정렬 없이 하드웨어 가속기(Posting List)가 밀어주는 순서 그대로 스트리밍(On-the-fly) 처리
        for candidate in candidates:
            cand_emb = np.asarray(candidate["embedding"], dtype=np.float32)
            dist_q_c = float(np.linalg.norm(query_emb - cand_emb))
            if not np.isfinite(dist_q_c):
                dist_q_c = 0.0

            has_context = bool(updated_s_buffer) or bool(anchors)
            if not has_context:
                score = -mmr_lambda_eff * dist_q_c
                entry = _make_entry(candidate, dist_q_c, None, score)
                self._sticky_buffer_try_insert(updated_s_buffer, entry, buf_k)
                continue

            # [Hard Filter] 다양성 페널티 계산 전, 완전 중복 문서 조기 제거 (tau_dup)
            min_dist_c_s = _min_dist_to_sets(candidate)

            if min_dist_c_s < tau_thresh:
                early_dedup_count += 1
                skipped_for_dedup.append((candidate, dist_q_c))
                continue

            # [Soft Penalty] MMR 점수 계산
            mmr_score = (
                -mmr_lambda_eff * dist_q_c
                + (1.0 - mmr_lambda_eff) * min_dist_c_s
            )
            if not np.isfinite(mmr_score):
                mmr_score = -mmr_lambda_eff * dist_q_c

            entry = _make_entry(candidate, dist_q_c, min_dist_c_s, mmr_score)
            self._sticky_buffer_try_insert(updated_s_buffer, entry, buf_k)

        if len(updated_s_buffer) < buf_k and skipped_for_dedup:
            skipped_for_dedup.sort(key=lambda x: x[1])
            for cand, d in skipped_for_dedup:
                if len(updated_s_buffer) >= buf_k:
                    break
                score = -mmr_lambda_eff * d
                entry = _make_entry(cand, d, 0.0, score)
                self._sticky_buffer_try_insert(updated_s_buffer, entry, buf_k)

        updated_s_buffer.sort(key=lambda x: float(x["score"]), reverse=True)

        min_dists_global = [
            float(e.get("_mmr_min_dist_s", float("inf"))) for e in updated_s_buffer
        ]
        if anchors:
            min_dists_accepted = [float("inf")] * len(anchors) + min_dists_global
            final_out = anchors + updated_s_buffer
            if len(final_out) > self.top_k:
                final_out = final_out[: self.top_k]
                min_dists_accepted = min_dists_accepted[: len(final_out)]
        else:
            final_out = updated_s_buffer
            min_dists_accepted = min_dists_global

        if self.verbose:
            print(f"[MMR] 최종 선택 문서 수: {len(final_out)}")

        stats = {
            "early_dedup_count": early_dedup_count,
            "min_dists_accepted": min_dists_accepted,
        }
        return final_out, stats

    def _compute_per_query_summary(
        self,
        s_buffer: List[Dict],
        early_dedup_count: int = 0,
        min_dists_accepted: Optional[List[float]] = None,
        nprobe_used: Optional[int] = None,
        hybrid_local_count: Optional[int] = None,
        hybrid_global_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Per-query summary (슬라이드 13페이지 기준)."""
        nprobe_used = nprobe_used if nprobe_used is not None else self.nprobe
        min_dists = min_dists_accepted or []
        min_dists_finite = [x for x in min_dists if np.isfinite(x)]
        unique_centroids = set()
        unique_doc_ids = set()
        for item in s_buffer:
            unique_centroids.add(item.get("centroid_id"))
            meta = item.get("metadata", {})
            doc_id = meta.get("doc_id")
            if doc_id is not None:
                unique_doc_ids.add(doc_id)

        unique_centroids_selected = len(unique_centroids)
        unique_centroids_ratio = (
            unique_centroids_selected / nprobe_used if nprobe_used else 0.0
        )
        try:
            p25_min_dist = (
                float(np.percentile(min_dists_finite, 25)) if len(min_dists_finite) > 0 else None
            )
            if p25_min_dist is not None and not np.isfinite(p25_min_dist):
                p25_min_dist = None
        except Exception:
            p25_min_dist = None

        # Top-K 문서 간 평균 쌍별 L2 거리 (Intra-List Distance, 의미적 풍성함)
        intra_list_distance = 0.0
        embs = []
        for item in s_buffer:
            e = item.get("embedding")
            if e is not None:
                embs.append(np.asarray(e, dtype=np.float32))
        if len(embs) > 1:
            n_vec = len(embs)
            total_dist = 0.0
            count = 0
            for i in range(n_vec):
                for j in range(i + 1, n_vec):
                    d = float(np.linalg.norm(embs[i] - embs[j]))
                    if np.isfinite(d):
                        total_dist += d
                        count += 1
            intra_list_distance = total_dist / count if count else 0.0

        gen_tokens = self._estimate_generation_input_tokens(s_buffer)
        local_vs_global_ratio: Optional[Dict[str, Any]] = None
        if (
            hybrid_local_count is not None
            and hybrid_global_count is not None
        ):
            tot_h = hybrid_local_count + hybrid_global_count
            local_vs_global_ratio = {
                "local_chunks": hybrid_local_count,
                "global_chunks": hybrid_global_count,
                "ratio_local": (
                    float(hybrid_local_count) / float(tot_h) if tot_h else 0.0
                ),
            }

        return {
            "unique_centroids_selected": unique_centroids_selected,
            "unique_centroids_ratio": unique_centroids_ratio,
            "unique_doc_ids": len(unique_doc_ids),
            "p25_min_dist_in_query": p25_min_dist,
            "early_dedup_count": early_dedup_count,
            "intra_list_distance": intra_list_distance,
            "generation_input_tokens": gen_tokens,
            "local_vs_global_ratio": local_vs_global_ratio,
        }

    def step6_coverage_check(
        self,
        s_buffer: List[Dict],
        early_dedup_count: int = 0,
    ) -> bool:
        """
        6. Coverage Check: Metadata 활용하여 MMR coverage 체크. [cite: 31]
        - Relevance/Redundancy floor, Centroid spread, Facet quota, Doc diversity 등 평가.
        """
        if not s_buffer:
            return False
        if len(s_buffer) < self.top_k:
            return False

        doc_ids = set()
        centroid_ids = set()

        for item in s_buffer:
            meta = item.get("metadata", {})
            doc_id = meta.get("doc_id")
            if doc_id is not None:
                doc_ids.add(doc_id)
            centroid_ids.add(item.get("centroid_id"))

        min_doc_div = min(2, self.top_k)
        min_centroid_div = min(2, self.nprobe, self.top_k)
        doc_diverse = len(doc_ids) >= min_doc_div
        centroid_spread = len(centroid_ids) >= min_centroid_div

        # 후보가 대부분 early-dedup으로 탈락하면 탐색 커버리지가 부족하다고 본다.
        early_dedup_limit = max(2, int(self.top_k * 1.5))
        dedup_ok = early_dedup_count <= early_dedup_limit

        coverage_satisfied = doc_diverse and centroid_spread and dedup_ok
        return coverage_satisfied

    def _run_l2_only(
        self,
        query_text: str,
        target_doc_id: Optional[str] = None,
    ) -> Tuple[List[Dict], Dict[str, Any], Dict[str, float]]:
        """
        순수 L2 Top-K 검색 (Intra/Inter 하이브리드 범위 분리 유지).
        """
        query_emb, _, _ = self._build_query_embedding_streams(query_text)
        centroid_ids = self.step3_coarse_search(query_emb, "query")
        tid_step4 = (
            str(target_doc_id).strip()
            if target_doc_id and str(target_doc_id).strip()
            else None
        )
        all_candidates = self.step4_candidate_load(centroid_ids, target_doc_id=tid_step4)
        query_emb_arr = np.asarray(query_emb, dtype=np.float32)
        for c in all_candidates:
            c["_dist_q"] = float(
                np.linalg.norm(query_emb_arr - np.asarray(c["embedding"], dtype=np.float32))
            )

        k = self.top_k
        use_hybrid = bool(target_doc_id and str(target_doc_id).strip())
        hybrid_l2_local: Optional[int] = None
        hybrid_l2_global: Optional[int] = None

        if use_hybrid:
            tid = str(target_doc_id).strip()
            local_c, global_c = self._split_candidates_local_global(all_candidates, tid)
            lq, gq = self._hybrid_local_global_quotas(k)
            if not local_c:
                pool = sorted(global_c, key=lambda x: x["_dist_q"])[:k]
                hybrid_l2_local, hybrid_l2_global = 0, len(pool)
            elif not global_c:
                pool = sorted(local_c, key=lambda x: x["_dist_q"])[:k]
                hybrid_l2_local, hybrid_l2_global = len(pool), 0
            else:
                loc_sorted = sorted(local_c, key=lambda x: x["_dist_q"])
                glob_sorted = sorted(global_c, key=lambda x: x["_dist_q"])
                pool, hybrid_l2_local, hybrid_l2_global = self._merge_hybrid_ranked_lists_for_doc(
                    loc_sorted, glob_sorted, k, lq, gq, tid
                )
            l2_topk_results = [{kk: v for kk, v in c.items() if kk != "_dist_q"} for c in pool]
        else:
            all_candidates.sort(key=lambda x: x["_dist_q"])
            k_take_l2 = min(k, len(all_candidates))
            l2_topk_results = [
                {kk: v for kk, v in c.items() if kk != "_dist_q"}
                for c in all_candidates[:k_take_l2]
            ]

        summary = self._compute_per_query_summary(
            l2_topk_results,
            early_dedup_count=0,
            min_dists_accepted=None,
            nprobe_used=len(centroid_ids),
            hybrid_local_count=hybrid_l2_local,
            hybrid_global_count=hybrid_l2_global,
        )
        latency = {
            "T_coarse_ms": T_COARSE_MS,
            "T_candidates_ms": T_CANDIDATES_MS,
            "T_L2_ms": T_L2_MS * max(len(all_candidates), 1),
        }
        return l2_topk_results, summary, latency

    def run_pipeline(
        self,
        user_query: str,
        target_doc_id: Optional[str] = None,
        log_summary: bool = True,
        mode: str = "PROPOSED",
        query_concat: Optional[str] = None,
        query_aspects: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        검색 모드 라우팅:
        - B1: 원본 user_query 기반 단일 쿼리 순수 L2 (싱글 대조군)
        - B2: user_query + query_aspects와 동일한 순차 루프이나, 매 라운드 λ=1·τ=0으로 순수 L2만 수행
        - PROPOSED: 순차 re-query + MMR(1라운드만 self.anchor_lambda/self.anchor_tau_dup, 이후 self.mmr_lambda/τ)

        - target_doc_id가 비어 있지 않으면(Core & Explore): Intra-doc 후보는 순수 L2로 상위 local_quota개를
          Core(s_loc)로 고정하고, Inter-doc 후보는 앵커(s_loc)를 기준으로 MMR만 수행하여 나머지 슬롯을 채운다.
        - target_doc_id가 없으면: 전체 후보로 단일 MMR 재정렬을 수행.
        """
        mode_upper = (mode or "PROPOSED").upper()
        if mode_upper == "B1":
            l2_results, l2_summary, l2_latency = self._run_l2_only(
                user_query,
                target_doc_id=target_doc_id,
            )
            self._last_summary = l2_summary
            self._last_latency_ms = l2_latency
            if log_summary and self.verbose:
                self._log_per_query_summary(l2_summary, l2_latency)
            return l2_results

        debug_log_path = "pipeline_debug_log.txt"
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== Query: {user_query} | Target ID: {target_doc_id} ===\n")

        def _append_debug_log(line: str) -> None:
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(line)

        latency = {"T_coarse_ms": 0.0, "T_candidates_ms": 0.0, "T_L2_ms": 0.0}

        aspect_texts = [a for a in (query_aspects or []) if isinstance(a, str) and a.strip()]
        queries_to_run_texts: List[str] = [user_query] + aspect_texts

        use_hybrid = bool(target_doc_id and str(target_doc_id).strip())
        s_buffer: List[Dict] = []
        total_early_dedup = 0
        all_min_dists: List[float] = []
        total_nprobe_used = 0
        hybrid_local_count: Optional[int] = None
        hybrid_global_count: Optional[int] = None
        loop_round = 0
        persistent_anchors: List[Dict] = []

        while queries_to_run_texts:
            loop_round += 1
            current_text = queries_to_run_texts.pop(0)
            # Lazy Evaluation: 해당 라운드에서만 인코딩
            current_query_emb, current_entities, current_kw = (
                self._build_query_embedding_streams(current_text)
            )
            current_query_emb = np.asarray(current_query_emb, dtype=np.float32)
            if loop_round == 1 and self.verbose and (log_summary or self.log_facets):
                # Priority(키워드 기반 정확 매칭) 로직은 제거되었으므로 entities만 표시합니다.
                print(
                    f"  추출된 핵심 엔티티: {current_entities if current_entities else '(없음)'}"
                )
            centroid_ids = self.step3_coarse_search(current_query_emb, "query")
            nprobe_used = len(centroid_ids)
            total_nprobe_used += nprobe_used
            latency["T_coarse_ms"] += T_COARSE_MS
            if self.verbose and (log_summary or self.log_facets):
                print(f"  [Re-query #{loop_round}] 탐색 nprobe 수: {nprobe_used}")

            tid_step4 = (
                str(target_doc_id).strip()
                if target_doc_id and str(target_doc_id).strip()
                else None
            )
            candidates = self.step4_candidate_load(centroid_ids, target_doc_id=tid_step4)
            latency["T_candidates_ms"] += T_CANDIDATES_MS
            latency["T_L2_ms"] += T_L2_MS * max(len(candidates), 1)
            _append_debug_log(
                f"Step 4 (Candidate Load, round={loop_round}): Total {len(candidates)} chunks loaded.\n"
            )

            def _step5_mmr_kw() -> Dict[str, Any]:
                """B2: 모든 라운드 순수 L2. PROPOSED: 1라운드만 지정된 anchor override 사용."""
                if mode_upper == "B2":
                    # B2는 무조건 순수 L2를 강제해야 하므로 하드코딩 유지
                    return {"mmr_lambda_override": 1.0, "tau_dup_override": 0.0}
                if mode_upper == "PROPOSED" and loop_round == 1:
                    # PROPOSED의 1라운드는 외부에서 주입된 파라미터 사용
                    return {
                        "mmr_lambda_override": self.anchor_lambda,
                        "tau_dup_override": self.anchor_tau_dup,
                    }
                return {}

            _s5 = _step5_mmr_kw()

            if use_hybrid:
                tid = str(target_doc_id).strip()
                local_c, global_c = self._split_candidates_local_global(candidates, tid)
                k = self.top_k
                local_q, global_q = self._hybrid_local_global_quotas(k)
                _append_debug_log(
                    f"Step 5 (Hybrid Core&Explore, round={loop_round}): local_pool={len(local_c)}, "
                    f"global_pool={len(global_c)}, quota local/global={local_q}/{global_q}.\n"
                )

                if loop_round == 1:
                    # Anchor Phase: 타겟 문서(local)에서만 gamma·local_q 슬롯까지 선발. 글로벌은 채우지 않음.
                    if local_c:
                        s_loc, stats = self.step5_on_the_fly_mmr(
                            current_query_emb,
                            local_c,
                            [],
                            buffer_top_k=local_q,
                            **_s5,
                        )
                        persistent_anchors = list(s_loc)
                        s_buffer = list(s_loc)
                    else:
                        persistent_anchors = []
                        s_buffer = []
                        stats = {"early_dedup_count": 0, "min_dists_accepted": []}
                else:
                    # Aspect round: 로컬/글로벌 구분 없이 전체 후보를 공정 경쟁시키고,
                    # 1라운드 앵커만 기준점(anchor)으로 유지.
                    s_buffer, stats = self.step5_on_the_fly_mmr(
                        current_query_emb,
                        candidates,
                        s_buffer,
                        buffer_top_k=k,
                        anchor_buffer=persistent_anchors,
                        **_s5,
                    )
                _append_debug_log(
                    f"Step 5 (MMR & Dedup, hybrid, round={loop_round}): {len(s_buffer)} chunks retained.\n"
                )
            else:
                s_buffer, stats = self.step5_on_the_fly_mmr(
                    current_query_emb, candidates, s_buffer, **_s5
                )
                _append_debug_log(
                    f"Step 5 (MMR & Dedup, round={loop_round}): {len(s_buffer)} chunks retained.\n"
                )

            total_early_dedup += int(stats.get("early_dedup_count", 0) or 0)
            iter_min_dists = stats.get("min_dists_accepted") or []
            all_min_dists.extend(iter_min_dists)

            # Coverage가 충분하면 조기 종료, 아니면 다음 aspect로 re-query
            coverage_ok = self.step6_coverage_check(
                s_buffer,
                early_dedup_count=total_early_dedup,
            )
            _append_debug_log(
                f"Step 6 (Coverage Check, round={loop_round}): "
                f"{'Satisfied' if coverage_ok else 'Insufficient'} | "
                f"queue_remaining={len(queries_to_run_texts)}\n"
            )
            if coverage_ok:
                break

        if use_hybrid and target_doc_id and str(target_doc_id).strip():
            tid = str(target_doc_id).strip()
            n_local = 0
            n_glob = 0
            for item in s_buffer:
                doc_id = (item.get("metadata") or {}).get("doc_id")
                if doc_id is not None and str(doc_id).strip() == tid:
                    n_local += 1
                else:
                    n_glob += 1
            hybrid_local_count, hybrid_global_count = n_local, n_glob

        k_take = min(self.top_k, len(s_buffer))
        final_top_k = s_buffer[:k_take]
        summary = self._compute_per_query_summary(
            final_top_k,
            early_dedup_count=total_early_dedup,
            min_dists_accepted=all_min_dists if all_min_dists else None,
            nprobe_used=max(total_nprobe_used, 1),
            hybrid_local_count=hybrid_local_count,
            hybrid_global_count=hybrid_global_count,
        )
        self._last_summary = summary
        self._last_latency_ms = latency

        if log_summary and self.verbose:
            self._log_per_query_summary(summary, latency)

        return final_top_k

    def _log_per_query_summary(
        self,
        summary: Dict[str, Any],
        latency: Dict[str, float],
    ) -> None:
        """Per-query summary 및 Latency breakdown 출력."""
        total_ms = sum(latency.values())
        print("[Per-query summary]")
        print(f"  unique_centroids_selected: {summary['unique_centroids_selected']}")
        print(f"  unique_centroids_ratio:   {summary['unique_centroids_ratio']:.4f}")
        print(f"  unique_doc_ids:           {summary['unique_doc_ids']} (Doc diversity)")
        print(f"  p25_min_dist_in_query:    {summary['p25_min_dist_in_query']}")
        print(f"  early_dedup_count:        {summary['early_dedup_count']}")
        print(f"  generation_input_tokens:  {summary.get('generation_input_tokens', 0)}")
        lvg = summary.get("local_vs_global_ratio")
        if lvg:
            print(
                f"  local_vs_global_ratio:    local={lvg['local_chunks']}, "
                f"global={lvg['global_chunks']}, ratio_local={lvg['ratio_local']:.4f}"
            )
        print("[Latency breakdown (simulated ms)]")
        print(f"  T_coarse:     {latency['T_coarse_ms']:.2f} ms")
        print(f"  T_candidates: {latency['T_candidates_ms']:.2f} ms")
        print(f"  T_L2:         {latency['T_L2_ms']:.2f} ms")
        print(f"  Total:        {total_ms:.2f} ms")
        print()

    def run_with_comparison(
        self,
        user_query: str,
        target_doc_id: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[List[Dict], List[Dict], Dict[str, Any], Dict[str, Any]]:
        """
        동일 쿼리에 대해 (1) 일반 L2 Top-K 검색 vs (2) On-the-fly MMR 검색을 수행하고
        지표를 나란히 비교.
        - target_doc_id가 있으면: run_pipeline과 동일하게 하이브리드(후보 분리·7:3 병합)로 L2·MMR 모두 정렬.
        - target_doc_id가 없으면: 전체 후보를 대상으로 L2 Top-K / MMR 재정렬을 수행.
        반환: (l2_topk_results, mmr_results, l2_summary, mmr_summary)
        """
        use_hybrid = bool(target_doc_id and str(target_doc_id).strip())
        query_emb, weighting_entities, cuad_keyword = self._build_query_embedding_streams(
            user_query
        )

        centroid_ids = self.step3_coarse_search(query_emb, "query")
        tid_step4 = (
            str(target_doc_id).strip()
            if target_doc_id and str(target_doc_id).strip()
            else None
        )
        all_candidates = self.step4_candidate_load(centroid_ids, target_doc_id=tid_step4)
        t_coarse_l2, t_cand_l2 = T_COARSE_MS, T_CANDIDATES_MS
        for c in all_candidates:
            c["_dist_q"] = float(
                np.linalg.norm(
                    np.asarray(query_emb, dtype=np.float32)
                    - np.asarray(c["embedding"], dtype=np.float32)
                )
            )
        k = self.top_k
        hybrid_l2_local: Optional[int] = None
        hybrid_l2_global: Optional[int] = None
        if use_hybrid:
            tid = str(target_doc_id).strip()
            local_c, global_c = self._split_candidates_local_global(all_candidates, tid)
            lq, gq = self._hybrid_local_global_quotas(k)
            if not local_c:
                pool = sorted(global_c, key=lambda x: x["_dist_q"])[:k]
                hybrid_l2_local, hybrid_l2_global = 0, len(pool)
            elif not global_c:
                pool = sorted(local_c, key=lambda x: x["_dist_q"])[:k]
                hybrid_l2_local, hybrid_l2_global = len(pool), 0
            else:
                loc_sorted = sorted(local_c, key=lambda x: x["_dist_q"])
                glob_sorted = sorted(global_c, key=lambda x: x["_dist_q"])
                pool, hybrid_l2_local, hybrid_l2_global = (
                    self._merge_hybrid_ranked_lists_for_doc(
                        loc_sorted, glob_sorted, k, lq, gq, tid
                    )
                )
            l2_topk_results = [
                {kk: v for kk, v in c.items() if kk != "_dist_q"} for c in pool
            ]
        else:
            all_candidates.sort(key=lambda x: x["_dist_q"])
            k_take_l2 = min(k, len(all_candidates))
            l2_topk_results = [
                {kk: v for kk, v in c.items() if kk != "_dist_q"}
                for c in all_candidates[:k_take_l2]
            ]
        t_l2_l2 = T_L2_MS * max(len(all_candidates), 1)

        l2_summary = self._compute_per_query_summary(
            l2_topk_results,
            early_dedup_count=0,
            min_dists_accepted=None,
            nprobe_used=self.nprobe,
            hybrid_local_count=hybrid_l2_local,
            hybrid_global_count=hybrid_l2_global,
        )
        l2_latency = {
            "T_coarse_ms": t_coarse_l2,
            "T_candidates_ms": t_cand_l2,
            "T_L2_ms": t_l2_l2,
        }

        # ---- (2) On-the-fly MMR
        self._last_summary = None
        self._last_latency_ms = None
        mmr_results = self.run_pipeline(
            user_query,
            target_doc_id=target_doc_id,
            log_summary=False,
            mode="PROPOSED",
        )
        mmr_summary = self._last_summary or {}
        mmr_latency = self._last_latency_ms or {}

        # ---- 나란히 출력 (verbose 시에만)
        if verbose:
            print("========== L2 Top-K vs On-the-fly MMR 비교 ==========")
            print(f"Query: {user_query[:80]}{'...' if len(user_query) > 80 else ''}")
            print(
                f"  추출된 핵심 엔티티: {weighting_entities if weighting_entities else '(없음)'}"
            )
            print(f"  단일 검색으로 탐색한 nprobe 수: {len(centroid_ids)}")
            print()
            print("--- (1) 일반 L2 Top-K ---")
            print(f"  unique_centroids_selected: {l2_summary['unique_centroids_selected']}")
            print(f"  unique_centroids_ratio:   {l2_summary['unique_centroids_ratio']:.4f}")
            print(f"  unique_doc_ids:           {l2_summary['unique_doc_ids']}")
            print(f"  p25_min_dist_in_query:    {l2_summary['p25_min_dist_in_query']}")
            print(f"  early_dedup_count:        {l2_summary['early_dedup_count']}")
            print(f"  Latency: T_coarse={l2_latency['T_coarse_ms']:.2f} ms, "
                  f"T_candidates={l2_latency['T_candidates_ms']:.2f} ms, "
                  f"T_L2={l2_latency['T_L2_ms']:.2f} ms")
            print()
            print("--- (2) On-the-fly MMR ---")
            print(f"  unique_centroids_selected: {mmr_summary['unique_centroids_selected']}")
            print(f"  unique_centroids_ratio:   {mmr_summary['unique_centroids_ratio']:.4f}")
            print(f"  unique_doc_ids:           {mmr_summary['unique_doc_ids']}")
            print(f"  p25_min_dist_in_query:    {mmr_summary['p25_min_dist_in_query']}")
            print(f"  early_dedup_count:        {mmr_summary['early_dedup_count']}")
            print(f"  Latency: T_coarse={mmr_latency['T_coarse_ms']:.2f} ms, "
                  f"T_candidates={mmr_latency['T_candidates_ms']:.2f} ms, "
                  f"T_L2={mmr_latency['T_L2_ms']:.2f} ms")
            print("=====================================================")

        return l2_topk_results, mmr_results, l2_summary, mmr_summary


# =============================================================================
# 데이터셋 로더 (CUAD 로컬 JSON만)
# =============================================================================

def load_dataset_for_domain(
    domain: str,
    split: str = "train",
    max_contexts: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, int, Optional[str], Optional[str], List[str]]]]:
    """
    로컬 processed_cuad/cuad_corpus.json, cuad_queries.json에서
    인덱싱용 문맥과 평가용 (question, ref_doc_id, answers_text, query_concat, query_aspects) 로드.
    """
    _ = split  # API 호환용 (미사용)
    if domain.lower() != "cuad":
        raise ValueError(f"이 모듈은 CUAD 전용입니다. domain='cuad'만 허용됩니다. 입력: {domain!r}")

    corpus_path = os.path.join(".", "processed_cuad", "cuad_corpus.json")
    queries_path = os.path.join(".", "processed_cuad", "cuad_queries.json")
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(f"CUAD 코퍼스 파일이 없습니다: {corpus_path}. preprocess_cuad.py를 먼저 실행하세요.")
    if not os.path.isfile(queries_path):
        raise FileNotFoundError(f"CUAD 쿼리 파일이 없습니다: {queries_path}. preprocess_cuad.py를 먼저 실행하세요.")
    with open(corpus_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)
    eval_pairs = [
        (
            item["query"],
            int(item["ref_doc_id"]),
            item.get("answers", ""),
            item.get("query_concat"),
            item.get("query_aspects", []) or [],
        )
        for item in queries_raw
    ]
    if max_contexts is not None and max_contexts > 0:
        contexts = contexts[:max_contexts]
        eval_pairs = [
            (q, ref_id, ans, q_concat, q_aspects)
            for (q, ref_id, ans, q_concat, q_aspects) in eval_pairs
            if ref_id < len(contexts)
        ]
    return contexts, eval_pairs


def load_hw_state_from_dir(dir_path: str) -> Dict[str, Any]:
    """도메인별 저장 폴더(offline_data_*)에서 Centroids, PQ codebooks, Posting lists, Metadata를 읽어 hw_ready_state 구성."""
    dir_path = os.path.abspath(dir_path)
    centroids = None
    pq_codebooks = None
    path_c = os.path.join(dir_path, "centroids.npy")
    path_pq = os.path.join(dir_path, "pq_codebooks.npy")
    path_pl = os.path.join(dir_path, "posting_lists.pkl")
    path_meta = os.path.join(dir_path, "metadata_db.pkl")
    path_config = os.path.join(dir_path, "config.pkl")
    path_opq = os.path.join(dir_path, "opq_rotation.npy")
    if os.path.isfile(path_c):
        centroids = np.load(path_c)
    if os.path.isfile(path_pq):
        pq_codebooks = np.load(path_pq)
    opq_rotation = None
    if os.path.isfile(path_opq):
        opq_rotation = np.load(path_opq)
    with open(path_pl, "rb") as f:
        posting_lists = pickle.load(f)
    with open(path_meta, "rb") as f:
        metadata_db = pickle.load(f)
    domain_mode = "cuad"
    if os.path.isfile(path_config):
        try:
            with open(path_config, "rb") as f:
                config = pickle.load(f)
            domain_mode = config.get("domain_mode", "cuad")
        except Exception:
            pass

    df_dict: Dict[str, int] = {}
    total_docs = 0
    path_df = os.path.join(dir_path, "df_stats.json")
    if os.path.isfile(path_df):
        try:
            with open(path_df, "r", encoding="utf-8") as f:
                df_stats = json.load(f)
            df_dict = df_stats.get("df_dict") or {}
            total_docs = int(df_stats.get("total_docs", 0))
        except Exception:
            pass

    return {
        "l2_accelerator_1_centroids": centroids,
        "l2_accelerator_2_pq_codes": posting_lists,
        "lut_memory": pq_codebooks,
        "metadata_sram": metadata_db,
        "embedding_dim": int(centroids.shape[1]) if centroids is not None else 768,
        "domain_mode": domain_mode,
        "df_dict": df_dict,
        "total_docs": total_docs,
        "opq_rotation": opq_rotation,
    }


def run_offline_indexing_for_domain(
    domain: str,
    output_base_dir: str = ".",
    max_docs: Optional[int] = 5000,
) -> str:
    """
    CUAD 코퍼스를 로드해 L2RAGOfflineIndexer로 인덱싱한 뒤 offline_data_cuad에 저장.
    반환: 저장된 디렉터리 경로.
    """
    from off_line_prepper import L2RAGOfflineIndexer  # noqa: F401

    if domain.lower() != "cuad":
        raise ValueError(f"CUAD 전용입니다. domain='cuad'만 허용됩니다. 입력: {domain!r}")
    contexts, _ = load_dataset_for_domain("cuad", max_contexts=max_docs)
    if not contexts:
        raise RuntimeError(f"도메인 {domain}에서 로드된 문맥이 없습니다.")

    indexer = L2RAGOfflineIndexer(
        domain_mode=domain,
        output_base_dir=output_base_dir,
    )
    save_dir = os.path.join(output_base_dir, f"offline_data_{domain}")
    indexer.run_pipeline(contexts, output_dir=save_dir)
    return save_dir


GT_COVERAGE_THRESHOLD = 1.0


def _ground_truth_token_coverage(ground_truth: str, retrieved_text: str) -> float:
    if not ground_truth or not str(ground_truth).strip():
        return 0.0

    # [SEP] 토큰으로 여러 개의 정답(Answers) 분리
    answers = [ans.strip() for ans in ground_truth.split("[SEP]") if ans.strip()]
    if not answers:
        return 0.0

    chunk_set = set(re.findall(r"[a-z0-9]+", (retrieved_text or "").lower()))
    if not chunk_set:
        return 0.0

    max_score = 0.0
    for ans in answers:
        gt_tokens = re.findall(r"[a-z0-9]+", ans.lower())
        if not gt_tokens:
            continue

        matched = sum(1 for w in gt_tokens if w in chunk_set)
        n_gt = len(gt_tokens)

        if n_gt <= 3:
            score = 1.0 if matched == n_gt else 0.0
        else:
            ratio = matched / float(n_gt)
            score = 1.0 if ratio >= 0.85 else 0.0

        if score > max_score:
            max_score = score

    return max_score

def _trim_broken_edges(text: str) -> str:
    """
    스티칭된 텍스트의 양 끝단에 있는 불완전한 단어를 잘라냅니다.
    """
    if not text:
        return text
    
    # 1. 앞쪽의 잘린 단어 날리기 (첫 번째 공백 이전 텍스트 제거)
    if " " in text:
        text = text.split(" ", 1)[1]
        
    # 2. 뒤쪽의 잘린 단어 날리기 (마지막 공백 이후 텍스트 제거)
    if " " in text:
        text = text.rsplit(" ", 1)[0]
        
    return text.strip()



def _merge_with_overlap(left: str, right: str, max_overlap: int = 2000) -> str:
    """
    strip()을 제거하여 원본 공백을 유지한 채로 오버랩을 찾습니다.
    """
    l = left or ""
    r = right or ""
    
    if not l.strip():
        return r
    if not r.strip():
        return l

    max_k = min(len(l), len(r), max_overlap)
    overlap_k = 0
    
    # 뒤에서부터 매칭 검사
    for k in range(max_k, 0, -1):
        if l[-k:] == r[:k]:
            overlap_k = k
            break
            
    if overlap_k > 0:
        return f"{l}{r[overlap_k:]}"
    
    # 오버랩을 못 찾았을 경우 안전하게 공백 하나만 두고 병합
    return f"{l.rstrip()} {r.lstrip()}"


def _build_doc_chunk_text_index(metadata_sram: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[int, str]]:
    """metadata_sram -> doc_id별 chunk_idx:text 인덱스."""
    out: Dict[str, Dict[int, str]] = {}
    if not isinstance(metadata_sram, dict):
        return out
    for _, meta in metadata_sram.items():
        if not isinstance(meta, dict):
            continue
        doc_id = meta.get("doc_id")
        chunk_idx = meta.get("chunk_idx")
        text = meta.get("text")
        if doc_id is None or chunk_idx is None or not isinstance(text, str):
            continue
        try:
            ci = int(chunk_idx)
        except Exception:
            continue
        ds = str(doc_id).strip()
        if not ds:
            continue
        out.setdefault(ds, {})[ci] = text
    return out


def _stitch_context_for_chunk(
    doc: Dict[str, Any],
    doc_chunk_text_index: Dict[str, Dict[int, str]],
    window_size: int = 1,
    debug_cb: Optional[Any] = None,
) -> str:
    """
    현재 chunk를 중심으로 같은 doc 내 인접 청크를 stitch해서 반환.
    앞쪽은 ci-1 … ci-window_size(역순), 뒤쪽은 ci+1 … ci+window_size(순서)로 최대 window_size개씩,
    인덱스가 없으면 해당 방향 루프를 중단. overlap 중복은 _merge_with_overlap으로 정리.
    """
    meta = (doc.get("metadata") or {}) if isinstance(doc, dict) else {}
    doc_id_raw = meta.get("doc_id")
    chunk_idx_raw = meta.get("chunk_idx")
    current_text = meta.get("text") or doc.get("text", "") or ""
    if doc_id_raw is None or chunk_idx_raw is None:
        return str(current_text)
    try:
        ci = int(chunk_idx_raw)
    except Exception:
        return str(current_text)
    ds = str(doc_id_raw).strip()
    if not ds:
        return str(current_text)

    per_doc = doc_chunk_text_index.get(ds, {})
    if not per_doc:
        return str(current_text)

    cur_text = per_doc.get(ci, str(current_text))

    stitched = str(cur_text)
    used_neighbors: List[int] = []

    # 앞쪽 문맥: ci-1부터 ci-window_size까지 역순으로 stitch
    for off in range(1, max(0, int(window_size)) + 1):
        ni = ci - off
        if ni not in per_doc:
            break
        prev_text = per_doc.get(ni)
        if isinstance(prev_text, str) and prev_text:
            stitched = _merge_with_overlap(prev_text, stitched)
            used_neighbors.append(ni)

    # 뒤쪽 문맥: ci+1부터 ci+window_size까지 순서대로 stitch
    for off in range(1, max(0, int(window_size)) + 1):
        ni = ci + off
        if ni not in per_doc:
            break
        next_text = per_doc.get(ni)
        if isinstance(next_text, str) and next_text:
            stitched = _merge_with_overlap(stitched, next_text)
            used_neighbors.append(ni)

    if used_neighbors and callable(debug_cb):
        nb = " and ".join(str(x) for x in used_neighbors)
        debug_cb(f"Stitched chunk {ci} with {nb} for {ds}")
    return stitched


def _redundancy_rate_from_embeddings(
    top_k_results: List[Dict],
    top_k: int,
    sim_threshold: float = 0.85,
) -> float:
    """
    RR(Redundancy Rate):
    RR = (# pairs with cosine similarity > sim_threshold) / (total # valid pairs).
    """
    k_take = min(int(top_k), len(top_k_results))
    ranked = top_k_results[:k_take]
    embs: List[np.ndarray] = []
    for doc in ranked:
        emb = doc.get("embedding")
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float32).flatten()
        if arr.size == 0:
            continue
        embs.append(arr)

    n = len(embs)
    if n < 2:
        return 0.0

    redundant_pairs = 0
    valid_pairs = 0
    for i in range(n):
        ni = float(np.linalg.norm(embs[i]))
        if not np.isfinite(ni) or ni <= 1e-9:
            continue
        for j in range(i + 1, n):
            nj = float(np.linalg.norm(embs[j]))
            if not np.isfinite(nj) or nj <= 1e-9:
                continue
            cos = float(np.dot(embs[i], embs[j]) / (ni * nj))
            if not np.isfinite(cos):
                continue
            valid_pairs += 1
            if cos > sim_threshold:
                redundant_pairs += 1
    if valid_pairs == 0:
        return 0.0
    return redundant_pairs / float(valid_pairs)


def run_mmr_evaluation(
    domain: str,
    offline_data_dir: Optional[str] = None,
    num_queries: int = 1,
    max_contexts: Optional[int] = None,
    top_k: int = 5,
    nprobe: int = 8,
    top_r: int = 128,
    tau_dup: float = 0.5,
    mmr_lambda: float = 0.85,
    gamma: float = 0.7,
    verbose: bool = False,
    show_docs: bool = False,
    export_json: Optional[str] = None,
    target_doc_id: Optional[str] = None,
    window_size: int = 1,
    retrieval_mode: str = "COMPARE",
    anchor_lambda: float = 1.0,
    anchor_tau_dup: float = 0.0,
) -> Dict[str, Any]:
    """
    데이터셋에서 질문 N개(num_queries)를 샘플링해 OnlinePipeline으로 검색.
    평가 목표: Recall@K(Strict coverage, 임계값 GT_COVERAGE_THRESHOLD), ILD, RR(Redundancy Rate), Search Breadth.
    검색은 항상 Step 4의 전체 후보를 기반으로 동작하며,
    target_doc_id(또는 ref_doc_id)는 하이브리드(Local/Global)에서 Local 그룹 구분에 사용됩니다.
    retrieval_mode=COMPARE일 때 B1(단일 L2)·B2(순차 다중 쿼리 순수 L2)·PROPOSED(MMR) 3-way 비교 및 export.
    그 외 단일 모드에서는 해당 모드만 집계한다.
    eval_pairs는 (question, ref_doc_id, answers_text, query_concat, query_aspects) 구조로,
    ref_doc_id는 문서 인덱스, answers_text는 실제 정답 구절(없으면 빈 문자열)입니다.
    window_size: export용 문맥 스티칭 시 동일 doc 인접 청크 반경(_stitch_context_for_chunk).
    anchor_lambda / anchor_tau_dup: PROPOSED 모드 1라운드(Anchor) step5에만 전달되는 오버라이드.
    """
    debug_log_path = "pipeline_debug_log.txt"
    with open(debug_log_path, "w", encoding="utf-8") as f:
        f.write("========== H-AIMDC Pipeline Debug Log ==========\n")
    def _append_eval_debug(line: str) -> None:
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    contexts, eval_pairs = load_dataset_for_domain(domain, max_contexts=max_contexts)
    if not eval_pairs:
        raise RuntimeError(f"도메인 {domain}에서 평가용 (question, ref_doc_id) 쌍이 없습니다.")
    if not contexts:
        raise RuntimeError(f"도메인 {domain}에서 contexts 리스트가 비어 있습니다. 데이터셋 로드를 확인하세요.")

    if len(eval_pairs) > num_queries:
        import random
        random.seed(42)
        eval_pairs = random.sample(eval_pairs, num_queries)

    if offline_data_dir is None:
        offline_data_dir = os.path.join(".", f"offline_data_{domain}")
    if not os.path.isdir(offline_data_dir):
        raise FileNotFoundError(
            f"오프라인 인덱스가 없습니다: {offline_data_dir}. 먼저 run_offline_indexing_for_domain('{domain}')을 실행하세요."
        )

    hw_state = load_hw_state_from_dir(offline_data_dir)
    pipeline = L2RAGOnlinePipeline(
        hw_state,
        domain_mode=domain,
        nprobe=nprobe,
        top_r=top_r,
        top_k=top_k,
        tau_dup=tau_dup,
        mmr_lambda=mmr_lambda,
        gamma=gamma,
        anchor_lambda=anchor_lambda,
        anchor_tau_dup=anchor_tau_dup,
        verbose=verbose,
    )
    doc_chunk_text_index = _build_doc_chunk_text_index(pipeline.metadata_sram or {})
    
    metrics_l2: List[Dict[str, Any]] = []  # B1(원본 단일 쿼리 L2) 집계
    metrics_b2: List[Dict[str, Any]] = []  # B2(순차 다중 쿼리 순수 L2) 집계
    metrics_mmr: List[Dict[str, Any]] = []  # PROPOSED(MMR) 집계
    export_data: List[Dict[str, Any]] = []

    mode_upper = (retrieval_mode or "PROPOSED").upper()

    def _format_for_llm(doc: Dict, stitched_text: str, current_target_id: Optional[str]) -> str:
        meta = doc.get("metadata", {})
        doc_id = str(meta.get("doc_id", "")).strip()
        section = str(meta.get("section_heading", "Unknown Section")).strip()

        if current_target_id and doc_id == str(current_target_id).strip():
            source_type = "TARGET CONTRACT (본 계약서)"
        else:
            source_type = f"REFERENCE CONTRACT (참고용 외부 계약서 ID: {doc_id})"

        header = f"--- [Source: {source_type} | Section: {section}] ---\n"
        return header + stitched_text

    for idx, (question, ref_doc_id, answers_text, query_concat, query_aspects) in enumerate(eval_pairs):
        current_target = (
            str(ref_doc_id) if ref_doc_id is not None else target_doc_id
        )
        target_id_for_tag = str(current_target).strip() if current_target else None
        if not target_id_for_tag:
            target_id_for_tag = None

        # 단일 모드 실행: B1/B2/PROPOSED
        if mode_upper in ("B1", "B2", "PROPOSED"):
            routed_results = pipeline.run_pipeline(
                question,
                target_doc_id=current_target,
                log_summary=False,
                mode=mode_upper,
                query_concat=None if mode_upper == "B2" else query_concat,
                query_aspects=query_aspects,
            )
            routed_summary = pipeline._last_summary or {}
            k_take = min(int(top_k), len(routed_results))
            ws = max(0, int(window_size))
            routed_stitched: List[str] = []
            routed_stitched_for_recall: List[str] = []
            for doc in routed_results[:k_take]:
                raw = _trim_broken_edges(
                    _stitch_context_for_chunk(
                        doc,
                        doc_chunk_text_index,
                        window_size=ws,
                        debug_cb=(lambda msg: _append_eval_debug(msg)),
                    )
                )
                routed_stitched_for_recall.append(raw)
                routed_stitched.append(_format_for_llm(doc, raw, target_id_for_tag))
            expected_text = ""
            if isinstance(answers_text, str) and answers_text.strip():
                expected_text = answers_text.strip()
            elif isinstance(ref_doc_id, int) and 0 <= ref_doc_id < len(contexts):
                expected_text = contexts[ref_doc_id] or ""
            recall_routed = 0.0
            for t in routed_stitched_for_recall:
                if _ground_truth_token_coverage(expected_text, t) >= GT_COVERAGE_THRESHOLD:
                    recall_routed = 1.0
                    break
            rr_routed = _redundancy_rate_from_embeddings(routed_results, top_k, sim_threshold=0.7)
            routed_summary = {
                **routed_summary,
                "recall_at_k_50": recall_routed,
                "redundancy_rate": rr_routed,
            }
            if mode_upper == "PROPOSED":
                metrics_mmr.append(routed_summary)
            elif mode_upper == "B2":
                metrics_b2.append(routed_summary)
            else:
                metrics_l2.append(routed_summary)
            export_data.append({
                "query": question,
                "ground_truth": expected_text,
                "mode": mode_upper,
                "contexts": routed_stitched,
            })
            if verbose:
                routed_docs = routed_summary.get("unique_doc_ids")
                routed_ild = routed_summary.get("intra_list_distance") or 0.0
                routed_rr = routed_summary.get("redundancy_rate") or 0.0
                routed_recall = routed_summary.get("recall_at_k_50") or 0.0
                print(
                    f"[Query {idx+1}/{len(eval_pairs)}] Mode={mode_upper} | "
                    f"Unique_Docs:{routed_docs} | Recall@K50:{routed_recall:.2f} | "
                    f"ILD:{routed_ild:.4f} | RR:{routed_rr:.4f}"
                )
            if show_docs:
                print(f"\n[질문] {question}")
                print(f"\n  === {mode_upper} Top-5 ===")
                for rank, doc in enumerate(routed_results[:5], start=1):
                    doc_id = doc.get("chunk_id") or (doc.get("metadata") or {}).get("doc_id") or "—"
                    text = (doc.get("metadata") or {}).get("text") or ""
                    text_flat = (text[:300] + "..." if len(text) > 300 else text).replace("\n", " ")
                    print(f"  [{rank}] doc_id={doc_id} | {text_flat}")
                print("-" * 80)
            continue

        # COMPARE: B1(단일 L2) + B2(순차 다중 순수 L2) + PROPOSED(MMR) 3-way
        b1_results = pipeline.run_pipeline(
            question,
            target_doc_id=current_target,
            log_summary=False,
            mode="B1",
        )
        b1_summary = dict(pipeline._last_summary or {})

        b2_results = pipeline.run_pipeline(
            question,
            target_doc_id=current_target,
            log_summary=False,
            mode="B2",
            query_aspects=query_aspects,
        )
        b2_summary = dict(pipeline._last_summary or {})

        mmr_results = pipeline.run_pipeline(
            question,
            target_doc_id=current_target,
            log_summary=False,
            mode="PROPOSED",
            query_aspects=query_aspects,
        )
        mmr_summary = dict(pipeline._last_summary or {})

        expected_text = ""
        if isinstance(answers_text, str) and answers_text.strip():
            expected_text = answers_text.strip()
        elif isinstance(ref_doc_id, int) and 0 <= ref_doc_id < len(contexts):
            expected_text = contexts[ref_doc_id] or ""

        ws = max(0, int(window_size))

        def _stitch_list_tagged(docs: List[Dict]) -> Tuple[List[str], List[str]]:
            k = min(int(top_k), len(docs))
            raw_list: List[str] = []
            tagged_list: List[str] = []
            for doc in docs[:k]:
                raw = _trim_broken_edges(
                    _stitch_context_for_chunk(
                        doc,
                        doc_chunk_text_index,
                        window_size=ws,
                        debug_cb=(lambda msg: _append_eval_debug(msg)),
                    )
                )
                raw_list.append(raw)
                tagged_list.append(_format_for_llm(doc, raw, target_id_for_tag))
            return raw_list, tagged_list

        b1_raw, b1_stitched = _stitch_list_tagged(b1_results)
        b2_raw, b2_stitched = _stitch_list_tagged(b2_results)
        mmr_raw, mmr_stitched = _stitch_list_tagged(mmr_results)

        def _recall_for(raw_blocks: List[str]) -> float:
            for t in raw_blocks:
                if _ground_truth_token_coverage(expected_text, t) >= GT_COVERAGE_THRESHOLD:
                    return 1.0
            return 0.0

        recall_b1 = _recall_for(b1_raw)
        recall_b2 = _recall_for(b2_raw)
        recall_mmr = _recall_for(mmr_raw)

        rr_b1 = _redundancy_rate_from_embeddings(b1_results, top_k, sim_threshold=0.7)
        rr_b2 = _redundancy_rate_from_embeddings(b2_results, top_k, sim_threshold=0.7)
        rr_mmr = _redundancy_rate_from_embeddings(mmr_results, top_k, sim_threshold=0.7)

        b1_summary = {**b1_summary, "recall_at_k_50": recall_b1, "redundancy_rate": rr_b1}
        b2_summary = {**b2_summary, "recall_at_k_50": recall_b2, "redundancy_rate": rr_b2}
        mmr_summary = {**mmr_summary, "recall_at_k_50": recall_mmr, "redundancy_rate": rr_mmr}

        metrics_l2.append(b1_summary)
        metrics_b2.append(b2_summary)
        metrics_mmr.append(mmr_summary)

        def _doc_text(doc: Dict) -> str:
            return (doc.get("metadata") or {}).get("text") or doc.get("text", "") or ""

        export_data.append({
            "query": question,
            "ground_truth": expected_text,
            "b1_contexts": b1_stitched,
            "b2_contexts": b2_stitched,
            "mmr_contexts": mmr_stitched,
        })
        if verbose:
            print(
                f"[Query {idx+1}/{len(eval_pairs)}] COMPARE | "
                f"Unique_Docs(B1:{b1_summary.get('unique_doc_ids')}|B2:{b2_summary.get('unique_doc_ids')}|"
                f"MMR:{mmr_summary.get('unique_doc_ids')}) | "
                f"Recall@K50(B1:{recall_b1:.2f}|B2:{recall_b2:.2f}|MMR:{recall_mmr:.2f}) | "
                f"ILD(B1:{b1_summary.get('intra_list_distance') or 0:.4f}|"
                f"B2:{b2_summary.get('intra_list_distance') or 0:.4f}|"
                f"MMR:{mmr_summary.get('intra_list_distance') or 0:.4f}) | "
                f"RR(B1:{rr_b1:.4f}|B2:{rr_b2:.4f}|MMR:{rr_mmr:.4f})"
            )
            for tag, res in (("B1", b1_results), ("B2", b2_results), ("MMR", mmr_results)):
                top_n = min(10, len(res))
                avg_c = (
                    sum(len(_doc_text(doc)) for doc in res[:top_n]) / float(top_n) if top_n else 0.0
                )
                print(f"  [Debug] AvgTop10Chars ({tag}:{avg_c:.1f} chars)")
        if show_docs:
            print(f"\n[질문] {question}")
            for tag, res in (("B1", b1_results), ("B2", b2_results), ("MMR", mmr_results)):
                print(f"\n  === {tag} Top-5 ===")
                for rank, doc in enumerate(res[:5], start=1):
                    doc_id = doc.get("chunk_id") or (doc.get("metadata") or {}).get("doc_id") or "—"
                    text = (doc.get("metadata") or {}).get("text") or ""
                    text_flat = (text[:300] + "..." if len(text) > 300 else text).replace("\n", " ")
                    print(f"  [{rank}] doc_id={doc_id} | {text_flat}")
            print("-" * 80)

    if export_json:
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"[Export] 검색 결과 저장: {export_json} ({len(export_data)} 쿼리)")

    n = len(eval_pairs)

    def _avg_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        out: Dict[str, float] = {}
        for k in [
            "unique_centroids_selected",
            "unique_centroids_ratio",
            "unique_doc_ids",
            "early_dedup_count",
            "intra_list_distance",
            "recall_at_k_50",
            "redundancy_rate",
        ]:
            vals = [m.get(k) for m in metrics_list if m.get(k) is not None]
            if vals:
                out[k] = sum(vals) / len(vals)
        p25_vals = [m.get("p25_min_dist_in_query") for m in metrics_list if m.get("p25_min_dist_in_query") is not None]
        p25_vals = [v for v in p25_vals if np.isfinite(v)]
        if p25_vals:
            out["p25_min_dist_in_query_avg"] = sum(p25_vals) / len(p25_vals)
        return out

    avg_l2 = _avg_metrics(metrics_l2)
    avg_b2 = _avg_metrics(metrics_b2)
    avg_mmr = _avg_metrics(metrics_mmr)

    report = {
        "domain": domain,
        "num_queries": n,
        "top_k": int(top_k),
        "window_size": int(window_size),
        "gamma": float(gamma),
        "avg_metrics_l2": avg_l2,
        "avg_metrics_b2": avg_b2,
        "avg_metrics_mmr": avg_mmr,
        # COMPARE: {query, ground_truth, b1_contexts, b2_contexts, mmr_contexts}
        "retrieved_context": export_data,
    }

    if verbose:
        print()
        print("========== H-AIMDC 다양성 및 효율성 평가 리포트 ==========")
        print(
            f"Domain: {domain} | Queries: {n} | Top-K: {top_k} | window_size: {window_size} | gamma: {gamma} | "
            f"GT coverage: Strict (<=3 tokens: 100%) | "
            f">=4 tokens: token ratio, pass if >= {GT_COVERAGE_THRESHOLD:.0%}"
        )
        print("  Recall@K (Strict Match):")
        print(
            f"    B1: {avg_l2.get('recall_at_k_50', 0):.4f}  |  B2: {avg_b2.get('recall_at_k_50', 0):.4f}  |  "
            f"MMR: {avg_mmr.get('recall_at_k_50', 0):.4f}"
        )
        print("  ILD (Intra-List Distance):")
        print(
            f"    B1: {avg_l2.get('intra_list_distance', 0):.4f}  |  B2: {avg_b2.get('intra_list_distance', 0):.4f}  |  "
            f"MMR: {avg_mmr.get('intra_list_distance', 0):.4f}"
        )
        print("  RR (Redundancy Rate, cosine > 0.7):")
        print(
            f"    B1: {avg_l2.get('redundancy_rate', 0):.4f}  |  B2: {avg_b2.get('redundancy_rate', 0):.4f}  |  "
            f"MMR: {avg_mmr.get('redundancy_rate', 0):.4f}"
        )
        print("  Search Breadth (탐색한 서랍 수):")
        b1_breadth = avg_l2.get("unique_centroids_selected", 0)
        b2_breadth = avg_b2.get("unique_centroids_selected", 0)
        mmr_breadth = avg_mmr.get("unique_centroids_selected", 0)
        print(
            f"    B1 평균: {b1_breadth:.2f}  |  B2 평균: {b2_breadth:.2f}  |  MMR 평균: {mmr_breadth:.2f}"
        )
        if mmr_breadth > b1_breadth:
            print(
                f"  Enhanced Exploratory Power: MMR breadth is higher than B1 ({mmr_breadth:.2f} > {b1_breadth:.2f})"
            )
        b1_rr_avg = avg_l2.get("redundancy_rate", 0)
        mmr_rr_avg = avg_mmr.get("redundancy_rate", 0)
        if mmr_rr_avg < b1_rr_avg:
            print(
                "  Efficiency Gain: MMR의 RR이 B1보다 낮아 LLM 입력 문맥의 정보 밀도가 높아질 수 있습니다 "
                f"({mmr_rr_avg:.4f} < {b1_rr_avg:.4f})."
            )
        else:
            print(
                "  Efficiency Note: RR 개선이 제한적입니다. lower RR일수록 LLM 정보 밀도가 높아집니다."
            )
        print("==================================================================")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline indexing + Online MMR evaluation (CUAD)")
    parser.add_argument("--domain", type=str, default="cuad", choices=["cuad"])
    parser.add_argument("--mode", type=str, default="eval", choices=["index", "eval", "both"])
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--max_docs", type=int, default=2000, help="오프라인 인덱싱 시 최대 문서 수")
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=32, help="탐색할 서랍(Centroid)의 개수")
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help=(
            "온라인 검색·재순위화 후 최종적으로 반환할 패시지(청크) 개수 Top-K. "
            "MMR/L2 버퍼 크기 및 export/recall 집계에 사용됩니다."
        ),
    )
    parser.add_argument("--top_r", type=int, default=128, help="Number of chunks to load per centroid")
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        dest="window_size",
        help=(
            "평가·export 시 동일 문서 내 인접 청크를 이어붙이는(stitch) 반경. "
            "중심 청크 기준 앞·뒤로 최대 이 개수만큼 이웃 chunk_idx 텍스트를 슬라이딩 윈도우처럼 포함합니다."
        ),
    )
    parser.add_argument("--tau_dup", type=float, default=0.5, help="MMR 중복 제거 L2 거리 임계값 (작을수록 덜 지움)")
    parser.add_argument(
        "--anchor_lambda",
        type=float,
        default=1.0,
        help="1라운드 Anchor 선발 시 강제할 MMR lambda 값 (기본 1.0: 순수 L2)",
    )
    parser.add_argument(
        "--anchor_tau_dup",
        type=float,
        default=0.0,
        help="1라운드 Anchor 선발 시 강제할 tau_dup 값 (기본 0.0: 조기탈락 없음)",
    )
    parser.add_argument("--show_docs", action="store_true", help="검색된 Top-K 문서의 실제 텍스트 내용을 터미널에 출력")
    parser.add_argument("--export_json", type=str, default=None, help="검색 결과를 JSON 파일로 저장")
    parser.add_argument(
        "--target_doc_id",
        type=str,
        default=None,
        help="지정 시 하이브리드에서 Local 그룹 기준 doc_id로 사용(없으면 Local/Global 분리 없이 전체 후보로 MMR)",
    )
    parser.add_argument(
        "--mmr_lambda",
        type=float,
        default=0.5,
        help="MMR 다양성 파라미터 (1.0에 가까울수록 유사도 중시, 작을수록 다양성 중시)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Hyperparameter to control the balance/trade-off between inter-doc and intra-doc (intra-doc quota ≈ gamma * top_k)",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="COMPARE",
        choices=["COMPARE", "B1", "B2", "PROPOSED"],
        help="검색 라우팅 모드: COMPARE(B1·B2·MMR 3-way), B1(단일 L2), B2(순차 다중 순수 L2), PROPOSED(MMR)",
    )
    args = parser.parse_args()

    if args.mode in ("index", "both"):
        print(f"[Offline] 도메인 '{args.domain}' 인덱싱 시작 (max_docs={args.max_docs})...")
        save_dir = run_offline_indexing_for_domain(
            args.domain,
            output_base_dir=args.output_dir,
            max_docs=args.max_docs,
        )
        print(f"[Offline] 저장 완료: {save_dir}")

    if args.mode in ("eval", "both"):
        print(
            f"[Online] 도메인 '{args.domain}' 검색 평가 "
            f"(retrieval_mode={args.retrieval_mode}, num_queries={args.num_queries}, top_k={args.top_k}, window_size={args.window_size})..."
        )
        run_mmr_evaluation(
            args.domain,
            offline_data_dir=os.path.join(args.output_dir, f"offline_data_{args.domain}"),
            num_queries=args.num_queries,
            max_contexts=args.max_docs,
            top_k=args.top_k,
            nprobe=args.nprobe,
            top_r=args.top_r,
            tau_dup=args.tau_dup,
            mmr_lambda=args.mmr_lambda,
            gamma=args.gamma,
            window_size=args.window_size,
            verbose=True,
            show_docs=args.show_docs,
            export_json=args.export_json,
            #export_json=None,
            target_doc_id=args.target_doc_id,
            retrieval_mode=args.retrieval_mode,
            anchor_lambda=args.anchor_lambda,
            anchor_tau_dup=args.anchor_tau_dup,
        )