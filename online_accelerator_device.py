from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING, Literal, Mapping, Union, cast
from dataclasses import replace

if TYPE_CHECKING:
    from aimdc_device_distance import DeviceDistance
    from aimdc_nvm_device_distance import NvmDeviceDistance
from collections import Counter
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
        gamma: float = 0.25,
        section_penalty_weight: float = 1.5,
        log_facets: bool = True,
        verbose: bool = True,
        dump_embeddings: bool = False,
        dump_dir: str = "./emb_dump",
        dump_tag: str = "q0",
        min_mmr_score: Optional[float] = None,
        max_requery_rounds: int = 3,
        use_centroid_filter: bool = True,
        device_distance: Optional["DeviceDistance"] = None,
        centroid_device_distance: Optional["NvmDeviceDistance"] = None,
        device_scale_k: Optional[float] = None,
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
        # Fraction of buffer_top_k reserved as pure-L2 anchors before MMR competes for the rest.
        self.gamma = float(max(0.0, min(1.0, gamma)))
        # 동일 section_heading 후보는 더 가깝게(중복으로) 보이도록 거리 축소 가중치
        self.section_penalty_weight = max(float(section_penalty_weight), 1e-9)

        # Coverage early-stop / re-query (only used when min_mmr_score is set; None preserves legacy behavior)
        self.min_mmr_score = min_mmr_score
        self.max_requery_rounds = int(max(0, max_requery_rounds))
        self.use_centroid_filter = bool(use_centroid_filter)

        self.device_distance: Optional["DeviceDistance"] = device_distance
        self.centroid_device_distance: Optional["NvmDeviceDistance"] = centroid_device_distance
        self._mmr_distance_backend: Literal["software", "analytical"] = (
            "analytical" if device_distance is not None else "software"
        )
        self.device_scale_k: Optional[float] = None
        if device_scale_k is not None:
            fk = float(device_scale_k)
            if np.isfinite(fk) and fk > 0.0:
                self.device_scale_k = fk

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
        self.raw_embeddings: Optional[np.ndarray] = hw_ready_state.get("raw_embeddings")
        # Full hw dict reference (raw row mapping, diagnostics)
        self._hw: Dict[str, Any] = hw_ready_state
        self.chunk_ids_order: Optional[List[str]] = hw_ready_state.get("chunk_ids_order")
        self.raw_emb_by_chunk_id: Dict[str, np.ndarray] = {}
        if self.raw_embeddings is not None and self.chunk_ids_order is not None:
            n = min(len(self.chunk_ids_order), len(self.raw_embeddings))
            for i in range(n):
                cid = str(self.chunk_ids_order[i])
                self.raw_emb_by_chunk_id[cid] = np.asarray(
                    self.raw_embeddings[i], dtype=np.float32
                )

        # 검색 품질 분석용 (마지막 쿼리 결과)
        self._last_summary: Optional[Dict[str, Any]] = None
        self._last_latency_ms: Optional[Dict[str, float]] = None
        # Last MMR centroid-filter vs full-doc chunk coverage (set in run_pipeline when verbose)
        self._last_centroid_filter_stats: Optional[Dict[str, Any]] = None

        # Optional embedding dump buffers for device/circuit modeling analysis.
        self._dump_enabled = bool(dump_embeddings)
        self._dump_dir = str(dump_dir)
        self._dump_tag = str(dump_tag)
        self._dump_records: Dict[str, Any] = {
            "query_emb_pre_opq": None,
            "query_emb_post_opq": None,
            "candidates": [],
            "mmr_steps": [],
        }
        if self._dump_enabled:
            os.makedirs(self._dump_dir, exist_ok=True)

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

    def _mmr_noise_enabled(self) -> bool:
        dd = self.device_distance
        return bool(dd is not None and dd.noise.enabled())

    def _dump_reset(self) -> None:
        self._dump_records = {
            "query_emb_pre_opq": None,
            "query_emb_post_opq": None,
            "candidates": [],
            "mmr_steps": [],
        }

    @staticmethod
    def _safe_stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
        a = np.asarray(arr)
        if a.size == 0:
            return {"min": None, "max": None, "mean": None, "std": None, "norm": None}
        aa = a.astype(np.float64, copy=False).ravel()
        return {
            "min": float(np.min(aa)),
            "max": float(np.max(aa)),
            "mean": float(np.mean(aa)),
            "std": float(np.std(aa)),
            "norm": float(np.linalg.norm(aa)),
        }

    @staticmethod
    def _safe_percentiles(arr: np.ndarray) -> Dict[str, Optional[float]]:
        a = np.asarray(arr)
        if a.size == 0:
            return {"p5": None, "p50": None, "p95": None, "min": None, "max": None}
        aa = a.astype(np.float64, copy=False).ravel()
        return {
            "p5": float(np.percentile(aa, 5)),
            "p50": float(np.percentile(aa, 50)),
            "p95": float(np.percentile(aa, 95)),
            "min": float(np.min(aa)),
            "max": float(np.max(aa)),
        }

    def _flush_dump(self) -> Optional[str]:
        if not self._dump_enabled:
            return None
        rec = self._dump_records or {}
        q_pre = rec.get("query_emb_pre_opq")
        q_post = rec.get("query_emb_post_opq")
        cands = rec.get("candidates") or []
        mmr_steps = rec.get("mmr_steps") or []
        if q_pre is None and q_post is None and not cands and not mmr_steps:
            return None

        os.makedirs(self._dump_dir, exist_ok=True)
        out_npz = os.path.join(self._dump_dir, f"{self._dump_tag}_emb_dump.npz")
        out_summary = os.path.join(self._dump_dir, f"{self._dump_tag}_emb_dump_summary.json")

        cand_embs = np.asarray(
            [np.asarray(c.get("cand_emb"), dtype=np.float32).copy() for c in cands],
            dtype=np.float32,
        ) if cands else np.zeros((0, max(int(self.embedding_dim), 1)), dtype=np.float32)
        cand_chunk_ids = np.asarray([str(c.get("chunk_id", "")) for c in cands], dtype=object)
        cand_centroid_ids = np.asarray(
            [int(c.get("centroid_id", -1)) for c in cands], dtype=np.int64
        ) if cands else np.asarray([], dtype=np.int64)
        cand_dists_q = np.asarray(
            [float(c.get("dist_q_c", 0.0)) for c in cands], dtype=np.float32
        ) if cands else np.asarray([], dtype=np.float32)

        mmr_step_cand_embs = np.asarray(
            [np.asarray(s.get("cand_emb"), dtype=np.float32).copy() for s in mmr_steps],
            dtype=object,
        )
        mmr_step_selected_embs = np.asarray(
            [np.asarray(s.get("anchor_and_mmr_embs"), dtype=np.float32).copy() for s in mmr_steps],
            dtype=object,
        )
        mmr_step_pair_dists = np.asarray(
            [np.asarray(s.get("per_pair_dists"), dtype=np.float32).copy() for s in mmr_steps],
            dtype=object,
        )
        mmr_step_min_dists = np.asarray(
            [float(s.get("min_dist_c_s_raw", np.nan)) for s in mmr_steps], dtype=np.float32
        )
        mmr_step_min_dists_used = np.asarray(
            [float(s.get("min_dist_c_s_used", np.nan)) for s in mmr_steps], dtype=np.float32
        )
        mmr_step_skipped_dedup = np.asarray(
            [bool(s.get("skipped_dedup", False)) for s in mmr_steps], dtype=bool
        )
        mmr_step_selected = np.asarray(
            [bool(s.get("selected", False)) for s in mmr_steps], dtype=bool
        )
        mmr_step_method = np.asarray(
            [str(s.get("method", "")) for s in mmr_steps], dtype=object
        )
        mmr_step_chunk_ids = np.asarray(
            [str(s.get("cand_chunk_id", "")) for s in mmr_steps], dtype=object
        )
        mmr_step_idx = np.asarray(
            [int(s.get("step_idx", -1)) for s in mmr_steps], dtype=np.int64
        )
        mmr_step_mmr_score = np.asarray(
            [
                (
                    np.nan
                    if s.get("mmr_score", np.nan) is None
                    else float(s.get("mmr_score", np.nan))
                )
                for s in mmr_steps
            ],
            dtype=np.float32,
        )
        mmr_step_dist_q = np.asarray(
            [float(s.get("dist_q_c_raw", np.nan)) for s in mmr_steps], dtype=np.float32
        )
        mmr_step_inserted = np.asarray(
            [bool(s.get("sticky_inserted", False)) for s in mmr_steps], dtype=bool
        )
        mmr_step_insert_event = np.asarray(
            [str(s.get("sticky_event", "")) for s in mmr_steps], dtype=object
        )

        np.savez(
            out_npz,
            query_emb_pre_opq=np.asarray(q_pre, dtype=np.float32).copy() if q_pre is not None else np.asarray([], dtype=np.float32),
            query_emb_post_opq=np.asarray(q_post, dtype=np.float32).copy() if q_post is not None else np.asarray([], dtype=np.float32),
            cand_embs=cand_embs,
            cand_chunk_ids=cand_chunk_ids,
            cand_centroid_ids=cand_centroid_ids,
            cand_dists_q=cand_dists_q,
            mmr_step_cand_embs=mmr_step_cand_embs,
            mmr_step_selected_embs=mmr_step_selected_embs,
            mmr_step_pair_dists=mmr_step_pair_dists,
            mmr_step_min_dists=mmr_step_min_dists,
            mmr_step_min_dists_used=mmr_step_min_dists_used,
            mmr_step_skipped_dedup=mmr_step_skipped_dedup,
            mmr_step_selected=mmr_step_selected,
            mmr_step_method=mmr_step_method,
            mmr_step_chunk_ids=mmr_step_chunk_ids,
            mmr_step_idx=mmr_step_idx,
            mmr_step_mmr_score=mmr_step_mmr_score,
            mmr_step_dist_q=mmr_step_dist_q,
            mmr_step_inserted=mmr_step_inserted,
            mmr_step_insert_event=mmr_step_insert_event,
        )

        pair_flat = np.concatenate(
            [
                np.asarray(v, dtype=np.float32).ravel()
                for v in mmr_step_pair_dists
                if isinstance(v, np.ndarray) and v.size > 0
            ],
            axis=0,
        ) if mmr_step_pair_dists.size > 0 else np.asarray([], dtype=np.float32)
        summary = {
            "dump_tag": self._dump_tag,
            "path_npz": os.path.abspath(out_npz),
            "query_emb_pre_opq_shape": list(np.asarray(q_pre).shape) if q_pre is not None else [0],
            "query_emb_post_opq_shape": list(np.asarray(q_post).shape) if q_post is not None else [0],
            "cand_embs_shape": list(cand_embs.shape),
            "cand_dists_q_shape": list(cand_dists_q.shape),
            "mmr_steps_count": int(len(mmr_steps)),
            "stats": {
                "query_emb_pre_opq": self._safe_stats(np.asarray(q_pre, dtype=np.float32)) if q_pre is not None else self._safe_stats(np.asarray([], dtype=np.float32)),
                "query_emb_post_opq": self._safe_stats(np.asarray(q_post, dtype=np.float32)) if q_post is not None else self._safe_stats(np.asarray([], dtype=np.float32)),
                "cand_dists_q": self._safe_stats(cand_dists_q),
                "mmr_step_min_dists_raw": self._safe_stats(mmr_step_min_dists),
                "mmr_step_pair_dists_raw": self._safe_stats(pair_flat),
            },
            "distance_percentiles": {
                "cand_dists_q": self._safe_percentiles(cand_dists_q),
                "mmr_step_min_dists_raw": self._safe_percentiles(mmr_step_min_dists),
                "mmr_step_pair_dists_raw": self._safe_percentiles(pair_flat),
            },
        }
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._dump_reset()
        return out_npz

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

    def step1_query_in_and_encode(self, user_query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        1. Query In: 사용자 쿼리 수신 및 Query encoder 수행 (시작 시각 t0).
        반환: (OPQ 적용 후 쿼리 임베딩, OPQ 적용 전 원본 임베딩).
        """
        query_emb = self._encode_query_general(user_query)
        # 오프라인에서 저장한 OPQ 회전행렬이 있으면 쿼리에도 동일 회전 적용
        query_emb = np.asarray(query_emb, dtype=np.float32).flatten()
        query_emb_pre_opq = query_emb.copy()
        if self.opq_rotation is not None:
            rot = np.asarray(self.opq_rotation, dtype=np.float32)
            if rot.ndim == 2 and rot.shape[0] == query_emb.shape[0]:
                query_emb = query_emb @ rot
        query_emb_post_opq = np.asarray(query_emb, dtype=np.float32).flatten()
        if self._dump_enabled:
            self._dump_records["query_emb_pre_opq"] = query_emb_pre_opq.copy()
            self._dump_records["query_emb_post_opq"] = query_emb_post_opq.copy()
        return query_emb_post_opq, query_emb_pre_opq

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
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
        """
        CUAD: Keyword-Context Fusion(search_text = keyword + Details).
        반환: (query_emb_post_opq, query_emb_pre_opq, weighting_entities, cuad_keyword 또는 None)
        """
        search_text, cuad_keyword = self._prepare_cuad_search_text(user_query)
        weighting_entities = self.extract_weighting_entities(search_text)
        query_emb_post_opq, query_emb_pre_opq = self.step1_query_in_and_encode(search_text)
        return query_emb_post_opq, query_emb_pre_opq, weighting_entities, cuad_keyword

    def extract_weighting_entities(self, user_query: str) -> List[str]:
        """
        CUAD 전용. NER 미사용 — 기존 파이프라인과 동일하게 가중 엔티티 리스트는 비어 있음.
        """
        min_len = 2
        entities_out: List[str] = []
        return entities_out

    def step3_coarse_search(self, query_emb: np.ndarray, facet: str) -> List[Tuple[int, float]]:
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
        if self.centroid_device_distance is not None:
            # NVM analytical device path: raw log10(I) sum as distance proxy.
            # Ranking-only — no software-L2 scale calibration applied here.
            q = query_emb.astype(np.float64)[None, :]
            c = valid_centroids.astype(np.float64)
            raw_d = self.centroid_device_distance.pair_matrix(q, c)[0]  # (C,)
            distances = raw_d.astype(np.float32)
        else:
            q = query_emb.astype(np.float32)[None, :]  # (1, D)
            x_norm = np.sum(q * q, axis=1, keepdims=True)            # (1, 1)
            c_norm = np.sum(valid_centroids * valid_centroids, axis=1, keepdims=True).T  # (1, C)
            distances = (x_norm + c_norm - 2.0 * (q @ valid_centroids.T))[0]

        nprobe = min(self.nprobe, valid_centroids.shape[0])
        nearest_idx = np.argpartition(distances, nprobe - 1)[:nprobe]
        sorted_order = np.argsort(distances[nearest_idx])
        nearest_idx_sorted = nearest_idx[sorted_order]
        
        # [신규 디버그] 전체 중 최소 거리 및 선택된 nprobe개 거리 확인
        min_dist_val = np.min(distances)
        min_dist_idx = np.argmin(distances)


        original_indices = np.where(non_zero_mask)[0]
        selected = [
            (int(original_indices[i]), float(distances[i]))
            for i in nearest_idx_sorted
        ]
        

        
        return selected

    def step4_candidate_load(
        self,
        target_doc_id: str,
        centroid_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        4. Candidate Load (T_candidates).
        Loads all chunk entries directly belonging to the target_doc_id.
        No global/inter-document posting lists are loaded.
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
        want = str(target_doc_id).strip()
        local_cands: List[Dict[str, Any]] = []
        centroid_ids_set = set(centroid_ids) if centroid_ids is not None else None

        if not want:
            return [] # In intra-doc mode, target_doc_id is mandatory

        for cid, entry in self.doc_to_pq_entries.get(want, []):
            if centroid_ids_set is not None and cid not in centroid_ids_set:
                continue
            ch = entry.get("chunk_id")
            if ch is None or ch in seen_chunk_ids:
                continue
            cand = _entry_to_candidate(int(cid), entry)
            if cand:
                local_cands.append(cand)
                seen_chunk_ids.add(ch)

        return local_cands

    @staticmethod
    def _chunk_key(item: Dict) -> Any:
        return item.get("chunk_id") if item.get("chunk_id") is not None else item.get("vector_id")

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

    @staticmethod
    def _candidate_dedupe_key(c: Dict[str, Any]) -> Any:
        """Stable key for de-duplicating candidates across re-query rounds."""
        cid = c.get("chunk_id")
        if cid is not None:
            return ("chunk_id", str(cid))
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        doc_id = meta.get("doc_id")
        cidx = meta.get("chunk_idx")
        if doc_id is not None and cidx is not None:
            try:
                ci = int(cidx)
            except (TypeError, ValueError):
                ci = cidx
            return ("doc_chunk", str(doc_id), ci)
        emb = np.asarray(c.get("embedding"), dtype=np.float32)
        return ("emb_bytes", hash(emb.tobytes()))

    def step5_on_the_fly_mmr(
        self,
        query_emb: np.ndarray,
        candidates: List[Dict],
        s_buffer: List[Dict],
        buffer_top_k: Optional[int] = None,
        gamma: float = 0.25,
        existing_anchors: Optional[List[Dict]] = None,
        requery_round: int = 0,
        opened_centroid_ids: Optional[List[int]] = None,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        On-the-fly MMR with L2 anchoring.
        1) All candidates are scored by L2 distance to the query and sorted ascending.
        2) The closest ``anchor_count`` chunks (at least 1) are added with pure L2 score (-dist),
           without tau_dup or MMR diversity (anchoring).
        3) Remaining candidates compete for the other buffer slots via tau_dup + MMR + sticky insert,
           using min-distance to both anchors and the MMR sub-buffer.
        buffer_top_k: total buffer size (default self.top_k).
        gamma: fraction of buffer_top_k reserved for L2 anchors (default 0.25).
        existing_anchors: if provided, keep anchors fixed and stream all candidates only through MMR loop.
        """
        buf_k = self.top_k if buffer_top_k is None else buffer_top_k
        if buf_k <= 0:
            return [], {
                "early_dedup_count": 0,
                "min_dists_accepted": [],
                "coverage_met_at_step": None,
            }
        mmr_lambda_eff = float(self.mmr_lambda)
        tau_thresh = float(self.tau_dup)
        gamma_eff = float(max(0.0, min(1.0, gamma)))

        query_emb = np.asarray(query_emb, dtype=np.float32)

        apply_k = self.device_distance is not None and self.device_scale_k is not None
        k_val = float(self.device_scale_k) if apply_k else 1.0

        def _k_dump_fields(min_dist_used: float) -> Dict[str, Any]:
            return {
                "device_scale_k": float(self.device_scale_k) if apply_k else None,
                "min_dist_c_s_scaled": (float(k_val * float(min_dist_used)) if apply_k else None),
            }

        scored: List[Tuple[Dict, float]] = []
        for candidate in candidates:
            cand_emb = np.asarray(candidate["embedding"], dtype=np.float32)
            # d(q, C) from PQ-LUT path (software L2 here): anchor sort + MMR relevance term.
            dist_q_c = float(np.linalg.norm(query_emb - cand_emb))
            if not np.isfinite(dist_q_c):
                dist_q_c = 0.0
            scored.append((candidate, dist_q_c))
            if self._dump_enabled:
                self._dump_records["candidates"].append(
                    {
                        "chunk_id": candidate.get("chunk_id"),
                        "centroid_id": candidate.get("centroid_id"),
                        "cand_emb": cand_emb.copy(),
                        "dist_q_c": float(dist_q_c),
                        "distance_backend": self._mmr_distance_backend,
                        "noise_enabled": self._mmr_noise_enabled(),
                    }
                )
        scored.sort(key=lambda x: x[1])

        early_dedup_count = 0
        skipped_for_dedup: List[Tuple[Dict, float]] = []

        def _section_heading_of(item: Dict[str, Any]) -> str:
            meta = item.get("metadata") if isinstance(item, dict) else None
            if isinstance(meta, dict):
                return str(meta.get("section_heading", "") or "").strip()
            return str(item.get("section_heading", "") or "").strip() if isinstance(item, dict) else ""

        def _pair_distances_single_readout(
            cand_emb: np.ndarray,
            pool: List[Dict[str, Any]],
            cand_heading: str,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
            """Single pair_matrix call. Returns (raw_dists, selected_embs, min_with_penalty)."""
            if not pool:
                return (
                    np.empty((0,), dtype=np.float32),
                    np.empty((0, cand_emb.shape[-1]), dtype=np.float32),
                    float("inf"),
                )
            selected_embs = np.stack(
                [np.asarray(s["embedding"], dtype=np.float32) for s in pool],
                axis=0,
            )
            if self.device_distance is not None:
                raw = self.device_distance.pair_matrix(
                    cand_emb.astype(np.float64)[np.newaxis, :],
                    selected_embs.astype(np.float64),
                )[0].astype(np.float32)
            else:
                raw = np.linalg.norm(
                    selected_embs - cand_emb[None, :], axis=1
                ).astype(np.float32)

            # Section-heading penalty applies ONLY to min, not to raw array (preserves dump semantics).
            min_d = float("inf")
            for k, s in enumerate(pool):
                d = float(raw[k])
                if not np.isfinite(d):
                    continue
                s_heading = _section_heading_of(s)
                if cand_heading and s_heading and cand_heading == s_heading:
                    d = d / self.section_penalty_weight
                if d < min_d:
                    min_d = d
            return raw, selected_embs, min_d

        def _make_entry(
            cand: Dict,
            dist_q_c: float,
            min_dist_c_s: Optional[float],
            mmr_score: float,
            method: str,
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
                "_retrieval_method": method,
            }

        if existing_anchors is None:
            anchor_count = max(1, int(buf_k * gamma_eff))
            anchor_count = min(anchor_count, buf_k, len(scored))
            anchor_entries: List[Dict] = []
            for i in range(anchor_count):
                cand, dist_q_c = scored[i]
                anchor_entries.append(
                    _make_entry(cand, dist_q_c, None, -dist_q_c, method="L2_Anchor")
                )
            mmr_slots = buf_k - anchor_count
            mmr_buffer: List[Dict] = []
            mmr_stream = scored[anchor_count:]
        else:
            anchor_entries = [dict(x) for x in (existing_anchors or [])]
            anchor_count = len(anchor_entries)
            mmr_slots = max(0, buf_k - anchor_count)
            mmr_buffer = [
                dict(x)
                for x in s_buffer
                if str(x.get("_retrieval_method", "")) != "L2_Anchor"
            ]
            mmr_buffer.sort(key=lambda x: float(x.get("score", -float("inf"))), reverse=True)
            if len(mmr_buffer) > mmr_slots:
                mmr_buffer = mmr_buffer[:mmr_slots]
            mmr_stream = scored
        coverage_met_at_step: Optional[int] = None

        if mmr_slots > 0:
            step_idx = 0
            for cand, dist_q_c in mmr_stream:
                step_idx += 1
                cand_heading = _section_heading_of(cand)
                pair_dists_raw, selected_embs, min_dist_c_s = _pair_distances_single_readout(
                    cand_emb=np.asarray(cand["embedding"], dtype=np.float32),
                    pool=(anchor_entries + mmr_buffer),
                    cand_heading=cand_heading,
                )
                min_dist_raw = float(np.min(pair_dists_raw)) if pair_dists_raw.size > 0 else float("inf")
                md_for_tau = float(k_val * float(min_dist_c_s)) if apply_k else float(min_dist_c_s)

                if md_for_tau < tau_thresh:
                    early_dedup_count += 1
                    skipped_for_dedup.append((cand, dist_q_c))
                    if self._dump_enabled:
                        self._dump_records["mmr_steps"].append(
                            {
                                "step_idx": step_idx,
                                "cand_chunk_id": cand.get("chunk_id"),
                                "cand_emb": np.asarray(cand["embedding"], dtype=np.float32).copy(),
                                "anchor_and_mmr_embs": selected_embs.copy(),
                                "per_pair_dists": pair_dists_raw.copy(),
                                "min_dist_c_s_raw": min_dist_raw,
                                "min_dist_c_s_used": float(min_dist_c_s),
                                "dist_q_c_raw": float(dist_q_c),
                                **_k_dump_fields(float(min_dist_c_s)),
                                "mmr_score": None,
                                "selected": False,
                                "skipped_dedup": True,
                                "method": "MMR",
                                "sticky_inserted": False,
                                "sticky_event": "skipped_tau_dup",
                                "requery_round": int(requery_round),
                                "opened_centroid_ids": list(opened_centroid_ids or []),
                                "distance_backend": self._mmr_distance_backend,
                                "noise_enabled": self._mmr_noise_enabled(),
                            }
                        )
                    continue

                md_for_score = float(k_val * float(min_dist_c_s)) if apply_k else float(min_dist_c_s)
                mmr_score = -mmr_lambda_eff * dist_q_c + (1.0 - mmr_lambda_eff) * md_for_score
                if not np.isfinite(mmr_score):
                    mmr_score = -mmr_lambda_eff * dist_q_c

                entry = _make_entry(
                    cand, dist_q_c, min_dist_c_s, mmr_score, method="MMR"
                )
                inserted, sticky_event = self._sticky_buffer_try_insert(mmr_buffer, entry, mmr_slots)
                step_rec: Optional[Dict[str, Any]] = None
                if self._dump_enabled:
                    step_rec = {
                        "step_idx": step_idx,
                        "cand_chunk_id": cand.get("chunk_id"),
                        "cand_emb": np.asarray(cand["embedding"], dtype=np.float32).copy(),
                        "anchor_and_mmr_embs": selected_embs.copy(),
                        "per_pair_dists": pair_dists_raw.copy(),
                                "min_dist_c_s_raw": min_dist_raw,
                                "min_dist_c_s_used": float(min_dist_c_s),
                                "dist_q_c_raw": float(dist_q_c),
                                **_k_dump_fields(float(min_dist_c_s)),
                                "mmr_score": float(mmr_score),
                                "selected": bool(inserted),
                        "skipped_dedup": False,
                        "method": "MMR",
                        "sticky_inserted": bool(inserted),
                        "sticky_event": str(sticky_event),
                        "requery_round": int(requery_round),
                        "opened_centroid_ids": list(opened_centroid_ids or []),
                        "distance_backend": self._mmr_distance_backend,
                        "noise_enabled": self._mmr_noise_enabled(),
                    }
                if inserted and self.min_mmr_score is not None:
                    cov = self.step6_coverage_check(
                        anchor_entries + mmr_buffer,
                        early_dedup_count,
                        min_mmr_score=self.min_mmr_score,
                    )
                    if step_rec is not None:
                        step_rec["coverage_checked"] = cov
                    if cov:
                        coverage_met_at_step = step_idx
                        if step_rec is not None:
                            self._dump_records["mmr_steps"].append(step_rec)
                        break
                if step_rec is not None:
                    self._dump_records["mmr_steps"].append(step_rec)

            if (
                coverage_met_at_step is None
                and len(mmr_buffer) < mmr_slots
                and skipped_for_dedup
            ):
                skipped_for_dedup.sort(key=lambda x: x[1])
                for cand, d_sw in skipped_for_dedup:
                    if len(mmr_buffer) >= mmr_slots:
                        break
                    step_idx += 1
                    score = -mmr_lambda_eff * d_sw
                    entry = _make_entry(cand, d_sw, 0.0, score, method="MMR")
                    inserted, sticky_event = self._sticky_buffer_try_insert(mmr_buffer, entry, mmr_slots)
                    cand_heading_fb = _section_heading_of(cand)
                    pair_dists_raw, selected_embs, _ = _pair_distances_single_readout(
                        cand_emb=np.asarray(cand["embedding"], dtype=np.float32),
                        pool=(anchor_entries + mmr_buffer),
                        cand_heading=cand_heading_fb,
                    )
                    min_dist_raw = float(np.min(pair_dists_raw)) if pair_dists_raw.size > 0 else float("inf")
                    step_rec_fb: Optional[Dict[str, Any]] = None
                    if self._dump_enabled:
                        step_rec_fb = {
                            "step_idx": step_idx,
                            "cand_chunk_id": cand.get("chunk_id"),
                            "cand_emb": np.asarray(cand["embedding"], dtype=np.float32).copy(),
                            "anchor_and_mmr_embs": selected_embs.copy(),
                            "per_pair_dists": pair_dists_raw.copy(),
                            "min_dist_c_s_raw": min_dist_raw,
                            "min_dist_c_s_used": 0.0,
                            "dist_q_c_raw": float(d_sw),
                            **_k_dump_fields(0.0),
                            "mmr_score": float(score),
                            "selected": bool(inserted),
                            "skipped_dedup": False,
                            "method": "MMR_FALLBACK",
                            "sticky_inserted": bool(inserted),
                            "sticky_event": str(sticky_event),
                            "requery_round": int(requery_round),
                            "opened_centroid_ids": list(opened_centroid_ids or []),
                            "distance_backend": self._mmr_distance_backend,
                            "noise_enabled": self._mmr_noise_enabled(),
                        }
                    if inserted and self.min_mmr_score is not None:
                        cov_fb = self.step6_coverage_check(
                            anchor_entries + mmr_buffer,
                            early_dedup_count,
                            min_mmr_score=self.min_mmr_score,
                        )
                        if step_rec_fb is not None:
                            step_rec_fb["coverage_checked"] = cov_fb
                        if cov_fb:
                            coverage_met_at_step = step_idx
                            if step_rec_fb is not None:
                                self._dump_records["mmr_steps"].append(step_rec_fb)
                            break
                    if step_rec_fb is not None:
                        self._dump_records["mmr_steps"].append(step_rec_fb)

            mmr_buffer.sort(key=lambda x: float(x["score"]), reverse=True)

        final_out = anchor_entries + mmr_buffer
        min_dists_accepted = [
            float(e.get("_mmr_min_dist_s", float("inf"))) for e in final_out
        ]


        stats = {
            "early_dedup_count": early_dedup_count,
            "min_dists_accepted": min_dists_accepted,
            "coverage_met_at_step": coverage_met_at_step,
        }
        return final_out, stats
    

    def step6_coverage_check(
        self,
        s_buffer: List[Dict],
        early_dedup_count: int = 0,
        *,
        min_mmr_score: Optional[float] = None,
    ) -> bool:
        """
        6. Coverage Check: Metadata 활용하여 MMR coverage 체크. [cite: 31]
        - Relevance/Redundancy floor, Centroid spread, Facet quota, Doc diversity 등 평가.
        - Optional min_mmr_score: buffer-wide MMR score floor (None = skip score gate).
        """
        if not s_buffer:
            return False
        if len(s_buffer) < self.top_k:
            return False

        thr = self.min_mmr_score if min_mmr_score is None else min_mmr_score
        if thr is not None:
            try:
                min_score_in_buffer = min(float(item["score"]) for item in s_buffer)
            except (KeyError, TypeError, ValueError):
                return False
            score_ok = min_score_in_buffer >= float(thr)
        else:
            score_ok = True

        doc_ids = set()
        centroid_ids = set()

        for item in s_buffer:
            meta = item.get("metadata", {})
            doc_id = meta.get("doc_id")
            if doc_id is not None:
                doc_ids.add(doc_id)
            centroid_ids.add(item.get("centroid_id"))

        # In intra-doc mode, doc diversity should not fail when all chunks come from one document.
        min_doc_div = 1 if len(doc_ids) <= 1 else min(2, self.top_k)
        min_centroid_div = min(2, self.nprobe, self.top_k)
        doc_diverse = len(doc_ids) >= min_doc_div
        centroid_spread = len(centroid_ids) >= min_centroid_div

        # 후보가 대부분 early-dedup으로 탈락하면 탐색 커버리지가 부족하다고 본다.
        early_dedup_limit = max(2, int(self.top_k * 1.5))
        dedup_ok = early_dedup_count <= early_dedup_limit

        coverage_satisfied = (
            len(s_buffer) >= self.top_k
            and score_ok
            and doc_diverse
            and centroid_spread
            and dedup_ok
        )
        return coverage_satisfied

    def _compute_per_query_summary(
        self,
        s_buffer: List[Dict],
        early_dedup_count: int = 0,
        min_dists_accepted: Optional[List[float]] = None,
        nprobe_used: Optional[int] = None,
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

        return {
            "unique_centroids_selected": unique_centroids_selected,
            "unique_centroids_ratio": unique_centroids_ratio,
            "unique_doc_ids": len(unique_doc_ids),
            "p25_min_dist_in_query": p25_min_dist,
            "early_dedup_count": early_dedup_count,
            "intra_list_distance": intra_list_distance,
            "generation_input_tokens": gen_tokens,
        }
        
    def _run_l2_only(
        self,
        query_text: str,
        target_doc_id: Optional[str] = None,
    ) -> Tuple[List[Dict], Dict[str, Any], Dict[str, float]]:
        """
        순수 L2 Top-K 검색.
        target_doc_id가 주어지면 해당 문서의 후보만 사용한다.
        MMR과 동일: OPQ 적용 쿼리 + PQ 복원 청크 벡터로 L2, centroid 필터(step3→step4)도 동일 조건.
        """
        query_emb, _, _, _ = self._build_query_embedding_streams(query_text)
        query_emb_arr = np.asarray(query_emb, dtype=np.float32)
        centroid_results = self.step3_coarse_search(query_emb, "query")
        centroid_ids = [cid for cid, _ in centroid_results]
        tid_step4 = (
            str(target_doc_id).strip()
            if target_doc_id and str(target_doc_id).strip()
            else None
        )
        all_candidates = self.step4_candidate_load(
            target_doc_id=(tid_step4 or ""),
            centroid_ids=(
                centroid_ids if (self.use_centroid_filter and centroid_ids) else None
            ),
        )
        for c in all_candidates:
            c["_dist_q"] = float(
                np.linalg.norm(
                    query_emb_arr - np.asarray(c["embedding"], dtype=np.float32)
                )
            )

        k = self.top_k
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
            nprobe_used=len(centroid_results),
        )
        latency = {
            "T_coarse_ms": T_COARSE_MS,
            "T_candidates_ms": T_CANDIDATES_MS,
            "T_L2_ms": T_L2_MS * max(len(all_candidates), 1),
        }
        return l2_topk_results, summary, latency

    def _build_raw_row_index(self) -> Dict[str, int]:
        """
        Map chunk_id -> row index in raw_embeddings.

        Offline indexer (``off_line_prepper``) stores the global row in metadata
        ``vector_id`` (same index as ``raw_embeddings`` / ``chunk_ids_order``).
        Tests may use ``raw_row`` / ``row_index`` / ``raw_idx`` instead.
        """
        raw_table = self.raw_embeddings
        if raw_table is None:
            raise RuntimeError("_build_raw_row_index requires raw_embeddings in hw_state")
        raw_arr = np.asarray(raw_table, dtype=np.float32)
        n_raw = int(raw_arr.shape[0])

        meta = self.metadata_sram if isinstance(self.metadata_sram, dict) else {}
        out: Dict[str, int] = {}
        for cid, rec in meta.items():
            if not isinstance(rec, dict):
                continue
            idx_any: Optional[object] = None
            for key in ("vector_id", "raw_row", "row_index", "raw_idx"):
                if key not in rec:
                    continue
                v = rec.get(key)
                if v is not None:
                    idx_any = v
                    break
            if idx_any is None:
                continue
            # vector_id / raw_row may be numpy scalar or plain int
            out[str(cid)] = int(np.asarray(idx_any).reshape(-1)[0])

        order = self.chunk_ids_order
        if isinstance(order, list) and len(order) == n_raw:
            for i, cid in enumerate(order):
                cid_s = str(cid)
                if cid_s not in out:
                    out[cid_s] = int(i)

        if not out:
            raise RuntimeError(
                "_build_raw_row_index: could not resolve any chunk_id -> raw row mapping. "
                "Expected metadata_sram entries with 'vector_id' (offline indexer) or "
                "chunk_ids_order aligned with raw_embeddings; inspect hw_state."
            )

        for cid_s, i in out.items():
            if i < 0 or i >= n_raw:
                raise RuntimeError(
                    f"_build_raw_row_index: row index {i} out of range for chunk_id={cid_s!r} "
                    f"(raw_embeddings rows={n_raw})."
                )
        return out

    def _save_l2raw_topk_npz(
        self,
        q_raw: np.ndarray,
        top_rows: List[Dict[str, Any]],
        dists_raw: np.ndarray,
        raw_mat: np.ndarray,
    ) -> str:
        """Write ``<dump_tag>_l2raw_topk.npz`` for device modeling (L2-raw path only)."""
        if not self._dump_enabled:
            raise RuntimeError("_save_l2raw_topk_npz called with dump_embeddings disabled")
        os.makedirs(self._dump_dir, exist_ok=True)
        out_npz = os.path.join(self._dump_dir, f"{self._dump_tag}_l2raw_topk.npz")
        cids = np.asarray([str(r.get("chunk_id", "")) for r in top_rows], dtype=object)
        np.savez(
            out_npz,
            chunk_ids=cids,
            dists_q_raw=np.asarray(dists_raw, dtype=np.float32).copy(),
            raw_embeddings=np.asarray(raw_mat, dtype=np.float32).copy(),
            query_emb_raw=np.asarray(q_raw, dtype=np.float32).reshape(-1).copy(),
        )
        return out_npz

    def _run_l2_only_raw(
        self,
        query_text: str,
        target_doc_id: Optional[str] = None,
    ) -> Tuple[List[Dict], Dict[str, Any], Dict[str, float]]:
        """
        L2 Top-K over RAW embeddings (no OPQ on query distance, no PQ reconstruction on candidates).

        Upper-bound reference: retrieval quality without PQ quantization loss on vectors.
        Uses the same candidate set (step3 -> step4) as ``_run_l2_only`` (OPQ-space coarse search)
        to isolate PQ reconstruction effect.
        """
        query_emb_post_opq, query_emb_pre_opq, _, _ = self._build_query_embedding_streams(query_text)
        q_raw = np.asarray(query_emb_pre_opq, dtype=np.float32).reshape(-1)

        centroid_results = self.step3_coarse_search(query_emb_post_opq, "query")
        centroid_ids = [cid for cid, _ in centroid_results]
        tid_step4 = (
            str(target_doc_id).strip()
            if target_doc_id and str(target_doc_id).strip()
            else None
        )
        all_candidates = self.step4_candidate_load(
            target_doc_id=(tid_step4 or ""),
            centroid_ids=(
                centroid_ids if (self.use_centroid_filter and centroid_ids) else None
            ),
        )

        raw_emb_table = self._hw.get("raw_embeddings")
        if raw_emb_table is None:
            raise RuntimeError("_run_l2_only_raw requires raw_embeddings.npy in hw_state")
        raw_emb_table = np.asarray(raw_emb_table, dtype=np.float32)

        if getattr(self, "_raw_row_by_chunk_id", None) is None:
            self._raw_row_by_chunk_id = self._build_raw_row_index()
        row_lookup: Dict[str, int] = self._raw_row_by_chunk_id

        missing: List[str] = []
        for c in all_candidates:
            cid = str(c.get("chunk_id", ""))
            row_i = row_lookup.get(cid)
            if row_i is None:
                missing.append(cid)
                continue
            c["_raw_embedding"] = raw_emb_table[row_i].astype(np.float32, copy=False)
            c["_dist_q_raw"] = float(np.linalg.norm(q_raw - raw_emb_table[row_i]))
        if missing:
            raise RuntimeError(
                f"_run_l2_only_raw: {len(missing)} candidate chunks not found in raw_embeddings "
                f"row map (first few: {missing[:3]}). Bundle may be missing or stale."
            )

        k = self.top_k
        all_candidates.sort(key=lambda x: x["_dist_q_raw"])
        k_take = min(k, len(all_candidates))
        top_slice = all_candidates[:k_take]

        if self._dump_enabled and top_slice:
            dists_arr = np.asarray([float(x["_dist_q_raw"]) for x in top_slice], dtype=np.float32)
            raw_mat = np.stack(
                [np.asarray(x["_raw_embedding"], dtype=np.float32) for x in top_slice],
                axis=0,
            )
            self._save_l2raw_topk_npz(q_raw, top_slice, dists_arr, raw_mat)

        l2raw_topk_results: List[Dict[str, Any]] = []
        for c in top_slice:
            row = {
                kk: v
                for kk, v in c.items()
                if kk not in ("_dist_q", "_dist_q_raw", "_raw_embedding")
            }
            row["_retrieval_method"] = "L2_Raw"
            l2raw_topk_results.append(row)

        summary = self._compute_per_query_summary(
            l2raw_topk_results,
            early_dedup_count=0,
            min_dists_accepted=None,
            nprobe_used=len(centroid_results),
        )
        dim_scale = float(raw_emb_table.shape[1]) / 768.0 if raw_emb_table.ndim == 2 else 1.0
        latency = {
            "T_coarse_ms": T_COARSE_MS,
            "T_candidates_ms": T_CANDIDATES_MS,
            "T_L2_ms": T_L2_MS * max(len(all_candidates), 1) * dim_scale,
        }
        return l2raw_topk_results, summary, latency

    def run_pipeline(
        self,
        user_query: str,
        target_doc_id: str,
        log_summary: bool = True,
        mode: str = "MMR"
    ) -> Union[
        List[Dict[str, Any]],
        Tuple[
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            Dict[str, Any],
            Dict[str, Any],
            Dict[str, Any],
        ],
    ]:
        """
        Executes single-query retrieval within a specific target document.
        """
        mode_upper = (mode or "MMR").upper()
        if mode_upper == "L2":
            self._last_centroid_filter_stats = None
            l2_results, l2_summary, l2_latency = self._run_l2_only(
                user_query,
                target_doc_id=target_doc_id,
            )
            self._last_summary = l2_summary
            self._last_latency_ms = l2_latency
            if log_summary and self.verbose:
                self._log_per_query_summary(l2_summary, l2_latency)
            return l2_results

        if mode_upper == "L2_RAW":
            self._last_centroid_filter_stats = None
            l2raw_results, l2raw_summary, l2raw_latency = self._run_l2_only_raw(
                user_query,
                target_doc_id=target_doc_id,
            )
            self._last_summary = l2raw_summary
            self._last_latency_ms = l2raw_latency
            if log_summary and self.verbose:
                self._log_per_query_summary(l2raw_summary, l2raw_latency)
            return l2raw_results

        if mode_upper == "COMPARE_ALL":
            self._last_centroid_filter_stats = None
            l2_results, l2_summary, _l2_lat = self._run_l2_only(
                user_query,
                target_doc_id=target_doc_id,
            )
            l2raw_results, l2raw_summary, _l2raw_lat = self._run_l2_only_raw(
                user_query,
                target_doc_id=target_doc_id,
            )
            mmr_results = cast(
                List[Dict[str, Any]],
                self.run_pipeline(
                    user_query,
                    target_doc_id=target_doc_id,
                    log_summary=False,
                    mode="MMR",
                ),
            )
            mmr_summary = dict(self._last_summary or {})
            return (
                l2_results,
                l2raw_results,
                mmr_results,
                dict(l2_summary),
                dict(l2raw_summary),
                mmr_summary,
            )

        if mode_upper != "MMR":
            raise ValueError("Use 'L2', 'L2_RAW', 'MMR', or 'COMPARE_ALL'.")

        latency = {"T_coarse_ms": 0.0, "T_candidates_ms": 0.0, "T_L2_ms": 0.0}

        # Embed query
        current_query_emb, _pre_q, current_entities, current_kw = self._build_query_embedding_streams(
            user_query
        )
        current_query_emb = np.asarray(current_query_emb, dtype=np.float32)

        # step3: nprobe centroids selected once (fixed), sorted by distance.
        centroid_results = self.step3_coarse_search(current_query_emb, "query")
        centroid_ids = [cid for cid, _ in centroid_results]
        latency["T_coarse_ms"] += T_COARSE_MS

        # step4: load target_doc chunks (optionally restricted to nprobe centroids).
        # 빈 centroid_ids는 기존과 같이 전체 로드(None)로 처리.
        candidates = self.step4_candidate_load(
            target_doc_id=target_doc_id,
            centroid_ids=(
                centroid_ids if (self.use_centroid_filter and centroid_ids) else None
            ),
        )
        latency["T_candidates_ms"] += T_CANDIDATES_MS
        latency["T_L2_ms"] += T_L2_MS * max(len(candidates), 1)

        self._last_centroid_filter_stats = None
        if self.verbose:
            # --- Centroid filtering coverage analysis (extra full load; does not affect MMR input) ---
            all_candidates = self.step4_candidate_load(
                target_doc_id=target_doc_id,
                centroid_ids=None,
            )
            all_chunk_ids = {c.get("chunk_id") for c in all_candidates}
            filtered_chunk_ids = {c.get("chunk_id") for c in candidates}
            missed_chunk_ids = all_chunk_ids - filtered_chunk_ids

            total = len(all_chunk_ids)
            filtered = len(filtered_chunk_ids)
            missed = len(missed_chunk_ids)
            coverage_pct = (filtered / total * 100) if total > 0 else 0.0

            # --- Centroid distribution analysis ---
            all_centroids_in_doc = {c.get("centroid_id") for c in all_candidates}
            selected_centroids_set = set(centroid_ids)
            hit_centroids = all_centroids_in_doc & selected_centroids_set

            centroid_chunk_counts = Counter(
                c.get("centroid_id") for c in all_candidates
            )
            top_centroids = centroid_chunk_counts.most_common(10)

            doc_centroids_n = len(all_centroids_in_doc)
            hit_centroids_n = len(hit_centroids)

            self._last_centroid_filter_stats = {
                "total": total,
                "loaded": filtered,
                "missed": missed,
                "doc_centroids": doc_centroids_n,
                "hit_centroids": hit_centroids_n,
            }

            for pct_target in (50, 70, 80, 90, 95, 100):
                needed = 0
                cum = 0
                for _cid, cnt in centroid_chunk_counts.most_common():
                    cum += int(cnt)
                    needed += 1
                    if cum >= total * pct_target / 100.0:
                        break
                
            # --- end ---

        # step5: anchor + MMR streaming (coverage early-stop only inside step5 when min_mmr_score set).
        s_buffer, stats = self.step5_on_the_fly_mmr(
            current_query_emb,
            candidates,
            [],
            buffer_top_k=self.top_k,
            gamma=self.gamma,
            existing_anchors=None,
            requery_round=0,
        )

        coverage_met = stats.get("coverage_met_at_step") is not None
        if self.min_mmr_score is not None and not coverage_met:
            if self.verbose:
                print(
                    f"[Coverage] Exhausted all candidates "
                    f"(buffer={len(s_buffer)}/{self.top_k}) — "
                    f"returning best-effort result"
                )

        final_top_k = s_buffer[: self.top_k]
        summary = self._compute_per_query_summary(
            final_top_k,
            early_dedup_count=stats.get("early_dedup_count", 0),
            min_dists_accepted=stats.get("min_dists_accepted", []),
            nprobe_used=max(len(centroid_results), 1)
        )
        if self.min_mmr_score is not None:
            coverage_summary = stats.get("coverage_met_at_step") is not None
            if not coverage_summary:
                coverage_summary = self.step6_coverage_check(
                    s_buffer,
                    stats.get("early_dedup_count", 0),
                    min_mmr_score=self.min_mmr_score,
                )
            summary["coverage_met"] = coverage_summary
            summary["centroids_opened"] = len(centroid_ids)
            summary["centroids_total"] = len(centroid_results)
            summary["requery_rounds_used"] = 0
            summary["best_effort"] = not coverage_summary
        
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
        반환: (l2_topk_results, mmr_results, l2_summary, mmr_summary)
        """
        query_emb, _pre_q, weighting_entities, cuad_keyword = self._build_query_embedding_streams(
            user_query
        )

        centroid_ids = self.step3_coarse_search(query_emb, "query")
        if not target_doc_id or not str(target_doc_id).strip():
            raise ValueError("run_with_comparison requires target_doc_id in intra-document mode.")
        tid_step4 = str(target_doc_id).strip()
        all_candidates = self.step4_candidate_load(target_doc_id=tid_step4)
        t_coarse_l2, t_cand_l2 = T_COARSE_MS, T_CANDIDATES_MS
        for c in all_candidates:
            c["_dist_q"] = float(
                np.linalg.norm(
                    np.asarray(query_emb, dtype=np.float32)
                    - np.asarray(c["embedding"], dtype=np.float32)
                )
            )
        k = self.top_k
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
        )
        l2_latency = {
            "T_coarse_ms": t_coarse_l2,
            "T_candidates_ms": t_cand_l2,
            "T_L2_ms": t_l2_l2,
        }

        # ---- (2) On-the-fly MMR
        self._last_summary = None
        self._last_latency_ms = None
        mmr_results = cast(
            List[Dict[str, Any]],
            self.run_pipeline(
                user_query,
                target_doc_id=target_doc_id,
                log_summary=False,
                mode="MMR",
            ),
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
) -> Tuple[List[str], List[Tuple[str, int, Optional[str], Optional[str]]]]:
    """
    로컬 processed_cuad/cuad_corpus.json, cuad_queries.json에서
    인덱싱용 문맥과 평가용 (question, ref_doc_id, answers_text, query_concat) 로드.
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
        )
        for item in queries_raw
    ]
    if max_contexts is not None and max_contexts > 0:
        contexts = contexts[:max_contexts]
        eval_pairs = [
            (q, ref_id, ans, q_concat)
            for (q, ref_id, ans, q_concat) in eval_pairs
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
    path_raw_emb = os.path.join(dir_path, "raw_embeddings.npy")
    path_chunk_ids = os.path.join(dir_path, "chunk_ids_order.pkl")
    if os.path.isfile(path_c):
        centroids = np.load(path_c)
    if os.path.isfile(path_pq):
        pq_codebooks = np.load(path_pq)
    opq_rotation = None
    if os.path.isfile(path_opq):
        opq_rotation = np.load(path_opq)
    raw_embeddings = None
    if os.path.isfile(path_raw_emb):
        raw_embeddings = np.load(path_raw_emb)
    chunk_ids_order = None
    if os.path.isfile(path_chunk_ids):
        with open(path_chunk_ids, "rb") as f:
            chunk_ids_order = pickle.load(f)
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
        "raw_embeddings": raw_embeddings,
        "chunk_ids_order": chunk_ids_order,
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
    indexer.run_pipeline(contexts, doc_ids=[str(i) for i in range(len(contexts))], output_dir=save_dir)
    return save_dir

GT_COVERAGE_THRESHOLD = 0.5

# Lazy singleton: RougeScorer is relatively expensive to construct.
_ROUGE_L_SCORER: Any = None


def _get_rouge_l_scorer() -> Any:
    """Build a stemmed ROUGE-L scorer once (legal morphology: e.g. transfer vs transferable)."""
    global _ROUGE_L_SCORER
    if _ROUGE_L_SCORER is None:
        try:
            from rouge_score import rouge_scorer
        except ImportError as e:
            raise ImportError(
                "rouge-score is required for ROUGE-L GT coverage. Install: pip install rouge-score"
            ) from e
        _ROUGE_L_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_L_SCORER


def _ground_truth_max_rouge_l_recall(ground_truth: str, retrieved_text: str) -> float:
    """
    Max ROUGE-L recall over [SEP]-separated GT spans (reference = span, prediction = retrieved_text).
    Returns 0.0 on empty/missing inputs or scorer errors.
    """
    gt = str(ground_truth or "").strip()
    cand = str(retrieved_text or "").strip()
    if not gt or not cand:
        return 0.0

    spans = [s.strip() for s in gt.split("[SEP]") if s.strip()]
    if not spans:
        return 0.0

    scorer = _get_rouge_l_scorer()
    max_recall = 0.0
    for span in spans:
        if not span:
            continue
        try:
            scores = scorer.score(span, cand)
            rec = float(scores["rougeL"].recall)
            if rec > max_recall:
                max_recall = rec
        except (ValueError, KeyError, TypeError):
            continue
    return max_recall


def _max_rouge_l_recall_over_blocks(ground_truth: str, raw_blocks: List[str]) -> float:
    """Best max span-recall among top-k stitched chunks for one query."""
    best = 0.0
    for t in raw_blocks:
        best = max(best, _ground_truth_max_rouge_l_recall(ground_truth, t))
    return best


def _ground_truth_token_coverage(ground_truth: str, retrieved_text: str) -> float:
    """
    Binary gate: 1.0 if _ground_truth_max_rouge_l_recall >= GT_COVERAGE_THRESHOLD, else 0.0.
    """
    r = _ground_truth_max_rouge_l_recall(ground_truth, retrieved_text)
    return 1.0 if r >= GT_COVERAGE_THRESHOLD else 0.0

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


def build_normalized_retrieval_record(
    *,
    query_idx: int,
    query: str,
    ground_truth: str,
    aspect_queries: List[str],
    l2_contexts: Optional[List[str]],
    mmr_contexts: Optional[List[str]],
    l2_chunk_lut: Optional[List[Dict[str, Any]]],
    mmr_chunk_lut: Optional[List[Dict[str, Any]]],
    domain: str,
    top_k: int,
    window_size: int,
    retrieval_mode: str,
    target_doc_id: Optional[str],
    ref_doc_id: Optional[int],
    mmr_lambda: float,
    tau_dup: float,
    gamma: float,
    l2_raw_contexts: Optional[List[str]] = None,
    l2_raw_chunk_lut: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one canonical retrieval record for downstream evaluation."""
    return {
        "query_idx": int(query_idx),
        "query": str(query or ""),
        "rewritten_query": None,
        "ground_truth": str(ground_truth or ""),
        "aspect_queries": [str(x) for x in (aspect_queries or [])],
        "retrieval": {
            "l2": {
                "contexts": list(l2_contexts or []),
                "chunk_lut": list(l2_chunk_lut or []),
            },
            "l2_raw": {
                "contexts": list(l2_raw_contexts or []),
                "chunk_lut": list(l2_raw_chunk_lut or []),
            },
            "mmr": {
                "contexts": list(mmr_contexts or []),
                "chunk_lut": list(mmr_chunk_lut or []),
            },
        },
        "generation": {
            "l2": {"answer": None},
            "mmr": {"answer": None},
        },
        "metadata": {
            "domain": str(domain or ""),
            "top_k": int(top_k),
            "window_size": int(window_size),
            "retrieval_mode": str(retrieval_mode or ""),
            "target_doc_id": None if target_doc_id is None else str(target_doc_id),
            "ref_doc_id": ref_doc_id,
            "mmr_lambda": float(mmr_lambda),
            "tau_dup": float(tau_dup),
            "gamma": float(gamma),
        },
    }


def save_normalized_retrieval_export(path: str, records: List[Dict[str, Any]]) -> None:
    """Save normalized retrieval records as JSON list."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_device_fit_json() -> str:
    return os.path.join(_REPO_ROOT, "h_aimdc_sim", "igzo_fit.json")


def _reconstruct_pq_all(
    lut: np.ndarray,
    pq_codes_map: Mapping[Any, Any],
    *,
    max_rows: int = 50_000,
) -> np.ndarray:
    """
    Iterate posting lists, decode each entry's ``pq_code`` via ``lut``, stack into (N, M*sub_dim).
    Stops once ``max_rows`` rows collected (stable dict/list iteration order).
    """
    if lut.ndim != 3:
        raise ValueError(f"lut shape expected (M,K,sub_dim), got {lut.shape}")
    m, k_size, sub_dim = int(lut.shape[0]), int(lut.shape[1]), int(lut.shape[2])
    out_rows: List[np.ndarray] = []
    if not isinstance(pq_codes_map, dict):
        return np.zeros((0, m * sub_dim), dtype=np.float32)
    for _cid, posting_list in pq_codes_map.items():
        if len(out_rows) >= max_rows:
            break
        if not isinstance(posting_list, list):
            continue
        for entry in posting_list:
            if len(out_rows) >= max_rows:
                break
            if not isinstance(entry, dict):
                continue
            pq_code = entry.get("pq_code")
            if pq_code is None:
                continue
            try:
                codes = np.asarray(pq_code, dtype=np.int64).reshape(m)
            except Exception:
                continue
            if np.any(codes < 0) or np.any(codes >= k_size):
                continue
            vec = np.empty(m * sub_dim, dtype=np.float32)
            for j in range(m):
                vec[j * sub_dim : (j + 1) * sub_dim] = lut[j, int(codes[j]), :].astype(
                    np.float32, copy=False
                )
            out_rows.append(vec)
    if not out_rows:
        return np.zeros((0, m * sub_dim), dtype=np.float32)
    return np.stack(out_rows, axis=0).astype(np.float32, copy=False)


def _configure_device_distance(
    hw_state: Dict[str, Any],
    *,
    device_fit_path: str,
    device_scaler_path: Optional[str],
    device_reduce: str,
    device_use_log: bool,
    device_v_ml: float,
    noise_config: Optional[str],
    noise_seed: Optional[int],
    device_scale_k_path: Optional[str] = None,
    device_scale_k_override: Optional[float] = None,
) -> Tuple["DeviceDistance", Optional[float]]:
    import hashlib

    from aimdc_device_distance import DeviceDistance, NoiseModel, noise_model_from_json_path

    fit_abs = os.path.abspath(device_fit_path)
    if not os.path.isfile(fit_abs):
        raise FileNotFoundError(
            f"[Device-L2] Fit JSON not found: {fit_abs}. "
            "Run: python h_aimdc_sim/aimdc_sim_core.py fit --csv h_aimdc_sim/igzo_measure.csv --out igzo_fit.json"
        )
    with open(fit_abs, "rb") as f:
        fit_sha256 = hashlib.sha256(f.read()).hexdigest()
    stem, _ext = os.path.splitext(fit_abs)
    scaler_abs = (
        os.path.abspath(device_scaler_path)
        if device_scaler_path
        else f"{stem}.scaler.json"
    )
    noise: NoiseModel
    if noise_config:
        nc = os.path.abspath(noise_config)
        if not os.path.isfile(nc):
            raise FileNotFoundError(f"[Device-L2] noise_config not found: {nc}")
        noise = noise_model_from_json_path(nc)
    else:
        noise = NoiseModel()
    if noise_seed is not None:
        noise = replace(noise, rng_seed=int(noise_seed))
    reduce_mode: Literal["sum", "mean"] = "mean" if device_reduce == "mean" else "sum"
    reduction_mode_key = str(reduce_mode)

    def _fit_scaler(dd: "DeviceDistance") -> None:
        lut = hw_state.get("lut_memory")
        pq_codes_map = hw_state.get("pq_codes")
        if not isinstance(pq_codes_map, dict):
            pq_codes_map = hw_state.get("l2_accelerator_2_pq_codes")
        if lut is None or not isinstance(lut, np.ndarray) or not isinstance(pq_codes_map, dict):
            raise RuntimeError(
                "[Device-L2] Scaler missing and hw_state has no (lut_memory, pq_codes); "
                "cannot auto-fit on PQ-recon."
            )
        pq_recon = _reconstruct_pq_all(lut, pq_codes_map, max_rows=50_000)
        if pq_recon.size == 0:
            raise RuntimeError("[Device-L2] PQ reconstruction yielded empty matrix.")
        dd.fit_scaler(pq_recon.astype(np.float64, copy=False))
        dd.save_scaler(scaler_abs)
        print(
            "[Device-L2] fitted single scaler on PQ-recon: "
            f"N={pq_recon.shape[0]}, D={pq_recon.shape[1]}, "
            f"x_lo={dd._x_lo:.6g}, x_hi={dd._x_hi:.6g}"
        )

    scaler_exists = os.path.isfile(scaler_abs)
    need_refit = False
    if scaler_exists:
        try:
            with open(scaler_abs, encoding="utf-8") as f:
                sc = json.load(f)
            sc_sha = sc.get("fit_sha256")
            if sc_sha is None or str(sc_sha) != fit_sha256:
                print(
                    f"[Device-L2] scaler fit_sha256 mismatch or missing "
                    f"(scaler={sc_sha!r}, fit={fit_sha256}); auto-refitting."
                )
                need_refit = True
        except (OSError, json.JSONDecodeError, TypeError):
            need_refit = True

    dd = DeviceDistance(
        fit_abs,
        scaler_path=None if need_refit else scaler_abs,
        reduce=reduce_mode,
        use_log=bool(device_use_log),
        v_ml=float(device_v_ml),
        noise=noise,
    )
    if need_refit or dd._x_lo is None:
        _fit_scaler(dd)
    else:
        print(f"[Device-L2] using existing scaler at {scaler_abs}.")
    ne = bool(dd.noise.enabled())
    print(
        "[Device-L2] analytical backend: "
        f"fit={fit_abs}, reduce={reduce_mode}, use_log={bool(device_use_log)}, "
        f"scaler=[x_lo={dd._x_lo:.6g}, x_hi={dd._x_hi:.6g}] → "
        f"[v_min={dd.v_min:.4g}, v_max={dd.v_max:.4g}], D={dd.embedding_dim}"
    )
    print(
        "[Device-L2] noise: "
        f"enabled={ne}, ktc_C={noise.ktc_C}, σα={noise.sigma_alpha}, "
        f"σvth={noise.sigma_vth}, retention_tau={noise.retention_tau}, adc={noise.adc_bits}"
    )
    print(
        "[Device-L2] scope: AIMDC is strictly the MMR diversity accelerator "
        "(V_SL = incoming candidate C, V_PRG = anchor + selected S). "
        "d(q, C) comes from PQ-LUT (software L2). Units differ by design; "
        "tune mmr_lambda experimentally."
    )
    k_val: Optional[float] = None
    if device_scale_k_override is not None:
        k_val = float(device_scale_k_override)
        if not np.isfinite(k_val) or k_val <= 0.0:
            raise ValueError(f"[Device-L2] invalid scale_k override: {k_val}")
        print(f"[Device-L2] scale_k override: k={k_val:.6g}")
    elif device_scale_k_path:
        kp = os.path.abspath(str(device_scale_k_path).strip())
        if not os.path.isfile(kp):
            raise FileNotFoundError(f"[Device-L2] scale_k JSON not found: {kp}")
        with open(kp, "r", encoding="utf-8") as f:
            kdata = json.load(f)
        # Reject pre-migration scale_k (log-domain calibration is invalid for
        # linear_sum / any reduction change).
        k_red = kdata.get("reduction_mode")
        k_sha = kdata.get("fit_sha256")
        if k_red is None or k_sha is None:
            raise ValueError(
                f"[Device-L2] scale_k JSON {kp} is missing reduction_mode/fit_sha256. "
                "Old log-domain calibrations are invalid; re-run "
                "scripts/calibrate_scale_k.py against the current fit."
            )
        if str(k_sha) != fit_sha256:
            raise ValueError(
                f"[Device-L2] scale_k fit_sha256 mismatch "
                f"(k={k_sha}, fit={fit_sha256}). Re-calibrate."
            )
        if str(k_red) != reduction_mode_key:
            raise ValueError(
                f"[Device-L2] scale_k reduction_mode mismatch "
                f"(k={k_red!r}, current={reduction_mode_key!r}). Re-calibrate."
            )
        k_val = float(kdata["k"])
        if not np.isfinite(k_val) or k_val <= 0.0:
            raise ValueError(f"[Device-L2] invalid k from {kp}: {k_val}")
        rs = kdata.get("ratio_stats") or {}
        print(
            f"[Device-L2] loaded scale_k from {kp}: k={k_val:.6g} "
            f"(ratio p5={rs.get('p5', 'NA')}, p95={rs.get('p95', 'NA')})"
        )
    if k_val is not None:
        print(
            f"[Device-L2] calibration: k={k_val:.6g}. "
            "tau_dup and MMR score are interpreted in software-L2 scale; "
            "raw AIMDC distances are preserved in dumps."
        )
    else:
        print(
            "[Device-L2] no scale_k set - tau_dup and MMR score use AIMDC raw scale. "
            "Run scripts/calibrate_scale_k.py to calibrate."
        )
    return dd, k_val


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
    gamma: float = 0.25,
    verbose: bool = True,
    show_docs: bool = False,
    export_normalized_json: Optional[str] = None,
    target_doc_id: Optional[str] = None,
    window_size: int = 1,
    retrieval_mode: str = "COMPARE",
    dump_embeddings: bool = False,
    dump_dir: str = "./emb_dump",
    dump_tag: str = "q0",
    dump_only_first_query: bool = False,
    min_mmr_score: Optional[float] = None,
    max_requery_rounds: int = 3,
    use_centroid_filter: bool = True,
    use_device_distance: bool = False,
    device_fit_path: str = "",
    device_scaler_path: Optional[str] = None,
    device_reduce: str = "sum",
    device_use_log: bool = True,
    device_v_ml: float = 0.0,
    noise_config: Optional[str] = None,
    noise_seed: Optional[int] = None,
    device_scale_k_path: Optional[str] = None,
    device_scale_k: Optional[float] = None,
    use_centroid_device_distance: bool = False,
    centroid_fit_path: str = "",
    centroid_scaler_path: Optional[str] = None,
    centroid_reduce: str = "linear_sum",
    centroid_use_log: bool = False,
    centroid_v_ml: float = 0.0,
    centroid_noise_config: Optional[str] = None,
    centroid_noise_seed: Optional[int] = None,
    centroid_inverter_vtc_path: Optional[str] = None,
    tau_dup_calibrated_for: Optional[str] = None,
) -> Dict[str, Any]:
    """
    데이터셋에서 질문 N개(num_queries)를 샘플링해 OnlinePipeline으로 검색.
    평가 목표: Recall@K(top_k 인자, ROUGE-L multi-span max recall 게이트, GT_COVERAGE_THRESHOLD), 평균 ROUGE-L max recall, ILD, RR, Search Breadth.
    검색은 항상 target document 후보를 기준으로 동작하며,
    target_doc_id(또는 ref_doc_id)는 검색 대상 문서를 지정합니다.
    retrieval_mode=COMPARE일 때 baseline L2 vs proposed MMR 2-way 비교 및 export.
    COMPARE_ALL이면 L2 + L2_RAW(raw 임베딩) + MMR 3-way 비교.
    L2_RAW는 PQ 없는 상한 기준선만 단독 집계한다.
    그 외 단일 모드에서는 해당 모드만 집계한다.
    eval_pairs는 (question, ref_doc_id, answers_text, query_concat) 구조로,
    ref_doc_id는 문서 인덱스, answers_text는 실제 정답 구절(없으면 빈 문자열)입니다.
    window_size: export용 문맥 스티칭 시 동일 doc 인접 청크 반경(_stitch_context_for_chunk).
    gamma: fraction of top_k reserved as pure-L2 anchors before MMR fills remaining slots.
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
    device_dd: Optional["DeviceDistance"] = None
    device_scale_k_val: Optional[float] = None
    if use_device_distance:
        fit_p = device_fit_path.strip() or _default_device_fit_json()
        device_dd, device_scale_k_val = _configure_device_distance(
            hw_state,
            device_fit_path=fit_p,
            device_scaler_path=device_scaler_path,
            device_reduce=device_reduce,
            device_use_log=device_use_log,
            device_v_ml=device_v_ml,
            noise_config=noise_config,
            noise_seed=noise_seed,
            device_scale_k_path=device_scale_k_path,
            device_scale_k_override=device_scale_k,
        )
    centroid_dd = None
    if use_centroid_device_distance:
        from aimdc_nvm_device_distance import (
            NvmDeviceDistance,
            NoiseModel as NvmNoiseModel,
            noise_model_from_json_path as nvm_noise_from_json,
        )
        fit_path_nvm = centroid_fit_path or os.path.join("h_aimdc_sim", "igzo_fit_v3.json")
        if not os.path.isfile(fit_path_nvm):
            raise FileNotFoundError(
                f"NVM fit JSON not found: {fit_path_nvm}. "
                f"Run: python -m h_aimdc_sim.aimdc_sim_core_v3 fit_v3 --csv h_aimdc_sim/igzo_measure.csv --out {fit_path_nvm}"
            )
        if centroid_noise_config:
            nvm_noise = nvm_noise_from_json(centroid_noise_config)
            if centroid_noise_seed is not None:
                nvm_noise = replace(nvm_noise, rng_seed=int(centroid_noise_seed))
        else:
            nvm_noise = NvmNoiseModel()
            if centroid_noise_seed is not None:
                nvm_noise = replace(nvm_noise, rng_seed=int(centroid_noise_seed))
        centroid_dd = NvmDeviceDistance(
            fit_path_nvm,
            scaler_path=centroid_scaler_path,
            reduce=centroid_reduce,
            use_log=centroid_use_log,
            v_ml=centroid_v_ml,
            noise=nvm_noise,
            inverter_vtc_path=centroid_inverter_vtc_path,
        )
        if centroid_dd._is_fefet:
            cal_for = tau_dup_calibrated_for
            if cal_for != "fefet_linear_sum":
                print("=" * 68)
                print(
                    "WARNING: FeFET-path fit loaded with tau_dup that was NOT "
                    "re-tuned for linear_sum reduction. "
                    "Old log-domain tau_dup / min_mmr_score thresholds are on a "
                    "different scale and must be re-calibrated. "
                    "Set tau_dup_calibrated_for='fefet_linear_sum' once retuned."
                )
                print(f"  tau_dup={tau_dup!r}  reduce={centroid_dd.reduce!r}  "
                      f"p={centroid_dd.p!r}")
                print("=" * 68)
        if centroid_dd._x_lo is None:
            # Auto-fit scaler from raw_embeddings if no scaler JSON was found.
            raw_emb = hw_state.get("raw_embeddings")
            if isinstance(raw_emb, np.ndarray) and raw_emb.size > 0:
                centroid_dd.fit_scaler(raw_emb)
                if not centroid_scaler_path:
                    auto_scaler = f"{os.path.splitext(fit_path_nvm)[0]}.scaler.json"
                    centroid_dd.save_scaler(auto_scaler)
                    print(f"[NVM] Auto-fit scaler saved to {auto_scaler}")
            else:
                raise RuntimeError(
                    "NVM scaler not loaded and raw_embeddings unavailable; "
                    "provide --centroid_scaler_path explicitly."
                )
    if centroid_dd is not None:
        cp = centroid_dd._circuit_params
        meta = centroid_dd._meta or {}
        cp_meta_strs = (meta.get("circuit_params") or {}) if isinstance(meta, dict) else {}
        # _validated_at and _op_source are string side-channels embedded by fit_v3 --operating-point
        validated_at = cp_meta_strs.get("_validated_at") if isinstance(cp_meta_strs, dict) else None
        op_source = cp_meta_strs.get("_op_source") if isinstance(cp_meta_strs, dict) else None

        def _fmt(key: str, fallback: str = "n/a") -> str:
            v = cp.get(key)
            if v is None:
                return fallback
            return f"{float(v):.3f}V"

        v_mbl = float(cp.get("V_MBL", float("nan")))
        v_ml_used = float(centroid_dd.v_ml)
        v_ds_eff = v_mbl - v_ml_used if np.isfinite(v_mbl) else float("nan")

        print("=" * 68)
        print("[NVM Centroid Accelerator] Operating point in use:")
        print(f"  Inverter:   VDD={_fmt('VDD')}  VGG={_fmt('VGG')}  VSS={_fmt('VSS')}")
        if np.isfinite(v_ds_eff):
            print(
                f"  Cell bias:  V_MBL={_fmt('V_MBL')}  V_ML={v_ml_used:.3f}V"
                f"  (V_DS_effective = {v_ds_eff:.3f}V)"
            )
        else:
            print(f"  Cell bias:  V_MBL={_fmt('V_MBL')}  V_ML={v_ml_used:.3f}V")
        print(f"  Embedding:  v_min={centroid_dd.v_min:.3f}V  v_max={centroid_dd.v_max:.3f}V")
        if centroid_dd._x_lo is not None and centroid_dd._x_hi is not None:
            print(
                f"  Scaler:     raw [{centroid_dd._x_lo:+.4g}, {centroid_dd._x_hi:+.4g}] "
                f"-> V [{centroid_dd.v_min:+.3f}, {centroid_dd.v_max:+.3f}]"
            )
        inverter_mode = "LUT (loaded)" if centroid_dd._inv_v_in is not None else "Ideal (-V_A)"
        print(f"  Inverter:   {inverter_mode}")
        if validated_at:
            print(f"  Validated:  {validated_at}")
        if op_source:
            print(f"  Op source:  {op_source}")
        print(f"  Fit JSON:   {centroid_dd.fit_json_path}")
        if centroid_dd.noise.enabled():
            # show only the active noise mechanisms
            active_noise = []
            n = centroid_dd.noise
            if n.ktc_C > 0.0:
                active_noise.append(f"kT/C={n.ktc_C:g}F")
            if n.sigma_alpha > 0.0:
                active_noise.append(f"sigma_alpha={n.sigma_alpha:g}")
            if n.sigma_vth > 0.0:
                active_noise.append(f"sigma_vth={n.sigma_vth:g}V")
            if n.sigma_log_mu > 0.0:
                active_noise.append(f"sigma_log_mu={n.sigma_log_mu:g}")
            if n.retention_tau is not None:
                active_noise.append(
                    f"retention_tau={n.retention_tau:g}s, t_hold={n.t_hold:g}s"
                )
            if n.flicker_sigma > 0.0:
                active_noise.append(f"flicker={n.flicker_sigma:g}")
            if n.rtn_sigma > 0.0:
                active_noise.append(f"RTN={n.rtn_sigma:g}")
            if n.shot_enable:
                active_noise.append(f"shot(BW={n.read_bandwidth_hz:g}Hz)")
            if n.adc_bits is not None:
                active_noise.append(f"ADC={n.adc_bits}b@{n.adc_i_fullscale:g}A")
            print(f"  Noise:      ENABLED — {', '.join(active_noise)}")
        else:
            print("  Noise:      disabled (deterministic)")
        print("=" * 68)
    pipeline = L2RAGOnlinePipeline(
        hw_state,
        domain_mode=domain,
        nprobe=nprobe,
        top_r=top_r,
        top_k=top_k,
        tau_dup=tau_dup,
        mmr_lambda=mmr_lambda,
        gamma=gamma,
        verbose=verbose,
        dump_embeddings=dump_embeddings,
        dump_dir=dump_dir,
        dump_tag=dump_tag,
        min_mmr_score=min_mmr_score,
        max_requery_rounds=max_requery_rounds,
        use_centroid_filter=use_centroid_filter,
        device_distance=device_dd,
        centroid_device_distance=centroid_dd,
        device_scale_k=device_scale_k_val,
    )
    doc_chunk_text_index = _build_doc_chunk_text_index(pipeline.metadata_sram or {})
    
    metrics_l2: List[Dict[str, Any]] = []  # L2 baseline
    metrics_l2_raw: List[Dict[str, Any]] = []  # L2 on raw embeddings (PQ-free upper bound)
    metrics_mmr: List[Dict[str, Any]] = []  # MMR proposed
    export_data: List[Dict[str, Any]] = []
    export_eval_data: List[Dict[str, Any]] = []

    mode_upper = (retrieval_mode or "MMR").upper()
    if mode_upper not in ("COMPARE", "COMPARE_ALL", "L2", "L2_RAW", "MMR"):
        raise ValueError(
            f"Unsupported retrieval_mode: {retrieval_mode}. "
            "Use one of COMPARE, COMPARE_ALL, L2, L2_RAW, MMR."
        )
    recall_k = int(top_k)

    def _format_for_llm(
        doc: Dict,
        stitched_text: str,
        current_target_id: Optional[str],
        mode: str = "L2",
    ) -> str:
        meta = doc.get("metadata", {})
        doc_id = str(meta.get("doc_id", "")).strip()
        section = str(meta.get("section_heading", "Unknown Section")).strip()
        method = str(doc.get("_retrieval_method", mode))
        _ = current_target_id, doc_id
        header = f"--- [Source: {method} | Section: {section}] ---\n"
        return header + stitched_text

    def _normalize_aspect_queries(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            t = value.strip()
            return [t] if t else []
        if isinstance(value, list):
            out: List[str] = []
            for x in value:
                if isinstance(x, str):
                    t = x.strip()
                    if t:
                        out.append(t)
                elif x is not None:
                    t = str(x).strip()
                    if t:
                        out.append(t)
            return out
        t = str(value).strip()
        return [t] if t else []

    def _extract_aspect_only_queries(value: Any) -> List[str]:
        """Return aspect-line sub-queries only (no leading main question).

        The CUAD ``query_concat`` field may concatenate the rewritten main
        question and four aspect lines with ``[SEP]``. Downstream evaluation
        (e.g. coverage) expects ``aspect_queries`` to contain only the aspect
        lines. When ``value`` is (or unpacks to) a ``[SEP]``-joined string,
        the first segment is dropped if it does not start with a known aspect
        label prefix (heuristic: main question vs. ``[Scope/...]``, etc.).
        """
        if value is None:
            return []

        # List of separate strings: treat as already-split aspects.
        if isinstance(value, list):
            if len(value) == 1:
                only = value[0]
                # Single entry may still be a packed "[SEP]" string from the loader.
                if isinstance(only, str) and "[SEP]" in only:
                    return _extract_aspect_only_queries(only)
            return _normalize_aspect_queries(value)

        if isinstance(value, str):
            t = value.strip()
            if not t:
                return []
            if "[SEP]" not in t:
                return [t]
            parts = [p.strip() for p in t.split("[SEP]")]
            parts = [p for p in parts if p]
            if len(parts) <= 1:
                return parts
            # Heuristic: real aspect lines in this pipeline start with these tags.
            aspect_label_prefixes = (
                "[Scope",
                "[Obligation",
                "[Exclusion",
                "[Procedure",
            )
            first = parts[0]
            if not any(first.startswith(prefix) for prefix in aspect_label_prefixes):
                parts = parts[1:]
            return parts

        return _normalize_aspect_queries(value)

    centroid_filter_mmr_samples: List[Dict[str, Any]] = []

    for idx, (question, ref_doc_id, answers_text, query_concat) in enumerate(eval_pairs):
        if dump_embeddings:
            pipeline._dump_reset()
            pipeline._dump_tag = f"{dump_tag}_q{idx}"
        current_target = (
            str(ref_doc_id) if ref_doc_id is not None else target_doc_id
        )
        aspect_queries = _extract_aspect_only_queries(query_concat)
        target_id_for_tag = str(current_target).strip() if current_target else None
        if not target_id_for_tag:
            target_id_for_tag = None

        # Single-mode execution: L2, L2_RAW, or MMR
        if mode_upper in ("L2", "L2_RAW", "MMR"):
            routed_results = cast(
                List[Dict[str, Any]],
                pipeline.run_pipeline(
                    question,
                    target_doc_id=current_target,
                    log_summary=False,
                    mode=mode_upper,
                ),
            )
            if mode_upper == "MMR":
                cfs = pipeline._last_centroid_filter_stats
                if cfs is not None:
                    centroid_filter_mmr_samples.append(dict(cfs))
            routed_summary = pipeline._last_summary or {}
            k_take = min(int(top_k), len(routed_results))
            ws = max(0, int(window_size))
            routed_stitched: List[str] = []
            routed_stitched_for_recall: List[str] = []
            routed_lut: List[Dict] = []
            fmt_mode = "L2_RAW" if mode_upper == "L2_RAW" else mode_upper
            for chunk_no, doc in enumerate(routed_results[:k_take], start=1):
                raw = _trim_broken_edges(
                    _stitch_context_for_chunk(
                        doc,
                        doc_chunk_text_index,
                        window_size=ws,
                        debug_cb=(lambda msg: _append_eval_debug(msg)),
                    )
                )
                routed_stitched_for_recall.append(raw)
                routed_stitched.append(
                    _format_for_llm(doc, raw, target_id_for_tag, mode=fmt_mode)
                )
                routed_lut.append({
                    "chunk_no": chunk_no,
                    "doc_id": str((doc.get("metadata") or {}).get("doc_id", "")).strip(),
                    "chunk_text": raw,
                })
            expected_text = ""
            if isinstance(answers_text, str) and answers_text.strip():
                expected_text = answers_text.strip()
            elif isinstance(ref_doc_id, int) and 0 <= ref_doc_id < len(contexts):
                expected_text = contexts[ref_doc_id] or ""
            recall_routed = 0.0
            for t in routed_stitched_for_recall:
                if _ground_truth_token_coverage(expected_text, t) >= 1.0:
                    recall_routed = 1.0
                    break
            rouge_routed = _max_rouge_l_recall_over_blocks(expected_text, routed_stitched_for_recall)
            rr_routed = _redundancy_rate_from_embeddings(routed_results, top_k, sim_threshold=0.7)
            k_take_mmr = min(int(top_k), len(routed_results))
            _scores_m = [
                float(x.get("score", float("nan"))) for x in routed_results[:k_take_mmr]
            ]
            _arr_m = np.asarray(_scores_m, dtype=np.float64)
            mean_mmr_topk = (
                float(np.nanmean(_arr_m)) if _arr_m.size > 0 else float("nan")
            )
            routed_summary = {
                **routed_summary,
                "recall_at_k_50": recall_routed,
                "rouge_l_max_recall": rouge_routed,
                "redundancy_rate": rr_routed,
                "mean_mmr_topk_score": mean_mmr_topk,
            }
            if mode_upper == "MMR":
                metrics_mmr.append(routed_summary)
            elif mode_upper == "L2_RAW":
                metrics_l2_raw.append(routed_summary)
            else:
                metrics_l2.append(routed_summary)
            lut_key = "l2_raw" if mode_upper == "L2_RAW" else mode_upper.lower()
            export_data.append({
                "query": question,
                "ground_truth": expected_text,
                "aspect_queries": aspect_queries,
                "mode": mode_upper,
                "contexts": routed_stitched,
                "chunk_lut": {
                    lut_key: routed_lut,
                },
            })
            export_eval_data.append(
                build_normalized_retrieval_record(
                    query_idx=idx,
                    query=question,
                    ground_truth=expected_text,
                    aspect_queries=aspect_queries,
                    l2_contexts=routed_stitched if mode_upper == "L2" else [],
                    mmr_contexts=routed_stitched if mode_upper == "MMR" else [],
                    l2_chunk_lut=routed_lut if mode_upper == "L2" else [],
                    mmr_chunk_lut=routed_lut if mode_upper == "MMR" else [],
                    l2_raw_contexts=routed_stitched if mode_upper == "L2_RAW" else [],
                    l2_raw_chunk_lut=routed_lut if mode_upper == "L2_RAW" else [],
                    domain=domain,
                    top_k=top_k,
                    window_size=window_size,
                    retrieval_mode=mode_upper,
                    target_doc_id=current_target,
                    ref_doc_id=ref_doc_id,
                    mmr_lambda=mmr_lambda,
                    tau_dup=tau_dup,
                    gamma=gamma,
                )
            )
            if verbose:
                routed_docs = routed_summary.get("unique_doc_ids")
                routed_ild = routed_summary.get("intra_list_distance") or 0.0
                routed_rr = routed_summary.get("redundancy_rate") or 0.0
                routed_recall = routed_summary.get("recall_at_k_50") or 0.0
                print(
                    f"[Query {idx+1}/{len(eval_pairs)}] Mode={mode_upper} | "
                    f"Unique_Docs:{routed_docs} | Recall@K{recall_k}:{routed_recall:.2f} | "
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

        if mode_upper == "COMPARE_ALL":
            triple = pipeline.run_pipeline(
                question,
                target_doc_id=current_target,
                log_summary=False,
                mode="COMPARE_ALL",
            )
            if not isinstance(triple, tuple) or len(triple) != 6:
                raise RuntimeError("internal: COMPARE_ALL must return a 6-tuple from run_pipeline")
            l2_results, l2raw_results, mmr_results, l2_summary, l2raw_summary, mmr_summary = triple
            l2_summary = dict(l2_summary)
            l2raw_summary = dict(l2raw_summary)
            mmr_summary = dict(mmr_summary)
            cfs_mmr = pipeline._last_centroid_filter_stats
            if cfs_mmr is not None:
                centroid_filter_mmr_samples.append(dict(cfs_mmr))

            expected_text = ""
            if isinstance(answers_text, str) and answers_text.strip():
                expected_text = answers_text.strip()
            elif isinstance(ref_doc_id, int) and 0 <= ref_doc_id < len(contexts):
                expected_text = contexts[ref_doc_id] or ""

            ws = max(0, int(window_size))

            def _stitch_list_tagged_all(
                docs: List[Dict], mode_tag: str
            ) -> Tuple[List[str], List[str], List[Dict]]:
                k = min(int(top_k), len(docs))
                raw_list: List[str] = []
                tagged_list: List[str] = []
                lut_list: List[Dict] = []
                for chunk_no, doc in enumerate(docs[:k], start=1):
                    raw = _trim_broken_edges(
                        _stitch_context_for_chunk(
                            doc,
                            doc_chunk_text_index,
                            window_size=ws,
                            debug_cb=(lambda msg: _append_eval_debug(msg)),
                        )
                    )
                    raw_list.append(raw)
                    tagged_list.append(
                        _format_for_llm(doc, raw, target_id_for_tag, mode=mode_tag)
                    )
                    lut_list.append({
                        "chunk_no": chunk_no,
                        "doc_id": str((doc.get("metadata") or {}).get("doc_id", "")).strip(),
                        "chunk_text": raw,
                    })
                return raw_list, tagged_list, lut_list

            l2_raw_blocks, l2_stitched, l2_lut = _stitch_list_tagged_all(l2_results, mode_tag="L2")
            l2raw_raw_blocks, l2raw_stitched, l2raw_lut = _stitch_list_tagged_all(
                l2raw_results, mode_tag="L2_RAW"
            )
            mmr_raw_blocks, mmr_stitched, mmr_lut = _stitch_list_tagged_all(mmr_results, mode_tag="MMR")

            def _recall_for_all(raw_blocks: List[str]) -> float:
                for t in raw_blocks:
                    if _ground_truth_token_coverage(expected_text, t) >= 1.0:
                        return 1.0
                return 0.0

            recall_l2 = _recall_for_all(l2_raw_blocks)
            recall_l2raw = _recall_for_all(l2raw_raw_blocks)
            recall_mmr = _recall_for_all(mmr_raw_blocks)

            rouge_l2 = _max_rouge_l_recall_over_blocks(expected_text, l2_raw_blocks)
            rouge_l2raw = _max_rouge_l_recall_over_blocks(expected_text, l2raw_raw_blocks)
            rouge_mmr = _max_rouge_l_recall_over_blocks(expected_text, mmr_raw_blocks)

            rr_l2 = _redundancy_rate_from_embeddings(l2_results, top_k, sim_threshold=0.7)
            rr_l2raw = _redundancy_rate_from_embeddings(l2raw_results, top_k, sim_threshold=0.7)
            rr_mmr = _redundancy_rate_from_embeddings(mmr_results, top_k, sim_threshold=0.7)

            l2_summary = {
                **l2_summary,
                "recall_at_k_50": recall_l2,
                "rouge_l_max_recall": rouge_l2,
                "redundancy_rate": rr_l2,
            }
            l2raw_summary = {
                **l2raw_summary,
                "recall_at_k_50": recall_l2raw,
                "rouge_l_max_recall": rouge_l2raw,
                "redundancy_rate": rr_l2raw,
            }
            k_take_m = min(int(top_k), len(mmr_results))
            _scores_mm = [
                float(x.get("score", float("nan"))) for x in mmr_results[:k_take_m]
            ]
            _arr_mm = np.asarray(_scores_mm, dtype=np.float64)
            mean_mmr_topk_c = (
                float(np.nanmean(_arr_mm)) if _arr_mm.size > 0 else float("nan")
            )
            mmr_summary = {
                **mmr_summary,
                "recall_at_k_50": recall_mmr,
                "rouge_l_max_recall": rouge_mmr,
                "redundancy_rate": rr_mmr,
                "mean_mmr_topk_score": mean_mmr_topk_c,
            }

            metrics_l2.append(l2_summary)
            metrics_l2_raw.append(l2raw_summary)
            metrics_mmr.append(mmr_summary)

            export_data.append({
                "query": question,
                "ground_truth": expected_text,
                "aspect_queries": aspect_queries,
                "l2_contexts": l2_stitched,
                "l2_raw_contexts": l2raw_stitched,
                "mmr_contexts": mmr_stitched,
                "chunk_lut": {
                    "l2": l2_lut,
                    "l2_raw": l2raw_lut,
                    "mmr": mmr_lut,
                },
            })
            export_eval_data.append(
                build_normalized_retrieval_record(
                    query_idx=idx,
                    query=question,
                    ground_truth=expected_text,
                    aspect_queries=aspect_queries,
                    l2_contexts=l2_stitched,
                    mmr_contexts=mmr_stitched,
                    l2_chunk_lut=l2_lut,
                    mmr_chunk_lut=mmr_lut,
                    domain=domain,
                    top_k=top_k,
                    window_size=window_size,
                    retrieval_mode=mode_upper,
                    target_doc_id=current_target,
                    ref_doc_id=ref_doc_id,
                    mmr_lambda=mmr_lambda,
                    tau_dup=tau_dup,
                    gamma=gamma,
                    l2_raw_contexts=l2raw_stitched,
                    l2_raw_chunk_lut=l2raw_lut,
                )
            )
            if verbose:
                print(
                    f"[Query {idx+1}/{len(eval_pairs)}] COMPARE_ALL | "
                    f"Unique_Docs(L2:{l2_summary.get('unique_doc_ids')}|"
                    f"L2-raw:{l2raw_summary.get('unique_doc_ids')}|MMR:{mmr_summary.get('unique_doc_ids')}) | "
                    f"Recall@K{recall_k}(L2:{recall_l2:.2f}|L2-raw:{recall_l2raw:.2f}|MMR:{recall_mmr:.2f}) | "
                    f"ILD(L2:{l2_summary.get('intra_list_distance') or 0:.4f}|"
                    f"L2-raw:{l2raw_summary.get('intra_list_distance') or 0:.4f}|"
                    f"MMR:{mmr_summary.get('intra_list_distance') or 0:.4f}) | "
                    f"RR(L2:{rr_l2:.4f}|L2-raw:{rr_l2raw:.4f}|MMR:{rr_mmr:.4f})"
                )
            if show_docs:
                print(f"\n[질문] {question}")
                for tag, res in (
                    ("L2", l2_results),
                    ("L2_Raw", l2raw_results),
                    ("MMR", mmr_results),
                ):
                    print(f"\n  === {tag} Top-5 ===")
                    for rank, doc in enumerate(res[:5], start=1):
                        doc_id = doc.get("chunk_id") or (doc.get("metadata") or {}).get("doc_id") or "—"
                        text = (doc.get("metadata") or {}).get("text") or ""
                        text_flat = (text[:300] + "..." if len(text) > 300 else text).replace("\n", " ")
                        print(f"  [{rank}] doc_id={doc_id} | {text_flat}")
                print("-" * 80)

            if dump_embeddings:
                out_dump = pipeline._flush_dump()
                if out_dump and verbose:
                    print(f"[dump] saved embeddings: {out_dump}")
                if dump_only_first_query:
                    break
            continue

        # COMPARE: L2 baseline + MMR proposed
        l2_results = cast(
            List[Dict[str, Any]],
            pipeline.run_pipeline(
                question,
                target_doc_id=current_target,
                log_summary=False,
                mode="L2",
            ),
        )
        l2_summary = dict(pipeline._last_summary or {})

        mmr_results = cast(
            List[Dict[str, Any]],
            pipeline.run_pipeline(
                question,
                target_doc_id=current_target,
                log_summary=False,
                mode="MMR",
            ),
        )
        cfs_mmr = pipeline._last_centroid_filter_stats
        if cfs_mmr is not None:
            centroid_filter_mmr_samples.append(dict(cfs_mmr))
        mmr_summary = dict(pipeline._last_summary or {})

        expected_text = ""
        if isinstance(answers_text, str) and answers_text.strip():
            expected_text = answers_text.strip()
        elif isinstance(ref_doc_id, int) and 0 <= ref_doc_id < len(contexts):
            expected_text = contexts[ref_doc_id] or ""

        ws = max(0, int(window_size))

        def _stitch_list_tagged(
            docs: List[Dict], mode_tag: str
        ) -> Tuple[List[str], List[str], List[Dict]]:
            k = min(int(top_k), len(docs))
            raw_list: List[str] = []
            tagged_list: List[str] = []
            lut_list: List[Dict] = []
            for chunk_no, doc in enumerate(docs[:k], start=1):
                raw = _trim_broken_edges(
                    _stitch_context_for_chunk(
                        doc,
                        doc_chunk_text_index,
                        window_size=ws,
                        debug_cb=(lambda msg: _append_eval_debug(msg)),
                    )
                )
                raw_list.append(raw)
                tagged_list.append(
                    _format_for_llm(doc, raw, target_id_for_tag, mode=mode_tag)
                )
                lut_list.append({
                    "chunk_no": chunk_no,
                    "doc_id": str((doc.get("metadata") or {}).get("doc_id", "")).strip(),
                    "chunk_text": raw,
                })
            return raw_list, tagged_list, lut_list

        l2_raw, l2_stitched, l2_lut = _stitch_list_tagged(l2_results, mode_tag="L2")
        mmr_raw, mmr_stitched, mmr_lut = _stitch_list_tagged(mmr_results, mode_tag="MMR")

        def _recall_for(raw_blocks: List[str]) -> float:
            for t in raw_blocks:
                if _ground_truth_token_coverage(expected_text, t) >= 1.0:
                    return 1.0
            return 0.0

        recall_l2 = _recall_for(l2_raw)
        recall_mmr = _recall_for(mmr_raw)

        rouge_l2 = _max_rouge_l_recall_over_blocks(expected_text, l2_raw)
        rouge_mmr = _max_rouge_l_recall_over_blocks(expected_text, mmr_raw)

        rr_l2 = _redundancy_rate_from_embeddings(l2_results, top_k, sim_threshold=0.7)
        rr_mmr = _redundancy_rate_from_embeddings(mmr_results, top_k, sim_threshold=0.7)

        l2_summary = {
            **l2_summary,
            "recall_at_k_50": recall_l2,
            "rouge_l_max_recall": rouge_l2,
            "redundancy_rate": rr_l2,
        }
        k_take_m = min(int(top_k), len(mmr_results))
        _scores_mm = [
            float(x.get("score", float("nan"))) for x in mmr_results[:k_take_m]
        ]
        _arr_mm = np.asarray(_scores_mm, dtype=np.float64)
        mean_mmr_topk_c = (
            float(np.nanmean(_arr_mm)) if _arr_mm.size > 0 else float("nan")
        )
        mmr_summary = {
            **mmr_summary,
            "recall_at_k_50": recall_mmr,
            "rouge_l_max_recall": rouge_mmr,
            "redundancy_rate": rr_mmr,
            "mean_mmr_topk_score": mean_mmr_topk_c,
        }

        metrics_l2.append(l2_summary)
        metrics_mmr.append(mmr_summary)

        def _doc_text(doc: Dict) -> str:
            return (doc.get("metadata") or {}).get("text") or doc.get("text", "") or ""

        export_data.append({
            "query": question,
            "ground_truth": expected_text,
            "aspect_queries": aspect_queries,
            "l2_contexts": l2_stitched,
            "mmr_contexts": mmr_stitched,
            "chunk_lut": {
                "l2": l2_lut,
                "mmr": mmr_lut,
            },
        })
        export_eval_data.append(
            build_normalized_retrieval_record(
                query_idx=idx,
                query=question,
                ground_truth=expected_text,
                aspect_queries=aspect_queries,
                l2_contexts=l2_stitched,
                mmr_contexts=mmr_stitched,
                l2_chunk_lut=l2_lut,
                mmr_chunk_lut=mmr_lut,
                domain=domain,
                top_k=top_k,
                window_size=window_size,
                retrieval_mode=mode_upper,
                target_doc_id=current_target,
                ref_doc_id=ref_doc_id,
                mmr_lambda=mmr_lambda,
                tau_dup=tau_dup,
                gamma=gamma,
            )
        )
        if verbose:
            print(
                f"[Query {idx+1}/{len(eval_pairs)}] COMPARE | "
                f"Unique_Docs(L2:{l2_summary.get('unique_doc_ids')}|MMR:{mmr_summary.get('unique_doc_ids')}) | "
                f"Recall@K{recall_k}(L2:{recall_l2:.2f}|MMR:{recall_mmr:.2f}) | "
                f"ILD(L2:{l2_summary.get('intra_list_distance') or 0:.4f}|"
                f"MMR:{mmr_summary.get('intra_list_distance') or 0:.4f}) | "
                f"RR(L2:{rr_l2:.4f}|MMR:{rr_mmr:.4f})"
            )
            for tag, res in (("L2", l2_results), ("MMR", mmr_results)):
                top_n = min(10, len(res))
                avg_c = (
                    sum(len(_doc_text(doc)) for doc in res[:top_n]) / float(top_n) if top_n else 0.0
                )
        if show_docs:
            print(f"\n[질문] {question}")
            for tag, res in (("L2", l2_results), ("MMR", mmr_results)):
                print(f"\n  === {tag} Top-5 ===")
                for rank, doc in enumerate(res[:5], start=1):
                    doc_id = doc.get("chunk_id") or (doc.get("metadata") or {}).get("doc_id") or "—"
                    text = (doc.get("metadata") or {}).get("text") or ""
                    text_flat = (text[:300] + "..." if len(text) > 300 else text).replace("\n", " ")
                    print(f"  [{rank}] doc_id={doc_id} | {text_flat}")
            print("-" * 80)

        if dump_embeddings:
            out_dump = pipeline._flush_dump()
            if out_dump and verbose:
                print(f"[dump] saved embeddings: {out_dump}")
            if dump_only_first_query:
                break

    if export_normalized_json:
        save_normalized_retrieval_export(export_normalized_json, export_eval_data)
        if verbose:
            print(
                f"[Export] 정규화 검색 결과 저장: {export_normalized_json} ({len(export_eval_data)} 쿼리)"
            )

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
            "rouge_l_max_recall",
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
    avg_l2_raw = _avg_metrics(metrics_l2_raw)
    avg_mmr = _avg_metrics(metrics_mmr)

    report = {
        "domain": domain,
        "num_queries": n,
        "top_k": int(top_k),
        "window_size": int(window_size),
        "avg_metrics_l2": avg_l2,
        "avg_metrics_l2_raw": avg_l2_raw,
        "avg_metrics_mmr": avg_mmr,
        # COMPARE: {query, ground_truth, l2_contexts, mmr_contexts}
        "retrieved_context": export_data,
    }

    if verbose:
        print()
        print("========== H-AIMDC 다양성 및 효율성 평가 리포트 ==========")
        print(
            f"Domain: {domain} | Queries: {n} | Top-K: {top_k} | window_size: {window_size} | "
            f"GT: max ROUGE-L recall over [SEP] spans vs each stitched chunk; "
            f"Recall@K{recall_k} pass if any chunk >= {GT_COVERAGE_THRESHOLD:.0%}"
        )
        has_l2_block = bool(metrics_l2)
        has_l2_raw_block = bool(metrics_l2_raw)
        has_mmr_block = bool(metrics_mmr)
        three_way = has_l2_block and has_l2_raw_block and has_mmr_block
        l2_raw_only = has_l2_raw_block and not has_l2_block and not has_mmr_block

        print(f"  Recall@K{recall_k} (binary, ROUGE-L gate above):")
        if three_way:
            print(
                f"    L2: {avg_l2.get('recall_at_k_50', 0):.4f}  |  "
                f"L2-raw: {avg_l2_raw.get('recall_at_k_50', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('recall_at_k_50', 0):.4f}"
            )
        elif l2_raw_only:
            print(f"    L2-raw: {avg_l2_raw.get('recall_at_k_50', 0):.4f}")
        else:
            print(
                f"    L2: {avg_l2.get('recall_at_k_50', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('recall_at_k_50', 0):.4f}"
            )
        print(
            "  ROUGE-L max recall (avg; per query: max over [SEP] refs, best stitched chunk; stemmed):"
        )
        if three_way:
            print(
                f"    L2: {avg_l2.get('rouge_l_max_recall', 0):.4f}  |  "
                f"L2-raw: {avg_l2_raw.get('rouge_l_max_recall', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('rouge_l_max_recall', 0):.4f}"
            )
        elif l2_raw_only:
            print(f"    L2-raw: {avg_l2_raw.get('rouge_l_max_recall', 0):.4f}")
        else:
            print(
                f"    L2: {avg_l2.get('rouge_l_max_recall', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('rouge_l_max_recall', 0):.4f}"
            )
        print("  ILD (Intra-List Distance):")
        if three_way:
            print(
                f"    L2: {avg_l2.get('intra_list_distance', 0):.4f}  |  "
                f"L2-raw: {avg_l2_raw.get('intra_list_distance', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('intra_list_distance', 0):.4f}"
            )
        elif l2_raw_only:
            print(f"    L2-raw: {avg_l2_raw.get('intra_list_distance', 0):.4f}")
        else:
            print(
                f"    L2: {avg_l2.get('intra_list_distance', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('intra_list_distance', 0):.4f}"
            )
        print("  RR (Redundancy Rate, cosine > 0.7):")
        if three_way:
            print(
                f"    L2: {avg_l2.get('redundancy_rate', 0):.4f}  |  "
                f"L2-raw: {avg_l2_raw.get('redundancy_rate', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('redundancy_rate', 0):.4f}"
            )
        elif l2_raw_only:
            print(f"    L2-raw: {avg_l2_raw.get('redundancy_rate', 0):.4f}")
        else:
            print(
                f"    L2: {avg_l2.get('redundancy_rate', 0):.4f}  |  "
                f"MMR: {avg_mmr.get('redundancy_rate', 0):.4f}"
            )
        print("  Search Breadth (탐색한 서랍 수):")
        l2_breadth = avg_l2.get("unique_centroids_selected", 0)
        l2raw_breadth = avg_l2_raw.get("unique_centroids_selected", 0)
        mmr_breadth = avg_mmr.get("unique_centroids_selected", 0)
        if three_way:
            print(
                f"    L2 평균: {l2_breadth:.2f}  |  L2-raw 평균: {l2raw_breadth:.2f}  |  MMR 평균: {mmr_breadth:.2f}"
            )
        elif l2_raw_only:
            print(f"    L2-raw 평균: {l2raw_breadth:.2f}")
        else:
            print(
                f"    L2 평균: {l2_breadth:.2f}  |  MMR 평균: {mmr_breadth:.2f}"
            )
        if has_l2_raw_block and has_l2_block:
            recall_l2r = float(avg_l2_raw.get("recall_at_k_50", 0) or 0.0)
            recall_l2p = float(avg_l2.get("recall_at_k_50", 0) or 0.0)
            pq_cost_pp = (recall_l2r - recall_l2p) * 100.0
            print(
                "  PQ Quantization Cost: L2-raw → L2 shows Recall loss of "
                f"{pq_cost_pp:.2f}%p from PQ reconstruction."
            )
        if not three_way and not l2_raw_only and mmr_breadth > l2_breadth:
            print(
                f"  Enhanced Exploratory Power: MMR breadth is higher than L2 ({mmr_breadth:.2f} > {l2_breadth:.2f})"
            )
        if has_l2_block and has_mmr_block:
            l2_rr_avg = avg_l2.get("redundancy_rate", 0)
            mmr_rr_avg = avg_mmr.get("redundancy_rate", 0)
            if mmr_rr_avg < l2_rr_avg:
                print(
                    "  Efficiency Gain: MMR의 RR이 L2보다 낮아 LLM 입력 문맥의 정보 밀도가 높아질 수 있습니다 "
                    f"({mmr_rr_avg:.4f} < {l2_rr_avg:.4f})."
                )
            else:
                print(
                    "  Efficiency Note: RR 개선이 제한적입니다. lower RR일수록 LLM 정보 밀도가 높아집니다."
                )
        if centroid_filter_mmr_samples:
            n_cf = len(centroid_filter_mmr_samples)
            avg_total = sum(s["total"] for s in centroid_filter_mmr_samples) / n_cf
            avg_loaded = sum(s["loaded"] for s in centroid_filter_mmr_samples) / n_cf
            avg_missed = sum(s["missed"] for s in centroid_filter_mmr_samples) / n_cf
            cov_pcts = [
                (s["loaded"] / s["total"] * 100.0) if s.get("total", 0) > 0 else 0.0
                for s in centroid_filter_mmr_samples
            ]
            miss_pcts = [
                (s["missed"] / s["total"] * 100.0) if s.get("total", 0) > 0 else 0.0
                for s in centroid_filter_mmr_samples
            ]
            avg_coverage = sum(cov_pcts) / len(cov_pcts) if cov_pcts else 0.0
            avg_miss_rate = sum(miss_pcts) / len(miss_pcts) if miss_pcts else 0.0
            print("  Centroid Filter Coverage:")
            print(
                f"    avg_total: {avg_total:.1f}, avg_loaded: {avg_loaded:.1f}, "
                f"avg_missed: {avg_missed:.1f}"
            )
            print(
                f"    avg_coverage: {avg_coverage:.1f}%, avg_miss_rate: {avg_miss_rate:.1f}%"
            )
            avg_doc_c = (
                sum(s.get("doc_centroids", 0) for s in centroid_filter_mmr_samples)
                / n_cf
            )
            avg_hit_c = (
                sum(s.get("hit_centroids", 0) for s in centroid_filter_mmr_samples)
                / n_cf
            )
            print(
                f"    avg_doc_centroids: {avg_doc_c:.1f}, avg_hit_centroids: {avg_hit_c:.1f}"
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
        default=10,
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
    parser.add_argument("--show_docs", action="store_true", help="검색된 Top-K 문서의 실제 텍스트 내용을 터미널에 출력")
    parser.add_argument(
        "--export_normalized_json",
        type=str,
        default=None,
        help="정규화된 평가용 검색 결과(JSON list) 저장 경로",
    )
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
        default=0.5,
        help="Fraction of top_k chunks reserved as pure L2 anchors before MMR reranks the rest (clamped to [0, 1]).",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="COMPARE",
        choices=["L2", "L2_RAW", "MMR", "COMPARE", "COMPARE_ALL"],
        help=(
            "L2_RAW: raw-embedding L2 (PQ-free upper bound). "
            "COMPARE_ALL: three-way L2 + L2_RAW + MMR. "
            "COMPARE: L2 vs MMR only."
        ),
    )
    parser.add_argument(
        "--dump_embeddings",
        action="store_true",
        help="Dump query/candidate/MMR embeddings to NPZ for device modeling.",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default="./emb_dump",
        help="Directory for embedding dump outputs.",
    )
    parser.add_argument(
        "--dump_tag",
        type=str,
        default="q0",
        help="Prefix tag for embedding dump files.",
    )
    parser.add_argument(
        "--dump_only_first_query",
        action="store_true",
        help="Flush dump after first evaluated query and stop loop early.",
    )
    parser.add_argument(
        "--min_mmr_score",
        type=float,
        default=None,
        help=(
            "If set, enables streaming coverage early-stop inside MMR and re-query rounds "
            "(see max_requery_rounds). None keeps legacy single-pass behavior."
        ),
    )
    parser.add_argument(
        "--max_requery_rounds",
        type=int,
        default=3,
        help="Max extra candidate-fetch rounds when coverage is not satisfied (only with --min_mmr_score).",
    )
    parser.add_argument(
        "--no_centroid_filter",
        action="store_true",
        help="centroid 필터링 비활성화 (target_doc 전체 청크 로드, 원본 동작).",
    )
    parser.add_argument(
        "--use_device_distance",
        action="store_true",
        help="MMR pair distances use analytical AIMDC device model (JSON fit; no Docker).",
    )
    parser.add_argument(
        "--device_fit_path",
        type=str,
        default="",
        help="Path to igzo_fit.json (default: h_aimdc_sim/igzo_fit.json when flag set).",
    )
    parser.add_argument(
        "--device_scaler_path",
        type=str,
        default=None,
        help="Scaler JSON path (default: <fit_stem>.scaler.json; fitted from raw_embeddings if missing).",
    )
    parser.add_argument(
        "--device_reduce",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="Reduction over embedding dims for device proxy.",
    )
    parser.add_argument(
        "--device_use_log",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use log10(I_bit+eps) per dim (default: true).",
    )
    parser.add_argument(
        "--device_v_ml",
        type=float,
        default=0.0,
        help="Read-line V_ML offset (V) for analytical device distance.",
    )
    parser.add_argument(
        "--noise_config",
        type=str,
        default=None,
        help="JSON file with NoiseModel fields (default: deterministic).",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=None,
        help="Override NoiseModel.rng_seed from JSON.",
    )
    parser.add_argument(
        "--device_scale_k_path",
        type=str,
        default=None,
        help=(
            "Path to scale_k JSON from scripts/calibrate_scale_k.py. "
            "When set, AIMDC distances are mapped to software-L2 scale for τ/MMR score. "
            "Raw values remain in dumps."
        ),
    )
    parser.add_argument(
        "--device_scale_k",
        type=float,
        default=None,
        help="Override scale_k with a direct float (takes precedence over --device_scale_k_path).",
    )
    parser.add_argument(
        "--use_centroid_device_distance",
        action="store_true",
        help="step3 cluster centroid L2 uses analytical NVM (FeFET) device model.",
    )
    parser.add_argument(
        "--centroid_fit_path",
        type=str,
        default="",
        help="Path to NVM v3 fit JSON (default: h_aimdc_sim/igzo_fit_v3.json when flag set).",
    )
    parser.add_argument(
        "--centroid_scaler_path",
        type=str,
        default=None,
        help="Scaler JSON for NVM (default: <fit_stem>.scaler.json; auto-fit from raw_embeddings if missing).",
    )
    parser.add_argument(
        "--centroid_reduce",
        type=str,
        default="linear_sum",
        choices=["linear_sum", "legacy_sum", "legacy_mean", "sum", "mean"],
        help="Reduction over embedding dims for NVM centroid distance "
             "(default: linear_sum = hardware ML current sum).",
    )
    parser.add_argument(
        "--centroid_use_log",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use log10(I_bit+eps) per dim for NVM (default: false; ignored for linear_sum).",
    )
    parser.add_argument(
        "--centroid_v_ml",
        type=float,
        default=0.0,
        help="Read-line V_ML offset (V) for NVM analytical model.",
    )
    parser.add_argument(
        "--centroid_noise_config",
        type=str,
        default=None,
        help="JSON file with NVM NoiseModel fields (default: deterministic).",
    )
    parser.add_argument(
        "--centroid_noise_seed",
        type=int,
        default=None,
        help="Override NVM NoiseModel.rng_seed.",
    )
    parser.add_argument(
        "--tau_dup_calibrated_for",
        type=str,
        default=None,
        help="Set to 'fefet_linear_sum' once tau_dup has been re-tuned for the FeFET path.",
    )
    parser.add_argument(
        "--centroid_inverter_vtc_path",
        type=str,
        default=None,
        help="Path to inverter VTC npz for column-shared inverter modeling "
             "(Phase 1.5). When omitted, NvmDeviceDistance falls back to ideal -V_A.",
    )
    # Legacy (no coverage gate / no re-query):
    #   python online_accelerator_device.py --mode eval --num_queries 1
    # Coverage + re-query:
    #   python online_accelerator_device.py --mode eval --num_queries 1 \
    #       --min_mmr_score -0.5 --max_requery_rounds 3
    # Example:
    # python online_accelerator_device.py --mode eval --num_queries 1 \
    #     --dump_embeddings --dump_tag q0 --dump_dir ./emb_dump --dump_only_first_query
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
            export_normalized_json=args.export_normalized_json,
            target_doc_id=args.target_doc_id,
            retrieval_mode=args.retrieval_mode,
            dump_embeddings=args.dump_embeddings,
            dump_dir=args.dump_dir,
            dump_tag=args.dump_tag,
            dump_only_first_query=args.dump_only_first_query,
            min_mmr_score=args.min_mmr_score,
            max_requery_rounds=args.max_requery_rounds,
            use_centroid_filter=not args.no_centroid_filter,
            use_device_distance=bool(args.use_device_distance),
            device_fit_path=str(args.device_fit_path or ""),
            device_scaler_path=args.device_scaler_path,
            device_reduce=str(args.device_reduce),
            device_use_log=bool(args.device_use_log),
            device_v_ml=float(args.device_v_ml),
            noise_config=args.noise_config,
            noise_seed=args.noise_seed,
            device_scale_k_path=args.device_scale_k_path,
            device_scale_k=args.device_scale_k,
            use_centroid_device_distance=bool(args.use_centroid_device_distance),
            centroid_fit_path=str(args.centroid_fit_path or ""),
            centroid_scaler_path=args.centroid_scaler_path,
            centroid_reduce=str(args.centroid_reduce),
            centroid_use_log=bool(args.centroid_use_log),
            centroid_v_ml=float(args.centroid_v_ml),
            centroid_noise_config=args.centroid_noise_config,
            centroid_noise_seed=args.centroid_noise_seed,
            centroid_inverter_vtc_path=args.centroid_inverter_vtc_path,
            tau_dup_calibrated_for=args.tau_dup_calibrated_for,
        )