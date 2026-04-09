#!/usr/bin/env python3
"""
저장된 오프라인 IVF 산출물(centroids.npy, posting_lists.pkl, metadata_db.pkl, config.pkl)을 사용해
특정 centroid에 할당된 passage 텍스트를 임베딩 파이프라인과 동일한 HuggingFace 토크나이저로
서브워드를 병합한 뒤 단어 빈도를 구하고 Word Cloud로 저장한다.

참고: IVF는 \"토큰 ID\"가 아니라 청크 벡터를 centroid에 할당한다. 본 스크립트는 해당 청크의
원문(`metadata_db`의 `text`)을 수집한 뒤, config.pkl의 `model_name`으로 AutoTokenizer를
불러 단어 단위 빈도를 계산한다 (off_line_prepper / SentenceTransformer 경로와 정합).

필요: pip install wordcloud matplotlib numpy transformers pillow
선택: pip install nltk  (영어 불용어; 최초 1회 nltk.download('stopwords') 필요할 수 있음)

예:
  python centroid_wordcloud_vis.py --offline_dir ./offline_data_cuad --centroid_ids 0 3 7
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import numpy as np

# ----- 추가로 제외할 토큰/단어 (소문자 기준 비교) -----
_EXTRA_BLOCKLIST = {
    "redacted_pad",
    "<redacted_pad>",
    "[cls]",
    "[sep]",
    "[pad]",
    "<pad>",
    "",
    "[UNK]",
    "[unk]",
    "cls",
    "sep",
    "pad",
}

# CUAD / 계약서에서 의미 분석에 거의 기여하지 않는 기능어·격식어 (일반 불용어와 별도)
_LEGAL_CONTRACT_STOPWORDS = frozenset(
    {
        "shall",
        "may",
        "must",
        "will",
        "would",
        "should",
        "could",
        "can",
        "might",
        "hereby",
        "hereto",
        "herein",
        "hereof",
        "hereafter",
        "hereinafter",
        "hereon",
        "herewith",
        "thereof",
        "therein",
        "thereon",
        "thereto",
        "thereby",
        "thereafter",
        "therefor",
        "therefrom",
        "therewith",
        "whereas",
        "whereof",
        "whereby",
        "wherein",
        "witnesseth",
        "aforesaid",
        "aforementioned",
        "abovementioned",
        "said",
        "undersigned",
        "party",
        "parties",
        "agreement",
        "contract",
        "article",
        "section",
        "clause",
        "exhibit",
        "schedule",
        "appendix",
    }
)


def _nltk_english_stopwords() -> Set[str]:
    """nltk.corpus.stopwords (영어). 코퍼스 없으면 자동 다운로드 1회 시도."""
    try:
        from nltk.corpus import stopwords as nltk_sw

        return set(nltk_sw.words("english"))
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords as nltk_sw

            return set(nltk_sw.words("english"))
        except Exception:
            return set()
    except Exception:
        return set()


def build_combined_stopwords(
    *,
    use_wordcloud_default: bool = True,
    use_nltk: bool = True,
    use_legal: bool = True,
) -> Set[str]:
    """
    wordcloud.STOPWORDS + (선택) NLTK 영어 불용어 + 법률 계약 기능어.
    모두 소문자로 통일해 단어 빈도 집계 전 필터에 사용한다.
    """
    combined: Set[str] = set()

    if use_wordcloud_default:
        try:
            from wordcloud import STOPWORDS as WC_STOP

            combined.update(str(w).lower() for w in WC_STOP)
        except ImportError:
            pass

    if use_nltk:
        combined.update(_nltk_english_stopwords())

    if use_legal:
        combined.update(_LEGAL_CONTRACT_STOPWORDS)

    return combined


def load_offline_ivf_bundle(offline_dir: str) -> tuple:
    """centroids.npy, posting_lists.pkl, metadata_db.pkl, config.pkl 로드."""
    d = os.path.abspath(offline_dir)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"offline_dir 없음: {d}")

    cpath = os.path.join(d, "centroids.npy")
    pl_path = os.path.join(d, "posting_lists.pkl")
    meta_path = os.path.join(d, "metadata_db.pkl")
    cfg_path = os.path.join(d, "config.pkl")

    for p in (cpath, pl_path, meta_path, cfg_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"필수 파일 없음: {p}")

    centroids = np.load(cpath)
    with open(pl_path, "rb") as f:
        posting_lists: Dict[Any, List[Dict[str, Any]]] = pickle.load(f)
    with open(meta_path, "rb") as f:
        metadata_db: Dict[str, Dict[str, Any]] = pickle.load(f)
    with open(cfg_path, "rb") as f:
        config: Dict[str, Any] = pickle.load(f)

    return centroids, posting_lists, metadata_db, config


def resolve_model_name(config: Dict[str, Any], override: Optional[str]) -> str:
    if override and override.strip():
        return override.strip()
    return (
        config.get("embedding_model_name")
        or config.get("model_name")
        or "sentence-transformers/all-mpnet-base-v2"
    )


def load_tokenizer(model_name: str):
    """프로젝트 임베딩과 동일 계열 HF 토크나이저 (BERT/MPNet/RoBERTa 등)."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        try:
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(model_name)
            tok = getattr(st, "tokenizer", None)
            if tok is not None:
                return tok
        except Exception:
            pass
        raise RuntimeError(
            f"토크나이저 로드 실패: {model_name!r} — transformers 또는 sentence-transformers 확인.\n"
            f"원인: {e}"
        ) from e


def _build_exclusion_set(tokenizer: Any) -> Set[str]:
    s: Set[str] = set(_EXTRA_BLOCKLIST)
    for attr in ("all_special_tokens", "special_tokens_map"):
        raw = getattr(tokenizer, attr, None)
        if raw is None:
            continue
        if isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, str):
                    s.add(v.lower())
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, str):
                            s.add(x.lower())
        elif isinstance(raw, (list, tuple)):
            for x in raw:
                if isinstance(x, str):
                    s.add(x.lower())
    # added_tokens 내 키
    am = getattr(tokenizer, "added_tokens_encoder", None) or {}
    for k in am:
        if isinstance(k, str):
            s.add(k.lower())
    return s


def texts_for_centroid(
    posting_lists: Dict[Any, List[Dict[str, Any]]],
    metadata_db: Dict[str, Dict[str, Any]],
    centroid_id: int,
) -> List[str]:
    """해당 centroid posting의 chunk_id로 metadata_db에서 본문 수집."""
    # 키가 int 또는 str로 섞일 수 있음
    entries = posting_lists.get(centroid_id)
    if entries is None:
        entries = posting_lists.get(str(centroid_id), [])
    out: List[str] = []
    for ent in entries or []:
        cid = ent.get("chunk_id")
        if cid is None:
            continue
        meta = metadata_db.get(str(cid)) or metadata_db.get(cid)
        if not meta:
            continue
        t = (meta.get("text") or "").strip()
        if t:
            out.append(t)
    return out


def word_frequencies_from_texts(
    texts: List[str],
    tokenizer: Any,
    *,
    max_length: int = 512,
    exclude_unk: bool = True,
    stopwords: Optional[Set[str]] = None,
) -> Counter:
    """
    passage들을 토크나이즈한 뒤 특수 토큰을 제거하고 convert_tokens_to_string으로
    복원한 문자열에서 단어(유니코드 \\w + 하이픈/아포스트로피) 빈도를 센다.
    stopwords: 빈도에 넣기 전에 완전히 제외(일반·NLTK·법률 불용어 등).
    """
    exclude_lower = _build_exclusion_set(tokenizer)
    sw = stopwords if stopwords is not None else set()
    unk_id = getattr(tokenizer, "unk_token_id", None)

    freq: Counter = Counter()
    word_re = re.compile(r"[\w'-]+", flags=re.UNICODE)

    for text in texts:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        ids = enc["input_ids"]
        if not ids:
            continue
        filtered: List[int] = []
        for tid in ids:
            if tid in getattr(tokenizer, "all_special_ids", ()):
                continue
            if exclude_unk and unk_id is not None and tid == unk_id:
                continue
            filtered.append(tid)

        if not filtered:
            continue
        tokens = tokenizer.convert_ids_to_tokens(filtered)
        # 토큰 문자열이 블록리스트에 있으면 제거 (일부 모델)
        tokens = [t for t in tokens if str(t).lower() not in exclude_lower]
        if not tokens:
            continue
        restored = tokenizer.convert_tokens_to_string(tokens)
        for w in word_re.findall(restored.lower()):
            if len(w) < 2:
                continue
            if w in exclude_lower:
                continue
            if w in sw:
                continue
            freq[w] += 1

    return freq


def render_wordcloud(
    frequencies: Dict[str, int],
    out_path: str,
    *,
    width: int = 1200,
    height: int = 800,
    background_color: str = "white",
    stopwords: Optional[Set[str]] = None,
) -> None:
    if not frequencies:
        raise ValueError("빈 빈도 딕셔너리 — Word Cloud를 만들 수 없습니다.")
    try:
        from wordcloud import WordCloud
    except ImportError as e:
        raise ImportError("pip install wordcloud") from e

    import matplotlib.pyplot as plt

    # 빈도 단계에서 이미 걸러졌으나, 라이브러리 단계에서도 동일 세트를 넘겨 이중 방어
    sw = stopwords if stopwords is not None else set()

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        prefer_horizontal=0.85,
        colormap="viridis",
        max_words=200,
        stopwords=sw,
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def run_for_centroids(
    offline_dir: str,
    centroid_ids: List[int],
    output_dir: str,
    *,
    model_name_override: Optional[str] = None,
    max_length: int = 512,
    use_wordcloud_stopwords: bool = True,
    use_nltk_stopwords: bool = True,
    use_legal_stopwords: bool = True,
) -> None:
    centroids, posting_lists, metadata_db, config = load_offline_ivf_bundle(offline_dir)
    model_name = resolve_model_name(config, model_name_override)
    print(f"[tokenizer] {model_name}", flush=True)
    tokenizer = load_tokenizer(model_name)

    stopwords = build_combined_stopwords(
        use_wordcloud_default=use_wordcloud_stopwords,
        use_nltk=use_nltk_stopwords,
        use_legal=use_legal_stopwords,
    )
    print(
        f"[stopwords] 합집합 크기={len(stopwords)} "
        f"(wordcloud={use_wordcloud_stopwords}, nltk={use_nltk_stopwords}, legal={use_legal_stopwords})",
        flush=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    n_cent = int(centroids.shape[0]) if centroids.ndim == 2 else 0

    for cid in centroid_ids:
        cid = int(cid)
        if n_cent and not (0 <= cid < n_cent):
            print(
                f"[경고] centroid_id={cid} 가 centroids.npy 행 범위 [0,{n_cent - 1}] 밖입니다. 계속 시도합니다.",
                file=sys.stderr,
            )

        texts = texts_for_centroid(posting_lists, metadata_db, cid)
        if not texts:
            print(f"[건너뜀] centroid {cid}: 할당된 청크/텍스트 없음", flush=True)
            continue

        freq = word_frequencies_from_texts(
            texts,
            tokenizer,
            max_length=max_length,
            stopwords=stopwords,
        )
        if not freq:
            print(f"[건너뜀] centroid {cid}: 유효 단어 빈도 없음", flush=True)
            continue

        out_path = os.path.join(output_dir, f"centroid_{cid}_wordcloud.png")
        render_wordcloud(dict(freq), out_path, stopwords=stopwords)
        print(
            f"[저장] centroid {cid}: passages={len(texts)}, unique_words={len(freq)} → {out_path}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IVF centroid별 passage 단어 Word Cloud (centroids.npy + posting_lists + metadata_db + config)"
    )
    parser.add_argument(
        "--offline_dir",
        type=str,
        required=True,
        help="offline_data_* 폴더 (centroids.npy, posting_lists.pkl, metadata_db.pkl, config.pkl)",
    )
    parser.add_argument(
        "--centroid_ids",
        type=int,
        nargs="+",
        required=True,
        help="시각화할 centroid ID (여러 개 가능)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="PNG 저장 디렉터리 (기본: 현재 디렉터리)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="토크나이저 강제 지정 (기본: config.pkl의 model_name)",
    )
    parser.add_argument("--max_length", type=int, default=512, help="토크나이저 truncation 길이")
    parser.add_argument(
        "--no_wordcloud_stopwords",
        action="store_true",
        help="wordcloud 기본 STOPWORDS 제외",
    )
    parser.add_argument(
        "--no_nltk_stopwords",
        action="store_true",
        help="NLTK 영어 불용어 제외",
    )
    parser.add_argument(
        "--no_legal_stopwords",
        action="store_true",
        help="법률 계약용 커스텀 불용어 제외",
    )
    args = parser.parse_args()

    run_for_centroids(
        args.offline_dir,
        args.centroid_ids,
        args.output_dir,
        model_name_override=args.model_name,
        max_length=args.max_length,
        use_wordcloud_stopwords=not args.no_wordcloud_stopwords,
        use_nltk_stopwords=not args.no_nltk_stopwords,
        use_legal_stopwords=not args.no_legal_stopwords,
    )


if __name__ == "__main__":
    main()
