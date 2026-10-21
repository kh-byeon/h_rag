"""
Run run_mmr_evaluation three ways (software / analytical / noisy device) and
emit a markdown table: Recall@K, MRR, mean MMR score, pair-distance p5/p50/p95,
wallclock per query.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from online_accelerator_device import (  # noqa: E402
    load_hw_state_from_dir,
    run_mmr_evaluation,
)
from aimdc_device_distance import DeviceDistance, NoiseModel  # noqa: E402


def _ensure_fit_json(path: Path) -> None:
    if path.is_file():
        return
    from h_aimdc_sim.aimdc_sim_core import (  # noqa: PLC0415
        compute_fit_residual_rmse,
        fit_igzo_model,
        save_fit_json,
    )

    csv_p = _REPO / "h_aimdc_sim" / "igzo_measure.csv"
    if not csv_p.is_file():
        raise FileNotFoundError(f"Need {csv_p} to build {path}")
    popt, _expr = fit_igzo_model(str(csv_p))
    rmse, vds = compute_fit_residual_rmse(str(csv_p), popt)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fit_json(
        popt,
        str(path),
        source_csv="igzo_measure.csv",
        vds_fit=vds,
        fit_rmse_log10=rmse,
    )


def _pair_dist_percentiles(
    *,
    use_device: bool,
    fit_path: Path,
    scaler_path: Path | None,
    noise: NoiseModel | None,
) -> tuple[float, float, float]:
    offline = _REPO / "offline_data_cuad"
    hw = load_hw_state_from_dir(str(offline))
    raw = hw.get("raw_embeddings")
    if raw is None:
        return float("nan"), float("nan"), float("nan")
    emb = np.asarray(raw, dtype=np.float64)
    rng = np.random.default_rng(42)
    n = min(200, len(emb))
    idx = rng.choice(len(emb), size=(n, 2), replace=True)
    d_list: list[float] = []
    if use_device:
        _ensure_fit_json(fit_path)
        dd = DeviceDistance(
            str(fit_path),
            scaler_path=str(scaler_path) if scaler_path and scaler_path.is_file() else None,
            noise=noise or NoiseModel(),
        )
        if dd._x_lo is None:
            dd.fit_scaler(emb)
            if scaler_path:
                dd.save_scaler(str(scaler_path))
        for i in range(n):
            a, b = int(idx[i, 0]), int(idx[i, 1])
            d_list.append(float(dd.distance(emb[a], emb[b])))
    else:
        for i in range(n):
            a, b = int(idx[i, 0]), int(idx[i, 1])
            d_list.append(float(np.linalg.norm(emb[a] - emb[b])))
    arr = np.asarray(d_list, dtype=np.float64)
    return (
        float(np.percentile(arr, 5)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
    )


def _mrr_from_report(report: dict[str, object]) -> float:
    ctxs = report.get("retrieved_context") or []
    ranks: list[float] = []
    for row in ctxs:
        if not isinstance(row, dict):
            continue
        gt = str(row.get("ground_truth") or "").strip()
        if not gt:
            ranks.append(0.0)
            continue
        mmr_ctx = row.get("mmr_contexts")
        if not isinstance(mmr_ctx, list):
            ranks.append(0.0)
            continue
        hit = 0.0
        for r, block in enumerate(mmr_ctx, start=1):
            if not isinstance(block, str):
                continue
            if gt and gt[: min(80, len(gt))] in block.replace("\n", " "):
                hit = 1.0 / float(r)
                break
        ranks.append(hit)
    return float(np.mean(np.asarray(ranks, dtype=np.float64))) if ranks else 0.0


def _run_row(
    label: str,
    *,
    use_device: bool,
    fit_path: Path,
    scaler_path: Path | None,
    noise: NoiseModel | None,
    noise_json: Path | None,
) -> dict[str, object]:
    t0 = time.perf_counter()
    out = run_mmr_evaluation(
        "cuad",
        offline_data_dir=str(_REPO / "offline_data_cuad"),
        num_queries=20,
        max_contexts=None,
        top_k=5,
        top_r=128,
        retrieval_mode="MMR",
        verbose=False,
        use_device_distance=use_device,
        device_fit_path=str(fit_path) if use_device else "",
        device_scaler_path=str(scaler_path) if scaler_path else None,
        device_reduce="sum",
        device_use_log=True,
        device_v_ml=0.0,
        noise_config=str(noise_json) if noise_json else None,
        noise_seed=42 if use_device else None,
    )
    wall = time.perf_counter() - t0
    p5, p50, p95 = _pair_dist_percentiles(
        use_device=use_device,
        fit_path=fit_path,
        scaler_path=scaler_path,
        noise=noise,
    )
    return {"label": label, "out": out, "wall_s": wall, "p5": p5, "p50": p50, "p95": p95}


def main() -> None:
    fit_path = _REPO / "h_aimdc_sim" / "igzo_fit.json"
    scaler_path = _REPO / "h_aimdc_sim" / "igzo_fit.scaler.json"
    _ensure_fit_json(fit_path)
    hw = load_hw_state_from_dir(str(_REPO / "offline_data_cuad"))
    raw = hw.get("raw_embeddings")
    if raw is not None and not scaler_path.is_file():
        dd0 = DeviceDistance(str(fit_path), scaler_path=None)
        dd0.fit_scaler(np.asarray(raw, dtype=np.float64))
        dd0.save_scaler(str(scaler_path))

    noisy = {"sigma_vth": 0.03, "sigma_alpha": 0.03, "adc_bits": 8, "rng_seed": 42}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(noisy, tmp)
        noisy_path = Path(tmp.name)

    rows: list[dict[str, object]] = []
    try:
        rows.append(
            _run_row(
                "software",
                use_device=False,
                fit_path=fit_path,
                scaler_path=scaler_path,
                noise=None,
                noise_json=None,
            )
        )
        rows.append(
            _run_row(
                "analytical",
                use_device=True,
                fit_path=fit_path,
                scaler_path=scaler_path,
                noise=NoiseModel(rng_seed=42),
                noise_json=None,
            )
        )
        rows.append(
            _run_row(
                "noisy device",
                use_device=True,
                fit_path=fit_path,
                scaler_path=scaler_path,
                noise=None,
                noise_json=noisy_path,
            )
        )
    finally:
        try:
            noisy_path.unlink(missing_ok=True)
        except OSError:
            pass

    lines = [
        "# MMR: software vs analytical vs noisy device",
        "",
        "| Row | Recall@K | MRR | mean MMR score | pair p5 | pair p50 | pair p95 | s/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        out = r["out"]  # type: ignore[assignment]
        mmr = out.get("avg_metrics_mmr") or {}
        nq = max(1, int(out.get("num_queries", 20)))
        mrr = _mrr_from_report(out)
        mean_score = float(mmr.get("mean_mmr_topk_score", float("nan")))
        lines.append(
            "| {lab} | {rk:.4f} | {mrr:.4f} | {ms:.4f} | {p5:.4f} | {p50:.4f} | {p95:.4f} | {ws:.3f} |".format(
                lab=str(r["label"]),
                rk=float(mmr.get("recall_at_k_50", 0.0)),
                mrr=mrr,
                ms=mean_score,
                p5=float(r["p5"]),  # type: ignore[arg-type]
                p50=float(r["p50"]),  # type: ignore[arg-type]
                p95=float(r["p95"]),  # type: ignore[arg-type]
                ws=float(r["wall_s"]) / nq,  # type: ignore[arg-type]
            )
        )

    md = "\n".join(lines)
    print(md)
    out_path = _REPO / "mmr_device_vs_soft.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
