"""Device-tolerance probe for the FeFET centroid tier (H-AIMDC).

Replays step 3 (IVF coarse search on the FeFET device model), builds the cross-scope
candidate pool from the posting lists of the nprobe centroids, ranks inside the pool
(L2 arm: exact L2; MMR arm: software MMR proxy, lambda = 0.7, window 50), and scores
supporting-passage recall@5 against gold_support. Perturbations are applied to the
STORED centroid voltages V_W only.

Outputs (in --out):
  tol_sweep.csv          one row per configuration
  tol_summary.json       baselines, fidelity, crossings
  per_query_<tag>.npz    hit/recall/full per query for every config (for pairing)
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aimdc_nvm_device_distance import NvmDeviceDistance  # noqa: E402

NPROBE = 32
TOPK = 5
MMR_LAMBDA = 0.7
MMR_WINDOW = 50


# ----------------------------------------------------------------------------- data
def load_index(root):
    d = os.path.join(root, "offline_data_hotpot_q1000")
    E = np.load(os.path.join(d, "raw_embeddings.npy")).astype(np.float64)
    R = np.load(os.path.join(d, "opq_rotation.npy")).astype(np.float64)
    C = np.load(os.path.join(d, "centroids.npy")).astype(np.float64)
    md = pickle.load(open(os.path.join(d, "metadata_db.pkl"), "rb"))
    order = pickle.load(open(os.path.join(d, "chunk_ids_order.pkl"), "rb"))
    pl = pickle.load(open(os.path.join(d, "posting_lists.pkl"), "rb"))
    X = E @ R                                   # chunk embeddings in the centroid (OPQ) space
    doc_of_chunk = np.array([int(md[c]["doc_id"]) for c in order])
    cid_of_chunk = np.array([int(md[c]["centroid_id"]) for c in order])
    vid = {c: i for i, c in enumerate(order)}
    # centroid -> array of chunk row indices (from posting lists; must agree with metadata)
    members = {}
    for cid, entries in pl.items():
        rows = np.array([vid[e["chunk_id"]] for e in entries], dtype=np.int64)
        members[int(cid)] = rows
    # sanity: posting list membership == metadata centroid_id
    for cid, rows in members.items():
        assert np.all(cid_of_chunk[rows] == cid), f"posting list / metadata mismatch at centroid {cid}"
    q = json.load(open(os.path.join(root, "processed_hotpot_q1000", "hotpot_queries.json"), encoding="utf-8"))
    gold = [np.array(sorted(set(int(g) for g in x["gold_support"]))) for x in q]
    return dict(X=X, C=C, R=R, doc_of_chunk=doc_of_chunk, cid_of_chunk=cid_of_chunk,
                members=members, gold=gold, queries=[x["query"] for x in q])


# ------------------------------------------------------------------- device model
class FefetCentroidTier:
    """Wraps NvmDeviceDistance and applies stored-side perturbations."""

    def __init__(self, fit_json, scaler_json, v_ml=1.4):
        self.dd = NvmDeviceDistance(fit_json, scaler_path=scaler_json, reduce="linear_sum", v_ml=v_ml)
        self.window = (self.dd.v_min, self.dd.v_max)

    def distances(self, Q, C, *, offset_v=0.0, gain=1.0, cell_offset=None, cell_gain=None, batch=100):
        """Return (nq, nc) match-line currents (larger = farther)."""
        dd = self.dd
        aa = dd._scale(Q)
        bb = dd._scale(C)
        bb = gain * bb + offset_v
        if cell_offset is not None:
            bb = bb + cell_offset
        out = np.empty((Q.shape[0], C.shape[0]))
        for s in range(0, Q.shape[0], batch):
            i_all = dd._bit_current(aa[s:s + batch, None, :], bb[None, :, :])   # (b, nc, D)
            if cell_gain is not None:
                i_all = i_all * cell_gain[None, :, :]
            out[s:s + batch] = i_all.sum(-1)
        return out


def software_l2(Q, C):
    return (Q * Q).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * Q @ C.T


# --------------------------------------------------------------------- retrieval
def probe_sets(dist, nprobe=NPROBE):
    idx = np.argpartition(dist, nprobe - 1, axis=1)[:, :nprobe]
    return np.sort(idx, axis=1)


def mmr_select(sim_qc, sim_cc, k, lam):
    """sim_qc: (n,) query-candidate cosine; sim_cc: (n,n). Standard MMR (Carbonell)."""
    n = sim_qc.shape[0]
    sel = [int(np.argmax(sim_qc))]
    cand = np.ones(n, bool); cand[sel[0]] = False
    while len(sel) < min(k, n):
        red = sim_cc[:, sel].max(1)
        score = lam * sim_qc - (1 - lam) * red
        score[~cand] = -np.inf
        j = int(np.argmax(score)); sel.append(j); cand[j] = False
    return sel


def evaluate(ix, Q, sets, idxs=None):
    """Per-query metrics for the L2 arm and the MMR proxy arm inside the pooled candidates.
    If idxs is given, only those query rows are computed (others left 0); caller merges."""
    X, doc, members, gold = ix["X"], ix["doc_of_chunk"], ix["members"], ix["gold"]
    nq = Q.shape[0]
    res = {k: np.zeros(nq) for k in ("l2_recall", "l2_hit", "l2_full", "mmr_recall", "mmr_hit", "mmr_full", "gold_cov")}
    rng_idx = range(nq) if idxs is None else idxs
    for qi in rng_idx:
        pool = np.concatenate([members[int(c)] for c in sets[qi]])
        pool_docs = doc[pool]
        g = gold[qi]
        res["gold_cov"][qi] = np.isin(g, pool_docs).mean()          # fraction of gold docs whose chunks are in the pool
        s = X[pool] @ Q[qi]                                        # cosine (normalised) == -0.5*L2 + const
        order = np.argsort(-s)
        top = pool[order[:TOPK]]
        found = np.isin(g, np.unique(doc[top]))
        res["l2_recall"][qi] = found.mean(); res["l2_hit"][qi] = found.any(); res["l2_full"][qi] = found.all()
        # MMR proxy over the top-MMR_WINDOW candidates
        win = pool[order[:MMR_WINDOW]]
        sw = s[order[:MMR_WINDOW]]
        cc = X[win] @ X[win].T
        sel = mmr_select(sw, cc, TOPK, MMR_LAMBDA)
        found = np.isin(g, np.unique(doc[win[sel]]))
        res["mmr_recall"][qi] = found.mean(); res["mmr_hit"][qi] = found.any(); res["mmr_full"][qi] = found.all()
    return res


def _binom_two_sided_p(k, n, p=0.5):
    """Exact two-sided binomial p-value (numpy only), for McNemar."""
    from math import comb
    probs = np.array([comb(n, i) for i in range(n + 1)], dtype=float) * (p ** np.arange(n + 1)) * ((1 - p) ** (n - np.arange(n + 1)))
    pk = probs[k]
    return float(probs[probs <= pk + 1e-12].sum())


def mcnemar_p(a, b):
    """Exact McNemar on binary per-query outcomes a (baseline) and b (perturbed)."""
    n01 = int(np.sum((a == 1) & (b == 0))); n10 = int(np.sum((a == 0) & (b == 1)))
    n = n01 + n10
    if n == 0:
        return 1.0
    return _binom_two_sided_p(min(n01, n10), n, 0.5)


def summarize(base, pert, sets_base, sets_pert):
    out = {}
    for arm in ("l2", "mmr"):
        for m in ("recall", "hit", "full"):
            k = f"{arm}_{m}"
            out[f"d_{k}"] = float(pert[k].mean() - base[k].mean())
            out[f"{k}"] = float(pert[k].mean())
        out[f"{arm}_flip"] = float(np.mean(pert[f"{arm}_hit"] != base[f"{arm}_hit"]))
        out[f"{arm}_mcnemar_p"] = mcnemar_p(base[f"{arm}_hit"], pert[f"{arm}_hit"])
    out["gold_cov"] = float(pert["gold_cov"].mean())
    out["d_gold_cov"] = float(pert["gold_cov"].mean() - base["gold_cov"].mean())
    inter = np.array([len(np.intersect1d(a, b)) for a, b in zip(sets_base, sets_pert)])
    out["set_jaccard"] = float(np.mean(inter / (2 * NPROBE - inter)))
    out["set_changed"] = float(np.mean(inter < NPROBE))
    out["top1_changed"] = float("nan")
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--query_emb", required=True, help="(1000,768) pre-OPQ normalised query embeddings")
    ap.add_argument("--fit", default="h_aimdc_sim/fefet_ref_fit_smu.json")
    ap.add_argument("--scaler", default="h_aimdc_sim/fefet_ref_fit_smu.fs1p31.scaler.json")
    ap.add_argument("--v_ml", type=float, default=1.4)
    ap.add_argument("--tag", default="fs1p31")
    ap.add_argument("--out", default="out")
    ap.add_argument("--nq", type=int, default=0, help="limit queries (debug)")
    ap.add_argument("--quick", action="store_true", help="reduced grid")
    ap.add_argument("--pq", default="", help="path to PQ-reconstructed chunk matrix (rotated space); if set, in-pool ranking uses it (pipeline regime)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    ix = load_index(args.root)
    if args.pq:
        ix["X"] = np.load(args.pq).astype(np.float64)
        print(f"[load] in-pool ranking on PQ-reconstructed vectors: {args.pq}")
    Qraw = np.load(args.query_emb).astype(np.float64)
    if args.nq:
        Qraw = Qraw[: args.nq]; ix["gold"] = ix["gold"][: args.nq]
    Q = Qraw @ ix["R"]
    C = ix["C"]
    tier = FefetCentroidTier(args.fit, args.scaler, v_ml=args.v_ml)
    fs = tier.window[1] - tier.window[0]
    print(f"[load] {Q.shape[0]} queries, {C.shape[0]} centroids, window {tier.window} (FS {fs:.4f} V), {time.time()-t0:.1f}s")

    # ---- baselines: software L2 (ideal) and device model at zero perturbation
    d_sw = software_l2(Q, C); sets_sw = probe_sets(d_sw)
    d_dev = tier.distances(Q, C); sets_dev = probe_sets(d_dev)
    base_sw = evaluate(ix, Q, sets_sw)
    base_dev = evaluate(ix, Q, sets_dev)
    inter = np.array([len(np.intersect1d(a, b)) for a, b in zip(sets_sw, sets_dev)])
    # device-vs-software distance fidelity (per query, Spearman over the 128 centroids; numpy-only)
    def _spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    rho = np.array([_spearman(d_sw[i], d_dev[i]) for i in range(Q.shape[0])])
    top1_agree = float(np.mean(d_sw.argmin(1) == d_dev.argmin(1)))
    fidelity = dict(nprobe_jaccard_sw_vs_dev=float(np.mean(inter / (2 * NPROBE - inter))),
                    spearman_median=float(np.median(rho)), spearman_p05=float(np.percentile(rho, 5)),
                    top1_agree=top1_agree)
    print("[baseline] software L2 : " + ", ".join(f"{k}={base_sw[k].mean():.4f}" for k in ("l2_recall", "l2_hit", "l2_full", "mmr_recall", "mmr_hit", "gold_cov")))
    print("[baseline] device  d=0 : " + ", ".join(f"{k}={base_dev[k].mean():.4f}" for k in ("l2_recall", "l2_hit", "l2_full", "mmr_recall", "mmr_hit", "gold_cov")))
    print(f"[fidelity] {fidelity}")

    # ---- grid
    if args.quick:
        offsets = [1, 3, 10, 30, 100]
        gains = [0.97, 0.9, 0.7]
        sigmas = [3, 9.5, 30]; seeds = [0]
        cgs = [0.05]
    else:
        offsets = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200]
        gains = [0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
        sigmas = [0.5, 1, 2, 3, 5, 7, 9.5, 15, 20, 30, 50]; seeds = [0, 1, 2]
        cgs = [0.017, 0.05, 0.1]
    grid = []
    for mv in offsets:
        for sgn in (+1, -1):
            grid.append(dict(axis="offset", value=sgn * mv, seed=0))
    for g in gains:
        grid.append(dict(axis="gain", value=g, seed=0))
    for s in sigmas:
        for sd in seeds:
            grid.append(dict(axis="cell_sigma", value=s, seed=sd))
    for s in cgs:
        for sd in seeds:
            grid.append(dict(axis="cell_gain_sigma", value=s, seed=sd))

    # resumable checkpoint: one JSON line per completed config
    ckpt = os.path.join(args.out, f"rows_{args.tag}.jsonl")
    done = set()
    if os.path.isfile(ckpt):
        for line in open(ckpt):
            try:
                r = json.loads(line); done.add((r["axis"], r["value"], r["seed"]))
            except Exception:
                pass
    print(f"[resume] {len(done)} configs already done in {ckpt}")

    def merge(ev_sub, idxs):
        out = {k: base_dev[k].copy() for k in base_dev}
        for k in out:
            out[k][idxs] = ev_sub[k][idxs]
        return out

    ck = open(ckpt, "a")
    for gi, cfg in enumerate(grid):
        key = (cfg["axis"], cfg["value"], cfg["seed"])
        if key in done:
            continue
        t1 = time.time()
        kw = {}
        rng = np.random.default_rng(1000 + cfg["seed"])
        if cfg["axis"] == "offset":
            kw["offset_v"] = cfg["value"] / 1000.0
        elif cfg["axis"] == "gain":
            kw["gain"] = cfg["value"]
        elif cfg["axis"] == "cell_sigma":
            kw["cell_offset"] = rng.normal(0.0, cfg["value"] / 1000.0, size=C.shape)
        elif cfg["axis"] == "cell_gain_sigma":
            kw["cell_gain"] = 1.0 + rng.normal(0.0, cfg["value"], size=C.shape)
        d = tier.distances(Q, C, **kw)
        sets = probe_sets(d)
        # only rescore queries whose probe set changed vs device baseline (recall is exact fn of the set)
        changed = np.array([not np.array_equal(a, b) for a, b in zip(sets, sets_dev)])
        idxs = np.where(changed)[0]
        ev_sub = evaluate(ix, Q, sets, idxs=idxs)
        ev = merge(ev_sub, idxs)
        s = summarize(base_dev, ev, sets_dev, sets)
        s["top1_changed"] = float(np.mean(d.argmin(1) != d_dev.argmin(1)))
        s["n_changed"] = int(changed.sum())
        row = dict(tag=args.tag, axis=cfg["axis"], value=cfg["value"], seed=cfg["seed"], **s)
        if cfg["axis"] == "offset":
            row["value_pct_fs"] = 100.0 * abs(cfg["value"]) / 1000.0 / fs
        row["_hit_mmr"] = ev["mmr_hit"].astype(int).tolist()
        row["_hit_l2"] = ev["l2_hit"].astype(int).tolist()
        ck.write(json.dumps(row) + "\n"); ck.flush()
        print(f"[{gi+1:3d}/{len(grid)}] {cfg['axis']:>15s}={cfg['value']:>8}  seed={cfg['seed']}  "
              f"d_mmr_recall={s['d_mmr_recall']:+.4f}  d_l2_recall={s['d_l2_recall']:+.4f}  "
              f"d_gold_cov={s['d_gold_cov']:+.4f}  flip_mmr={s['mmr_flip']:.3f}  chg={s['n_changed']}  "
              f"p={s['mmr_mcnemar_p']:.3f}  ({time.time()-t1:.1f}s)")
    ck.close()

    import pandas as pd
    rows = [json.loads(l) for l in open(ckpt)]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(os.path.join(args.out, f"tol_sweep_{args.tag}.csv"), index=False)
    summary = dict(tag=args.tag, window=list(tier.window), full_scale_v=fs, v_ml=args.v_ml, nprobe=NPROBE, topk=TOPK,
                   mmr_lambda=MMR_LAMBDA, mmr_window=MMR_WINDOW, n_queries=int(Q.shape[0]), in_pool_vectors=("pq" if args.pq else "exact"),
                   baseline_software={k: float(v.mean()) for k, v in base_sw.items()},
                   baseline_device={k: float(v.mean()) for k, v in base_dev.items()},
                   fidelity=fidelity)
    json.dump(summary, open(os.path.join(args.out, f"tol_summary_{args.tag}.json"), "w"), indent=2)
    print(f"[done] {len(rows)} configs in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
