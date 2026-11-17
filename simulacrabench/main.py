"""SimulacraBench submission v15 (team junlong): v14 + label-free CA re-ordering of >=16-level GIVEN codes for the FM members (UNHCR governorates); v14 = v12 + more FM ensemble members in the hidden-row full fit for high-K items (KC42); v6 + TabPFN-3 member
(Prior Labs tabpfn_3 v3_default checkpoint, bundled; numeric GIVEN codes;
up to 160 classes per fit, beyond that top-9 + recursive 'other') + TabICLv2
member (Soda/Inria tabicl-classifier-v2 checkpoint, bundled, BSD-3).

Per instrument, fitted at runtime on the visible rows only, nothing bundled:
  * crowd marginal (+0.5)
  * multi-task MLPs on one-hot GIVEN features (shared trunk, one softmax head
    per PREDICT item, per-item early stopping on the held-out fold)
  * per-item gradient boosting (LightGBM) on the GIVEN codes
  * hierarchical Dirichlet backoff over the most informative GIVEN columns
All models share K folds; out-of-fold probabilities feed a per-item linear pool
fitted by EM on the out-of-fold log score (optionally a per-item temperature).
Items gated on a GIVEN parent whose gate holds without exception in the visible
rows get the sentinel slot from the rule of succession.

predict() is pure: no files, sockets or subprocesses (threads only).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

T_IMPORT = time.time()

import torch  # noqa: E402

try:
    import lightgbm as lgb  # noqa: E402
except Exception:  # noqa: BLE001
    lgb = None

STRICT = os.environ.get("SIMU_STRICT") == "1"
VERBOSE = os.environ.get("SIMU_VERBOSE") == "1"
N_THREADS = 8
LGB_MODE = os.environ.get("SIMU_LGB_MODE", "seq")
SHARE = {"unicef": 0.20, "world_bank": 0.35, "unhcr": 0.45, "other": 0.33}
SAFE_FRACTION = 0.70
_DONE = []

MLP_A = dict(hidden=512, layers=2, dropout=0.3, lr=2e-3, wd=1e-4, batch=256,
             epochs=200, patience=12, seeds=1)
LGB_A = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
             lambda_l2=1.0, feature_fraction=0.9, bagging_fraction=0.8,
             bagging_freq=1, cat_smooth=10, cat_l2=10, min_data_per_group=20,
             max_cat_to_onehot=8, early_stopping=30, rounds=1000)
BACKOFF_A = dict(depth=3, alphas=(1.0, 3.0, 10.0, 30.0))


def _v(base, **kw):
    d = dict(base)
    d.update(kw)
    return d


# Ordered by value per second: a model the deadline reaches is skipped.
CONFIG = dict(
    folds=5,
    models=[("marginal", "marginal", {}),
            ("mlpb_s3", "mlpb", _v(MLP_A, seeds=3)),
            ("lgb_d1", "lgb", _v(LGB_A, num_leaves=2, learning_rate=0.1)),
            ("backoff", "backoff", BACKOFF_A),
            ("mlpb_wide_s3", "mlpb", _v(MLP_A, hidden=1024, dropout=0.5, wd=1e-3,
                                        seeds=3)),
            ("lgb_small", "lgb", _v(LGB_A, num_leaves=7, min_data_in_leaf=40)),
            # KC20: +0.0017..+0.0019 (GSS) / +0.0008..+0.0011 (NHIS) on 6/6
            # proxy splits; runs after the core models so the deadline guard
            # cuts it item by item rather than dropping a core member
            # KC24 TabPFN-3 (numeric codes, <=160 classes natively): over the
            # v9 TabPFN-v2 member GSS +0.0001..0.0005, NHIS +0.0010..0.0013
            ("tabpfn", "tabpfn", dict(est=4, cat="none", mc="hier", ver="v3",
                                      maxc=160, hcm="given_vis")),
            # KC29: TabICLv2 as a second in-context member next to TabPFN-3:
            # GSS +0.0005..0.0008, NHIS +0.00013..0.00019, gss_mc +0.0005..0.0009
            ("tabicl", "tabicl", dict(est=8, hcm="given_vis"))],
    # appended to `models` only when the phase budget is the 3600 s test budget
    # (KC12: +0.0001..+0.00025 on 6/6 proxy splits); ordered cheap -> costly so
    # the deadline guard drops the most expensive member first
    # (KC27: with both in-context members present, lgb_l4 / lgb_reg add
    # -0.00002..+0.00003 on 6 proxy splits for ~370 s of test runtime: dropped)
    models_extra=[("mlp_small", "mlp", _v(MLP_A, hidden=256, layers=1, dropout=0.2)),
                  ("mlp_lowlr", "mlp", _v(MLP_A, lr=5e-4, patience=20, epochs=400))],
    extra_min_budget=3000.0,
    temperature=True,
    loglin=True,
    shrink=0.0,
    gate_pool=True,
    class_pool=True,
    gate_rule=True,
    unmix=True,
)


def _log(*args):
    if VERBOSE:
        print("[sim %7.1fs]" % (time.time() - T_IMPORT), *args, flush=True)


def _instrument(schema):
    items = schema["items"]
    if "s1_2_country_asylum_lbl" in items or "wave" in items:
        return "unhcr"
    if "sc_nstd_extra" in items or "group" in items:
        return "world_bank"
    if "Country" in items and "Q06" in items:
        return "unicef"
    return "other"


def _options(schema, name):
    rec = schema["items"][name]
    opts = list(rec["values"])
    if rec.get("gate"):
        opts.append(schema.get("gated_value", "NA_GATED"))
    return opts


# ---------------------------------------------------------------- prepare --

def prepare(frame, schema, n_folds=5, fold_seed=12345):
    recs = schema["items"]
    items = [k for k, r in recs.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [k for k in items if recs[k]["class"] == "GIVEN"]
    targets = [k for k in items if recs[k]["class"] == "PREDICT"]
    opts = {k: _options(schema, k) for k in items}

    hidden_mask = frame[targets].isna().any(axis=1).to_numpy()
    vis = frame.loc[~hidden_mask].reset_index(drop=True)
    hid = frame.loc[hidden_mask].reset_index(drop=True)
    n, n_test = len(vis), len(hid)

    Xc = np.zeros((n, len(given)), np.int32)
    Xc_test = np.zeros((n_test, len(given)), np.int32)
    card = []
    blocks, blocks_t = [], []
    for gi, g in enumerate(given):
        lut = {v: i for i, v in enumerate(opts[g])}
        unk = len(opts[g])
        a = vis[g].map(lut).fillna(unk).astype(np.int64).to_numpy()
        b = hid[g].map(lut).fillna(unk).astype(np.int64).to_numpy()
        Xc[:, gi], Xc_test[:, gi] = a, b
        card.append(unk + 1)
        eye = np.eye(unk + 1, dtype=np.float32)
        blocks.append(eye[a])
        blocks_t.append(eye[b])
    Xoh = np.concatenate(blocks, 1)
    Xoh_test = np.concatenate(blocks_t, 1)
    keep = Xoh.std(0) > 0
    Xoh, Xoh_test = Xoh[:, keep], Xoh_test[:, keep]
    # ordinal copies of the codes (level order as listed in the schema)
    scale = np.maximum(np.array(card, np.float32) - 2.0, 1.0)
    Xnum = (Xc / scale).astype(np.float32)
    Xnum_test = (Xc_test / scale).astype(np.float32)

    J = len(targets)
    K = np.array([len(opts[t]) for t in targets])
    Kmax = int(K.max())
    Y = np.zeros((n, J), np.int64)
    for j, t in enumerate(targets):
        lut = {v: i for i, v in enumerate(opts[t])}
        y = vis[t].map(lut)
        if y.isna().any():
            if STRICT:
                raise ValueError("unknown level in %s" % t)
            y = y.fillna(0)
        Y[:, j] = y.astype(np.int64).to_numpy()

    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    folds = []
    for f in range(n_folds):
        va = np.sort(perm[f::n_folds])
        tr = np.setdiff1d(np.arange(n), va)
        folds.append((tr, va))
    return dict(recs=recs, items=items, given=given, targets=targets, opts=opts,
                hidden_mask=hidden_mask, vis=vis, hid=hid, n=n, n_test=n_test,
                Xc=Xc, Xc_test=Xc_test, card=np.array(card), Xoh=Xoh,
                Xoh_test=Xoh_test, Xnum=Xnum, Xnum_test=Xnum_test, Y=Y, K=K,
                Kmax=Kmax, J=J, folds=folds)


# ----------------------------------------------------------------- models --

def m_marginal(P, cfg, deadline):
    Y, K, folds, Kmax = P["Y"], P["K"], P["folds"], P["Kmax"]
    n, J = Y.shape
    oof = np.zeros((n, J, Kmax), np.float32)
    test = np.zeros((P["n_test"], J, Kmax), np.float32)
    for j in range(J):
        k = K[j]
        for tr, va in folds:
            c = np.bincount(Y[tr, j], minlength=k).astype(np.float64) + 0.5
            oof[va, j, :k] = c / c.sum()
        c = np.bincount(Y[:, j], minlength=k).astype(np.float64) + 0.5
        test[:, j, :k] = c / c.sum()
    return oof, test


class _MLP(torch.nn.Module):
    def __init__(self, d, hidden, layers, dropout, out):
        super().__init__()
        mods, width = [], d
        for _ in range(layers):
            mods += [torch.nn.Linear(width, hidden), torch.nn.GELU(),
                     torch.nn.Dropout(dropout)]
            width = hidden
        mods.append(torch.nn.Linear(width, out))
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def m_mlp(P, cfg, deadline):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xoh, Xoh_test, Y, K, folds, Kmax = (P["Xoh"], P["Xoh_test"], P["Y"], P["K"],
                                        P["folds"], P["Kmax"])
    if cfg.get("ordinal"):
        Xoh = np.concatenate([Xoh, P["Xnum"]], 1)
        Xoh_test = np.concatenate([Xoh_test, P["Xnum_test"]], 1)
    n, J = Y.shape
    d = Xoh.shape[1]
    valid = np.zeros((J, Kmax), bool)
    for j in range(J):
        valid[j, :K[j]] = True
    neg = torch.tensor(np.where(valid, 0.0, -1e4), dtype=torch.float32, device=dev)
    Xa = torch.tensor(Xoh, dtype=torch.float32, device=dev)
    Xt = torch.tensor(Xoh_test, dtype=torch.float32, device=dev)
    Ya = torch.tensor(Y, dtype=torch.long, device=dev)
    oof = np.zeros((n, J, Kmax), np.float32)
    test = np.zeros((Xoh_test.shape[0], J, Kmax), np.float32)
    seeds = cfg.get("seeds", 1)
    base_seed = cfg.get("seed", 1)
    bs = cfg["batch"]
    for fi, (tr, va) in enumerate(folds):
        if time.time() > deadline:
            raise TimeoutError("mlp ran out of time")
        tr_t = torch.tensor(tr, device=dev)
        va_t = torch.tensor(va, device=dev)
        acc_va = torch.zeros((len(va), J, Kmax), device=dev)
        acc_te = torch.zeros((Xt.shape[0], J, Kmax), device=dev)
        for s in range(seeds):
            seed = base_seed * 1000 + fi * 10 + s
            torch.manual_seed(seed)
            g = torch.Generator(device="cpu").manual_seed(seed)
            model = _MLP(d, cfg["hidden"], cfg["layers"], cfg["dropout"],
                         J * Kmax).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["wd"])
            best = torch.full((J,), float("inf"), device=dev)
            best_va = torch.zeros((len(va), J, Kmax), device=dev)
            best_te = torch.zeros((Xt.shape[0], J, Kmax), device=dev)
            stale = 0
            for epoch in range(cfg["epochs"]):
                model.train()
                perm = tr_t[torch.randperm(len(tr), generator=g).to(dev)]
                for b0 in range(0, len(perm), bs):
                    idx = perm[b0:b0 + bs]
                    logits = model(Xa[idx]).view(-1, J, Kmax) + neg
                    lp = torch.log_softmax(logits, -1)
                    loss = -lp.gather(2, Ya[idx].unsqueeze(2)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    lpv = torch.log_softmax(model(Xa[va_t]).view(-1, J, Kmax) + neg, -1)
                    lossj = -lpv.gather(2, Ya[va_t].unsqueeze(2)).squeeze(2).mean(0)
                    imp = lossj < best - 1e-5
                    if bool(imp.any()):
                        stale = 0
                        pte = torch.softmax(model(Xt).view(-1, J, Kmax) + neg, -1)
                        best = torch.where(imp, lossj, best)
                        best_va[:, imp] = lpv.exp()[:, imp]
                        best_te[:, imp] = pte[:, imp]
                    else:
                        stale += 1
                if stale >= cfg["patience"]:
                    break
            acc_va += best_va / seeds
            acc_te += best_te / seeds
        oof[va] = acc_va.cpu().numpy()
        test += acc_te.cpu().numpy() / len(folds)
    return oof, test


def _init_linear(F, fan_in, fan_out, dev, gen):
    bound = 1.0 / np.sqrt(fan_in)
    w = (torch.rand((F, fan_in, fan_out), generator=gen, device="cpu") * 2 - 1) * bound
    b = (torch.rand((F, 1, fan_out), generator=gen, device="cpu") * 2 - 1) * bound
    return (w.to(dev).requires_grad_(), b.to(dev).requires_grad_())


def m_mlpb(P, cfg, deadline):
    """Fold- and seed-batched multi-task MLP: F = folds x seeds nets trained at
    once with batched matmuls; per (net, item) early stopping on that net's
    held-out fold."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xoh, Xoh_test, Y, K, folds, Kmax = (P["Xoh"], P["Xoh_test"], P["Y"], P["K"],
                                        P["folds"], P["Kmax"])
    if cfg.get("ordinal"):
        Xoh = np.concatenate([Xoh, P["Xnum"]], 1)
        Xoh_test = np.concatenate([Xoh_test, P["Xnum_test"]], 1)
    n, J = Y.shape
    d = Xoh.shape[1]
    nf = len(folds)
    S = cfg.get("seeds", 1)
    F = nf * S
    H, layers, drop = cfg["hidden"], cfg["layers"], cfg["dropout"]
    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 1) * 7919)
    valid = np.zeros((J, Kmax), bool)
    for j in range(J):
        valid[j, :K[j]] = True
    neg = torch.tensor(np.where(valid, 0.0, -1e4), dtype=torch.float32, device=dev)
    Xa = torch.tensor(Xoh, dtype=torch.float32, device=dev)
    Xt = torch.tensor(Xoh_test, dtype=torch.float32, device=dev)
    Ya = torch.tensor(Y, dtype=torch.long, device=dev)
    params = []
    width = d
    for _ in range(layers):
        params.append(_init_linear(F, width, H, dev, gen))
        width = H
    params.append(_init_linear(F, width, J * Kmax, dev, gen))
    flat = [t for wb in params for t in wb]
    opt = torch.optim.AdamW(flat, lr=cfg["lr"], weight_decay=cfg["wd"])

    def forward(x, train):
        h = x
        for li, (w, b) in enumerate(params):
            h = torch.baddbmm(b, h, w)
            if li < len(params) - 1:
                h = torch.nn.functional.gelu(h)
                if train and drop > 0:
                    h = torch.nn.functional.dropout(h, drop, True)
        return h.view(h.shape[0], h.shape[1], J, Kmax) + neg

    tr_list = [torch.tensor(folds[f % nf][0], device=dev) for f in range(F)]
    va_np = [folds[f % nf][1] for f in range(F)]
    nva = max(len(v) for v in va_np)
    va_idx = torch.zeros((F, nva), dtype=torch.long, device=dev)
    va_msk = torch.zeros((F, nva), dtype=torch.bool, device=dev)
    for f, v in enumerate(va_np):
        va_idx[f, :len(v)] = torch.tensor(v, device=dev)
        va_msk[f, :len(v)] = True
    ntr = min(len(t) for t in tr_list)
    bs = cfg["batch"]
    steps = max(1, ntr // bs)
    best = torch.full((F, J), float("inf"), device=dev)
    best_va = torch.zeros((F, nva, J, Kmax), device=dev)
    best_te = torch.zeros((F, Xt.shape[0], J, Kmax), device=dev)
    Xte = Xt.unsqueeze(0).expand(F, -1, -1)
    stale = 0
    for epoch in range(cfg["epochs"]):
        if time.time() > deadline:
            raise TimeoutError("mlpb ran out of time")
        perms = torch.stack([t[torch.randperm(len(t), generator=gen)[:steps * bs].to(dev)]
                             for t in tr_list])
        for s in range(steps):
            idx = perms[:, s * bs:(s + 1) * bs]
            logits = forward(Xa[idx], True)
            lp = torch.log_softmax(logits, -1)
            loss = -lp.gather(3, Ya[idx].unsqueeze(3)).mean() * F
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            lpv = torch.log_softmax(forward(Xa[va_idx], False), -1)
            nll = -lpv.gather(3, Ya[va_idx].unsqueeze(3)).squeeze(3)
            nll = (nll * va_msk.unsqueeze(2)).sum(1) / va_msk.sum(1, keepdim=True)
            imp = nll < best - 1e-5
            if bool(imp.any()):
                stale = 0
                pte = torch.softmax(forward(Xte, False), -1)
                best = torch.where(imp, nll, best)
                m = imp[:, None, :, None]
                best_va = torch.where(m, lpv.exp(), best_va)
                best_te = torch.where(m, pte, best_te)
            else:
                stale += 1
        if stale >= cfg["patience"]:
            break
    oof = np.zeros((n, J, Kmax), np.float32)
    bv = best_va.cpu().numpy()
    for f in range(F):
        v = va_np[f]
        oof[v] += bv[f, :len(v)] / S
    test = best_te.mean(0).cpu().numpy()
    return oof, test


def _lgb_task(Xc, Xc_test, y, k, tr, va, params, cat):
    ytr = y[tr]
    present = np.unique(ytr)
    out_va = np.zeros((len(va), k), np.float32)
    out_te = np.zeros((Xc_test.shape[0], k), np.float32)
    if len(present) == 1:
        c = np.bincount(ytr, minlength=k).astype(np.float64) + 0.5
        out_va[:] = c / c.sum()
        out_te[:] = c / c.sum()
        return out_va, out_te
    remap = np.full(k, -1, np.int64)
    remap[present] = np.arange(len(present))
    p = dict(params)
    stop = p.pop("early_stopping")
    rounds = p.pop("rounds")
    p.setdefault("num_threads", 1)
    p.update(verbose=-1, seed=7, deterministic=True)
    if len(present) == 2:
        p.update(objective="binary")
    else:
        p.update(objective="multiclass", num_class=len(present))
    yv = remap[y[va]]
    keep = yv >= 0
    dtr = lgb.Dataset(Xc[tr], label=remap[ytr], categorical_feature=cat,
                      free_raw_data=False)
    if keep.sum() > 0:
        dva = lgb.Dataset(Xc[va][keep], label=yv[keep], reference=dtr,
                          categorical_feature=cat)
        booster = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(stop, verbose=False)])
    else:
        booster = lgb.train(p, dtr, num_boost_round=100)
    it = booster.best_iteration or None
    pv = booster.predict(Xc[va], num_iteration=it)
    pt = booster.predict(Xc_test, num_iteration=it)
    if len(present) == 2:
        pv = np.stack([1 - pv, pv], 1)
        pt = np.stack([1 - pt, pt], 1)
    out_va[:, present] = pv
    out_te[:, present] = pt
    return out_va, out_te


def _hgb_task(Xc, Xc_test, y, k, tr, va, params, cat):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ytr = y[tr]
    present = np.unique(ytr)
    out_va = np.zeros((len(va), k), np.float32)
    out_te = np.zeros((Xc_test.shape[0], k), np.float32)
    if len(present) == 1:
        c = np.bincount(ytr, minlength=k).astype(np.float64) + 0.5
        out_va[:] = c / c.sum()
        out_te[:] = c / c.sum()
        return out_va, out_te
    yva = y[va]
    keep = np.isin(yva, present)
    clf = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_leaf_nodes=max(2, params["num_leaves"]),
        min_samples_leaf=params["min_data_in_leaf"], l2_regularization=params["lambda_l2"],
        max_iter=params["rounds"], early_stopping=bool(keep.sum() > 0),
        n_iter_no_change=params["early_stopping"], categorical_features=cat,
        random_state=7)
    if keep.sum() > 0:
        clf.fit(Xc[tr], ytr, X_val=Xc[va][keep], y_val=yva[keep])
    else:
        clf.fit(Xc[tr], ytr)
    out_va[:, clf.classes_] = clf.predict_proba(Xc[va])
    out_te[:, clf.classes_] = clf.predict_proba(Xc_test)
    return out_va, out_te


def m_lgb(P, cfg, deadline):
    Xc, Xc_test, Y, K, folds, Kmax = (P["Xc"], P["Xc_test"], P["Y"], P["K"],
                                      P["folds"], P["Kmax"])
    n, J = Y.shape
    cat = list(range(Xc.shape[1]))
    use_hgb = cfg.get("engine") == "hgb" or lgb is None
    if lgb is None and cfg.get("engine") != "hgb" and STRICT:
        raise ImportError("lightgbm missing")
    task_fn = _hgb_task if use_hgb else _lgb_task
    if use_hgb:
        cat = np.ones(Xc.shape[1], bool)
    oof = np.zeros((n, J, Kmax), np.float32)
    test = np.zeros((Xc_test.shape[0], J, Kmax), np.float32)
    t0 = time.time()
    threaded = cfg.get("mode", LGB_MODE) == "threads"
    run_cfg = dict(cfg)
    run_cfg.pop("mode", None)
    run_cfg.pop("engine", None)
    run_cfg["num_threads"] = 1 if threaded else N_THREADS
    pool = ThreadPoolExecutor(max_workers=N_THREADS) if threaded else None
    try:
        for fi, (tr, va) in enumerate(folds):
            if fi == 1:
                per_fold = time.time() - t0
                if time.time() + per_fold * (len(folds) - 1) > deadline:
                    raise TimeoutError("lgb projected past its deadline")
            call = (lambda j, tr=tr, va=va: task_fn(Xc, Xc_test, Y[:, j], K[j], tr,
                                                    va, run_cfg, cat))
            res = pool.map(call, range(J)) if threaded else map(call, range(J))
            for j, (pv, pt) in enumerate(res):
                oof[va, j, :K[j]] = pv
                test[:, j, :K[j]] += pt / len(folds)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return oof, test


def _mi(x, y, nx, ny):
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(np.float64)
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _backoff_fit_predict(Xtr, ytr, Xev_list, k, card, order, alpha):
    """Recursive Dirichlet smoothing along `order`; returns preds for each Xev."""
    c0 = np.bincount(ytr, minlength=k).astype(np.float64) + 0.5
    p0 = c0 / c0.sum()
    outs = [np.tile(p0, (len(X), 1)) for X in Xev_list]
    key_tr = np.zeros(len(ytr), np.int64)
    keys_ev = [np.zeros(len(X), np.int64) for X in Xev_list]
    # running prediction for training rows too (needed as the next prior)
    ptr = np.tile(p0, (len(ytr), 1))
    size = 1
    for col in order:
        key_tr = key_tr * card[col] + Xtr[:, col]
        keys_ev = [ke * card[col] + X[:, col] for ke, X in zip(keys_ev, Xev_list)]
        size *= card[col]
        uniq, inv = np.unique(key_tr, return_inverse=True)
        tab = np.zeros((len(uniq), k))
        np.add.at(tab, (inv, ytr), 1.0)
        tot = tab.sum(1)
        # prior for each group: mean of the previous-level prediction in it
        prior = np.zeros((len(uniq), k))
        np.add.at(prior, inv, ptr)
        prior /= np.maximum(tot, 1)[:, None]
        grp = (tab + alpha * prior) / (tot + alpha)[:, None]
        ptr = grp[inv]
        new_outs = []
        for ke, prev in zip(keys_ev, outs):
            pos = np.searchsorted(uniq, ke)
            pos = np.minimum(pos, len(uniq) - 1)
            hit = uniq[pos] == ke
            o = prev.copy()
            o[hit] = (tab[pos[hit]] + alpha * prev[hit]) / (tot[pos[hit]] + alpha)[:, None]
            new_outs.append(o)
        outs = new_outs
    return outs


def m_backoff(P, cfg, deadline):
    Xc, Xc_test, Y, K, folds, Kmax, card = (P["Xc"], P["Xc_test"], P["Y"], P["K"],
                                            P["folds"], P["Kmax"], P["card"])
    n, J = Y.shape
    depth = min(cfg["depth"], Xc.shape[1])
    alphas = cfg["alphas"]
    oof = np.zeros((n, J, Kmax), np.float32)
    test = np.zeros((Xc_test.shape[0], J, Kmax), np.float32)

    def one(j):
        k = K[j]
        y = Y[:, j]
        cand = np.zeros((len(alphas), n, k))
        for tr, va in folds:
            mi = [_mi(Xc[tr, c], y[tr], card[c], k) for c in range(Xc.shape[1])]
            order = list(np.argsort(mi)[::-1][:depth])
            for ai, a in enumerate(alphas):
                cand[ai, va] = _backoff_fit_predict(Xc[tr], y[tr], [Xc[va]], k,
                                                    card, order, a)[0]
        ll = [np.log(np.maximum(cand[ai, np.arange(n), y], 1e-12)).mean()
              for ai in range(len(alphas))]
        ai = int(np.argmax(ll))
        mi = [_mi(Xc[:, c], y, card[c], k) for c in range(Xc.shape[1])]
        order = list(np.argsort(mi)[::-1][:depth])
        pt = _backoff_fit_predict(Xc, y, [Xc_test], k, card, order, alphas[ai])[0]
        return j, cand[ai], pt

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for j, po, pt in pool.map(one, range(J)):
            if time.time() > deadline:
                raise TimeoutError("backoff ran out of time")
            oof[:, j, :K[j]] = po
            test[:, j, :K[j]] = pt
    return oof, test



def m_kernel(P, cfg, deadline):
    """Learned Hamming-kernel smoother. w(x, x') = exp(-sum_c lam_c [x_c != x'_c]);
    p_j(x) = (sum_l w(x, x_l) onehot(y_lj) + a_j m_j) / (sum_l w(x, x_l) + a_j).
    lam (shared across items) and a global a are fitted by Adam on the
    fold-masked out-of-fold log score of a row subset; then a per-item a_j is
    picked on the full OOF from a grid."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xc, Xc_test, Y, K, folds, Kmax = (P["Xc"], P["Xc_test"], P["Y"], P["K"],
                                      P["folds"], P["Kmax"])
    n, J = Y.shape
    G = Xc.shape[1]
    fold_id = np.zeros(n, np.int64)
    for fi, (_, va) in enumerate(folds):
        fold_id[va] = fi
    off = np.concatenate([[0], np.cumsum(K)])
    SK = int(off[-1])
    Yoh = np.zeros((n, SK), np.float32)
    for j in range(J):
        Yoh[np.arange(n), off[j] + Y[:, j]] = 1.0
    marg = Yoh.sum(0) + 0.5
    for j in range(J):
        marg[off[j]:off[j + 1]] /= marg[off[j]:off[j + 1]].sum()
    Xa = torch.tensor(Xc, dtype=torch.int32, device=dev)
    Xt = torch.tensor(Xc_test, dtype=torch.int32, device=dev)
    Fa = torch.tensor(fold_id, device=dev)
    Ya_oh = torch.tensor(Yoh, device=dev)
    M = torch.tensor(marg, dtype=torch.float32, device=dev)
    ycols = torch.tensor(off[:-1][None, :] + Y, device=dev)       # n, J

    def logw(xq, lam, fq=None):
        lw = torch.zeros((xq.shape[0], n), device=dev)
        for c in range(G):
            lw -= lam[c] * (xq[:, c:c + 1] != Xa[None, :, c]).float()
        if fq is not None:
            lw = lw.masked_fill(fq[:, None] == Fa[None, :], float("-inf"))
        return lw

    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 3))
    q = torch.randperm(n, generator=gen)[:min(n, cfg.get("q", 2048))].to(dev)
    theta = torch.full((G,), float(np.log(np.expm1(cfg.get("lam0", 1.0)))), device=dev,
                       requires_grad=True)
    loga = torch.tensor(float(np.log(cfg.get("a0", 5.0))), device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta, loga], lr=cfg.get("lr", 0.05))
    for step in range(cfg.get("steps", 150)):
        lam = torch.nn.functional.softplus(theta).clamp(max=6.0)
        lw = logw(Xa[q], lam, Fa[q])
        w = torch.exp(lw)
        a = torch.exp(loga)
        num = w @ Ya_oh + a * M[None, :]
        den = w.sum(1, keepdim=True) + a
        lp = torch.log(torch.gather(num, 1, ycols[q]) / den)
        loss = -lp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 and time.time() > deadline:
            raise TimeoutError("kernel ran out of time")
    lam = torch.nn.functional.softplus(theta).clamp(max=6.0).detach()
    _log("kernel lam", np.round(lam.cpu().numpy(), 2), "a", round(float(torch.exp(loga)), 2))
    grid = torch.tensor(cfg.get("alphas", (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)), device=dev)

    def stats(xq, fq):
        A, D = [], []
        for s in range(0, xq.shape[0], 1024):
            w = torch.exp(logw(xq[s:s + 1024], lam, None if fq is None else fq[s:s + 1024]))
            A.append(w @ Ya_oh)
            D.append(w.sum(1, keepdim=True))
        return torch.cat(A), torch.cat(D)

    with torch.no_grad():
        Ao, Do = stats(Xa, Fa)
        At, Dt = stats(Xt, None)
        oof = np.zeros((n, J, Kmax), np.float32)
        test = np.zeros((Xc_test.shape[0], J, Kmax), np.float32)
        for j in range(J):
            sl = slice(int(off[j]), int(off[j + 1]))
            yj = torch.tensor(Y[:, j], device=dev)
            best, ba = None, None
            for a in grid:
                p = (Ao[:, sl] + a * M[None, sl]) / (Do + a)
                v = torch.log(p.gather(1, yj[:, None])).mean().item()
                if best is None or v > best:
                    best, ba = v, a
            oof[:, j, :K[j]] = ((Ao[:, sl] + ba * M[None, sl]) / (Do + ba)).cpu().numpy()
            test[:, j, :K[j]] = ((At[:, sl] + ba * M[None, sl]) / (Dt + ba)).cpu().numpy()
    return oof, test



def m_wide(P, cfg, deadline):
    """Multinomial GLM on one-hot GIVEN singles + all pairwise one-hot crosses,
    multi-output (one softmax per item), L2 penalty. F = folds x lambdas models
    trained full-batch at once (Adam); per item the lambda with the best OOF
    log score is kept; test = mean over that lambda's fold models."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xc, Xc_test, Y, K, folds, Kmax, card = (P["Xc"], P["Xc_test"], P["Y"], P["K"],
                                            P["folds"], P["Kmax"], P["card"])
    n, J = Y.shape
    G = Xc.shape[1]
    cols, cols_t = [P["Xoh"]], [P["Xoh_test"]]
    max_cells = cfg.get("max_cells", 2000)
    for a in range(G):
        for b in range(a + 1, G):
            size = int(card[a] * card[b])
            if size > max_cells:
                continue
            ka = Xc[:, a].astype(np.int64) * card[b] + Xc[:, b]
            kb = Xc_test[:, a].astype(np.int64) * card[b] + Xc_test[:, b]
            cnt = np.bincount(ka, minlength=size)
            keep = np.flatnonzero(cnt >= cfg.get("min_count", 3))
            if len(keep) == 0:
                continue
            lut = -np.ones(size, np.int64)
            lut[keep] = np.arange(len(keep))
            A = np.zeros((n, len(keep)), np.float32)
            B = np.zeros((len(kb), len(keep)), np.float32)
            ia, ib = lut[ka], lut[kb]
            A[np.flatnonzero(ia >= 0), ia[ia >= 0]] = 1.0
            B[np.flatnonzero(ib >= 0), ib[ib >= 0]] = 1.0
            cols.append(A)
            cols_t.append(B)
    X = np.concatenate(cols, 1)
    Xt_np = np.concatenate(cols_t, 1)
    d = X.shape[1]
    lams = list(cfg.get("lams", (1e-4, 3e-4, 1e-3, 3e-3)))
    nf, NL = len(folds), len(lams)
    F = nf * NL
    valid = np.zeros((J, Kmax), bool)
    for j in range(J):
        valid[j, :K[j]] = True
    neg = torch.tensor(np.where(valid, 0.0, -1e4), dtype=torch.float32, device=dev)
    Xa = torch.tensor(X, device=dev)
    Xt = torch.tensor(Xt_np, device=dev)
    Ya = torch.tensor(Y, dtype=torch.long, device=dev)
    trm = torch.zeros((F, n), device=dev)
    for f in range(F):
        trm[f, torch.tensor(folds[f % nf][0], device=dev)] = 1.0
    trm = trm / trm.sum(1, keepdim=True)
    lam_t = torch.tensor([lams[f // nf] for f in range(F)], device=dev)
    W = torch.zeros((F, d, J * Kmax), device=dev, requires_grad=True)
    b = torch.zeros((F, 1, J * Kmax), device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=cfg.get("lr", 0.05))
    steps = cfg.get("steps", 300)
    chunk = max(1, int(cfg.get("chunk_rows", 4096)))
    for step in range(steps):
        if step % 25 == 0 and time.time() > deadline:
            raise TimeoutError("wide ran out of time")
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for s0 in range(0, n, chunk):
            xs = Xa[s0:s0 + chunk]
            lg = (torch.einsum("nd,fdk->fnk", xs, W) + b).view(F, xs.shape[0], J, Kmax) + neg
            lp = torch.log_softmax(lg, -1)
            nll = -lp.gather(3, Ya[s0:s0 + chunk].unsqueeze(0).unsqueeze(3)
                             .expand(F, -1, -1, -1)).squeeze(3).sum(2)       # F, rows
            loss = (nll * trm[:, s0:s0 + chunk]).sum()
            loss.backward()
            tot += float(loss)
        (0.5 * (lam_t[:, None, None] * W * W).sum() * J).backward()
        opt.step()
    with torch.no_grad():
        oofF = np.zeros((NL, n, J, Kmax), np.float32)
        testF = np.zeros((NL, Xt.shape[0], J, Kmax), np.float32)
        for f in range(F):
            li, fi = f // nf, f % nf
            va = folds[fi][1]
            xv = Xa[torch.tensor(va, device=dev)]
            pv = torch.softmax((xv @ W[f] + b[f]).view(len(va), J, Kmax) + neg, -1)
            oofF[li, va] = pv.cpu().numpy()
            pt = torch.softmax((Xt @ W[f] + b[f]).view(Xt.shape[0], J, Kmax) + neg, -1)
            testF[li] += pt.cpu().numpy() / nf
    oof = np.zeros((n, J, Kmax), np.float32)
    test = np.zeros((Xt_np.shape[0], J, Kmax), np.float32)
    rows = np.arange(n)
    pick = []
    for j in range(J):
        ll = [np.log(np.maximum(oofF[li, rows, j, Y[:, j]], 1e-12)).mean() for li in range(NL)]
        li = int(np.argmax(ll))
        pick.append(li)
        oof[:, j] = oofF[li, :, j]
        test[:, j] = testF[li, :, j]
    _log("wide d=%d lam picks %s" % (d, np.bincount(pick, minlength=NL).tolist()))
    return oof, test



HERE = os.path.dirname(os.path.abspath(__file__))


class TabPFNEnvError(RuntimeError):
    """Raised past run_models' fallback: the run fails instead of degrading."""


TABPFN_CKPT = "tabpfn-v3-classifier-v3_default.ckpt"  # Prior-Labs/tabpfn_3 @24a16a89


def _tpf_no_cudnn_attention():
    """tabpfn 8.5.0 runs each attention call under its own sdpa_kernel list,
    which includes cuDNN attention; that backend builds a new graph for every
    new training-row count (50-100 s each, KF9), which the 'other' sub-fits
    below hit constantly. Flash / memory-efficient / math kernels have no
    per-shape cost. Returns the previous list (restored after the member)."""
    from torch.nn.attention import SDPBackend
    import tabpfn.architectures.shared.scaled_dot_product_attention as tsd
    prev = list(tsd._SDPA_BACKENDS)
    tsd._SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    return tsd, prev


def _ca_scores(N, a=5.0, d=2):
    """Row standard coordinates (first d dims) of a correspondence analysis of the
    count table N (levels x columns), rows smoothed toward the column margins with
    a pseudo-rows; empty rows get score 0 (the centroid)."""
    N = np.asarray(N, np.float64)
    nl = N.sum(1)
    cm = N.sum(0) / max(N.sum(), 1e-12)
    D = (N + a * cm) / (nl[:, None] + a)
    r = (nl + a) / (nl + a).sum()
    Z = (D - cm) / np.sqrt(np.maximum(cm, 1e-12)) * np.sqrt(r)[:, None]
    u, sv, _ = np.linalg.svd(Z, full_matrices=False)
    sc = u[:, :d] / np.sqrt(r)[:, None]
    sc[nl == 0] = 0.0
    return sc


def _hc_remap(P, cfg, X, Xt):
    mode = cfg.get("hcm")
    if not mode:
        return X, Xt
    card, Xc, Xct = P["card"], P["Xc"], P["Xc_test"]
    cols = [c for c, m in enumerate(card) if m >= cfg.get("hc_min", 16)]
    if not cols:
        return X, Xt
    key = (mode, tuple(cols))
    cache = P.setdefault("_hc_cache", {})
    if key not in cache:
        out = {}
        for c in cols:
            L = int(card[c])
            if mode in ("given", "given_vis"):
                A = Xc if mode == "given_vis" else np.concatenate([Xc, Xct], 0)
                blocks = []
                for g in range(A.shape[1]):
                    if g == c:
                        continue
                    T = np.zeros((L, int(card[g])))
                    np.add.at(T, (A[:, c], A[:, g]), 1.0)
                    blocks.append(T)
            else:
                Y, K = P["Y"], P["K"]
                blocks = []
                for j in range(Y.shape[1]):
                    T = np.zeros((L, int(K[j])))
                    np.add.at(T, (Xc[:, c], Y[:, j]), 1.0)
                    blocks.append(T)
            sc = _ca_scores(np.concatenate(blocks, 1))
            out[c] = sc
            _log("[hc2] mode", mode, "col", c, "card", L, "score sd",
                 np.round(sc.std(0), 3).tolist())
        cache[key] = out
    out = cache[key]
    X, Xt = X.copy(), Xt.copy()
    if cfg.get("hca"):
        ex = [out[c][X[:, c].astype(np.int64)] for c in cols]
        ext = [out[c][Xt[:, c].astype(np.int64)] for c in cols]
        return (np.concatenate([X] + ex, 1).astype(np.float32),
                np.concatenate([Xt] + ext, 1).astype(np.float32))
    for c in cols:
        rank = np.argsort(np.argsort(out[c][:, 0], kind="stable"), kind="stable")
        X[:, c] = rank[X[:, c].astype(np.int64)]
        Xt[:, c] = rank[Xt[:, c].astype(np.int64)]
    return X, Xt


def _tpf_fit_proba(clf_cls, version, Xtr, ytr, Xq, cfg, ckpt, cat):
    """One in-context fit + predict; returns (classes, proba over them)."""
    clf = clf_cls.create_default_for_version(
        version, model_path=ckpt, device="cuda",
        n_estimators=cfg.get("est", 4), categorical_features_indices=cat,
        ignore_pretraining_limits=True, random_state=cfg.get("seed", 0),
        n_preprocessing_jobs=1)
    clf.fit(Xtr, ytr)
    return clf.classes_.astype(int), clf.predict_proba(Xq)


def _tpf_hier(fit, Xtr, ytr, Xq, k, min_rest=30, depth=0, maxc=10):
    """P(y | x) over k classes for the rows Xq. At most maxc observed classes:
    one fit. More: the 9 most frequent classes stay and the rest merge into
    'other'; one fit gives P(top_i) and P(other); P(c | other) comes from a
    recursive fit on the 'other' rows (smoothed counts when they are fewer
    than min_rest or depth runs out)."""
    cls, cnt = np.unique(ytr, return_counts=True)
    out = np.zeros((len(Xq), k))
    if len(cls) == 1:
        out[:, cls[0]] = 1.0
        return out
    if len(cls) <= maxc:
        c, pr = fit(Xtr, ytr, Xq)
        out[:, c] = pr
        return out
    order = cls[np.argsort(-cnt, kind="stable")]
    top, rest = order[:9], order[9:]
    g = np.full(len(ytr), 9)
    for i, c in enumerate(top):
        g[ytr == c] = i
    c, pr = fit(Xtr, g, Xq)
    pg = np.zeros((len(Xq), 10))
    pg[:, c] = pr
    out[:, top] = pg[:, :9]
    m = g == 9
    if m.sum() >= min_rest and len(np.unique(ytr[m])) >= 2 and depth < 3:
        sub = _tpf_hier(fit, Xtr[m], ytr[m], Xq, k, min_rest, depth + 1, maxc)
    else:
        cr = np.bincount(ytr[m], minlength=k).astype(float)
        cr[rest] += 0.5
        sub = np.tile(cr / cr.sum(), (len(Xq), 1))
    out += pg[:, 9:10] * sub
    return out


def m_tabpfn(P, cfg, deadline):
    """Per-item TabPFN classifier (Prior Labs TabPFN-3, weights bundled:
    Prior-Labs/tabpfn_3 @24a16a89, v3_default; TABPFN-3 Non-Commercial
    License, see LICENSE_TABPFN3 / NOTICE), one in-context fit per fold for the
    held-out-fold predictions and one on all visible rows for the hidden rows,
    on the GIVEN codes as numeric features (cat="none"; "all" = categorical). Items
    with more than maxc classes in a fold go through _tpf_hier's top-9 +
    'other' split (mc="hier"; else the fold marginal); items left when the
    deadline passes get the fold marginal (the blend then gives this member no
    weight there)."""
    if not torch.cuda.is_available():
        raise RuntimeError("tabpfn needs a GPU")  # skipped (e.g. CPU smoke check)
    # With a GPU present, a broken install or a missing checkpoint is an
    # environment fault that must fail the run loudly rather than silently
    # dropping this member (a failed attempt can be resubmitted).
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except Exception as exc:  # noqa: BLE001
        raise TabPFNEnvError("tabpfn import failed: %r" % (exc,)) from exc
    ckpt = os.path.join(HERE, TABPFN_CKPT)
    if not os.path.exists(ckpt):
        raise TabPFNEnvError("missing " + TABPFN_CKPT)
    try:
        tsd, prev_backends = _tpf_no_cudnn_attention()
    except Exception as exc:  # noqa: BLE001  (speed only; deadline guard covers it)
        if STRICT:
            raise
        _log("tabpfn attention patch failed", repr(exc))
        tsd = prev_backends = None
    Xc, Xc_test, Y, K, folds, Kmax = (P["Xc"], P["Xc_test"], P["Y"], P["K"],
                                      P["folds"], P["Kmax"])
    X = Xc.astype(np.float32)
    Xt = Xc_test.astype(np.float32)
    X, Xt = _hc_remap(P, cfg, X, Xt)
    n, J = Y.shape
    eps = cfg.get("eps", 1e-3)
    cat = list(range(X.shape[1])) if cfg.get("cat", "none") == "all" else None
    hier = cfg.get("mc", "hier") == "hier"
    maxc = int(cfg.get("maxc", 10))
    version = {"v2": ModelVersion.V2, "v3": ModelVersion.V3}[cfg.get("ver", "v3")]
    oof = np.zeros((n, J, Kmax), np.float32)
    test = np.zeros((len(Xt), J, Kmax), np.float32)
    n_fit = n_late = n_many = 0

    def fit(Xtr, ytr, Xq):
        return _tpf_fit_proba(TabPFNClassifier, version, Xtr, ytr, Xq,
                              cfg, ckpt, cat)

    n_full = 0
    try:
        for j in range(J):
            k = K[j]
            late = time.time() > deadline
            n_late += int(late)
            # held-out-fold predictions (OOF, for the blend) from one fit per fold
            for tr, va in folds:
                ytr = Y[tr, j]
                marg = (np.bincount(ytr, minlength=k) + 0.5) / (len(ytr) + 0.5 * k)
                ncls = len(np.unique(ytr))
                many = ncls > maxc
                if late or ncls < 2 or (many and not hier):
                    oof[va, j, :k] = marg
                else:
                    try:
                        pv = _tpf_hier(fit, X[tr], ytr, X[va], k, maxc=maxc)
                    except Exception as exc:  # noqa: BLE001
                        raise TabPFNEnvError("tabpfn fit failed: %r" % (exc,)) from exc
                    oof[va, j, :k] = (1 - eps) * pv + eps * marg
                    n_fit += 1
                    n_many += int(many)
            # hidden-row predictions from one fit on ALL visible rows (KC32:
            # +0.0001..0.0005 over the fold average on 9/9 proxy splits)
            y = Y[:, j]
            marg = (np.bincount(y, minlength=k) + 0.5) / (n + 0.5 * k)
            ncls = len(np.unique(y))
            if late or ncls < 2 or (ncls > maxc and not hier) or len(Xt) == 0:
                test[:, j, :k] = marg
            else:
                try:
                    fitf = fit
                    if ncls >= HIK_MIN:
                        cfg_h = dict(cfg, est=HIK_EST)
                        fitf = (lambda a_, b_, q_: _tpf_fit_proba(
                            TabPFNClassifier, version, a_, b_, q_, cfg_h, ckpt, cat))
                    pt = _tpf_hier(fitf, X, y, Xt, k, maxc=maxc)
                except Exception as exc:  # noqa: BLE001
                    raise TabPFNEnvError("tabpfn fit failed: %r" % (exc,)) from exc
                test[:, j, :k] = (1 - eps) * pt + eps * marg
                n_full += 1
    finally:
        if tsd is not None:
            tsd._SDPA_BACKENDS = prev_backends
    _log("tabpfn item-folds fitted %d (>maxc classes %d), full fits %d, items past deadline %d/%d"
         % (n_fit, n_many, n_full, n_late, J))
    if n_fit == 0:
        raise RuntimeError("tabpfn produced no fits")
    return oof, test

# KC42: high-K items (>= HIK_MIN observed classes) get HIK_EST ensemble members in
# the single all-visible-rows FM fit that predicts the hidden rows
HIK_MIN, HIK_EST = 11, 16
TABICL_CKPT = "tabicl-classifier-v2-20260212.ckpt"  # jingang/TabICL @4dcd344e (BSD-3)
_TICL_MODEL = {}


def _ticl_classifier():
    """TabICLClassifier whose checkpoint is read from disk once per process
    (the stock _load_model re-reads the 110 MB file on every fit)."""
    from tabicl import TabICLClassifier
    from tabicl._model import TabICL

    class _Cached(TabICLClassifier):
        def _load_model(self):
            key = str(self.model_path)
            if key not in _TICL_MODEL:
                ck = torch.load(self.model_path, map_location="cpu", weights_only=True)
                m = TabICL(**ck["config"])
                m.load_state_dict(ck["state_dict"])
                m.eval()
                _TICL_MODEL[key] = (m, ck["config"])
            self.model_, self.model_config_ = _TICL_MODEL[key]
            self.model_path_ = self.model_path
    return _Cached


def m_tabicl(P, cfg, deadline):
    """Per-item TabICLv2 classifier (Soda/Inria, BSD-3; weights bundled:
    jingang/TabICL @4dcd344e tabicl-classifier-v2-20260212.ckpt), one
    in-context fit per fold for the held-out-fold predictions and one on all
    visible rows for the hidden rows, on the GIVEN codes as numeric features; items with
    more classes than the model's native limit use tabicl's own hierarchical
    classification. Attention runs under flash / memory-efficient / math SDPA
    kernels only: PyTorch's default choice includes cuDNN attention, which
    builds a new graph for every new input shape (tens of seconds each).
    Items left when the deadline passes get the fold marginal."""
    if not torch.cuda.is_available():
        raise RuntimeError("tabicl needs a GPU")  # skipped (e.g. CPU smoke check)
    try:
        Cls = _ticl_classifier()
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:  # noqa: BLE001
        raise TabPFNEnvError("tabicl import failed: %r" % (exc,)) from exc
    ckpt = os.path.join(HERE, TABICL_CKPT)
    if not os.path.exists(ckpt):
        raise TabPFNEnvError("missing " + TABICL_CKPT)
    Xc, Xc_test, Y, K, folds, Kmax = (P["Xc"], P["Xc_test"], P["Y"], P["K"],
                                      P["folds"], P["Kmax"])
    X = Xc.astype(np.float32)
    Xt = Xc_test.astype(np.float32)
    X, Xt = _hc_remap(P, cfg, X, Xt)
    n, J = Y.shape
    eps = cfg.get("eps", 1e-3)
    oof = np.zeros((n, J, Kmax), np.float32)
    test = np.zeros((len(Xt), J, Kmax), np.float32)
    n_fit = n_late = n_err = 0
    err = None
    kernels = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
               SDPBackend.MATH]
    def fit_proba(Xtr, ytr, Xq, k, est=None):
        clf = Cls(n_estimators=est or cfg.get("est", 8), model_path=ckpt,
                  allow_auto_download=False, device="cuda",
                  random_state=cfg.get("seed", 42),
                  norm_methods=cfg.get("norm"))
        clf.fit(Xtr, ytr)
        pr = clf.predict_proba(Xq)
        out = np.zeros((len(Xq), k))
        out[:, clf.classes_.astype(int)] = pr
        return out

    n_full = 0
    with sdpa_kernel(kernels):
        for j in range(J):
            k = K[j]
            late = time.time() > deadline
            n_late += int(late)
            # held-out-fold predictions (OOF, for the blend) from one fit per fold
            for tr, va in folds:
                ytr = Y[tr, j]
                marg = (np.bincount(ytr, minlength=k) + 0.5) / (len(ytr) + 0.5 * k)
                if late or len(np.unique(ytr)) < 2:
                    oof[va, j, :k] = marg
                    continue
                try:
                    pv = fit_proba(X[tr], ytr, X[va], k)
                    oof[va, j, :k] = (1 - eps) * pv + eps * marg
                    n_fit += 1
                except Exception as exc:  # noqa: BLE001
                    # one odd item-fold falls back to its marginal; a
                    # systematic failure is raised below
                    if STRICT:
                        raise
                    n_err += 1
                    err = exc
                    oof[va, j, :k] = marg
            # hidden-row predictions from one fit on ALL visible rows (KC32:
            # +0.0001..0.0005 over the fold average on 9/9 proxy splits)
            y = Y[:, j]
            marg = (np.bincount(y, minlength=k) + 0.5) / (n + 0.5 * k)
            if late or len(np.unique(y)) < 2 or len(Xt) == 0:
                test[:, j, :k] = marg
                continue
            try:
                pt = fit_proba(X, y, Xt, k,
                               est=HIK_EST if len(np.unique(y)) >= HIK_MIN else None)
                test[:, j, :k] = (1 - eps) * pt + eps * marg
                n_full += 1
            except Exception as exc:  # noqa: BLE001
                if STRICT:
                    raise
                n_err += 1
                err = exc
                # keep the item consistent for the blend: marginal everywhere
                test[:, j, :k] = marg
                for tr, va in folds:
                    ytr = Y[tr, j]
                    oof[va, j, :k] = ((np.bincount(ytr, minlength=k) + 0.5)
                                      / (len(ytr) + 0.5 * k))
    _log("tabicl item-folds fitted %d, full fits %d, failed %d, items past deadline %d/%d"
         % (n_fit, n_full, n_err, n_late, J))
    if n_err > max(5, n_fit):
        raise TabPFNEnvError("tabicl fits failing: %d of %d, last %r"
                             % (n_err, n_err + n_fit, err))
    if n_fit == 0:
        raise RuntimeError("tabicl produced no fits")
    return oof, test


MODEL_FNS = {"marginal": m_marginal, "mlp": m_mlp, "mlpb": m_mlpb,
             "lgb": m_lgb, "backoff": m_backoff, "kernel": m_kernel,
             "wide": m_wide, "tabpfn": m_tabpfn,
             "tabicl": m_tabicl}


# ------------------------------------------------------------------ blend --

def em_weights(L, iters=500, tol=1e-9, prior=None, c=0.0):
    """Mixture weights maximising mean log(sum_m w_m L[:, m]) (+ c * sum_m
    prior_m log w_m, a Dirichlet MAP pull towards `prior`); L >= 0."""
    n, M = L.shape
    w = np.full(M, 1.0 / M) if prior is None else np.asarray(prior, float).copy()
    L = np.maximum(L, 1e-300)
    prev = -np.inf
    for _ in range(iters):
        mix = L @ w
        ll = np.log(mix).mean()
        r = (L * w / mix[:, None]).sum(0)
        if prior is not None and c > 0:
            w = (r + c * prior) / (n + c)
        else:
            w = r / n
        if ll - prev < tol:
            break
        prev = ll
    return w


def _temper(p, T):
    lp = np.log(np.maximum(p, 1e-300)) / T
    lp -= lp.max(-1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(-1, keepdims=True)


def _tempfit(po, y, rows):
    best = np.log(np.maximum(po[rows, y], 1e-300)).mean()
    T = 1.0
    for t in (0.8, 0.85, 0.9, 0.95, 1.05, 1.1):
        v = np.log(np.maximum(_temper(po, t)[rows, y], 1e-300)).mean()
        if v > best + 1e-6:
            best, T = v, t
    return T, best


def _loglin(LO, LT, y, rows):
    """Log-linear pool: p ~ exp(sum_m w_m log p_m), w = softplus(theta) >= 0,
    fitted by L-BFGS on the OOF log score. LO/LT: (M, n, k) log-probs."""
    from scipy.optimize import minimize
    n = LO.shape[1]
    onehot = np.zeros(LO.shape[1:])
    onehot[rows, y] = 1.0

    def f(theta):
        w = np.log1p(np.exp(theta))
        sc = np.tensordot(w, LO, 1)
        mx = sc.max(-1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(sc - mx).sum(-1))
        ll = sc[rows, y] - lse
        p = np.exp(sc - lse[:, None])
        g_w = ((onehot - p)[None] * LO).sum((1, 2)) / n
        return -ll.mean(), -g_w / (1.0 + np.exp(-theta))

    M = LO.shape[0]
    th0 = np.full(M, np.log(np.expm1(1.0 / M)))
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 200})
    w = np.log1p(np.exp(r.x))

    def pool(L):
        sc = np.tensordot(w, L, 1)
        sc -= sc.max(-1, keepdims=True)
        e = np.exp(sc)
        return e / e.sum(-1, keepdims=True)
    return pool(LO), pool(LT), w


def blend(P, models, temperature=False, names=None, loglin=False):
    """Per item: EM linear pool (+temperature) and, if loglin, a log-linear pool
    (+temperature); the one with the better OOF log score is kept. Returns
    (test, oof, info)."""
    Y, K, n, J = P["Y"], P["K"], P["n"], P["J"]
    names = names or list(models)
    final = np.zeros((P["n_test"], J, P["Kmax"]), np.float64)
    foof = np.zeros((n, J, P["Kmax"]), np.float64)
    info = []
    rows = np.arange(n)
    shrink = float(CONFIG.get("shrink", 0.0))
    wbar = None
    if shrink > 0:
        wbar = np.mean([em_weights(np.stack([models[m][0][rows, j, Y[:, j]] for m in names],
                                            1).astype(np.float64)) for j in range(J)], 0)
        wbar = np.maximum(wbar, 1e-4)
        wbar /= wbar.sum()
    for j in range(J):
        k = K[j]
        y = Y[:, j]
        L = np.stack([models[m][0][rows, j, y] for m in names], 1).astype(np.float64)
        w = em_weights(L, prior=wbar, c=shrink)
        po = sum(wi * models[m][0][:, j, :k].astype(np.float64) for wi, m in zip(w, names))
        pt = sum(wi * models[m][1][:, j, :k].astype(np.float64) for wi, m in zip(w, names))
        T = 1.0
        best = np.log(np.maximum(po[rows, y], 1e-300)).mean()
        if temperature:
            T, best = _tempfit(po, y, rows)
            if T != 1.0:
                po, pt = _temper(po, T), _temper(pt, T)
        kind = "lin"
        if loglin and len(names) > 1:
            try:
                LO = np.stack([np.log(np.maximum(models[m][0][:, j, :k].astype(np.float64), 1e-12))
                               for m in names])
                LT = np.stack([np.log(np.maximum(models[m][1][:, j, :k].astype(np.float64), 1e-12))
                               for m in names])
                qo, qt, w2 = _loglin(LO, LT, y, rows)
                T2 = 1.0
                b2 = np.log(np.maximum(qo[rows, y], 1e-300)).mean()
                if temperature:
                    T2, b2 = _tempfit(qo, y, rows)
                    if T2 != 1.0:
                        qo, qt = _temper(qo, T2), _temper(qt, T2)
                if np.isfinite(b2) and b2 > best + 1e-6:
                    po, pt, w, T, kind = qo, qt, w2, T2, "log"
            except Exception:  # noqa: BLE001
                if STRICT:
                    raise
        final[:, j, :k] = pt
        foof[:, j, :k] = po
        info.append((w, T, kind))
    return final, foof, info


def _bin_temper(pi, T):
    pi = np.clip(pi, 1e-9, 1 - 1e-9)
    return 1.0 / (1.0 + np.exp(-np.log(pi / (1 - pi)) / T))


def _apply_open(Pm, j, k, pi):
    body = Pm[:, j, :k - 1]
    body = body / np.maximum(body.sum(1, keepdims=True), 1e-12)
    Pm[:, j, :k - 1] = body * pi[:, None]
    Pm[:, j, k - 1] = 1.0 - pi


def _lp_items(Pm, Y, K, js):
    out = 0.0
    r = np.arange(len(Y))
    for j in js:
        p = Pm[:, j, :K[j]]
        p = p / p.sum(1, keepdims=True)
        p = 1e-3 + (1 - K[j] * 1e-3) * p
        out += np.log(p[r, Y[:, j]]).mean()
    return out


def gate_pool(P, final, foof):
    """Gated items whose NA pattern on the visible rows is identical share one
    open-event. Candidates for P(open|x): members' mean (pool), an exactly
    determining PREDICT item's blended mass on its opening set (par), their
    mean (mix), each also with a binary temperature. Kept per family only when
    the family's summed OOF log score improves."""
    recs, targets, Y, K = P["recs"], P["targets"], P["Y"], P["K"]
    J = len(targets)
    fam = {}
    for j, t in enumerate(targets):
        if not recs[t].get("gate"):
            continue
        na = Y[:, j] == K[j] - 1
        if na.all() or not na.any():
            continue
        fam.setdefault(na.tobytes(), []).append(j)
    n_used = 0
    for js in fam.values():
        j0 = js[0]
        openv = Y[:, j0] != K[j0] - 1
        cand = {}
        po = np.mean([1 - foof[:, j, K[j] - 1] for j in js], 0)
        pt = np.mean([1 - final[:, j, K[j] - 1] for j in js], 0)
        cand["pool"] = (po, pt)
        for p in range(J):
            if p in js:
                continue
            S, ok = [], True
            for a in range(K[p]):
                m = Y[:, p] == a
                if not m.any():
                    continue
                f = openv[m].mean()
                if f == 1.0:
                    S.append(a)
                elif f != 0.0:
                    ok = False
                    break
            if ok and S and len(S) < K[p]:
                qo, qt = foof[:, p, S].sum(1), final[:, p, S].sum(1)
                cand["par"] = (qo, qt)
                cand["mix"] = ((qo + po) / 2, (qt + pt) / 2)
                break
        for nm in list(cand):
            co, ct = cand[nm]
            best, bT = None, 1.0
            for T in (1.0, 0.7, 0.8, 0.9, 1.1, 1.25):
                c = _bin_temper(co, T)
                v = np.mean(np.log(np.where(openv, c, 1 - c)))
                if best is None or v > best + 1e-7:
                    best, bT = v, T
            if bT != 1.0:
                cand[nm + "_t"] = (_bin_temper(co, bT), _bin_temper(ct, bT))
        base = _lp_items(foof, Y, K, js)
        bestv, bestc = base + 1e-6 * len(js), None
        for nm, (co, ct) in cand.items():
            tmp = foof[:, js, :].copy()
            for ii, j in enumerate(js):
                _apply_open(tmp, ii, K[j], np.clip(co, 1e-6, 1 - 1e-6))
            v = _lp_items(tmp, Y[:, js], K[js], range(len(js)))
            if v > bestv:
                bestv, bestc = v, nm
        if bestc is not None:
            co, ct = cand[bestc]
            for j in js:
                _apply_open(final, j, K[j], np.clip(ct, 1e-6, 1 - 1e-6))
                _apply_open(foof, j, K[j], np.clip(co, 1e-6, 1 - 1e-6))
            n_used += 1
    _log("gate_pool families %d used %d" % (len(fam), n_used))
    return final, foof


def unmix(p, eps=1e-3):
    """Floor compensation. The grader scores eps + (1-K eps) q; return q so that
    the scored vector is the log-optimal p' >= eps for belief p (water-filling)."""
    p = p / p.sum(-1, keepdims=True)
    k = p.shape[-1]
    lam = np.ones(p.shape[:-1] + (1,))
    for _ in range(8):
        low = p / lam <= eps
        num = np.where(low, 0.0, p).sum(-1, keepdims=True)
        den = 1.0 - low.sum(-1, keepdims=True) * eps
        lam = num / np.maximum(den, 1e-12)
    t = np.maximum(eps, p / lam)
    t = t / t.sum(-1, keepdims=True)
    q = np.maximum(t - eps, 0.0) / (1 - k * eps)
    return q + 1e-12



def class_pool(P, final, foof, min_size=2, skip_gated_na=True):
    """Generalised gate pooling for any (item, option) event: options whose
    indicator pattern on the visible rows is identical across >= min_size
    (item, option) pairs (e.g. a shared 'Not answered' from follow-up
    attrition) share one event probability (members' mean, optionally
    temperature-scaled). Adopted per group only if the members' summed OOF log
    score improves. Gated items' sentinel slot is left to gate_pool."""
    recs, targets, Y, K = P["recs"], P["targets"], P["Y"], P["K"]
    n = len(Y)
    groups = {}
    for j, t in enumerate(targets):
        for c in range(K[j]):
            if skip_gated_na and recs[t].get("gate") and c == K[j] - 1:
                continue
            ev = Y[:, j] == c
            m = int(ev.sum())
            if m < 5 or m > n - 5:
                continue
            groups.setdefault(ev.tobytes(), []).append((j, c))
    groups = [g for g in groups.values() if len({j for j, _ in g}) >= min_size]
    n_used = 0

    def put(Pm, j, c, pi, k):
        p = Pm[:, j, :k]
        rest = 1.0 - p[:, c]
        scale = (1.0 - pi) / np.maximum(rest, 1e-12)
        q = p * scale[:, None]
        q[:, c] = pi
        Pm[:, j, :k] = q

    for g in groups:
        js = [j for j, _ in g]
        if len(set(js)) != len(js):
            continue
        j0, c0 = g[0]
        ev = Y[:, j0] == c0
        po = np.clip(np.mean([foof[:, j, c] for j, c in g], 0), 1e-6, 1 - 1e-6)
        pt = np.clip(np.mean([final[:, j, c] for j, c in g], 0), 1e-6, 1 - 1e-6)
        cand = {"pool": (po, pt)}
        best, bT = None, 1.0
        for T in (1.0, 0.7, 0.8, 0.9, 1.1, 1.25):
            q = _bin_temper(po, T)
            v = np.mean(np.log(np.where(ev, q, 1 - q)))
            if best is None or v > best + 1e-7:
                best, bT = v, T
        if bT != 1.0:
            cand["pool_t"] = (_bin_temper(po, bT), _bin_temper(pt, bT))
        base = _lp_items(foof, Y, K, js)
        bestv, bestc = base + 1e-6 * len(js), None
        for nm, (co, ct) in cand.items():
            tmp = foof[:, js, :].copy()
            for ii, (j, c) in enumerate(g):
                put(tmp, ii, c, np.clip(co, 1e-6, 1 - 1e-6), K[j])
            v = _lp_items(tmp, Y[:, js], K[js], range(len(js)))
            if v > bestv:
                bestv, bestc = v, nm
        if bestc is not None:
            co, ct = cand[bestc]
            for j, c in g:
                put(final, j, c, np.clip(ct, 1e-6, 1 - 1e-6), K[j])
                put(foof, j, c, np.clip(co, 1e-6, 1 - 1e-6), K[j])
            n_used += 1
    _log("class_pool groups %d used %d" % (len(groups), n_used))
    return final, foof


def gate_rule(P, final):
    recs, given, targets, vis, hid, Y, K = (P["recs"], P["given"], P["targets"],
                                            P["vis"], P["hid"], P["Y"], P["K"])
    for j, t in enumerate(targets):
        gate = recs[t].get("gate")
        if not gate or gate["parent"] not in given:
            continue
        par = gate["parent"]
        opened_v = vis[par].isin(gate["observed_if"]).to_numpy()
        is_na = Y[:, j] == K[j] - 1
        if np.any(opened_v == is_na):
            continue
        n_closed = int((~opened_v).sum())
        n_open = int(opened_v.sum())
        opened_t = hid[par].isin(gate["observed_if"]).to_numpy()
        p = final[:, j, :K[j]]
        body = p[:, :-1] / np.maximum(p[:, :-1].sum(1, keepdims=True), 1e-12)
        p_na = np.where(opened_t, 1.0 / (n_open + 2.0), 1.0 - 1.0 / (n_closed + 2.0))
        final[:, j, :K[j]] = np.concatenate([body * (1 - p_na)[:, None],
                                             p_na[:, None]], 1)
    return final


def emit(frame, P, final, fallback):
    items, targets, opts, K = P["items"], P["targets"], P["opts"], P["K"]
    tpos = {t: j for j, t in enumerate(targets)}
    grid = frame[items].isna().to_numpy()
    hid_rows = np.flatnonzero(P["hidden_mask"])
    hid_index = {r: i for i, r in enumerate(hid_rows)}
    out = []
    for r in np.flatnonzero(grid.any(1)):
        row = grid[r]
        for ci in np.flatnonzero(row):
            item = items[ci]
            if item in tpos and r in hid_index:
                j = tpos[item]
                v = final[hid_index[r], j, :K[j]]
                if not np.all(np.isfinite(v)) or v.sum() <= 0 or np.any(v < 0):
                    if STRICT:
                        raise ValueError("bad vector")
                    v = fallback[0, j, :K[j]].astype(np.float64)
                out.append(v.tolist())
            else:
                c = frame[item].value_counts()
                w = np.array([c.get(o, 0) + 0.5 for o in opts[item]], float)
                out.append((w / w.sum()).tolist())
    return out


# ---------------------------------------------------------------- predict --

def run_models(P, deadline, config=CONFIG, budget=900.0):
    models = {}
    lib = list(config["models"])
    if budget >= config.get("extra_min_budget", 1e9):
        lib += list(config.get("models_extra", []))
    for name, kind, cfg in lib:
        if time.time() > deadline and kind != "marginal":
            _log("skip", name, "(deadline)")
            continue
        t1 = time.time()
        try:
            models[name] = MODEL_FNS[kind](P, cfg, deadline)
            _log(name, "done %.1fs" % (time.time() - t1))
        except TabPFNEnvError:
            raise
        except Exception as exc:  # noqa: BLE001
            if STRICT:
                raise
            _log(name, "failed", type(exc).__name__)
    return models


def predict(frame, schema):
    t0 = time.time()
    inst = _instrument(schema)
    P = prepare(frame, schema, n_folds=CONFIG["folds"])
    split = schema.get("split", {})
    # phase 2 (test, 3600 s): TEST rows hidden, TRAIN+DEV visible (kit/score.py
    # sample_rows takes every row of each role); either count identifies it
    n_tr, n_dv, n_te = (split.get(k) for k in ("n_train", "n_dev", "n_test"))
    phase2 = (P["n_test"] == n_te) or (
        n_tr is not None and n_dv is not None and P["n"] == n_tr + n_dv
        and P["n_test"] != n_dv)
    budget = 3600.0 if phase2 else 900.0
    safe_end = T_IMPORT + SAFE_FRACTION * budget
    remaining = [i for i in ("unicef", "world_bank", "unhcr")
                 if i not in _DONE and i != inst]
    share = SHARE.get(inst, 0.33)
    tot = share + sum(SHARE[i] for i in remaining)
    deadline = t0 + max(30.0, (safe_end - t0) * share / tot)
    _log(inst, "n_vis", P["n"], "n_hid", P["n_test"], "budget", budget,
         "deadline in %.0fs" % (deadline - t0))
    models = run_models(P, deadline, budget=budget)
    final, foof, info = blend(P, models, temperature=CONFIG["temperature"],
                              loglin=CONFIG.get("loglin", False))
    if VERBOSE:
        names = list(models)
        W = np.array([w for w, _, _ in info])
        _log("mean blend weights", dict(zip(names, np.round(W.mean(0), 3))),
             "loglin items", sum(1 for _, _, kd in info if kd == "log"))
    if CONFIG.get("gate_pool"):
        try:
            final, foof = gate_pool(P, final, foof)
        except Exception as exc:  # noqa: BLE001
            if STRICT:
                raise
            _log("gate_pool failed", type(exc).__name__)
    if CONFIG.get("class_pool"):
        try:
            final, foof = class_pool(P, final, foof,
                                     skip_gated_na=bool(CONFIG.get("gate_pool")))
        except Exception as exc:  # noqa: BLE001
            if STRICT:
                raise
            _log("class_pool failed", type(exc).__name__)
    if CONFIG["gate_rule"]:
        final = gate_rule(P, final)
    if CONFIG.get("unmix"):
        for j in range(P["J"]):
            k = P["K"][j]
            final[:, j, :k] = unmix(final[:, j, :k])
    out = emit(frame, P, final, models["marginal"][1])
    _DONE.append(inst)
    _log(inst, "done in %.1fs" % (time.time() - t0))
    return out
