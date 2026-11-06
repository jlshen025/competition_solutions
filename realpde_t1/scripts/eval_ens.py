"""Shared inference helpers for the calibration scripts (newtab.py, vfit.py):
load a trained run dir and run members over windows in physical units."""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rpde.model import Net

# Fixed global normalisation (the same constants as train.py).
MEAN_R = np.array([0.154960856, -0.000513992854], np.float32)
STD_R = np.array([0.0968056545, 0.015960684], np.float32)
# Options of the research code that this release does not implement; a run
# whose results.json enables any of them cannot be loaded here.
_UNSUPPORTED = ("spectral", "multiscale", "tshare", "n_glob", "n_aux",
                "aux_fine", "ar_detach", "cascade")


def spec_of(run):
    """Net constructor arguments recorded in a run dir's results.json."""
    rj = json.load(open(os.path.join(run, "results.json")))["args"]
    bad = [k for k in _UNSUPPORTED if rj.get(k)]
    if bad:
        raise ValueError(f"{run}: unsupported options {bad}")
    return dict(width=rj["width"], depth=rj["depth"],
                delta=bool(rj.get("delta") or 0), ar_m=int(rj.get("ar_m") or 0))


def load(run):
    net = Net(**spec_of(run)).cuda()
    net.load_state_dict(torch.load(os.path.join(run, "model.pth"), map_location="cuda"))
    return net.eval()


def infer(nets, x, batch=64):
    """-> per-member means (M,N,20,H,W,2) and the members' mean predicted
    variance (N,20,H,W,2), both in physical units."""
    mean = torch.tensor(MEAN_R, device="cuda"); std = torch.tensor(STD_R, device="cuda")
    lg = float(np.log(STD_R).mean())
    MU, VAR = [], []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch, ..., :2])).cuda()
            xn = (xb - mean) / std
            outs = [n(xn) for n in nets]
            mus = torch.stack([o[0] for o in outs], 0) * std + mean
            var = torch.stack([(o[1] + lg).exp() ** 2 for o in outs], 0).mean(0)
            MU.append(mus.float().cpu().numpy()); VAR.append(var.float().cpu().numpy())
    return np.concatenate(MU, 1), np.concatenate(VAR, 0)
