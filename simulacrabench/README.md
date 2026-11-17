# SimulacraBench 2026 (NeurIPS 2026 Competition Track) — team junlong

Predict the answer distribution of every blank survey cell for held-out respondents of three
unreleased instruments (UNICEF Faith & Immunisation, World Bank Skills Assessment, UNHCR ERPIS),
scored by log score normalised against the crowd marginal (skill = 1 + L/R). Codabench user
**JLShen**, team name **junlong**; Test submission 1053622.

| phase | skill (mean) | UNICEF | World Bank | UNHCR |
|---|---|---|---|---|
| Test (exact) | **0.48456** | 0.34250 | 0.23630 | 0.87488 |

Method and evaluation: [`paper/writeup_v15.pdf`](paper/writeup_v15.pdf) (the 4-page method
description filed on OpenReview).

## What is here

This directory is the submitted Test archive (`v15.zip`, sha256 `5a818b36…6f5f88536`) unpacked,
minus the two pretrained checkpoints, which `fetch_weights.py` downloads from their pinned commits.

| file | role |
|---|---|
| `main.py` | the whole submission: `predict(frame, schema)`, called once per instrument. All fitting happens inside it on the visible rows; nothing is pre-trained by us. |
| `requirements.txt` | `lightgbm==4.6.0`, `tabpfn==8.5.0`, `tabicl==2.2.0` (on top of the organizers' image: numpy, pandas, scipy, scikit-learn, torch). |
| `fetch_weights.py` | downloads the two checkpoints next to `main.py` and checks their sha256 (stdlib only). |
| `make_zip.sh` | rebuilds the submission archive from this directory. |
| `NOTICE`, `LICENSE_TABPFN3`, `LICENSE_TABICL` | the third-party notices shipped inside the archive. |
| `paper/` | method description. |

sha256 of the archive's members (`make_zip.sh` prints the same list):

```
e6a71ef4b5f4d7553e12d4595c111520c86786950f538cbd4f0d7f43a2fd5e98  main.py
c5246e1db39863171cca9b65f3f5908457c8bc1078332f1a15aee93295c4df03  requirements.txt
d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988  tabpfn-v3-classifier-v3_default.ckpt
bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0  tabicl-classifier-v2-20260212.ckpt
```

## Pretrained models (used in context, not fine-tuned)

- TabPFN-3 classifier: Hugging Face `Prior-Labs/tabpfn_3` @ `24a16a89d245878b846555110985634aa2e656d7`,
  `tabpfn-v3-classifier-v3_default.ckpt`, TABPFN-3 Non-Commercial License v1.0. Built with TabPFN.
- TabICLv2 classifier: Hugging Face `jingang/TabICL` @ `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`,
  `tabicl-classifier-v2-20260212.ckpt`, BSD 3-Clause.

No LLM and no external data are used by the code. Public GSS 2022 and NHIS 2022 files were used
offline during development only, as format-matched proxies for model selection (see the paper).

## Reproduce

The microdata are not released; the organizers run the code against them. To run it locally on
synthetic frames, use the organizers' starter kit (`SituatedEvals/public`, commit
`aa03186f8f083c7785a1a566c802c2077925e265`): its `make_sandbox.py` writes schema-shaped practice
data and its `score.py` runs a submission the way the server does.

```sh
python fetch_weights.py          # ~323 MB
./make_zip.sh                    # -> simulacrabench_v15.zip
# in the starter kit:
python tools/check_submission_zip.py simulacrabench_v15.zip
python make_sandbox.py --schema data/unhcr.json --out sandbox/unhcr
python score.py --submission <unzipped dir> --data sandbox/unhcr --schema data/unhcr.json --phase 2
```

Runtime target: one H100, 16 GB RAM, Python 3.13, no network; the three `predict()` calls share
3600 s in the Test phase (900 s in development). `main.py` keeps itself inside that budget with
per-instrument deadlines: the foundation-model members stop item by item when time runs out, and
those items fall back to the core library. A full Test-shaped run took about 2250 s on a whole
H100. Environment flags: `SIMU_VERBOSE=1` logs progress; `SIMU_STRICT=1` raises instead of
falling back.

## License

Code (`main.py`, `fetch_weights.py`, `make_zip.sh`): MIT, see `LICENSE`. The checkpoints keep
their own licences (above); the TabPFN-3 weights are for non-commercial use only.
