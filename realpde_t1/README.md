# NeurIPS 2026 RealPDE Challenge — Track 1 (Sim2Real) · Team JLShen

Training and inference code for our Track 1 entry: 20-frame forecasts of real PIV
measurements of flow past a foil, with per-element uncertainty bounds. Codabench user
**junlong**, team **JLShen**.

**9th place in the final ranking**, decided by the organizers retraining each finalist's
code on a private set of unseen flow conditions.

| phase | final_score | rel_l2 | tke | mvpe | time | sps | rank |
|---|---|---|---|---|---|---|---|
| Final (organizer retrain, private set) | **82.3055** ± 0.0061 | | | | | | **9** of 10 finalists |
| Main Development Phase (submission 947215) | 81.885 | 94.849 | 79.412 | 93.941 | 91.537 | 44.309 | 10 of 182 |

The final ranking publishes only the total score and its standard deviation. On the
development board the entry ranked 2nd of 182 on rel_l2 and 5th on sps.

## Method in brief

- **No simulation data.** Four 1.16M-parameter 2-D convolutional operators (time folded into
  channels, output predicted as a change from the last input frame) are trained on the 81
  real trajectories only. Two map 20 frames to 20 in one step; two roll a 20→10 operator out
  twice.
- **Reynolds-similarity augmentation**, `u'(x,t) = k·u(x,kt)` at `Re' = k²·Re` with continuous
  `k`, plus a cross-stream mirror.
- **Score-optimal intervals.** The challenge's interval reward is independent per element,
  so the optimal half-width solves `max_h exp(-2h/σ)·P(|r| ≤ h)`. A 64-bucket table maps the
  predicted scale to that argmax, fitted on residuals of four cross-validation twins that
  never saw the held-out Reynolds numbers. Refitting this table alone, with predictions
  bit-identical, was worth +0.78 final score on the hidden test.
- **Variance head**: a small ridge model rescales the predicted temporal anomaly per pixel.
- **Fast inference**: the four members run as one fused fp16 grouped-conv trunk with CUDA
  graphs per batch-size bucket.

`METHOD.md` gives the full recipe. `paper/` holds the technical report, *Sim2Real
Without the Sim*. It explains the interval score, the hidden-test submissions, and the
negative results on simulation data, capacity and refiners.

## Layout

| path | contents |
|---|---|
| `solution.py` | organizer interface: `train(data_dir, output_dir, device)`, `load_predictor(output_dir, device)` |
| `scripts/release/run_all.sh` | the pipeline: `prep`, `train <NAME>` (8 nets), `calibrate` |
| `scripts/train.py`, `scripts/rpde/` | training, model, data windows, scorer mirror |
| `scripts/newtab.py`, `scripts/vfit.py` | width-table and variance-head fits |
| `scripts/sub/submission.py` | inference code packed into the predictor bundle |
| `logs/codabench_942156_training_run/` | logs of the training run behind 947215 |
| `paper/` | report source, figures (`figs/make_figs.py`) and PDF |

## Reproduce

```bash
docker build --platform linux/amd64 -t realpde-t1 .
# inside the image, with the release data at /data:
python -c "import solution; solution.train('/data', '/output', 'cuda')"
python -c "import solution, numpy as np; \
  m = solution.load_predictor('/output', 'cuda'); \
  print({k: v.shape for k, v in m.predict(np.zeros((5,20,32,64,3), np.float32)).items()})"
```

`predict` takes `(N, 20, 32, 64, 3)` float32 in physical units and returns `prediction`,
`lower` and `upper` of the same shape in float16. Outside Docker, install
`requirements.txt`. `bash scripts/release/run_all.sh all` runs the same steps; the eight
networks are independent, so `train <NAME>` can run in parallel, one GPU each. Training
takes about 26 min per one-step net and 46 min per rollout net on an H100 MIG 2g.20gb
slice, roughly 4.8 GPU-hours in total; peak GPU memory is about 3 GB.
`RPDE_STEPS=<n>` shortens every run for smoke tests (default 60000, the ranked recipe).

GPU training is not bit-deterministic, so a retrain is a fresh draw of the same recipe. Two
from-scratch draws scored 81.861 (942156, the weights behind 947215) and 81.846 (945473,
other seeds, trained inside the evaluation image).

## Data

Train on `train_real/` of the public release
([AI4Science-WestlakeU/RealPDE-Competition-Data](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data),
CC BY-NC 4.0; not redistributed here). `train_sim/` is not used. `train_real/7575_0.h5`
duplicates `6300_0.h5` and is skipped. No pretrained weights or external data are used.

## Relation to the submitted package

This is the package sent to the organizers, with the exploratory options the shipped
configuration never reaches removed and the internal comments rewritten. We checked it
against the submitted code on an H100 MIG slice, using synthetic data in the release
format:
- the assembled `config.json` and checkpoints are byte-identical;
- predictions and bounds are bitwise identical for N = 1, 5, 7, 16, 23 and 70;
- the width-table and variance-head fits are byte-identical;
- 1000 training steps of each recipe give bitwise-identical weights.

A full 60000-step run on the real data was not repeated. The logs in `logs/` were written by
the code as submitted, so some printed diagnostics differ from what the current scripts
print.

## License

MIT — see `LICENSE`.
