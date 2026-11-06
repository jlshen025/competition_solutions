# JLShen: Track 1 method

## 1. Summary
We use an ensemble of four small 2D convolutional operators, trained on real PIV data only. Each one maps 20 input frames of (u, v) to 20 output frames, with time folded into channels and the output predicted as a change from the last input frame. Each network also has a learned per-element log-scale head. The loss adds the scored statistics (TKE, MVPE and an interval-coverage term) to relative L2. Most of the score comes from the calibration after training. A width table maps predicted scale to interval half-width. It is fitted to maximise the SPS element reward on held-out-Reynolds residuals. A small variance head rescales the predicted temporal anomaly. On the development board, the width table alone moved sps_score from 40.6 to 43.7 and final_score by +0.78.

## 2. Model
- Architecture and number of parameters: four members of the same net, `rpde/model.py:Net` with width 64, depth 8 and no spectral layers. Each has about 1.16 M parameters, 4.6 M in total. Two members use recipe A and two use recipe B.
  - Input and lift: the input is 20 frames × (u, v) = 40 channels, plus 2 coordinate channels. A 1×1 convolution lifts this to 64 channels.
  - Trunk: 8 residual blocks. Each block computes `x + GroupNorm(GELU(pw1×1(x) + conv3×3(x) + conv5×5_dilated2(x)))`.
  - Mean head: 1×1 → GELU → 1×1 to `h_out × 2` channels, added to the last observed frame (delta parameterisation).
  - Scale head: 1×1 → GELU → 1×1 to a per-element log standard deviation, clamped to [−9, 2].
  - Recipe A (`shipA`) maps 20 frames to 20 in one step.
  - Recipe B (`shipB`) uses `ar_m = 10`. A 20→10 operator runs twice, and the second step's context is the last 10 inputs plus its own first 10 predictions. Gradients flow through both steps. Inference runs the same recursion.
- Inputs and outputs, including preprocessing and normalization: (u, v) are z-scored with fixed global real-data constants: mean (0.15496, −0.00051), std (0.09681, 0.01596). These are computed from `train_real` and written in the code, not taken per window. The pressure channel is ignored on input and returned as 0. Resolution is 32×64, the release subsampled `[::2, ::2]`, the same as the official loader.
- Starting point: random initialisation. No official or external checkpoint is used.

## 3. Data
- Training data used: `train_real/` only, 81 trajectories and 65,926 windows at stride 1. Each window is 20 input frames and 20 target frames. `train_sim` is not used, because joint sim+real training and sim pretraining both lost on held-out Reynolds numbers in our development.
- Validation split carved out of the training data: all trajectories at Re 13950 and Re 22875, 8 trajectories. They are held out for the four CV nets, which are used only for calibration. The four ship nets train on all 81 trajectories.
- Augmentation, on the fly, 4 of the 32 windows in each batch for each kind:
  - Reynolds-similarity rescaling: `u'(x, t) = k·u(x, k·t)`, which is an exact Navier–Stokes solution at Re' = k²·Re. k is drawn log-uniform in [0.75, 1.6], with linear interpolation in time between released frames.
  - Cross-stream mirror: flip the y axis and negate v.
- Data left out, and why: `train_real/7575_0.h5`, which duplicates `6300_0.h5` per the organizers' notice.

## 4. Training
`train()` runs these stages in this order, one network at a time.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 0 prep | `train_real/*.h5` | – | – | – | – |
| 1 shipA_s250, shipA_s350 | all real windows + aug | L (below) | AdamW, lr 2e-3, wd 1e-4, OneCycle (5% warm-up, cosine anneal), bf16 autocast, grad-clip 1.0 | 60,000 steps | 32 |
| 2 shipB_s250, shipB_s350 | same | L, through the 2-step rollout | same | 60,000 | 32 |
| 3 cvA_s250, cvA_s270 | real without Re 13950 / 22875 + aug | L | same | 60,000 | 32 |
| 4 cvB_s250, cvB_s270 | same as 3 | L, rollout | same | 60,000 | 32 |
| 5 width table | CV nets on the held-out-Re windows (stride 20) | per-bucket argmax of the SPS element reward | closed form | – | – |
| 6 variance head | ship nets on all training windows (stride 20) | ridge regression, λ = 1e-3 | closed form | – | – |

L is computed in physical units, where r = target − μ and ŝ = exp(scale head):
- `relL2(μ, y)`: relative L2 per window
- `+ 0.15 · relL2(TKE field)`: the spatial map of the temporal variance of u and v
- `+ 0.249 · MVPE`: the time-mean velocity profile error at the scorer's 9×4 wake probes
- `+ 3.0 · mean(1 − exp(−(r / 2ŝ)²))`: a coverage term, with ŝ detached, so it trains μ to keep residuals inside the predicted scale
- `+ 0.2 · Gaussian NLL(r, ŝ)`: with r detached, so it trains only the scale head

- How runs or checkpoints are selected: each network scores its checkpoint every 500 steps as `0.737 · rel_l2_score + 0.123 · tke_score`, on the Re 13950 / 22875 windows, and keeps the best one. For the CV nets those windows are held out. For the ship nets they are training data, so the pick is in-sample. In the run behind 947215 it landed between steps 46,500 and 59,500, near the end of the anneal. There is no selection across runs: the ensemble is always the four ship nets.
- Seeds, and the run-to-run spread: ship seeds 250 and 350, CV seeds 250 and 270. Two full from-scratch retrains with different seeds scored 81.861 and 81.846 on the development board (rel_l2 94.849 vs 94.794, tke 79.411 vs 79.491). On held-out Re, single-net rel_l2_score varied by about 0.03 across seeds (96.489 to 96.522 in the CV logs).

## 5. Uncertainty bounds
Every step below uses only the window being answered.
1. Predictive scale per element: `ŝ² = mean_m exp(scale_m)² + 0.75 · Var_m(μ_m)`. This is the members' own variance plus κ = 0.75 times their disagreement.
2. Half-width: `h = 0.95 · table(ŝ)`, linearly interpolated between bucket centres. The table has 64 quantile buckets of ŝ. In each bucket, h is the value that maximises the scorer's element reward `exp(−2h/σ) · P(|r| ≤ h)`, with σ = 0.0563870, computed on residuals of the four CV nets on the two held-out Reynolds numbers. The factor 0.95 and κ were chosen by cross-fitting between the two held-out Re values: fit on one, score on the other, in both directions.
3. Bounds: `lower = μ − h`, `upper = μ + h`, computed on the GPU.

## 6. Inference
- Track 1: each `predict` call does the following:
  1. Z-score the input.
  2. Run the four members as one fused grouped-convolution trunk in fp16 (see §7). Statistics are computed in fp32.
  3. Average the members to get μ, and compute ŝ as in §5.
  4. Amplify μ's temporal anomaly about its window mean: `μ = m̄ + 1.05·(μ − m̄)`.
  5. Apply the variance head. A degree-2 polynomial ridge model on 7 features estimates the log temporal variance of the true anomaly at each pixel. The features, all from this window and the model's own output on it, are: predicted anomaly variance, mean ŝ², input anomaly variance, its 3×3 smoothed version, the y and x coordinates, and the window-mean predicted variance. The anomaly is rescaled by the gain `1 + t·(g − 1)`, with g clamped to [0.2, 5]. t is 0.7 where g ≥ 1 and 0.2 where g < 1. The gain is applied only to pixels above the 70th percentile of predicted variance in that window.
  6. Compute the §5 bounds.
  7. Return prediction, lower and upper as float16.
- The model is built and warmed up in `__init__`, and a CUDA graph is captured there for each batch-size bucket. Every N is split into captured buckets, so no batch is padded.

## 7. Speed
- The 2D operator folds time into channels. It is about 100× cheaper than the released 3D baselines, and each member is only 1.16 M parameters.
- CUDA graphs per batch-size bucket (16, 12, 8, 6, 5, 4, 3, 2, 1), captured and warmed outside the timed region. One predict is one graph replay per bucket.
- A fused ensemble trunk: the four members run as grouped convolutions, and each block's 1×1 convolution is folded exactly into its 3×3 convolution. The trunk runs in fp16 and the statistics in fp32. This makes the forward pass about 1.7× faster, and held-out accuracy changes by at most 2e-4.
- The output path writes prediction and bounds into one pre-allocated pinned host buffer in float16, with a single contiguous device-to-host copy per array.
- Effects on the development board: CUDA graphs plus the contiguous output path moved time_score from 89.9 to 91.3. Per-bucket graphs moved it from 91.2 to 92.0.

## 8. What mattered
Effects are on the development-board final_score. Each row changes one thing relative to the previous best.
- Refitting the width table on honest held-out residuals, with the λ scale cross-fitted: +0.78. sps_score went from 40.6 to 43.7. This was the largest single gain.
- Spectral-free 2D conv members instead of a 2D FNO-style net, plus ensembling: 79.92 to 80.56.
- Continuous Reynolds-similarity augmentation with the mirror: +0.19 (80.61 to 80.80).
- A four-member blend with two direct and two rollout members: +0.06 (81.64 to 81.70). This gave our best rel_l2_score, 94.95.
- The per-pixel variance head, with λ 0.95: +0.11 (81.70 to 81.81). The asymmetric dose added +0.05 (81.81 to 81.86).
- Speed work on the inference path: time_score went from 89.9 to 92.0 at best, worth about +0.2 final.
- Things that did not help: sim pretraining or joint sim+real training, larger or spectral nets, per-lead amplitude gains, and warm-start fine-tuning.

## 9. Compute
The run behind 947215 trained on 8 × NVIDIA H100 80GB HBM3 MIG `2g.20gb` slices, one per network, in parallel, with 4 CPU cores and 32 GB RAM each. Each recipe-A net took 26 min and each recipe-B net took 46 min, for 60,000 steps. Preprocessing took 5 min, and calibration took a few minutes. Run one after another, the 8 nets add up to about 4.8 h on these slices. Peak GPU memory is about 3.0 GB per training process, measured in the evaluation image.

## 10. References
- Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021. This is the starting point we moved away from.
- Brandstetter et al., *Message Passing Neural PDE Solvers*, ICLR 2022. It covers rollout training for PDE surrogates.
- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017.
- Libraries: PyTorch, NumPy, h5py. No third-party model code.
