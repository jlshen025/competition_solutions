# Competition code releases

Training and inference code for the challenge entries of team **JLShen**. One directory per entry,
each self-contained and released under the MIT license; the Codabench username is stated per
challenge below.

A directory is the code archive as it was sent to that challenge's organizers, unpacked
verbatim unless its entry below says otherwise: its own `README.md` (method, environment, the
exact commands that produced the submitted file, and the local validation protocol), `LICENSE`,
requirements, and `code/`.
Start from the directory's README — this page is only an index.

No challenge dataset is included or redistributable here; request the data from the organizers
of the challenge in question. Each entry reads its dataset root through a path constant or
environment variable documented in its own README.

## MoCha 2026 (Benchmark and Challenge on Parkinsonian Gait @ ECCV 2026)

MDS-UPDRS gait severity {0,1,2,3} from canonicalized SMPL motion, scored on held-out clinical
sites. Codabench user **JLShen**.

**First place, 58 entries.**

| directory | task | server score | rank |
|---|---|---|---|
| [`mocha`](mocha/) | cross-site gait severity classification | macro-F1 **0.69447** | **1** |

Runner-up 0.5807; the organizers' released baseline 0.4289. A public motion encoder is frozen and a
single 4×512 linear layer is trained on top of it, so the whole trained model is 14 KB and is in the
directory. Most of the margin comes from reproducing the reference benchmark's exact head recipe and
from aggregating per-walk posteriors over the subject grouping the organizers ship in the input
format, not from representation learning.

Unlike the entries below, this directory is not the submitted archive unpacked verbatim: that
archive also bundles the SMPL body model and the pretrained encoder, which are not ours to
redistribute. `mocha/assets/README.md` lists all three third-party binaries with their checksums and
where to obtain them, and `mocha/verify.py` reassembles the exact runtime layout and runs it
end-to-end once they are in place. The dataset (CARE-PD, CC BY-NC 4.0) is likewise not redistributed.

## StanceEval-2026 (ArabicNLP @ EMNLP 2026)

Arabic stance detection on Mawqif-v2, two tracks. Codabench user **JLShen**. Both entries are
zero-shot frontier-LLM ensembles: the models are called through an API, nothing is fine-tuned into
the shipped decision on Track 2, and on Track 1 the only trained component is a tie-breaking
arbiter whose standalone score (0.7762) is the weakest in the system.

**First place on Track 1, second on Track 2.**

| directory | track | server score | rank |
|---|---|---|---|
| [`stanceeval_t1`](stanceeval_t1/) | held-out target ("Women Driving") | Overall_Favg2 **0.899400** | **1** of 24 |
| [`stanceeval_t2`](stanceeval_t2/) | unseen targets (Ecars, Trimester) | Unseen_Overall_Favg2 **0.935700** | **2** of 21 |

Favg2 is the macro-F1 over Favor and Against; None is excluded from the score but retained as a
row, which is what makes the abstain class the dominant error source in both tracks. Each directory
rebuilds its submitted `predictions.txt` **bit-exactly** with `numpy` alone, offline and on CPU, and
ships the per-row output of every LLM call behind that file — all inference was programmatic, so
those arrays and call logs are the record in place of chat transcripts. The dataset is not
redistributed here; request it from the task organizers.

## ECCV 2026 ChaLearn UDIVA-HHOI Challenge

Human–human–object interaction in dyadic Lego-assembly sessions, five tracks. Codabench user
**junlong**. Every model is
trained on the 21 annotated development sessions only; pretrained backbones are downloaded from
HuggingFace by name and run locally, and no dataset content is sent anywhere.

**First place on all five tracks.**

| directory | task | server score | rank |
|---|---|---|---|
| [`udiva_t1`](udiva_t1/) | exocentric event recognition | mAP 0.0318 (verbal 0.0237 / non-verbal 0.0398) | **1** |
| [`udiva_t2`](udiva_t2/) | egocentric event recognition | mAP 0.0178 (verbal 0.0200 / non-verbal 0.0155) | **1** |
| [`udiva_t3`](udiva_t3/) | exocentric event anticipation | next 0.375 / verbal 0.635 / non-verbal 0.434 / full 0.344 | **1** |
| [`udiva_t4`](udiva_t4/) | egocentric event anticipation | next 0.4641 / verbal 0.6548 / non-verbal 0.5301 / full 0.3872 | **1** |
| [`udiva_t5`](udiva_t5/) | exocentric causal event grounding | temporal 0.4274 / MC 0.7564 | **1** (joint) |

Recognition scores (T1, T2) are the official mAP; anticipation scores (T3, T4) are the four
subtask columns of the official normalized-SDL metric. Track 5 is ranked on the average of the
per-column ranks, where our entry ties for first — leading on temporal accuracy, behind on
multiple-choice accuracy. The tracks differ sharply in what they
need to run: T3 and T4 reproduce their submissions with the Python standard library alone,
while T1, T2 and T5 need a GPU and PyTorch. `requirements_shared_venv.txt`, where present, is
the frozen environment of the cluster used during the challenge — a provenance record, not a
portable install.

This code remains publicly accessible for at least three years, as stated in the fact sheets.

## Sea Winds Predictions 2026 (Capgemini "Prediction of Sea Winds" hackathon)

Probabilistic downscaling of coarse reanalysis and ECMWF-HRES to 1.3 km AROME winds over the North
Sea, plus a 1.2 GW wind-farm siting task and a written financial report. Codabench user
**junlong**.

**First place of the 3 published finalist entries** in the Phase-2 final window.

| directory | task | server score | rank |
|---|---|---|---|
| [`seawinds`](seawinds/) | Phase-2 final: forecast + siting + report | mean rank **1.167** | **1** |

Runner-up 2.000, third 2.833. The ranking column is the mean of six per-dimension ranks
({speed, direction} x {d+1, d+7, d+14}), so each sub-dimension carries equal weight; our entry
leads five of the six. Scores are the organisers' Winkler and circular-Winkler metrics on the
withheld 2022 set, where lower is better.

Direction, not speed, is where the entry is won: an 8-member Pangu-Weather initial-condition
ensemble supplies the direction centre, except at d+14, where the foundation-model centre measured
*below* a calibrated no-skill floor and is switched off in favour of a monthly climatology centre
with a solved arc half-width — worth 22.5 points on that column alone, taking it from last place to
first. The fitted artifacts are not redistributed here: two of the six are derived from the
organisers' competition data, and `seawinds/REPRODUCE.md` gives the command that regenerates each.

## NeurIPS 2026 RealPDE Challenge, Track 1 (Sim2Real)

Forecast 20 frames of real PIV measurements of flow past a foil, with per-element uncertainty
bounds, scored on accuracy, turbulence statistics, interval quality and latency. Codabench user
**junlong**.

**Ninth place in the final ranking**, decided by the organizers retraining every finalist's code
on a private set of unseen flow conditions.

| directory | task | server score | rank |
|---|---|---|---|
| [`realpde_t1`](realpde_t1/) | Sim2Real forecast + uncertainty bounds | final **82.3055** ± 0.0061 (dev board 81.885) | **9** (dev board 10 of 182) |

The entry uses no simulation data. Four small 2-D conv operators are trained on the 81 real
trajectories, with exact Reynolds-similarity augmentation. A post-hoc width table is fitted
to the challenge's interval reward, whose optimum is a per-element closed form rather than a
fixed coverage level. Refitting that table alone, with predictions bit-identical, was worth
+0.78 on the hidden test. `realpde_t1/paper/` is the technical report, including the negative
results on simulation pretraining and joint training. The directory is the submitted package
with exploratory options removed, verified bitwise against it.

## License

MIT — see `LICENSE`, and a copy in each entry directory.
