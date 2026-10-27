# TODO

Internal checklist. Not part of the submission. Each item is marked `(agent)` or `(you)`. Items marked `gate:` must not be executed before the named condition holds (see `AGENTS.md` Section 4). Tick items with `[x]` in the same session they are finished. When a new task or decision appears, add it here.

Legend for "you" items: the report section named is where your text goes in the corresponding `reports/stageN/report.tex`. Nothing in the report marks it; this list is the only reminder.

## Before every agent session (you)

- [ ] Tell the agent which phase to work on.
- [ ] If the phase runs an experiment, confirm your prediction for it is already written in the report's "Prediction vs. result" section.
- [ ] Mention any decision in `planning/PRD.md` Section 4 marked "Proposed" that you want changed.

## Phase P - Planning documents

- [x] (agent) `AGENTS.md`
- [x] (agent) `planning/PRD.md`
- [x] (agent) `planning/TODO.md`
- [x] (agent) `planning/TEXTBOOK.md` Parts A-D
- [x] (agent) `reports/preamble.tex`, `reports/stageN/report.tex` + `body.tex`, `reports/final/report.tex`
- [x] (agent) `.gitignore`, `README.md`
- [ ] (you) Put your name in `\newcommand{\projectauthor}{}` in `reports/preamble.tex`.
- [ ] (you) Read `planning/PRD.md` Section 4 and say which "Proposed" decisions you accept; the agent flips them to "Confirmed by student".
- [ ] (you) Read `planning/TEXTBOOK.md` Part A once, start to finish.
- [ ] (you) Decide whether the old `docs/Dataset/` layout and the real-time-recognition sentence in the original README should be kept anywhere (the new README drops both).

## Phase 0 - Foundation

- [ ] (agent) Create `.venv` with `py -3.14`; install `requirements.txt`; verify `import torch` works. If it does not, follow `AGENTS.md` Section 9 and record the outcome here and in TEXTBOOK Part C.
- [ ] (agent) Move CSVs and zip to `data/raw/`; remove duplicate CSVs under `docs/Dataset/sign_mnist_*/`; keep PNGs.
- [ ] (agent) `src/signlang/__init__.py`, `seed.py`, `data.py`, `timing.py`, `metrics.py`, `plotting.py`, `perturb.py`, `split.py`.
- [ ] (agent) `tools/build_tables.py`.
- [ ] (agent) `notebooks/stage0_data_and_split.ipynb`: class histograms, example grid (5 per letter), duplicate-distance histogram, threshold choice, group statistics, split construction, save `data/splits/split_v1.npz` and `split_v1.json`.
- [ ] (agent) Save `reports/figures/stage1/examples_grid.pdf`, `class_hist.pdf`, `dup_distance_hist.pdf`.
- [ ] (agent) If duplicate groups are not recoverable, apply the D-4 fallback and write deviation C-1.
- [ ] (agent) Run the notebook headlessly once; confirm it completes.
- [ ] (you) Read ISLP 4.3 (logistic regression, multinomial) and 5.1 (cross-validation).
- [ ] (you) Write the three Stage 1 self-check answers in the "Self-check" section of `reports/stage1/report.tex`.
- [ ] (you) Look at `reports/figures/stage1/examples_grid.pdf` and write in the "Data exploration" section which letters you expect to be confused and why. Do this before Phase 1 runs.
- [ ] (you) Write your predictions for Stage 1 questions 1, 2, 3 in the "Prediction vs. result" section of `reports/stage1/report.tex` (what accuracy range you expect, what the weight images will look like, which pairs will be confused).

## Phase 1 - Stage 1, raw pixels

- [ ] (agent) gate: Stage 1 predictions written. `notebooks/stage1_raw_pixels.ipynb` C sweep on the group split and on the random split; `reports/tables/s1_C_sweep.tex`; `reports/figures/stage1/c_sweep.pdf`.
- [ ] (agent) Choose C by group-validation accuracy; refit; single test evaluation; `reports/tables/s1_test.tex`; save `models/stage1_softmax.joblib`.
- [ ] (agent) Weight images for 5 letters; `reports/figures/stage1/weights.pdf`. Default letters A, B, C, L, Y unless you name others.
- [ ] (agent) Confusion matrix and top-5 confused pairs with example images; `reports/figures/stage1/confusion.pdf`, `confused_pairs.pdf`; `reports/tables/s1_confused_pairs.tex`.
- [ ] (agent) Headless run; report compiles with empty prose sections.
- [ ] (you) Name the 5 letters for the weight images, or accept the default.
- [ ] (you) "Results": discuss validation and test accuracy, the choice of C, and explain the gap.
- [ ] (you) "Weight images": interpret the 5 images.
- [ ] (you) "Confusions": explain why the top pairs are hard for a linear classifier on raw pixels.
- [ ] (you) "Prediction vs. result": state whether each of your three predictions was right.
- [ ] (you) "AI-use note".
- [ ] (you) Read ISLP 12.2 (PCA). Optional: Turk and Pentland 1991.
- [ ] (you) Write the three Stage 2 self-check answers in `reports/stage2/report.tex`.
- [ ] (you) Write your Stage 2 predictions for questions 1, 2, 3 (what the first eigen-hand looks like, how many components you expect to need, whether PCA will help under noise).

## Phase 2 - Stage 2, PCA

- [ ] (agent) gate: Stage 2 predictions written. `notebooks/stage2_pca.ipynb` PCA fit on train split; explained variance; eigen-hands; reconstructions; figures `explained_variance.pdf`, `eigenhands.pdf`, `reconstructions.pdf`.
- [ ] (agent) k sweep with the Stage 1 C; `reports/tables/s2_k_sweep.tex`; figures `k_sweep_acc.pdf`, `k_sweep_time.pdf`.
- [ ] (agent) Propose k by the D-10 rule and tell you the proposal with the numbers.
- [ ] (you) Decide k (accept or override) and tell the agent.
- [ ] (agent) Refit at chosen k; single test evaluation; save `models/stage2_pca.joblib`, `models/stage2_softmax.joblib`.
- [ ] (agent) Noise test at sigma 0.1, 0.2, 0.3 for raw-pixel and PCA-k models; `reports/tables/s2_noise.tex`; figures `noise_examples.pdf`, `noise_acc.pdf`.
- [ ] (agent) Headless run; report compiles.
- [ ] (you) "Components": interpret explained variance, eigen-hands, reconstructions.
- [ ] (you) "Number of components": justify k, compare with Stage 1 on accuracy and time.
- [ ] (you) "Noise": explain the results.
- [ ] (you) "Prediction vs. result".
- [ ] (you) "AI-use note".
- [ ] (you) Read ISLP 10.3, 10.7, 10.9 (CNN parts).
- [ ] (you) Write the three Stage 3 self-check answers in `reports/stage3/report.tex`. Self-check 1 is a calculation; do it by hand first, the notebook will print the parameter counts later so you can check.
- [ ] (you) Write your Stage 3 predictions for questions 1, 2, 3 (CNN test accuracy vs Stages 1-2, whether the 2D feature plot will be more separated, how accuracy will fall with shift for each model).

## Phase 3 - Stage 3, CNN

- [ ] (agent) gate: Stage 3 predictions written; `torch` importable. `src/signlang/cnn.py`.
- [ ] (agent) `notebooks/stage3_cnn.ipynb` training with early stopping; `reports/figures/stage3/curves.pdf`; `reports/tables/s3_training.tex`; parameter counts printed; save `models/stage3_cnn.pt`.
- [ ] (agent) Single test evaluation; three-model comparison `reports/tables/s3_comparison.tex`.
- [ ] (agent) Feature-space plots `feature_space_raw.pdf`, `feature_space_cnn.pdf`; `reports/tables/s3_silhouette.tex`.
- [ ] (agent) Shift test; `reports/tables/s3_shift.tex`, `s3_shift_directions.tex`; figures `shift_examples.pdf`, `shift_acc.pdf`.
- [ ] (agent) Optional bonus: `tools/prepare_own_hands.py`; notebook cell that runs only if `data/own_hands/` is non-empty; `own_hands.pdf`.
- [ ] (agent) Headless run; report compiles.
- [ ] (you) Optional bonus: photograph your hand signing several letters against a plain background, crop to the hand, put the files in `data/own_hands/` named like `A_1.jpg`, `B_1.jpg`.
- [ ] (you) "Training": epochs, how overfitting was avoided.
- [ ] (you) "Comparison": discuss the three test accuracies.
- [ ] (you) "Learned feature space": interpret the two plots and the silhouette numbers.
- [ ] (you) "Robustness": explain accuracy vs shift for the three models.
- [ ] (you) "Prediction vs. result".
- [ ] (you) "AI-use note".

## Phase 4 - Final report and defense

- [ ] (agent) Verify `reports/final/report.tex` compiles with the three stage bodies.
- [ ] (agent) `planning/DEFENSE_PREP.md` with likely oral questions mapped to textbook and report sections. Questions only.
- [ ] (agent) Clean-venv reproducibility run: fresh venv, install, execute all notebooks headlessly, rebuild all PDFs.
- [ ] (agent) Sweep submission-facing files for markers or references to `planning/`.
- [ ] (you) Write the final narrative section in `reports/final/report.tex`: what you learned at each stage and how each answer led to the next question.
- [ ] (you) Write the final AI-use note.
- [ ] (you) Decide whether to uncomment the `planning/` and `AGENTS.md` lines in `.gitignore` before sharing the repository.
- [ ] (you) Rehearse with `planning/DEFENSE_PREP.md`.

## Reruns and changes log

Record here any time final cells were rerun after prose was written, any change to `requirements.txt`, and any change of the split. Point to the TEXTBOOK Part C entry.

- (none yet)
