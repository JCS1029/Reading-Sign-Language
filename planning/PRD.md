# PRD: Reading Sign Language (Sign Language MNIST, three stages)

Internal planning document. Not part of the submission. See `AGENTS.md` for the rules that govern agents working on this repository.

## 1. Goal

Deliver the three-stage course project described in `Project_Sign_Language.pdf`:

1. Stage 1: softmax classifier on raw pixels. Accuracy, weight images, confusion analysis.
2. Stage 2: PCA features + the same softmax classifier. Explained variance, eigen-hands, reconstructions, accuracy and training time vs `k`, noise robustness.
3. Stage 3: a fixed small CNN (learned features + softmax). Training curves, comparison with Stages 1-2, 2D feature-space plots, shift robustness.

Plus a combined final report and a 20-minute oral defense.

The project is about features: the classifier (softmax / multinomial logistic regression) is identical in all three stages, so every difference in results is attributable to the features.

## 2. Non-goals

- No models beyond those the PDF allows for each stage.
- No real-time webcam application (the original README idea). Out of scope for the course deliverable; could be a later personal extension.
- No hyperparameter search beyond what the PDF asks (`C` in Stage 1, `k` in Stage 2, epochs via early stopping in Stage 3).
- No data augmentation during training. Shifts and noise are evaluation-time perturbations only.
- No automated writing of report prose. The student writes every answer, prediction, interpretation and note.

## 3. Constraints from the course

| Constraint | Consequence for us |
| --- | --- |
| Allowed models only | Softmax; PCA + softmax; the fixed CNN. Enforced in `AGENTS.md` Section 5. |
| Analysis and written answers must be the student's own; oral checks | Agents never write report prose (`AGENTS.md` Section 3). |
| AI-use disclosure at the end of each report | A section exists in every report skeleton; the student writes it. |
| Predict before you run | Gate in `AGENTS.md` Section 4; "Prediction vs. result" section in each report. |
| Report what did not work | Negative results are kept, never deleted from notebooks or tables. |
| Reproducibility: one notebook per stage, fixed seeds | One `.ipynb` per stage, seed 42, headless execution verified in Phase 4. |
| Same split and metrics across stages | Single `data/splits/split_v1.npz`; shared `src/signlang/metrics.py`. |
| Official test set used once per stage | Final cells of each notebook only. |
| Near-duplicate training images | Group-aware validation split (D-4); random split kept only as a diagnostic. |
| Hands are roughly centred | Shift test in Stage 3 is meaningful; implementation in D-6. |

## 4. Decisions log

| ID | Decision | Status | Rationale |
| --- | --- | --- | --- |
| D-1 | Code as Jupyter notebooks, one per stage, over a shared `src/signlang/` package | Confirmed by student | PDF asks for one notebook or script per stage; notebooks are what graders open. Shared package keeps split, metrics and plots identical across stages. |
| D-2 | Reports in LaTeX, one `report.tex` per stage plus a combined final report, sharing `reports/preamble.tex` | Confirmed by student | Clean PDFs, tables generated as `.tex` and `\input`, figures as PDF. |
| D-3 | Stage 3 in plain PyTorch (no Lightning, no ISLP helper, no Keras) | Confirmed by student | Training loop is explicit and defensible orally; smallest install risk on Python 3.14; penultimate-layer features are one line. |
| D-4 | Validation split is group-aware: near-duplicate groups are recovered from the images and split with `GroupShuffleSplit` 80/20, stratification checked after the fact. A stratified random split is kept only as a Stage 1 diagnostic. | Proposed; verified in Phase 0 | Training set is ~1,700 photos expanded into 27,455 altered copies; a random split leaks copies and inflates validation accuracy. Details in Section 7. |
| D-5 | Pixel scaling is `x / 255.0` only. No per-pixel standardisation. | Proposed | Keeps Stage 1 weight images interpretable in pixel space and matches the PDF's loading snippet. |
| D-6 | Shift test: integer pixel shifts of 1, 2, 3 in each of the four directions, implemented as pad-with-edge-replication then crop back to 28x28. Report accuracy per direction and averaged over directions. | Proposed | Edge replication avoids introducing a black bar that would itself be a new feature; averaging over directions reduces direction-specific noise while per-direction numbers are kept for honesty. |
| D-7 | Noise test: additive Gaussian noise with sigma in {0.1, 0.2, 0.3} on [0,1]-scaled test images, clipped back to [0,1], fixed seed. | Proposed | Three levels show the trend rather than a single point; clipping keeps images valid. |
| D-8 | No visible scaffolding in submission-facing files: no placeholder macros, IDs, or "write here" markers. Tasks for the student live only in `planning/TODO.md`. | Confirmed by student | The submitted project must read as ordinary student work. |
| D-9 | Stage 1 `C` grid: {0.001, 0.01, 0.1, 1, 10, 100}; solver `lbfgs`, `max_iter` 2000 or until converged. | Proposed | Six values over five decades satisfy "several values of C"; lbfgs is the scikit-learn default for multinomial. |
| D-10 | Stage 2 `k` grid: {10, 20, 50, 100, 150, 200, 300, 500}, trained with the Stage 1 `C`. Chosen `k` is proposed by the rule "smallest `k` whose validation accuracy is within 0.5 percentage points of the best"; the student makes and justifies the final choice. | Proposed | Includes the PDF's required 10, 50, 150 and spans up to most of the variance. Rule gives a defensible default without hiding the trade-off. |
| D-11 | Stage 3 training: Adam, lr 1e-3, batch 128, max 30 epochs, early stopping on validation loss with patience 5, restore best epoch. Padding 1 in both conv layers so feature maps are 28->14->7 and the flatten size is 32*7*7 = 1568. | Proposed | Standard, small, CPU-friendly. Padding choice keeps arithmetic clean; without padding the flatten size would be 32*5*5 = 800, which is also valid, but one must be fixed. |
| D-12 | Feature-space separability is reported visually (2D PCA scatter) and numerically (silhouette score on the 2D projection and on the full features) for raw pixels and CNN features. | Proposed | "Better separated" should not rest on eyeballing one scatter plot. |

Status values: Proposed (agent default, student may change before the phase runs), Confirmed by student, Superseded (see `planning/TEXTBOOK.md` Part C).

## 5. Repository layout (target)

```
Reading-Sign-Language/
  AGENTS.md                     agent rules (internal)
  README.md                     normal project README (submission-facing)
  requirements.txt
  .gitignore
  data/
    raw/                        sign_mnist_train.csv, sign_mnist_test.csv (gitignored)
    splits/split_v1.npz         train/val indices and duplicate-group ids
    own_hands/                  optional Stage 3 bonus photos
  docs/
    Dataset/*.png               reference images of the ASL alphabet (kept)
  notebooks/
    stage0_data_and_split.ipynb EDA, duplicate analysis, split construction
    stage1_raw_pixels.ipynb
    stage2_pca.ipynb
    stage3_cnn.ipynb
  src/signlang/
    __init__.py
    seed.py                     set_seed(42)
    data.py                     load CSVs, scale, label remap, letter names
    split.py                    duplicate grouping, GroupShuffleSplit, save/load
    metrics.py                  accuracy, macro-F1, confusion matrix, confused pairs
    plotting.py                 image grids, weight images, curves, scatter
    timing.py                   wall-clock context manager
    perturb.py                  shift and noise functions (evaluation only)
    cnn.py                      the fixed CNN, train loop, early stopping, feature extraction
  tools/
    build_tables.py             pandas DataFrame -> LaTeX table files
  models/                       saved classifiers and CNN checkpoint (gitignored)
  reports/
    preamble.tex
    figures/stage1/ stage2/ stage3/
    tables/
    stage1/report.tex
    stage2/report.tex
    stage3/report.tex
    final/report.tex
  planning/                     internal: PRD.md, TODO.md, TEXTBOOK.md, DEFENSE_PREP.md
```

## 6. Metrics (identical in all stages)

- Primary: accuracy on validation split (for choices) and on the official test set (reported once per stage).
- Secondary: macro-F1 (guards against the mild class imbalance), full 24x24 confusion matrix.
- Cost: wall-clock training time in seconds (`fit` only, excluding data loading), measured on the same machine, single run, reported with the machine description.
- Robustness: accuracy under perturbation minus clean accuracy (percentage points), for noise (Stage 2) and shifts (Stage 3).

## 7. Split strategy (D-4) in detail

Problem: the training CSV has no photo identifier, yet the ~27k images are altered copies of ~1,700 originals. Copies of one photo on both sides of a random split make validation accuracy optimistic (Stage 1 self-check 3).

Procedure (Phase 0):

1. Scale training images to [0,1]; fit PCA with 50 components on all training images (unsupervised, labels unused, so no leakage concern for grouping).
2. Build a k-nearest-neighbour graph (k = 10, cosine distance) in the 50-dim space, restricted to pairs with the same label.
3. Plot the histogram of nearest-neighbour distances. If it is bimodal (a tight mode of near-copies and a looser mode of different photos), set the threshold at the valley between them.
4. Keep edges below the threshold; connected components are the duplicate groups. Report the number of groups and the distribution of group sizes; the expectation from the PDF is on the order of 1,700 groups averaging ~16 images.
5. `GroupShuffleSplit(test_size=0.2, random_state=42)` on group ids. Check that every letter has at least 100 validation images; if not, re-draw with another seed and record it.
6. Save `train_idx`, `val_idx`, `group_id` and the threshold to `data/splits/split_v1.npz`, plus a JSON sidecar with the statistics.

Fallback: if the histogram is unimodal (duplicate structure not recoverable), use `StratifiedShuffleSplit` 80/20 with seed 42, record this as deviation C-1, and let Stage 1's random-vs-group comparison collapse to a single split with an explanation by the student.

Diagnostic (Stage 1 only): run the same `C` sweep on a plain stratified random 80/20 split and show both validation curves against the single test accuracy. The gap between the two validation curves and the test number is the evidence behind self-check 3.

## 8. Phases

### Phase 0 - Foundation

Entry: this PRD, `TODO.md` and `TEXTBOOK.md` exist; student has approved.

Agent tasks:
- Create venv and `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, seaborn, torch, jupyter, nbconvert, ipykernel). Pin major versions after a successful install.
- Move `docs/Dataset/*.csv` and `archive.zip` to `data/raw/`; delete the duplicate copies inside `docs/Dataset/sign_mnist_train/` and `sign_mnist_test/`; gitignore `data/raw/`. Keep the three PNGs in `docs/Dataset/`.
- Write `src/signlang/` modules listed in Section 5 (all except `cnn.py`, which is Phase 3).
- `notebooks/stage0_data_and_split.ipynb`: load, class histogram (train and test), grid of 5 examples per letter, duplicate analysis and histogram, split construction and statistics, save split.
- `tools/build_tables.py`.
- Create `reports/figures/*` and `reports/tables/` directories.

Student tasks:
- Read ISLP 4.3 and 5.1.
- Write the three Stage 1 self-check answers in the Self-check section of `reports/stage1/report.tex`.
- Look at the example grid and write down, in the Data exploration section, which letters you expect to be confused and why. This is a prediction and must be written before Stage 1 runs.

Exit: notebook runs headlessly; `data/splits/split_v1.npz` exists with statistics; EDA figures saved to `reports/figures/stage1/` (they belong to the Stage 1 report's Data exploration section).

### Phase 1 - Stage 1, raw pixels

Entry: Phase 0 exit; the student's predictions for Stage 1 questions 1-3 exist in `reports/stage1/report.tex`.

Agent tasks:
- `notebooks/stage1_raw_pixels.ipynb`:
  - Sweep `C` (D-9) on the group split; record validation accuracy, macro-F1, fit time, convergence flag per `C`.
  - Same sweep on the stratified random split (diagnostic).
  - Pick `C` by best group-validation accuracy; refit on train split; evaluate once on the official test set.
  - Weight images for 5 letters chosen by the student (default A, B, C, L, Y if not specified), diverging colormap with symmetric colour limits, plus a mean image per letter for comparison.
  - Confusion matrix (counts and row-normalised); top-5 confused pairs (off-diagonal, symmetric sum); for each pair a row of 4 true-letter and 4 predicted-letter example images from the misclassified test items.
  - Save `models/stage1_softmax.joblib`.
- Export: `reports/tables/s1_C_sweep.tex` (C, group-val acc, random-val acc, fit time), `reports/tables/s1_confused_pairs.tex`, `reports/tables/s1_test.tex` (chosen C, val acc, test acc, gap). Figures: `reports/figures/stage1/examples_grid.pdf`, `class_hist.pdf`, `dup_distance_hist.pdf`, `c_sweep.pdf`, `weights.pdf`, `confusion.pdf`, `confused_pairs.pdf`.

Student tasks (after the run): Results prose including the choice of `C` and the explanation of the validation/test gap; Weight images interpretation; Confusions interpretation; Prediction vs. result; AI-use note.

Exit: notebook runs headlessly; all tables and figures exist; report compiles with empty prose sections.

### Phase 2 - Stage 2, PCA

Entry: Phase 1 exit; Stage 2 self-check answers and predictions written.

Agent tasks:
- `notebooks/stage2_pca.ipynb`:
  - Fit `PCA(n_components=500, random_state=42)` on the train split only. Save mean and components.
  - Explained variance ratio (scree, first 50) and cumulative curve (to 500) with markers at 10, 50, 150.
  - First 10 eigen-hands as 28x28 images (diverging colormap) and the mean hand.
  - Reconstructions of 4 fixed images (one each of 4 letters) at k in {10, 50, 150} next to the originals, with per-image reconstruction MSE.
  - Sweep `k` (D-10): project, fit softmax with the Stage 1 `C`, record val acc, macro-F1, fit time (PCA transform time reported separately). Raw-pixel result from Stage 1 as a reference line.
  - Propose `k` by the D-10 rule; student confirms or overrides in chat; refit; evaluate once on the test set.
  - Noise test (D-7): for each sigma, accuracy of the Stage 1 raw-pixel model and the PCA-k model on the noisy test set; also a grid showing one image at each sigma, and its PCA reconstruction, to make "noise removal" visible.
  - Save `models/stage2_pca.joblib` and `models/stage2_softmax.joblib`.
- Export: `reports/tables/s2_k_sweep.tex` (raw, 10, 50, 150, chosen k, and the other grid values: val acc, test acc where evaluated, fit time), `reports/tables/s2_noise.tex` (sigma x {raw, PCA-k}). Figures: `reports/figures/stage2/explained_variance.pdf`, `eigenhands.pdf`, `reconstructions.pdf`, `k_sweep_acc.pdf`, `k_sweep_time.pdf`, `noise_examples.pdf`, `noise_acc.pdf`.

Student tasks: Components interpretation; Number of components discussion with the chosen `k` and comparison with Stage 1; Noise explanation; Prediction vs. result; AI-use note.

Exit: as Phase 1.

### Phase 3 - Stage 3, CNN

Entry: Phase 2 exit; Stage 3 self-check answers and predictions written; `torch` installed and importable.

Agent tasks:
- `src/signlang/cnn.py`: `SmallCNN` (architecture in D-11), `train()` with mini-batches, per-epoch train/val loss and accuracy logging, early stopping, best-state restore, `extract_features()` returning the 1568-d penultimate activations, `predict()`.
- `notebooks/stage3_cnn.ipynb`:
  - Deterministic setup (`torch.manual_seed`, `torch.use_deterministic_algorithms(True)` where supported, `num_workers=0`).
  - Train; plot loss and accuracy curves with the chosen epoch marked.
  - Parameter count printed per layer (relates to self-check 1).
  - Evaluate once on the test set; load Stage 1 and Stage 2 models; three-model clean comparison.
  - Feature space (D-12): PCA to 2D of raw pixels and of CNN features on the validation set, scatter coloured by letter with one shared legend; silhouette scores.
  - Shift test (D-6): shifts 1, 2, 3 in four directions for the three frozen models; per-direction and averaged accuracy; a grid showing one image shifted.
  - Optional bonus: `tools/prepare_own_hands.py` converts photos in `data/own_hands/` to 28x28 grayscale and the notebook predicts them if the folder is non-empty.
  - Save `models/stage3_cnn.pt`.
- Export: `reports/tables/s3_training.tex` (epochs run, best epoch, final train/val loss and acc), `reports/tables/s3_comparison.tex` (test acc of three models), `reports/tables/s3_shift.tex` (three models x shift 0,1,2,3, averaged, with per-direction appendix table `s3_shift_directions.tex`), `reports/tables/s3_silhouette.tex`. Figures: `reports/figures/stage3/curves.pdf`, `feature_space_raw.pdf`, `feature_space_cnn.pdf`, `shift_examples.pdf`, `shift_acc.pdf`, optionally `own_hands.pdf`.

Student tasks: Training section prose (epochs, how overfitting was avoided); Comparison; Learned feature space interpretation; Robustness explanation; Prediction vs. result; AI-use note.

Exit: as Phase 1, plus `models/stage3_cnn.pt` saved.

### Phase 4 - Final report and defense

Entry: Phases 1-3 exit; all stage report prose written by the student.

Agent tasks:
- `reports/final/report.tex` already `\input`s the three stage bodies; verify it compiles and page-breaks sensibly.
- `planning/DEFENSE_PREP.md`: a list of likely oral questions, each pointing to the textbook section and the report section that cover it. Questions only.
- Clean-venv reproducibility run: fresh venv, `pip install -r requirements.txt`, execute all four notebooks headlessly, rebuild all four PDFs, confirm figures and tables are regenerated byte-stably where determinism allows.
- Final sweep of submission-facing files for leftover markers or references to `planning/`.

Student tasks: write the final narrative section (what you learned at each stage and how each answer led to the next question) and the final AI-use note; rehearse with `DEFENSE_PREP.md`.

Exit: four PDFs built from a clean environment; `TODO.md` fully ticked.

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| PyTorch has no wheel for Python 3.14 | Medium | Blocks Phase 3 | Install the newest supported Python side by side; create venv from it; record in Part C. Check early in Phase 0, not Phase 3. |
| Duplicate groups not recoverable from pixels | Low-medium | D-4 fallback | Histogram check in Phase 0; fallback documented in Section 7. |
| lbfgs slow or non-convergent on 784 features with large `C` | Medium | Long Stage 1 runs | `max_iter` 2000, record convergence flag, report it honestly; it is itself a result. |
| Validation accuracy near 100 percent regardless of split | Medium | Weak evidence for self-check 3 | Group split is the mitigation; if still saturated, the test gap is the evidence. Negative results are allowed. |
| Non-determinism in CNN training on CPU | Low | Figures change slightly on rerun | Deterministic flags, seed, `num_workers=0`; document residual variation. |
| Student writes prose that contradicts tables after a rerun | Medium | Report inconsistency | Never rerun final cells after prose is written without telling the student; log reruns in `TODO.md`. |
| Time: Stage 3 shift test over 3 models x 3 shifts x 4 directions | Low | Minutes | Vectorised numpy shifts; models are small. |
