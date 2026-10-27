# Agent rules for this repository

This file is loaded automatically by Cursor. Read it fully before doing anything else.

## 1. Read order at the start of every session

1. `AGENTS.md` (this file)
2. `planning/PRD.md` - goals, constraints, decisions log, phases
3. `planning/TODO.md` - what is done, what is next, what is blocked
4. `planning/TEXTBOOK.md` Part C - deviations already made from the original plan
5. `planning/TEXTBOOK.md` Part B - the exact procedure for the phase you are about to work on

Ask the user which phase they want to work on if `planning/TODO.md` does not make it obvious.

## 2. Two kinds of files

- Submission-facing (what the instructor sees): `README.md`, `notebooks/`, `src/`, `reports/`, `requirements.txt`, `.gitignore`, `tools/`.
- Internal (for the student and for agents): `planning/` and this file.

Submission-facing files must look like an ordinary student project. They must not reference `planning/`, this file, or the fact that an AI was involved beyond what the student writes in the AI-use note. Never leave `TODO`, `FIXME`, `XXX`, "write here", "fill in", or similar markers in submission-facing files. Never add placeholder IDs or marker macros to the reports.

## 3. No ghostwriting

The course requires that analysis, interpretation and written answers be the student's own and tests them orally. Therefore agents:

- Do write: code, figures, auto-generated LaTeX tables, notebook code cells, short factual notebook markdown headings ("Sweep over C", "Confusion matrix"), the README, the planning documents, the textbook.
- Do not write: self-check answers, predictions, interpretations of figures, explanations of results, the gap explanation, the choice justification for `k`, the AI-use note, the final narrative, or any report prose that answers a course question. Leave those sections empty in `reports/*/report.tex`; the student fills them in.
- Do not paraphrase numbers into prose anywhere in `reports/`. Results enter the reports only via `\input{tables/...}` and `\includegraphics{figures/...}` wired to files produced by the notebooks.
- May explain results to the student in chat, and may explain concepts in `planning/TEXTBOOK.md`, as much as helpful. Chat is not the report.

If the student asks you to write report prose for them, remind them of this rule once, then follow their instruction if they insist. Record that in `planning/TEXTBOOK.md` Part C.

## 4. Predict-before-you-run gate

Every work question in every stage requires the student to write a prediction before the experiment runs. Before executing a work-question's experiment end-to-end (full data, final numbers):

1. Open `reports/stageN/report.tex` and check that the "Prediction vs. result" section contains the student's prediction for that question.
2. If it is empty, stop and ask the student to write it. Do not show them final numbers first.
3. Until the prediction exists you may only smoke-test code on a small subsample (for example 500 training rows) and must not report the resulting accuracy to the student as if it were a result.

## 5. Modelling constraints (from the course PDF)

- Stage 1: multinomial logistic regression (softmax) on raw pixels only. scikit-learn `LogisticRegression`.
- Stage 2: PCA fitted on the training split only, then the same softmax classifier. Nothing else.
- Stage 3: exactly the fixed CNN: Conv(1->16, 3x3) -> ReLU -> MaxPool 2x2 -> Conv(16->32, 3x3) -> ReLU -> MaxPool 2x2 -> Flatten -> Linear(->24). Plain PyTorch (decision D-3). No extra layers, no dropout, no batch norm, no data augmentation during training. Early stopping and checkpointing are allowed.
- Shift and noise perturbations are evaluation-only. Never train on shifted or noisy images.
- No other models, ensembles, or pretrained weights anywhere.

## 6. Reproducibility rules

- Seed 42 everywhere: `numpy`, `random`, `torch`, scikit-learn `random_state`. Use `src/signlang/seed.py`.
- One split for the whole project: `data/splits/split_v1.npz`, built once in Phase 0 by `src/signlang/split.py`. All three stages load it. Never rebuild it silently; if it must change, that is a deviation (Section 8) and every stage must be rerun.
- The official test set (`sign_mnist_test.csv`) is evaluated once per stage, in the last cells of that stage's notebook, with the already-chosen hyperparameters. Never tune on it.
- Pixel scaling is `x / 255.0` only (decision D-5). No standardisation, so Stage 1 weight images stay interpretable.
- Every notebook must run top to bottom headlessly: `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.
- Figures are saved as PDF under `reports/figures/stageN/`, tables as `.tex` under `reports/tables/`, with the exact file names listed in `planning/TEXTBOOK.md` Part B.

## 7. Working style

- Tick items in `planning/TODO.md` as they are finished, in the same session. Add new items when new decisions or tasks appear.
- Keep notebooks thin: logic lives in `src/signlang/`, notebooks call it and display results.
- Record wall-clock training times with `src/signlang/timing.py`; they are a required deliverable in Stage 2.
- Do not install packages outside the project venv. Do not change `requirements.txt` without noting it in `planning/TODO.md`.

## 8. Deviation rule

`planning/TEXTBOOK.md` Part B describes exactly how each phase is supposed to be done. If you implement anything differently (different hyperparameter, different split method, different file name, an extra diagnostic, a dropped step, a fallback), append an entry to `planning/TEXTBOOK.md` Part C in the same session, using the template there: date, phase, Part B section affected, what was planned, what was done, why, effect on results, who decided. Then add a one-line pointer to the affected Part B section ("See deviation C-n"). Future agents and the student rely on Part C to know the true state of the project.

## 9. Environment (Windows, PowerShell)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch has no wheel for the installed Python version, install the newest Python that it supports side by side and create the venv with that interpreter; record this in `planning/TODO.md` and Part C.

Run a notebook headlessly:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/stage1_raw_pixels.ipynb
```

Build a report (from `reports/stage1/`):

```powershell
latexmk -pdf report.tex
```

The GPU (GeForce MX350) is optional; the CNN trains in minutes on CPU. Code must run with `device = "cpu"`.
