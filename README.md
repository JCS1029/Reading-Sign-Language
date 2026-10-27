# Reading Sign Language

Classifying 28x28 grayscale images of American Sign Language letters (Sign Language MNIST, 24 static letters) with three kinds of features and one classifier:

1. Stage 1: softmax classifier (multinomial logistic regression) on raw pixels.
2. Stage 2: PCA features followed by the same softmax classifier.
3. Stage 3: a small convolutional neural network (two 3x3 convolutional layers with 16 and 32 filters, each followed by ReLU and 2x2 max pooling, then a softmax layer).

The classifier is the same in every stage, so differences in accuracy, training time and robustness come from the features alone.

## Data

Sign Language MNIST from Kaggle: <https://www.kaggle.com/datasets/datamunge/sign-language-mnist>. Download and unzip, then place `sign_mnist_train.csv` and `sign_mnist_test.csv` in `data/raw/`. The data files are not committed.

## Setup (Windows, PowerShell)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Run the notebooks in order; each one writes its figures to `reports/figures/` and its tables to `reports/tables/`, and saves fitted models to `models/`.

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/stage0_data_and_split.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/stage1_raw_pixels.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/stage2_pca.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/stage3_cnn.ipynb
```

All random seeds are fixed (42). The validation split used by every stage is built once by the first notebook and stored in `data/splits/split_v1.npz`. The official test set is evaluated once per stage.

## Reports

LaTeX sources are in `reports/stage1/`, `reports/stage2/`, `reports/stage3/` and `reports/final/`. Build any of them with `latexmk -pdf report.tex` from inside its directory. The final report combines the three stage reports.

## Layout

```
data/        raw CSVs (not committed) and the saved train/validation split
notebooks/   one notebook per stage
src/signlang/ shared code: data loading, split, metrics, plotting, perturbations, CNN
tools/       helpers for building LaTeX tables
models/      fitted models (not committed)
reports/     LaTeX reports, figures and tables
docs/        reference images of the ASL alphabet
```
