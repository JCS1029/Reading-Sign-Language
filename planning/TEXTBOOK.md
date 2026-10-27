# Textbook: Reading Sign Language with Three Kinds of Features

Internal document. Not part of the submission.

How to use this file:

- Part A explains the whole plan the way you would explain it to a curious 7th grader, but keeps every formula that the project actually uses. Read it once before Phase 0 and again before the defense.
- Part B describes each phase in boring, exact detail: inputs, steps, parameters, output file names, and what each figure and table is for. Agents implement from Part B. You read Part B for the phase you are about to run so nothing surprises you.
- Part C is the deviations log. Whenever something is implemented differently from Part B, the agent appends an entry here in the same session. Part C is the truth about what was actually done; Part B is the original plan.
- Part D is a glossary of every symbol used.

Rule for agents: read Part C before Part B. Append to Part C whenever you deviate. Never edit Part B retroactively to hide a deviation; add a pointer "See deviation C-n" instead.

---

# Part A - The whole plan, explained simply (with the real math)

## A.1 What we are trying to do

Someone makes a letter of the American Sign Language alphabet with one hand, and a camera takes a tiny black-and-white photo of it, 28 pixels wide and 28 pixels tall. We want the computer to look at that photo and say which letter it is. There are 24 letters to choose from, because J and Z need the hand to move and a single photo cannot show motion.

We have 27,455 photos to learn from (the training set) and 7,172 photos kept in a sealed envelope for the final exam (the test set). We are only allowed to open the envelope once per stage.

The project is not really about sign language. It is about **features**: what the computer looks at when it decides. We will try three kinds of features and keep the decision-maker the same, so that any change in the score is caused by the features alone.

1. Stage 1: raw pixels. The computer looks at all 784 brightness numbers directly.
2. Stage 2: PCA features. We first squeeze the 784 numbers into a smaller set of numbers that capture the big patterns, then the computer looks at those.
3. Stage 3: CNN features. The computer learns its own little pattern detectors from the labels, then looks at what the detectors say.

The decision-maker in all three stages is the same thing: a **softmax classifier**. So let us understand that first.

## A.2 An image is a list of numbers

A 28 x 28 grayscale image has 784 pixels. Each pixel is a brightness from 0 (black) to 255 (white). We divide every number by 255 so they are between 0 and 1. Then we unroll the grid, row by row, into one long list:

$$
\mathbf{x} = (x_1, x_2, \dots, x_{784}), \qquad 0 \le x_j \le 1.
$$

That list is the "input vector". When we say "the image is a point in 784-dimensional space", we just mean it is a list of 784 numbers. You cannot picture 784 dimensions, and you do not need to; every formula below treats the list the same way it would treat a list of 2 or 3 numbers.

**What this means for our hands:** pixel number 300 is always the pixel at row 10, column 20 (300 = 10 x 28 + 20, counting from 0). If the hand in one photo is 2 pixels to the right of the hand in another photo, the brightness values have moved to different positions in the list. To a method that looks at positions, those two photos look quite different, even though a person would say "same hand, slightly moved". Remember this; it is the whole reason Stage 3 exists.

## A.3 The softmax classifier: 24 scorecards

Imagine 24 judges, one per letter. Each judge has a scorecard with 784 slots, one per pixel, and in each slot a number called a **weight** (positive, negative, or zero). To score an image, the judge multiplies each pixel brightness by the weight in that slot, adds everything up, and adds a personal bonus called the **bias**. For judge $c$:

$$
z_c = \mathbf{w}_c \cdot \mathbf{x} + b_c = \sum_{j=1}^{784} w_{cj}\, x_j + b_c .
$$

A big positive weight at slot $j$ means "if this pixel is bright, I like this image more". A big negative weight means "if this pixel is bright, this is probably not my letter". A weight near zero means the judge ignores that pixel.

Now we have 24 scores $z_1, \dots, z_{24}$. To turn them into probabilities that add up to 1, we use the **softmax** function:

$$
p_c = \frac{e^{z_c}}{\sum_{k=1}^{24} e^{z_k}} .
$$

Why $e^{z}$ instead of just $z$? Three reasons. First, $e^{z}$ is always positive, so probabilities are never negative even if a score is negative. Second, dividing by the sum makes them add to 1. Third, the exponential exaggerates differences: a judge whose score is 3 points higher than another gets about $e^3 \approx 20$ times the probability, so the classifier becomes confident when one judge clearly wins. The predicted letter is the one with the largest $p_c$ (equivalently the largest $z_c$).

This is also called **multinomial logistic regression**. It is a **linear** classifier: each score is a weighted sum of the inputs, nothing more.

**What this means for our hands:** the weight vector $\mathbf{w}_c$ has 784 numbers, so we can reshape it back into a 28 x 28 picture and look at it. That picture shows which pixels judge $c$ wants bright (positive) and which it wants dark (negative). Stage 1 question 2 is exactly this.

## A.4 How the judges learn: cross-entropy loss

At the start, all weights are random and the judges are terrible. Learning means adjusting the weights so that, for training images, the correct judge gets a high probability.

We need a number that says how bad the current weights are. For one training image with true letter $y$, we use the **cross-entropy**:

$$
\ell = -\log p_y .
$$

If the model gives the correct letter probability 1, the loss is $-\log 1 = 0$: perfect. If it gives the correct letter probability 0.01, the loss is $-\log 0.01 \approx 4.6$: very bad. You can think of the loss as "how surprised the model is when told the right answer". The total loss is the average over all $n$ training images:

$$
L(\mathbf{W}, \mathbf{b}) = \frac{1}{n} \sum_{i=1}^{n} -\log p_{y_i}(\mathbf{x}_i) .
$$

We then find the weights that make $L$ as small as possible. The loss is smooth and bowl-shaped (convex) for this model, so an optimiser such as L-BFGS (what scikit-learn uses) finds the bottom of the bowl reliably. You do not need to know how L-BFGS works; it repeatedly computes which direction makes $L$ decrease fastest (the gradient) and steps that way.

## A.5 Regularisation: the knob called C

If the judges are allowed any weights at all, they will memorise the training set, including its quirks, and do worse on new photos. We discourage huge weights by adding a penalty on their size (**L2 regularisation**):

$$
\text{minimise} \quad C \cdot \sum_{i=1}^{n} -\log p_{y_i}(\mathbf{x}_i) \;+\; \frac{1}{2} \sum_{c}\lVert \mathbf{w}_c \rVert_2^2 .
$$

This is the form scikit-learn uses. $C$ multiplies the data-fit term, so:

- Small $C$ (like 0.001): the penalty dominates. Weights stay small and smooth, the model is simple, it may **underfit**.
- Large $C$ (like 100): the data term dominates. Weights can grow, the model fits the training set tightly, it may **overfit**.

We cannot pick $C$ by looking at the test set (that would be cheating on the exam). We pick it by looking at a **validation set** carved out of the training set.

**What this means for our hands:** with small $C$ the weight images in A.3 will look like smooth, blurry hand shapes; with large $C$ they will look speckled, with individual pixels having large weights. Stage 1 question 1 is choosing $C$, and the choice will be visible in the pictures.

## A.6 Validation, and why near-copies are a trap

The honest way to choose $C$ is: split the training images into a **train part** (say 80 percent) that the judges learn from, and a **validation part** (20 percent) that we only score on. The validation accuracy tells us how well the model generalises, and we pick the $C$ with the best validation accuracy. Only afterwards, once, we open the envelope and compute test accuracy.

Here is the trap. The dataset was made from only about 1,700 original photos. Each original was copied about 16 times with small changes (brightness, contrast, a little rotation, a little pixelation). If we split randomly, almost every validation image has a near-twin in the train part. The model has effectively seen it already. Validation accuracy will look wonderful and the test set, which was made from different original photos, will look worse.

The fix is a **group-aware split**: figure out which images are copies of the same original (a "group"), and put the whole group on one side. We do not have the original photo IDs, so we must recover the groups from the pixels themselves. We will do it by measuring distances between images: copies of one photo are much closer to each other than to anything else. Images closer than a threshold are linked; connected clumps are groups.

**What this means for our hands:** Stage 1 self-check 3 asks exactly why a random split gives a higher number than the test set. Our evidence will be two validation curves (random split vs group split) next to one test number.

## A.7 Measuring: accuracy, confusion matrix, confused pairs

**Accuracy** is the fraction of images where the predicted letter equals the true letter.

The **confusion matrix** is a 24 x 24 table. Row = true letter, column = predicted letter, cell = how many images. The diagonal is correct answers. A big off-diagonal number in row M, column N means "many real M's were called N". The most confused pairs are the largest off-diagonal entries (we add cell (M, N) and cell (N, M) to get a symmetric count).

**Macro-F1** is a second score that treats every letter equally even if some letters have more images; we report it as a sanity check.

## A.8 Why a shifted hand confuses a scorecard

Take judge $c$'s scorecard. Its score is $\sum_j w_{cj} x_j$: pixel $j$ is matched with weight $j$. Shift the image 2 pixels right. Now the brightness that used to be at slot $j$ is at slot $j+2$, so it gets multiplied by $w_{c,j+2}$ instead of $w_{cj}$. Unless the scorecard happens to be almost the same at neighbouring slots (smooth), the score changes a lot. A sharp edge in the hand that used to land on a positive weight may now land on a negative one.

Mathematically: the classifier is linear in pixel **positions**, and shifting permutes the positions. There is nothing in the model that says "position $j$ and position $j+2$ are related". It must learn the relation from data, and the training data (hands roughly centred) never showed it.

**What this means for our hands:** Stage 3 question 3 shifts the test images by 1, 2 and 3 pixels. We expect the raw-pixel judge to suffer, PCA to suffer similarly (it is still linear in positions), and the CNN to suffer much less, for reasons in A.12.

## A.9 PCA: finding the directions the data wiggles most

Stage 2 asks whether we can replace the 784 raw pixels by a smaller number $k$ of better features, found without using the labels at all.

Think of all 27,455 training images as 27,455 dots in 784-dimensional space. They do not fill the space evenly. Hands are mostly in the middle, mostly similar; the dots form a flattened cloud. **Principal Component Analysis** finds the directions along which the cloud is longest. The first direction (first principal component) is where the dots spread out the most; the second is the next most, at right angles to the first; and so on.

The steps:

1. **Centre.** Compute the mean image $\boldsymbol{\mu} = \frac{1}{n}\sum_i \mathbf{x}_i$ (the "average hand") and subtract it from every image: $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \boldsymbol{\mu}$.
2. **Covariance.** Build the 784 x 784 matrix that measures how each pair of pixels varies together:
   $$
   \boldsymbol{\Sigma} = \frac{1}{n-1} \sum_{i=1}^{n} \tilde{\mathbf{x}}_i \tilde{\mathbf{x}}_i^{\top}.
   $$
   Entry $(j, l)$ is positive if pixels $j$ and $l$ tend to be bright together, negative if one tends to be bright when the other is dark.
3. **Eigenvectors.** Find vectors $\mathbf{v}_1, \mathbf{v}_2, \dots$ and numbers $\lambda_1 \ge \lambda_2 \ge \dots \ge 0$ such that $\boldsymbol{\Sigma}\mathbf{v}_m = \lambda_m \mathbf{v}_m$. Each $\mathbf{v}_m$ is a direction (a unit-length list of 784 numbers, so it is also an image: an **eigen-hand**). $\lambda_m$ is the variance of the data along that direction: how much the dots spread out that way.
4. **Explained variance ratio.** The fraction of all the spread captured by the first $k$ directions is
   $$
   \text{EVR}(k) = \frac{\sum_{m=1}^{k} \lambda_m}{\sum_{m=1}^{784} \lambda_m}.
   $$
   If EVR(50) = 0.9, then 50 numbers capture 90 percent of how images differ from each other.
5. **Project.** The new features of an image are its coordinates along the first $k$ directions:
   $$
   z_m = \mathbf{v}_m \cdot (\mathbf{x} - \boldsymbol{\mu}), \qquad m = 1, \dots, k.
   $$
   So the image becomes a list of $k$ numbers instead of 784.
6. **Reconstruct.** Going back is
   $$
   \hat{\mathbf{x}} = \boldsymbol{\mu} + \sum_{m=1}^{k} z_m \mathbf{v}_m .
   $$
   With $k = 784$ you get the image back exactly. With small $k$ you get a smoothed version that keeps the big shapes and drops the fine detail. The dropped part is, among other things, noise.

Important facts the course asks about:

- **PCA must be fitted on the training part only.** The mean and the directions are learned quantities. If we computed them with validation or test images included, information from the exam would leak into the features.
- **PCA cannot make classes more linearly separable.** Projection is a linear map $\mathbf{z} = \mathbf{V}^{\top}(\mathbf{x} - \boldsymbol{\mu})$. Any linear classifier on $\mathbf{z}$ is $\mathbf{u} \cdot \mathbf{z} + b = (\mathbf{V}\mathbf{u}) \cdot (\mathbf{x} - \boldsymbol{\mu}) + b$, which is a linear classifier on $\mathbf{x}$. So every boundary you can draw in PCA space you could already draw in pixel space; the best possible training accuracy cannot go up. It **can** still improve test accuracy, because throwing away the low-variance directions removes dimensions in which the classifier could overfit to noise, and because fewer features means fewer weights to estimate from the same data.
- **PCA as noise removal.** Random pixel noise spreads its energy evenly across all 784 directions, while the signal (the hand) lives mostly in the first few dozen. Keeping only the top $k$ directions keeps most of the signal and only $k/784$ of the noise.

**What this means for our hands:** the first eigen-hand will probably look like a blurry hand-versus-background contrast or a brightness gradient; later ones will look like edges and finger positions. Training the softmax on 50 numbers instead of 784 should be much faster. The noise test will show whether the raw-pixel judge or the PCA judge falls apart first when we sprinkle noise on the test images.

## A.10 Convolution: a tiny scorecard that slides

Stage 3 lets the computer learn features from the labels. The tool is a **convolutional layer**.

Instead of a scorecard with 784 slots covering the whole image, take a tiny 3 x 3 scorecard with 9 weights (a **filter** or **kernel**). Slide it over the image. At each position, multiply the 9 weights by the 9 pixels under them, add up, add a bias, write the result into an output grid at that position. The output grid is a **feature map**: a picture of "how strongly does this 3 x 3 pattern appear here".

For a filter $\mathbf{K}$ (3 x 3) with bias $b$, the feature map at position $(r, s)$ is

$$
h_{r,s} = b + \sum_{a=-1}^{1} \sum_{d=-1}^{1} K_{a,d}\; x_{r+a,\, s+d} .
$$

We pad the image border with zeros so the output stays 28 x 28.

The same 9 weights are used at every position. This is **weight sharing**. Two consequences:

- Very few parameters. The first layer has 16 filters, each 3 x 3 x 1 (one input channel, grayscale) plus a bias: $16 \times (3 \times 3 \times 1) + 16 = 160$ parameters. A fully connected layer from 784 pixels to 16 units would have $784 \times 16 + 16 = 12{,}560$. That is Stage 3 self-check 1. (The second layer has 32 filters, each 3 x 3 x 16, plus 32 biases: $32 \times 144 + 32 = 4{,}640$.)
- **Equivariance to shifts.** If the input moves 2 pixels right, the feature map moves 2 pixels right too, because the same filter is applied everywhere. The detector does not care where the pattern is; it just reports where it found it.

After each convolution we apply **ReLU**, $\text{ReLU}(h) = \max(0, h)$, which sets negative responses to zero. This is what makes the features **nonlinear**: a stack of linear operations would collapse into one linear operation (like PCA + softmax did), but a $\max$ in between breaks that and lets later layers combine "pattern present / not present" signals in ways a single scorecard cannot.

## A.11 Pooling: shrinking and forgiving

**Max pooling 2 x 2** cuts the feature map into 2 x 2 blocks and keeps only the largest value in each block. A 28 x 28 map becomes 14 x 14. Two effects:

- Fewer numbers to process downstream.
- **Tolerance to small shifts.** If a strong response moves by one pixel but stays inside the same 2 x 2 block, the pooled output does not change at all. Two pooling layers in a row give tolerance to shifts of a few pixels in total.

This is the answer to Stage 3 self-check 2: convolution makes the response move with the pattern instead of changing, and pooling makes the downstream numbers insensitive to small moves.

## A.12 The whole CNN, and why its last layer is Stage 1 again

Our fixed network:

```
input 1 x 28 x 28
Conv 3x3, 16 filters, pad 1   -> 16 x 28 x 28     (160 parameters)
ReLU
MaxPool 2x2                   -> 16 x 14 x 14
Conv 3x3, 32 filters, pad 1   -> 32 x 14 x 14     (4,640 parameters)
ReLU
MaxPool 2x2                   -> 32 x 7 x 7
Flatten                       -> 1568
Linear 1568 -> 24             -> 24 scores        (1568 x 24 + 24 = 37,656 parameters)
softmax
```

Total about 42,456 parameters. Look at the last two lines: a list of 1,568 numbers goes into a linear layer that produces 24 scores and a softmax turns them into probabilities. That is exactly the Stage 1 classifier, only now its input is the 1,568 learned features instead of 784 raw pixels. In Stage 2 the input was $k$ PCA features. Same judge, three different sets of eyes.

The 1,568 numbers just before the last layer are "what the CNN sees". Stage 3 question 2 takes those numbers for many images, squeezes them to 2D with PCA (A.9 again, used only for drawing), and colours the dots by letter. If the colours form tighter, more separate clumps than the same picture made from raw pixels, the learned features are better organised for telling letters apart. We also compute a **silhouette score**, a number between -1 and 1 that is higher when points sit close to their own letter's clump and far from other clumps, so the comparison is not just visual.

## A.13 Training a network: mini-batches, epochs, learning rate, early stopping

The CNN has about 42,000 weights, and the loss is no longer a simple bowl, so we use **stochastic gradient descent** with the **Adam** variant. The loop:

1. Shuffle the training images. Take a **mini-batch** of 128.
2. Run them through the network, compute the cross-entropy loss (A.4) averaged over the batch.
3. Compute the gradient of the loss with respect to every weight (backpropagation; PyTorch does this automatically).
4. Move every weight a small step against its gradient. The step size is the **learning rate** (we use 0.001; Adam adapts it per weight).
5. Next batch. After all batches have been used once, one **epoch** is done.
6. At the end of each epoch, compute loss and accuracy on the validation set without updating weights. Record both.

Plotting training and validation loss per epoch gives the **training curves**. Normal: both go down. **Overfitting**: training loss keeps going down but validation loss turns around and goes up; the network is memorising. That is Stage 3 self-check 3.

**Early stopping**: watch the validation loss; if it has not improved for 5 epochs in a row, stop, and go back to the weights from the best epoch. Simple and honest; it is the only anti-overfitting device we use, because the course fixes the architecture.

## A.14 The three robustness tests and what we expect

| Perturbation | Stage | Raw pixels | PCA + softmax | CNN |
| --- | --- | --- | --- | --- |
| Gaussian noise on test images | 2 | Hurt: every noisy pixel multiplies a weight | Less hurt: most noise lies in discarded directions | not tested in Stage 2 |
| Shift by 1-3 pixels | 3 | Hurt: positions no longer match weights (A.8) | Hurt similarly: still linear in positions | Least hurt: equivariance + pooling (A.10-A.11) |

These are expectations, not results. The course requires you to write your own predictions before running, and to report whether they were right. The table above is the textbook's reasoning; your prediction in the report must be in your words and may differ.

## A.15 The story in one paragraph

We start with the simplest thing: a linear judge looking at raw pixels. It works surprisingly well on this dataset, but the weight images and the confusion pairs show it is matching pixel positions, not hand shapes. We ask whether unsupervised compression (PCA) can keep the accuracy with fewer, less noisy numbers and faster training; the explained-variance curve and the noise test answer that. Then we ask whether letting the model learn its own nonlinear, position-tolerant features (a CNN) beats both; the feature-space plots and the shift test answer that. Each experiment's answer motivates the next question, and that chain is the final report.

---

# Part B - Each phase in exact detail

Conventions used in Part B:

- All paths are relative to the repository root.
- "Train split" and "val split" mean the indices in `data/splits/split_v1.npz`. "Test set" means `sign_mnist_test.csv`.
- Seed is 42 unless stated.
- Figures are PDF, saved with `bbox_inches="tight"`, default size 6 x 4 inches unless stated. Tables are LaTeX fragments (a `tabular` environment only, no `table` wrapper) produced by `tools/build_tables.py`, so the report controls captions.
- Letters: the 24 classes are A-I and K-Y. Labels in the CSV are 0-25 with 9 and 25 absent. We remap to 0-23 for scikit-learn and PyTorch, and keep a `LETTERS` list for display.

## B.0 Phase 0 - Foundation

### B.0.1 Environment

1. `py -3.14 -m venv .venv`; activate; `pip install --upgrade pip`.
2. `requirements.txt` contents (unpinned at first, pinned to the installed versions after the first successful install): `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `jupyter`, `nbconvert`, `ipykernel`, `joblib`.
3. Check `python -c "import torch; print(torch.__version__)"`. If it fails, install the newest Python that PyTorch supports, recreate the venv with it, update `AGENTS.md` Section 9 and the README, log deviation.

### B.0.2 Data placement

1. Move `docs/Dataset/sign_mnist_train.csv`, `docs/Dataset/sign_mnist_test.csv`, `docs/Dataset/archive.zip` to `data/raw/`.
2. Delete `docs/Dataset/sign_mnist_train/` and `docs/Dataset/sign_mnist_test/` (duplicate copies of the CSVs).
3. Keep `docs/Dataset/*.png` (the ASL alphabet reference pictures; useful in the defense).
4. `.gitignore` covers `data/raw/`.

### B.0.3 `src/signlang/` modules

- `seed.py`: `set_seed(seed=42)` sets `random`, `numpy`, and `torch` (if importable) seeds and `PYTHONHASHSEED`.
- `data.py`:
  - `LETTERS = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]`.
  - `load_csv(path) -> (X, y_raw)`: `X` float32 of shape (n, 784) scaled by 1/255; `y_raw` int labels 0-25.
  - `remap_labels(y_raw) -> y`: maps the 24 present labels to 0-23 in letter order. `label_to_letter(y)`.
  - `load_train()`, `load_test()` using `data/raw/`.
  - `to_images(X) -> (n, 28, 28)`.
- `timing.py`: `class Timer` context manager; `.seconds` attribute after exit.
- `metrics.py`: `accuracy`, `macro_f1`, `confusion(y_true, y_pred) -> 24x24`, `top_confused_pairs(cm, n=5) -> list of (letter_a, letter_b, count_ab, count_ba)`, `summary(y_true, y_pred) -> dict`.
- `plotting.py`: `image_grid(images, titles, ncols, cmap="gray")`, `weight_images(W, letters, ...)` with `cmap="RdBu_r"` and symmetric `vmin/vmax = ±max|W|` per panel, `confusion_heatmap`, `curve(x, ys, labels, ...)`, `scatter_2d(Z, y, ...)` using a 24-colour `tab20`+`tab20b` palette and a shared legend, `savefig(fig, path)` that creates parent directories.
- `perturb.py`: `shift(images, dx, dy)` by integer pixels with edge replication (`np.pad(mode="edge")` then crop); `add_gaussian_noise(X, sigma, rng)` clipped to [0, 1]. Both operate on arrays shaped (n, 28, 28) or (n, 784) and return the same shape.
- `split.py`:
  - `duplicate_groups(X, y, n_components=50, n_neighbors=10, threshold=None, random_state=42) -> (group_id, distances, threshold)`: PCA to 50 dims (fit on all training X; labels not used for the projection), per-label `NearestNeighbors(metric="cosine")`, histogram of 1st-neighbour distances, threshold = valley between the two modes if `threshold is None` (found as the minimum of a smoothed histogram between the two highest peaks), edges below threshold, connected components via `scipy.sparse.csgraph.connected_components` (scipy comes with scikit-learn).
  - `make_split(group_id, y, val_frac=0.2, random_state=42) -> (train_idx, val_idx)`: `GroupShuffleSplit`; assert each class has at least 100 val images; otherwise retry with `random_state+1` up to 10 times and record the seed used.
  - `make_random_split(y, val_frac=0.2, random_state=42)`: `StratifiedShuffleSplit`, the diagnostic split.
  - `save_split(path, **arrays)` and `load_split(path) -> dict`. Also writes `split_v1.json` with: number of groups, group-size quantiles, threshold, seed used, class counts per side, and the random split indices under separate keys `rand_train_idx`, `rand_val_idx`.

### B.0.4 `tools/build_tables.py`

`df_to_tex(df, path, float_fmt="%.3f", column_format=None)` writes a `tabular` using `DataFrame.to_latex(index=False, escape=True)`. Column headers are given in the DataFrame so the agent controls wording. Nothing else.

### B.0.5 `notebooks/stage0_data_and_split.ipynb`

Cells, in order:

1. Imports, `set_seed(42)`, load train and test.
2. Print shapes, label sets, confirm 9 and 25 absent, remap. Class count bar chart for train and test side by side -> `reports/figures/stage1/class_hist.pdf`.
3. Example grid: 5 random training images per letter, 24 rows x 5 columns, letter label on the left -> `reports/figures/stage1/examples_grid.pdf` (size 6 x 20 inches).
4. Duplicate analysis: call `duplicate_groups` with `threshold=None`; plot the histogram of nearest-neighbour distances with the chosen threshold as a vertical line -> `reports/figures/stage1/dup_distance_hist.pdf`. Print: number of groups, min/median/mean/max group size, fraction of images in groups of size 1.
   - Pass criterion: histogram visibly bimodal and number of groups between 800 and 4,000. If not, apply the fallback (Section 7 of PRD) and log deviation C-1.
5. Show one group: a row with up to 10 images from one randomly chosen group of size >= 8, to make "altered copies" visible -> `reports/figures/stage1/dup_group_example.pdf`.
6. `make_split`; print class counts on each side; `make_random_split`; `save_split("data/splits/split_v1.npz", ...)`.
7. Sanity: load the split back, assert no index overlap, assert `len(train)+len(val) == 27455`.

### B.0.6 Your steps in Phase 0

After cell 3's figure exists and before Phase 1 runs: look at `examples_grid.pdf`, write in `reports/stage1/report.tex` Data exploration which letters you expect to be confused. Write the Stage 1 self-check answers. Write the Stage 1 predictions.

## B.1 Phase 1 - Stage 1, raw pixels

### B.1.1 Inputs

`data/raw/*.csv`, `data/splits/split_v1.npz`, your predictions in `reports/stage1/report.tex`.

### B.1.2 `notebooks/stage1_raw_pixels.ipynb`

1. Imports, seed, load data and split. `Xtr, ytr, Xva, yva` from the group split; `Xrtr, yrtr, Xrva, yrva` from the random split; `Xte, yte` test.
2. C sweep on group split: for `C in [0.001, 0.01, 0.1, 1, 10, 100]`, `LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42)` (multinomial is the default for lbfgs in current scikit-learn), fit inside `Timer`, record val accuracy, macro-F1, seconds, `n_iter_`, and whether it converged (no `ConvergenceWarning`). Store in a DataFrame.
3. Same sweep on the random split (val accuracy only is needed, but record the same fields).
4. Plot both validation-accuracy curves vs `C` on a log-x axis -> `reports/figures/stage1/c_sweep.pdf`. Table -> `reports/tables/s1_C_sweep.tex` with columns: C, Val acc (group split), Val acc (random split), Fit time (s), Converged.
5. Choose `C*` = argmax of group-split val accuracy (ties -> smaller `C`). Print it.
6. Refit at `C*` on the group train split (this is the model used everywhere later). Save `models/stage1_softmax.joblib`.
7. **Test, once:** predict on `Xte`; accuracy and macro-F1. Table `reports/tables/s1_test.tex`: rows "Validation (group split)", "Validation (random split)", "Test", columns Accuracy, Macro-F1. The gap is visible by reading the rows.
8. Weight images: for the 5 chosen letters, reshape `coef_[c]` to 28 x 28; `RdBu_r`, symmetric limits per panel; below each, the mean training image of that letter in gray for comparison -> `reports/figures/stage1/weights.pdf` (2 rows x 5 columns, 10 x 4.5 inches).
9. Confusion matrix on test: counts heatmap, and row-normalised heatmap, side by side -> `reports/figures/stage1/confusion.pdf` (12 x 5.5 inches).
10. Top-5 confused pairs: `top_confused_pairs(cm, 5)`. Table `reports/tables/s1_confused_pairs.tex`: columns True, Predicted, Count, Reverse count, Total. Figure: for each pair, one row of 8 images: 4 test images whose true letter is A and were predicted B, and 4 training images of letter B (so you can see what the model confused them with) -> `reports/figures/stage1/confused_pairs.pdf`.
11. Also print the ASL reference image path `docs/Dataset/american_sign_language.PNG` so you can compare the confused pairs with the real alphabet when writing.

### B.1.3 Outputs checklist

Figures: `class_hist.pdf`, `examples_grid.pdf`, `dup_distance_hist.pdf`, `dup_group_example.pdf` (from Phase 0), `c_sweep.pdf`, `weights.pdf`, `confusion.pdf`, `confused_pairs.pdf`. Tables: `s1_C_sweep.tex`, `s1_test.tex`, `s1_confused_pairs.tex`. Model: `models/stage1_softmax.joblib`.

### B.1.4 Checks

- Val accuracy differs between random and group split; if not, that is a result, not a bug, as long as the split sanity checks passed.
- lbfgs converged for all `C`, or the Converged column says no and the report will explain it.
- Weight images are not all-zero (a sign of over-regularisation) and not pure speckle (a sign the `C*` chosen is at the large end; still valid, just note it).

### B.1.5 Your steps in Phase 1

After the notebook has run: Results, Weight images, Confusions, Prediction vs. result, AI-use note. Then read ISLP 12.2 and write Stage 2 self-check answers and predictions before Phase 2 runs.

## B.2 Phase 2 - Stage 2, PCA

### B.2.1 Inputs

Same data and split; `models/stage1_softmax.joblib` and `C*`; your Stage 2 predictions.

### B.2.2 `notebooks/stage2_pca.ipynb`

1. Imports, seed, load data and split, load Stage 1 model and `C*`.
2. `pca = PCA(n_components=500, svd_solver="randomized", random_state=42).fit(Xtr)`; record fit time. Save `models/stage2_pca.joblib` (includes mean and components). Note: fitted on the **group train split**, not the val or test.
3. Explained variance: left panel `explained_variance_ratio_[:50]` as bars; right panel cumulative sum to 500 with vertical markers at 10, 50, 150 and horizontal markers at 0.90, 0.95 -> `reports/figures/stage2/explained_variance.pdf` (12 x 4.5 inches). Print EVR at 10, 50, 150 and the smallest `k` reaching 0.90 and 0.95.
4. Eigen-hands: mean image plus first 10 components, each reshaped 28 x 28, `RdBu_r` symmetric limits -> `reports/figures/stage2/eigenhands.pdf` (11 panels in one row or 2 rows).
5. Reconstructions: fix 4 validation images (first occurrence of letters A, F, R, Y in the val split), show original and reconstructions at k = 10, 50, 150 in a 4 x 4 grid with per-image MSE in the title -> `reports/figures/stage2/reconstructions.pdf`.
6. k sweep: for `k in [10, 20, 50, 100, 150, 200, 300, 500]`: `Ztr = pca.transform(Xtr)[:, :k]`, `Zva = pca.transform(Xva)[:, :k]` (transform once at 500 and slice), fit `LogisticRegression(C=C*, solver="lbfgs", max_iter=2000, random_state=42)` inside `Timer`, record val accuracy, macro-F1, fit seconds. Also record the Stage 1 raw-pixel fit time and val accuracy as a reference row `k = 784 (raw)`.
7. Plot val accuracy vs `k` (log-x) with a horizontal dashed line at the raw-pixel val accuracy -> `k_sweep_acc.pdf`; plot fit time vs `k` with the raw-pixel time as a dashed line -> `k_sweep_time.pdf`.
8. Propose `k*`: smallest `k` in the grid whose val accuracy is within 0.5 percentage points of the best val accuracy in the grid. Print the proposal and the full table. **Stop and tell the student.** The student decides `k*` (may differ). Set `K_STAR` in the next cell to the decided value.
9. Refit at `K_STAR`; save `models/stage2_softmax.joblib`. **Test, once:** accuracy and macro-F1 on `pca.transform(Xte)[:, :K_STAR]`. Also compute test accuracy at k = 10, 50, 150 for the required table (these are additional test evaluations of fixed, non-tuned models; the PDF's required table asks for them explicitly).
10. Table `reports/tables/s2_k_sweep.tex`: rows raw (784), 10, 50, 150, `K_STAR` and the other grid values; columns k, EVR, Val acc, Test acc (only where evaluated, else blank), Fit time (s).
11. Noise test: `rng = np.random.default_rng(42)`; for `sigma in [0.1, 0.2, 0.3]`: `Xte_noisy = add_gaussian_noise(Xte, sigma, rng)`; accuracy of the Stage 1 model on `Xte_noisy`; accuracy of the Stage 2 model on `pca.transform(Xte_noisy)[:, :K_STAR]`. Table `reports/tables/s2_noise.tex`: columns sigma, Raw-pixel acc, PCA-k acc, Drop (raw), Drop (PCA), with a sigma = 0 row for the clean numbers.
12. Figure `noise_examples.pdf`: one test image, 2 rows x 4 columns: top row the image at sigma 0, 0.1, 0.2, 0.3; bottom row the PCA-`K_STAR` reconstruction of each, to show the smoothing. Figure `noise_acc.pdf`: accuracy vs sigma for the two models.

### B.2.3 Outputs checklist

Figures: `explained_variance.pdf`, `eigenhands.pdf`, `reconstructions.pdf`, `k_sweep_acc.pdf`, `k_sweep_time.pdf`, `noise_examples.pdf`, `noise_acc.pdf`. Tables: `s2_k_sweep.tex`, `s2_noise.tex`. Models: `stage2_pca.joblib`, `stage2_softmax.joblib`.

### B.2.4 Checks

- EVR is monotone, reaches about 1.0 at 500 or less.
- Reconstruction MSE decreases with `k`.
- Fit time at small `k` is much smaller than raw; if not, record the machine state (other load) and rerun once.
- The k=784 raw row matches the Stage 1 numbers exactly (same model file).

### B.2.5 Your steps in Phase 2

Decide `k*` when asked. After the run: Components, Number of components, Noise, Prediction vs. result, AI-use note. Then read ISLP 10.3, 10.7, 10.9 and write Stage 3 self-check answers and predictions.

## B.3 Phase 3 - Stage 3, CNN

### B.3.1 Inputs

Same data and split; Stage 1 and Stage 2 model files; your Stage 3 predictions; `torch` importable.

### B.3.2 `src/signlang/cnn.py`

```python
class SmallCNN(nn.Module):
    conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    pool  = nn.MaxPool2d(2)
    fc    = nn.Linear(32 * 7 * 7, 24)

    def features(self, x):  # x: (n, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.flatten(1)   # (n, 1568)

    def forward(self, x):
        return self.fc(self.features(x))   # logits; softmax lives in the loss
```

- `count_parameters(model) -> list of (name, count)` and total.
- `train(model, Xtr, ytr, Xva, yva, max_epochs=30, batch_size=128, lr=1e-3, patience=5, seed=42, device="cpu") -> history` where history is a DataFrame with columns epoch, train_loss, train_acc, val_loss, val_acc, seconds. Uses `torch.optim.Adam`, `nn.CrossEntropyLoss`, shuffles with a seeded generator, `num_workers=0`. Early stopping on val_loss with the given patience; restores the best state dict; records `best_epoch`.
- `predict(model, X, batch_size=1024) -> y_pred` and `predict_proba`.
- `extract_features(model, X) -> (n, 1568)` numpy.
- `save(model, path)` / `load(path)`.

### B.3.3 `notebooks/stage3_cnn.ipynb`

1. Imports, seed, `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(True, warn_only=True)`, load data and split, reshape to (n, 1, 28, 28).
2. Build model; print `count_parameters` per layer and total. (Compare with your self-check 1 answer: conv1 = 160, conv2 = 4,640, fc = 37,656.)
3. Train with the B.3.2 defaults inside `Timer`. Print best epoch and total epochs. Save `models/stage3_cnn.pt`.
4. Curves: loss (train, val) and accuracy (train, val) vs epoch, two panels, best epoch marked with a vertical line -> `reports/figures/stage3/curves.pdf`. Table `reports/tables/s3_training.tex`: epochs run, best epoch, train/val loss and acc at best epoch, training seconds, parameter count.
5. **Test, once:** CNN test accuracy and macro-F1.
6. Load Stage 1 and Stage 2 models; compute (or reload from the Stage 1/2 tables) their test accuracy; table `reports/tables/s3_comparison.tex`: Model, Features, Number of features, Test acc, Macro-F1, Training time (s).
7. Feature space on the **validation split** (not test, to keep the test set for numbers only): `F_cnn = extract_features(model, Xva)`; `PCA(2).fit_transform` on `F_cnn` and separately on `Xva`. Scatter each with the 24-colour palette, one shared legend, same figure size -> `feature_space_raw.pdf`, `feature_space_cnn.pdf`. Silhouette score (`sklearn.metrics.silhouette_score`, labels = letters) on the 2D projections and on the full-dimensional features, computed on a fixed random subsample of 5,000 val images for speed -> `reports/tables/s3_silhouette.tex`: rows Raw pixels, CNN features; columns Silhouette (2D), Silhouette (full).
8. Shift test: for `s in [1, 2, 3]` and direction in `[(s,0), (-s,0), (0,s), (0,-s)]`: `Xs = shift(Xte_images, dx, dy)`; accuracy of Stage 1 model on `Xs.reshape(-1, 784)`, Stage 2 model on `pca.transform(...)[:, :K_STAR]`, CNN on `Xs[:, None]`. Table `reports/tables/s3_shift.tex`: rows Raw pixels, PCA + softmax, CNN; columns Shift 0, 1, 2, 3 (averaged over the four directions). Appendix table `s3_shift_directions.tex` with every direction. Figure `shift_acc.pdf`: accuracy vs shift for the three models. Figure `shift_examples.pdf`: one test image at shift 0, 1, 2, 3 to the right.
9. Optional bonus: if `data/own_hands/` contains images, run `tools/prepare_own_hands.py` logic (convert to grayscale, crop to square, resize to 28 x 28 with `PIL`, scale to [0,1]; optionally invert if the background is dark) and predict with the CNN; grid of your images with predicted and true letter in titles -> `own_hands.pdf`. Add `pillow` to `requirements.txt` only if this is used (log in TODO).

### B.3.4 Outputs checklist

Figures: `curves.pdf`, `feature_space_raw.pdf`, `feature_space_cnn.pdf`, `shift_examples.pdf`, `shift_acc.pdf`, optionally `own_hands.pdf`. Tables: `s3_training.tex`, `s3_comparison.tex`, `s3_silhouette.tex`, `s3_shift.tex`, `s3_shift_directions.tex`. Model: `models/stage3_cnn.pt`.

### B.3.5 Checks

- Parameter counts match the hand calculation.
- Early stopping triggered before 30 epochs, or the curves show no overfitting by epoch 30 (then say so).
- Shift 0 column equals the clean test accuracies in `s3_comparison.tex`.
- Shifts are implemented with edge replication (no black bars) — inspect `shift_examples.pdf`.

### B.3.6 Your steps in Phase 3

Optional photos. After the run: Training, Comparison, Learned feature space, Robustness, Prediction vs. result, AI-use note.

## B.4 Phase 4 - Final report and defense

1. `reports/final/report.tex` `\input`s `../stage1/body.tex`, `../stage2/body.tex`, `../stage3/body.tex` (each stage report is split into `report.tex` = wrapper with preamble and `body.tex` = sections, so the same body compiles alone and inside the final report), then the narrative section and the final AI-use note, which you write.
2. `planning/DEFENSE_PREP.md`: around 30 questions an examiner could ask, each with a pointer to the Part A section and the report section where the answer lives. No answers.
3. Clean-venv reproducibility: `py -3.14 -m venv .venv_check`, install, `jupyter nbconvert --execute` on all four notebooks in order, `latexmk -pdf` on all four reports. Compare the regenerated tables with the committed ones; small floating-point differences in times are expected, accuracies should match exactly for Stages 1-2 and to within 0.5 pp for Stage 3 (CPU determinism caveats).
4. Grep submission-facing files for `planning`, `AGENTS`, `TODO`, `FIXME`, `fill`, `write here`. Must be empty.

---

# Part C - Deviations from the plan

Rule: whenever an agent (or the student) implements something differently from Part B, append an entry here **in the same session**. Use the template. Number entries C-1, C-2, ... in order. Add a one-line pointer "See deviation C-n" to the affected Part B section. Never rewrite Part B to match; Part B stays the original plan so the difference is visible.

Future agents: read this part before Part B. Entries here override Part B.

## Template

```
### C-n: <short title>
- Date: YYYY-MM-DD
- Phase: 0 / 1 / 2 / 3 / 4
- Part B section affected: B.x.y
- Planned: <what Part B says>
- Done instead: <what was actually implemented>
- Why: <reason: bug, library limitation, result-driven, student decision, time>
- Effect on results: <none / which tables and figures changed and how>
- Decided by: agent / student / both
- Follow-up: <none / items added to TODO.md>
```

## Entries

(none yet)

---

# Part D - Glossary

| Symbol or term | Meaning |
| --- | --- |
| $\mathbf{x}$, $x_j$ | one image as a list of 784 numbers in [0,1]; the $j$-th pixel |
| $n$ | number of training images in the split being used |
| $y$ | true letter of an image, as an integer 0-23 |
| $\mathbf{w}_c$, $w_{cj}$, $b_c$ | weights and bias of the softmax judge for letter $c$ |
| $z_c$ | score (logit) of letter $c$: $\mathbf{w}_c \cdot \mathbf{x} + b_c$ |
| $p_c$ | softmax probability of letter $c$ |
| $\ell$, $L$ | cross-entropy loss for one image; average over the training set |
| $C$ | inverse regularisation strength in scikit-learn; larger $C$ = weaker penalty |
| $\lVert \mathbf{w} \rVert_2^2$ | sum of squared weights (L2 norm squared) |
| validation set | part of the training data held out for choosing hyperparameters |
| group split | a validation split where near-duplicate images stay on one side |
| accuracy | fraction of correct predictions |
| macro-F1 | mean over letters of the per-letter F1 score |
| confusion matrix | 24 x 24 table of true letter (row) vs predicted letter (column) |
| $\boldsymbol{\mu}$ | mean training image |
| $\boldsymbol{\Sigma}$ | 784 x 784 covariance matrix of the centred training images |
| $\mathbf{v}_m$, $\lambda_m$ | $m$-th principal component (eigen-hand) and its variance (eigenvalue) |
| EVR($k$) | fraction of total variance explained by the first $k$ components |
| $k$ | number of principal components kept |
| $z_m$ | coordinate of an image along component $m$ |
| $\hat{\mathbf{x}}$ | reconstruction of an image from $k$ components |
| $\sigma$ | standard deviation of the Gaussian noise added in the noise test |
| filter / kernel $\mathbf{K}$ | 3 x 3 grid of weights slid over the image |
| feature map $h_{r,s}$ | output of a filter at position $(r,s)$ |
| padding | zeros added around the border so the output keeps the input size |
| weight sharing | using the same filter weights at every position |
| ReLU | $\max(0, h)$ |
| max pooling | keeping the largest value in each 2 x 2 block |
| equivariance | the output moves when the input moves |
| logits | the 24 scores before softmax |
| epoch | one pass over all training images |
| mini-batch | the 128 images processed per gradient step |
| learning rate | step size of the optimiser (0.001) |
| Adam | an adaptive variant of stochastic gradient descent |
| early stopping | stop training when validation loss stops improving; keep the best weights |
| overfitting | training loss falls while validation loss rises |
| silhouette score | number in [-1, 1] measuring how tightly points cluster by label |
| shift test | evaluating on test images moved by 1-3 pixels, edge replicated |
| seed | the fixed random-number starting point (42) |
