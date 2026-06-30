# Weakly-Supervised Hyperspectral Image Segmentation Using Deep Neural Networks

Pixel-level landcover classification on the Indian Pines hyperspectral benchmark using a CNN + MLP architecture with PCA-based dimensionality reduction, spatial patch extraction, weak-class oversampling, and data augmentation.

**84.4% test accuracy · 0.90 macro F1 · 16 landcover classes · 2,563 test samples**

---

## Problem

Hyperspectral imaging (HSI) captures 200+ spectral bands per pixel — far more information than RGB — making it well-suited for precise landcover classification in agriculture, environmental monitoring, and remote sensing. The challenge is that raw HSI data is extremely high-dimensional, spatially correlated, and class-imbalanced (some landcover types have fewer than 10 labeled samples).

This project classifies every labeled pixel in the Indian Pines scene into one of 16 landcover classes using a deep learning pipeline that handles dimensionality reduction, spatial context encoding, class imbalance, and evaluation — end to end.

---

## Dataset

**Indian Pines** — a standard hyperspectral benchmark captured by the AVIRIS sensor over northwestern Indiana.

| Property | Value |
|---|---|
| Spatial dimensions | 145 × 145 pixels |
| Spectral bands (original) | 220 (200 after water absorption band removal) |
| Labeled pixels | 10,249 |
| Classes | 16 landcover types |
| Source | [Purdue University MultiSpec Site](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html) |

Download `Indian_pines_corrected.mat` and `Indian_pines_gt.mat` and place them in a `Data/` directory in the project root before running.

---

## Architecture

The pipeline is split across three notebooks that must be run in order:

```
CreatetheDatasets.ipynb  →  TrainTheModel.ipynb  →  Validation_and_Classification_Maps.ipynb
```

### Stage 1 — Preprocessing & Dataset Creation (`CreatetheDatasets.ipynb`)

1. **PCA dimensionality reduction** — reduces 200 spectral bands to 30 principal components, preserving >99% of variance while making the feature space tractable for a CNN
2. **Spatial patch extraction** — extracts a 5×5 pixel neighborhood around each labeled pixel, giving the model local spatial context rather than treating each pixel in isolation
3. **Train/test split** — stratified 75/25 split (`random_state=345`) to preserve class distribution
4. **Weak-class oversampling** — resamples underrepresented classes (e.g., Alfalfa: 54 total, Oats: 20 total) to prevent the model ignoring minority landcover types
5. **Data augmentation** — rotation-based augmentation applied to training patches

Hyperparameters are stored in `global_variables.txt` and loaded at runtime across all notebooks:
```
windowSize = 5
numPCAcomponents = 30
testRatio = 0.25
```

### Stage 2 — Model Training (`TrainTheModel.ipynb`)

```
Input: (N, 30, 5, 5)  — N patches, 30 PCA channels, 5×5 spatial window

Conv2D(90, 3×3, relu)
Conv2D(270, 3×3, relu)
Dropout(0.25)
Flatten()
Dense(180, relu)
Dropout(0.5)
Dense(16, softmax)
```

- **Optimizer:** SGD with Nesterov momentum (`lr=0.0001`, `momentum=0.9`, `decay=1e-6`)
- **Loss:** Categorical cross-entropy
- **Epochs:** 15, batch size 32
- Trained weights saved to `my_model5PCA30testRatio0.25.h5`

The number of filters in the first Conv2D layer (`C1 = 3 × numPCAcomponents = 90`) is derived from the PCA component count, keeping the spatial encoder proportional to the spectral input.

### Stage 3 — Evaluation & Classification Maps (`Validation_and_Classification_Maps.ipynb`)

- Loads saved model and held-out test patches
- Generates full classification report (precision, recall, F1 per class) and confusion matrix
- Reconstructs the full spatial classification map by running sliding-window inference over the original image and plots predicted vs. ground truth landcover maps using the `spectral` library

---

## Results

Test set: 2,563 labeled pixels, 25% holdout

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | 84.39% (84.4%) |
| **Macro F1** | 0.90 |
| **Macro Precision** | 0.89 |
| **Macro Recall** | 0.92 |
| **Test Loss** | 48.67% |

### Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Alfalfa | 1.00 | 1.00 | 1.00 | 11 |
| Corn-notill | 0.71 | 0.74 | 0.73 | 357 |
| Corn-mintill | 0.75 | 0.84 | 0.79 | 208 |
| Corn | 0.79 | 0.92 | 0.85 | 59 |
| Grass-pasture | 0.99 | 0.94 | 0.97 | 121 |
| Grass-trees | 0.99 | 0.98 | 0.99 | 183 |
| Grass-pasture-mowed | 1.00 | 1.00 | 1.00 | 7 |
| Hay-windrowed | 1.00 | 1.00 | 1.00 | 120 |
| Oats | 0.83 | 1.00 | 0.91 | 5 |
| Soybean-notill | 0.74 | 0.80 | 0.77 | 243 |
| Soybean-mintill | 0.84 | 0.72 | 0.78 | 614 |
| Soybean-clean | 0.73 | 0.78 | 0.76 | 148 |
| Wheat | 1.00 | 1.00 | 1.00 | 51 |
| Woods | 0.98 | 0.98 | 0.98 | 316 |
| Buildings-Grass-Trees-Drives | 0.87 | 0.95 | 0.91 | 97 |
| Stone-Steel-Towers | 0.96 | 1.00 | 0.98 | 23 |

**Observations:**
- Perfect or near-perfect F1 on spectrally distinct classes (Alfalfa, Hay-windrowed, Wheat, Grass-pasture-mowed, Stone-Steel-Towers)
- Main confusion is between Corn-notill, Corn-mintill, and Soybean variants — spectrally similar crops at different growth stages, a known challenge in this benchmark
- Macro recall (0.92) exceeds macro precision (0.89), meaning the model slightly over-predicts on ambiguous crop boundaries rather than under-predicting

### Confusion Matrix

```
                              Alfalfa  Corn-nt  Corn-mt  Corn  G-past  G-tree  G-p-m  Hay-w  Oats  Soy-nt  Soy-mt  Soy-cl  Wheat  Woods  BGTD  SST
Alfalfa                       [ 11       0        0       0      0       0       0      0      0      0       0       0       0      0      0     0 ]
Corn-notill                   [  0     262       17       7      0       0       0      0      0     22      37      11       0      0      1     0 ]
Corn-mintill                  [  0      13      173       3      0       0       0      0      0      4      11       4       0      0      0     0 ]
Corn                          [  0       4        1      54      0       0       0      0      0      0       0       0       0      0      0     0 ]
Grass-pasture                 [  0       1        0       1    114       0       0      0      0      0       0       1       0      1      3     0 ]
Grass-trees                   [  0       0        0       0      0     180       0      0      0      0       0       0       0      0      3     0 ]
Grass-pasture-mowed           [  0       0        0       0      0       0       7      0      0      0       0       0       0      0      0     0 ]
Hay-windrowed                 [  0       0        0       0      0       0       0    120      0      0       0       0       0      0      0     0 ]
Oats                          [  0       0        0       0      0       0       0      0      5      0       0       0       0      0      0     0 ]
Soybean-notill                [  0       4        2       0      0       1       0      0      1    194      37       2       0      0      2     0 ]
Soybean-mintill               [  0      73       33       2      0       0       0      0      0     35     450      20       0      0      1     0 ]
Soybean-clean                 [  0      18        3       0      0       1       0      0      0      8       1     116       0      0      0     1 ]
Wheat                         [  0       0        0       0      0       0       0      0      0      0       0       0      51      0      0     0 ]
Woods                         [  0       0        0       0      2       0       0      0      0      0       0       0       0    311      3     0 ]
Buildings-Grass-Trees-Drives  [  0       0        0       0      0       1       0      0      0      0       0       0       0      4     92     0 ]
Stone-Steel-Towers            [  0       0        0       0      0       0       0      0      0      0       0       0       0      0      0    23 ]
```

---

## How to Reproduce

### 1. Clone and set up environment

```bash
git clone https://github.com/hetvishah208/Weakly-Supervised-HyperSpectral-Image-Segmentation-Using-Deep-Neural-Network.git
cd Weakly-Supervised-HyperSpectral-Image-Segmentation-Using-Deep-Neural-Network
pip install -r requirements.txt
```

### 2. Download the dataset

Download from the [Purdue MultiSpec site](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html):
- `Indian_pines_corrected.mat`
- `Indian_pines_gt.mat`

Place both files in a `Data/` directory at the project root.

### 3. Configure hyperparameters (optional)

Edit `global_variables.txt` to change window size, PCA components, or test ratio:
```
windowSize = 5
numPCAcomponents = 30
testRatio = 0.25
```

### 4. Run notebooks in order

```
1. CreatetheDatasets.ipynb        — preprocessing, patch extraction, train/test split
2. TrainTheModel.ipynb            — model training, saves weights to .h5
3. Validation_and_Classification_Maps.ipynb  — evaluation report + spatial maps
```

Pre-computed artifacts for the default configuration (`windowSize=5, PCA=30, testRatio=0.25`) are already committed:
- `X_testPatches_5PCA30testRatio0.25.npy` — test patches
- `y_testPatches_5PCA30testRatio0.25.npy` — test labels
- `my_model5PCA30testRatio0.25.h5` — trained model weights
- `reportWindowSize5PCA30testRatio0.25.txt` — full classification report

You can skip to notebook 3 to run evaluation and generate classification maps immediately without retraining.

---

## Requirements

```
numpy
scipy
scikit-learn
keras (2.x)
tensorflow
spectral
matplotlib
h5py
scikit-image
```

> **Note:** This project uses Keras 2.x with TensorFlow backend and Theano-style channel ordering (`K.set_image_dim_ordering('th')`). If running on a newer TensorFlow version, set `data_format='channels_first'` in Conv2D layers or update to `channels_last` ordering with the corresponding reshape logic.

---

## Project Structure

```
├── Data/                                          # Dataset directory (download separately)
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── CreatetheDatasets.ipynb                        # Stage 1: preprocessing & patch extraction
├── TrainTheModel.ipynb                            # Stage 2: CNN+MLP training
├── Validation_and_Classification_Maps.ipynb       # Stage 3: evaluation & spatial maps
├── global_variables.txt                           # Shared hyperparameters across notebooks
├── my_model5PCA30testRatio0.25.h5                 # Trained model weights
├── X_testPatches_5PCA30testRatio0.25.npy          # Held-out test patches
├── y_testPatches_5PCA30testRatio0.25.npy          # Held-out test labels
├── y_trainPatches_5PCA30testRatio0.25.npy         # Training labels
└── reportWindowSize5PCA30testRatio0.25.txt        # Full classification report
```

---

## Key Design Decisions

**Why PCA before the CNN?** Raw HSI has 200 bands with high inter-band correlation. PCA reduces to 30 uncorrelated components while retaining the spectral variance that discriminates landcover types — this makes training faster and reduces overfitting on the small labeled set.

**Why 5×5 spatial patches?** Each pixel is classified using its local neighborhood rather than in isolation. A 5×5 window captures enough spatial context (texture, boundary information) without including pixels from distant, unrelated regions.

**Why oversampling instead of class weighting?** Several classes have fewer than 20 labeled samples total (Oats: 20, Grass-pasture-mowed: 28). Oversampling at the patch level ensures the model sees sufficient examples of minority classes during training rather than effectively ignoring them.

---

## Reference

This project implements and extends the CNN+MLP architecture for hyperspectral classification as described in:

> Paoletti, M.E., et al. "Deep learning classifiers for hyperspectral imaging: A review." *ISPRS Journal of Photogrammetry and Remote Sensing*, 2019.

The Indian Pines dataset is a standard benchmark used extensively in remote sensing literature for comparing HSI classification methods.
