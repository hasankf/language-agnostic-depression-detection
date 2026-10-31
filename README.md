# Speech Depression Detection via Siamese Forward-Backward Spectrogram Feature Invariance

Automated depression detection using a regularized ResNet-18 feature extractor on 3.0-second participant audio segments from the DAIC-WOZ dataset. The pipeline enforces acoustic feature symmetry across forward and time-flipped Mel-spectrograms using a directional Cosine Invariance Loss.

---

## 📌 1. Methodology & Pipeline Overview

The framework processes 3-second speech slices to classify speech as **Normal (PHQ-8 < 10)** or **Depressed (PHQ-8 ≥ 10)**.

### Model Architecture (`train.py`)
* **Backbone:** Torchvision `ResNet18` (initialized with default ImageNet weights).
* **Layer Freezing:** `conv1`, `bn1`, `layer1`, `layer2`, and `layer3` parameters are completely frozen (`requires_grad = False`) to retain low-level acoustic edge representations.
* **Trainable Parameters:** Only `layer4` and the final classification head (`fc`) are updated during training.
* **Bottleneck Embeddings:** A 512-dimensional vector $z$ extracted via `base.avgpool` and flattened prior to the dropout (`p = 0.5`) and linear classifier.

### Forward-Backward Siamese Invariance Formulation
Rather than training on passive audio augmentation, the training loop passes forward and time-reversed spectrogram representations in parallel:
1. **Forward Spec ($x_{\text{fwd}}$):** The original normalized 3-channel Mel-spectrogram.
2. **Backward Spec ($x_{\text{bwd}}$):** The Mel-spectrogram horizontally flipped along the temporal axis via `torch.flip(images, dims=[-1])`.
3. **Embeddings:** $z_{\text{fwd}} = f_{\theta}(x_{\text{fwd}})$ and $z_{\text{bwd}} = f_{\theta}(x_{\text{bwd}})$.
4. **Classification Loss:** Computed exclusively on forward features:
   $$\mathcal{L}_{\text{cls}} = \text{CrossEntropyLoss}(\hat{y}_{\text{fwd}}, y; w=[1.0, 2.2])$$
5. **Invariance Loss:** Measures directional divergence in bottleneck space:
   $$\mathcal{L}_{\text{inv}} = 1 - \frac{1}{B}\sum_{i=1}^{B} \text{CosineSimilarity}(z_{\text{fwd}}^{(i)}, z_{\text{bwd}}^{(i)})$$
6. **Total Loss:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{inv}} \quad (\lambda = 1.5)$$

---

## 🎛️ 2. Acoustic Processing & Data Augmentation (`dataset.py`)

* **Input Specifications:** Mono WAV audio sampled at $16\text{ kHz}$. Audio clips are padded or clipped to exactly 48,000 samples (3.0 seconds).
* **Spectrogram Generation:**
  * Mel Bins: 128
  * FFT Size ($N_{\text{fft}}$): 1024
  * Hop Length: 512
  * Decibel Conversion: `torchaudio.transforms.AmplitudeToDB()`
  * Min-Max Normalization: $(x - x_{\min}) / (x_{\max} - x_{\min} + 10^{-6})$
  * Channel Expansion: 1 channel repeated across 3 channels (`spec.repeat(3, 1, 1)`).
  * Standardization: Normalized using standard ImageNet RGB statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
* **Training Augmentations:**
  * **Pitch Shifting:** Applied with a 35% probability (`p < 0.35`) via `torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=2)` (shifts pitch upward by $+2$ semitones).
  * **SpecAugment:** `FrequencyMasking(freq_mask_param=20)` and `TimeMasking(time_mask_param=30)`.

---

## 📂 3. Repository File Structure

```
.
├── extract_daic.py         # Utility to scan ~/Downloads and unpack DAIC session WAVs/transcripts
├── preprocess_daic.py      # Slices participant speech into 3s chunks; creates metadata.csv
├── dataset.py              # PyTorch Dataset, transforms, and DataLoader definitions
├── train.py                # Siamese ResNet-18 training with Cosine Invariance Loss
├── evaluate.py             # Computes segment & patient metrics; generates confusion matrix
├── explain.py              # Grad-CAM on Layer 4 and vocal energy profile for Patient 343
├── SAD_AI Registration.pdf # DTU Youth Innovation Challenge registration document
├── data/                   # (Unreferenced) TESS dataset audio clips (OAF_*_sad.wav, etc.)
└── output/                 # Artifact directory
    ├── best_model.pth      # Saved checkpoint with highest validation Macro-F1
    ├── loss_curve.png      # Training & validation loss vs. epochs
    ├── confusion_matrix.png# Confusion matrix heatmap on the test set
    └── gradcam_sample.png  # Dual Grad-CAM + vocal energy contour visualization
```

> **Note on Data Directories:** 
> * `data_daic/` (the target folder for preprocessed DAIC-WOZ audio segments and `metadata.csv`) is **not included** in this repository due to DAIC-WOZ clinical data access restrictions.
> * The `data/` directory in the repository contains ~800 files from the Toronto Emotional Speech Set (TESS) (`OAF_*_sad.wav`, `OAF_*_neutral.wav`). These files are **not utilized** by the existing codebase.

---

## ⚙️ 4. Data Partitioning & Ground Truth Labels

Preprocessed via `preprocess_daic.py`:
* **Dataset:** DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz), participants 300 to 350.
* **Speaker Isolation:** Parses participant timestamps from `{pid}_TRANSCRIPT.csv`, extracting participant intervals $\ge 0.5\text{ s}$ and dropping interviewer audio.
* **Label Mapping:** Binary depression defined by official AVEC 2016 ground truth ($\text{PHQ-8} \ge 10 \implies 1$, $\text{PHQ-8} < 10 \implies 0$).
* **Speaker-Independent Split:**
  * **Test (7 Patients):**
    * Depressed: 343 (M), 345 (M), 347 (M)
    * Normal: 344 (F), 346 (F), 348 (M), 349 (F)
  * **Validation (7 Patients):** `{335, 336, 337, 338, 339, 340, 341}`
  * **Train (Remaining Patients):** All sessions between 300 and 334.

---

## 🚀 5. Installation & Execution Guide

### Prerequisites
Install dependencies in a Python 3.10+ environment:
```bash
pip install torch torchaudio torchvision scikit-learn seaborn matplotlib pandas pytorch-grad-cam
```

### Step 1: Extract Raw DAIC Sessions
If DAIC-WOZ session archives are located in your `~/Downloads` folder:
```bash
python extract_daic.py
```
This extracts audio (`*_AUDIO.wav`) and transcripts (`*_TRANSCRIPT.csv`) into `daic_raw/`.

### Step 2: Preprocess Audio Segments
Slice participant-only speech into 3-second segments and generate the metadata index:
```bash
python preprocess_daic.py
```
*Outputs: `data_daic/segments/*.wav` and `data_daic/metadata.csv`.*

### Step 3: Train the Model
Train for 15 epochs using AdamW (`backbone lr=1e-5`, `fc lr=1e-3`), Cosine Annealing, and $\lambda_{\text{inv}}=1.5$:
```bash
python train.py
```
*Outputs: `output/best_model.pth` and `output/loss_curve.png`.*

### Step 4: Evaluate Performance
Evaluate segment-level accuracy, patient-level majority voting, and subgroup gender accuracy on the test set:
```bash
python evaluate.py
```
*Outputs: Terminal summary table and `output/confusion_matrix.png`.*

### Step 5: Generate Explainability Dashboard
Compute Grad-CAM heatmaps on Layer 4 and display vocal energy contours:
```bash
python explain.py
```
*Outputs: `output/gradcam_sample.png`.*

---

## 🔍 6. Evaluation Logic & Explainability

### Patient-Level Aggregation (`evaluate.py`)
For each test patient, segment prediction probabilities $\hat{p}$ are averaged across all 3-second interview slices:
$$\hat{y}_{\text{patient}} = \begin{cases} 1 & \text{if } \frac{1}{N}\sum_{i=1}^{N} \hat{p}_i \ge 0.50 \\ 0 & \text{otherwise} \end{cases}$$

### Clinical Explainability (`explain.py`)
1. **Grad-CAM:** Hooks into `model.backbone[-2][-1]` (the terminal `BasicBlock` of ResNet-18 Layer 4) using `pytorch_grad_cam.GradCAM`.
2. **Vocal Energy Contour:** Averages normalized Mel-spectrogram energy across frequency bins to produce a 1D temporal energy contour and calculates temporal standard deviation ($\sigma$) to measure vocal energy dynamism.

---

## 📜 7. Citation & Academic Reference

```bibtex
@article{hasan2026depression,
  title={Speech Depression Detection via Siamese Forward-Backward Spectrogram Feature Invariance},
  author={Hasan et al.},
  journal={B.Tech Capstone Project / Research Candidate},
  year={2026}
}
```
