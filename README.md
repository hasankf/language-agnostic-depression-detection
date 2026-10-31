# Language-Invariant Speech Depression Detection via Forward-Backward Siamese Feature Invariance

> **B.Tech Final-Year Project & Research Paper Candidate**  
> *Enforcing True Acoustic Invariance, Resolving Gender Pitch Bias, and Eliminating Overfitting on DAIC-WOZ*

---

## 📌 1. Executive Summary & The Research Gap

Automated detection of Major Depressive Disorder (MDD) from speech is often hindered by **linguistic dependency**, **data scarcity**, and **gender pitch bias**. Previous state-of-the-art approaches—such as *Kim et al. (ICASSP 2026)*—attempt to make models language-independent using **passive audio augmentation** (e.g., appending reversed audio clips to the training set) or prompt-driven reprogramming on large acoustic transformers (AST, Whisper).

### 🔍 The Core Gap Identified in Literature
1. **Flaw of Passive Augmentation:** Simply feeding reversed audio as standard training samples **does not mathematically force feature invariance**. Deep neural networks still learn to memorize forward English phonetic sequences while treating reversed clips as separate noisy data.
2. **Gender Pitch Bias & Frequency Shortcuts:** Standard 2D Convolutional Networks (like ResNet18) easily lock onto absolute fundamental frequencies ($F_0$). Male voices ($85{-}180\text{ Hz}$) and female voices ($165{-}255\text{ Hz}$) occupy completely different vertical Mel-spectrogram bins. Models often over-adapt to female formants, causing male test accuracy to collapse to near-random chance ($\sim 47{-}50\%$).
3. **Feature Vector Scale Collapse (The MSE Illusion):** Minimizing Mean Squared Error ($\\text{MSE}$) between forward and backward feature vectors allows optimizers to "cheat" by shrinking vector scales toward zero ($\\|z\\| \\to 0$) rather than aligning feature directions.

### 💡 Our Proposed Solution
We implement an **explicit Forward-Backward Siamese Feature Invariance Constraint**:
- **Explicit Cosine Invariance Loss ($\\lambda = 1.5$):** Forces the directional Cosine Similarity between forward speech $z_{\\text{fwd}}$ and time-reversed speech $z_{\\text{bwd}}$ to approach $1.0$ ($\\cos \\theta \\to 1.0$).
- **Random Pitch-Shifting Augmentation ($\\pm 2$ semitones):** Destroys absolute $F_0$ pitch shortcuts, forcing early convolutional filters to learn pitch-invariant acoustic dynamics.
- **Layer-4 Fine-Tuned ResNet18 Architecture:** Freezes Conv1 through Layer 3 to retain low-level acoustic edge filters while fine-tuning Layer 4 with differential learning rates ($1\\times 10^{-5}$ backbone, $1\\times 10^{-3}$ head).

---

## 🏗️ 2. System Architecture & Methodology

```
                       ┌───────────────────────────────┐
                       │   3.0s Audio Spectrogram (x)  │
                       └───────────────┬───────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             [ Forward Pass: x ]              [ Time-Reversed: x_bwd ]
                      │                                 │
                      ▼                                 ▼
         ┌────────────────────────┐        ┌────────────────────────┐
         │ LanguageInvariantResNet│        │ LanguageInvariantResNet│
         │  (Layer 4 Fine-Tuned)  │        │  (Shared Weights)      │
         └────────────┬───────────┘        └────────────┬───────────┘
                      │                                 │
                      ▼                                 ▼
             Embedding: z_fwd                  Embedding: z_bwd
                      │                                 │
                      ├─────────────────────────────────┘
                      │   Cosine Invariance Loss:
                      │   L_inv = 1 - CosSim(z_fwd, z_bwd)
                      ▼
             [ Linear Classifier ]
                      │
                      ▼
            Output: Normal / Depressed
```

### 1. Data Pipeline & Preprocessing (`dataset.py`)
- **Dataset:** DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz).
- **Segmentation:** Audio interviews are windowed into non-overlapping **3.0-second participant-only clips** using transcript timestamps (interviewer voice and silence removed).
- **Spectrogram Transformation:** Log-Mel Spectrograms ($128$ Mel bins, $N_{\\text{fft}} = 1024$, hop length $= 512$).
- **Augmentations (Training Only):**
  - **ImageNet Feature Standardization:** $(\\mathbf{x} - \\mu) / \\sigma$ using standard ImageNet parameters ($\\mu=[0.485, 0.456, 0.406], \\sigma=[0.229, 0.224, 0.225]$).
  - **SpecAugment:** Frequency masking ($F=20$) and Time masking ($T=30$).
  - **Dynamic Pitch Shifting:** Randomly applied to $35\%$ of training samples ($\\pm 2$ semitones) via `torchaudio.transforms.PitchShift`.

### 2. Loss Formulation (`train.py`)
The model is trained end-to-end using a dual-objective loss function:

$$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{CE}}(y, \\hat{y}) + \\lambda \\cdot \\mathcal{L}_{\\text{inv}}$$

Where:
- $\\mathcal{L}_{\\text{CE}}$ is **Weighted Cross-Entropy Loss** with class weights $[1.0, 2.2]$ to counteract class imbalance (77% Normal / 23% Depressed).
- $\\mathcal{L}_{\\text{inv}} = 1 - \\text{CosineSimilarity}(z_{\\text{fwd}}, z_{\\text{bwd}})$ penalizes directional divergence in the $512$-dimensional bottleneck embedding space.
- $\\lambda = 1.5$ heavily enforces linguistic symmetry.

---

## 📊 3. Experimental Results & Improvements

### Segment-Level Performance (952 Test Clips)

| Metric | Baseline (Unregularized) | Final Proposed Pipeline | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Segment Accuracy** | 62.00% | **65.86% (~66%)** | **+3.86%** |
| **Segment Macro F1** | 0.59 | **0.62** | **+0.03** |
| **Depressed Class F1** | 0.48 | **0.51** | **+0.03** |
| **Male Test Accuracy** | 47.58% | **53.76%** | **+6.18%** |
| **Female Test Accuracy** | 71.38% | **73.62%** | **+2.24%** |
| **Invariance Distance ($1 - \\cos \\theta$)** | ~0.624 (MSE trick) | **0.1744 (82.56% similarity)** | **Massive Alignment** |

### Confusion Matrix Breakdown (Final Run)

```
                       Predicted Normal    Predicted Depressed
Actual Normal               460                  216
Actual Depressed            109                  167
```
- **True Negatives ($TN$):** 460 clips correctly identified as Normal.
- **True Positives ($TP$):** 167 clips correctly identified as Depressed (60.5% segment recall).

---

## 🔬 4. Clinical Explainability & Visual Dashboards

Running `python3 explain.py` generates a dual clinical explanation dashboard (`output/gradcam_sample.png`):

1. **Grad-CAM Acoustic Focus (Left Panel):**
   - Highlights feature importance in `Layer 4`.
   - **Key Finding:** The heatmap isolates **Mel Frequency Bins 40–80 (2.5 kHz – 5.0 kHz)**, corresponding to upper vocal tract formants ($F_3, F_4$) and prosodic energy dynamics, rather than lower phonetic frequencies.
2. **Clinical Biomarker Diagnostics (Right Panel):**
   - Extracts the **Vocal Energy Contour** across 3.0-second time frames.
   - **Depression Biomarker:** Depressed patients exhibit low vocal energy dynamism ($\\sigma = 0.0712$), visually demonstrating monotone speech patterns associated with psychomotor retardation.

---

## 📁 5. Project Structure

```
depressionDetection/
├── data_daic/              # Processed DAIC-WOZ dataset segments & metadata
│   ├── metadata.csv
│   └── segments/
├── output/                 # Model checkpoints and visual plots
│   ├── best_model.pth
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── gradcam_sample.png
├── preprocess_daic.py      # Audio slicing & speaker-centric timestamp extraction
├── dataset.py              # PyTorch Dataset, SpecAugment, PitchShift & DataLoader
├── train.py                # Siamese ResNet18 training with Cosine Invariance Loss
├── evaluate.py             # Patient-level & subgroup evaluation script
└── explain.py              # Grad-CAM & vocal energy contour dashboard generator
```

---

## 🚀 6. Quickstart & Execution Guide

### Prerequisites
```bash
pip install torch torchaudio torchvision scikit-learn seaborn matplotlib pandas pytorch-grad-cam
```

### Execution Pipeline

1. **Preprocess Audio Data:**
   ```bash
   python3 preprocess_daic.py
   ```
2. **Train Model (15 Epochs with Cosine Invariance Loss):**
   ```bash
   python3 train.py
   ```
3. **Evaluate Performance (Segment & Patient Level):**
   ```bash
   python3 evaluate.py
   ```
4. **Generate Clinical Explainability Report:**
   ```bash
   python3 explain.py
   ```

---

## 📜 Citation & Research Reference

If utilizing this codebase or architecture for research, please cite:

```bibtex
@article{hasan2026depression,
  title={Language-Invariant Speech Depression Detection via Forward-Backward Siamese Feature Invariance},
  author={Hasan et al.},
  journal={B.Tech Capstone Project / IEEE Manuscript Candidate},
  year={2026}
}
```
