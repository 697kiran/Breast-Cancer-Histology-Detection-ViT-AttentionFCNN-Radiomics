# Multimodal Breast Cancer Histopathological Classification

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Integration](#dataset-integration)
3. [System Architecture](#system-architecture)
4. [Mathematical & Theoretical Foundation](#mathematical--theoretical-foundation)
5. [Repository Structure](#repository-structure)
6. [Installation & Requirements](#installation--requirements)
7. [Usage Pipeline](#usage-pipeline)
8. [Cross-Validation Results](#cross-validation-results)
9. [Troubleshooting & Known Issues](#troubleshooting--known-issues)

---

## Project Overview

This repository provides an end-to-end multimodal deep learning framework for the binary classification of breast cancer using histopathological images. The pipeline identifies tissues as either **Benign** or **Malignant** by fusing raw pixel data with highly specific radiomic texture features.

Traditional models often rely solely on raw imagery. This framework bridges the gap between deep visual feature extraction and quantitative radiomics, utilizing a Custom Vision Transformer (ViT) paired with Sparse group-regularized Canonical Correlation Analysis (sgR-CCA).

## Dataset Integration

The system is configured to process the **BreakHis (Breast Cancer Histopathological Database)**.

* **Classes:** Benign (0), Malignant (1)
* **Magnification Levels:** 40X, 100X, 200X, 400X
* **Total Images Processed:** 7,712
* **Feature Metadata:** The pipeline strictly requires a paired CSV file (`radiomic_features_numeric.csv`) containing pre-extracted original shape, first-order, GLCM, GLDM, GLRLM, GLSZM, and NGTDM features for every image.

## System Architecture

The pipeline consists of four primary stages:

### 1. Image Preprocessing and Standardization
Histopathological slides often suffer from color inconsistencies due to different staining processes. The system utilizes **Macenko Stain Normalization** to map the color distribution of all input images to a predefined target reference. After patching the images to 224x224, a **Discrete Cosine Transform (DCT)** low-pass filter is applied to remove high-frequency noise artifacts while preserving structural cellular details.

### 2. Feature Selection
Instead of overwhelming the model with redundant data, a Random Forest algorithm analyzes the radiomics CSV. It ranks and isolates the top 10 most critical texture features (e.g., Small Area High Gray Level Emphasis, Short Run High Gray Level Emphasis) to feed into the transformer.

### 3. Custom Vision Transformer (ViT)
We modify the standard `google/vit-base-patch16-224` architecture. In addition to the standard image patch tokens, the model architecture introduces three independent, learnable positional embeddings. These represent the specific radiomic families:
* GLSZM (Gray Level Size Zone Matrix)
* GLRLM (Gray Level Run Length Matrix)
* GLDM (Gray Level Dependence Matrix)

### 4. sgR-CCA Fusion & Classification
The extracted deep visual features from the ViT and the PCA-reduced radiomic features are passed into the sgR-CCA module. This algorithm maximizes the correlation between the two data modalities, projecting them into a unified feature space. Finally, a standard Feed-Forward Neural Network (FCNN) outputs the final classification probabilities.

---

## Mathematical & Theoretical Foundation

To understand the core mechanisms, here are the mathematical principles driving the feature fusion and normalization.

### Macenko Normalization
The algorithm converts RGB images into Optical Density (OD) space to isolate the specific absorption properties of the stains (typically Hematoxylin and Eosin):

OD = -log10(I / 255)

It then calculates the singular value decomposition on the OD tuples to find the robust extremes of the stain vectors, allowing the source image to be mathematically projected onto the target image's color space.

### Canonical Correlation Analysis (CCA)
Given two centered data matrices, image features X and radiomic features Y, CCA seeks projection vectors u and v to maximize the correlation between the linear combinations Xu and Yv. The Sparse group-regularized (sgR) variant introduces L1 and L2 penalties to handle high-dimensional, grouped data, ensuring the model ignores irrelevant texture inputs without overfitting.

---

## Repository Structure

```text
├── dataset/
│   ├── 40X/
│   ├── 100X/
│   ├── 200X/
│   └── 400X/
├── metadata/
│   └── radiomic_features_numeric.csv
├── main_pipeline.py
├── requirements.txt
└── README.md
```

---

## Installation & Requirements

Ensure you have Anaconda or a standard Python 3.9+ virtual environment configured. 

1.  **Clone the repository and navigate to the directory.**
2.  **Install the exact dependencies:**

```bash
pip install torch torchvision transformers pandas numpy scikit-learn Pillow scipy opencv-python matplotlib seaborn joblib
```

> **Note on Hardware:** A CUDA-enabled GPU with at least 8GB of VRAM is highly recommended. The scripts will automatically detect and utilize CUDA if available.

---

## Usage Pipeline

Update the paths at the top of the main script to point to your local dataset directories:

```python
base_data_directory = 'path/to/dataset_cancer_v1/classificacao_binaria'
radiomics_metadata_path = 'path/to/radiomic_features_numeric.csv'
```

Execute the primary script:

```bash
python main_pipeline.py
```

The script will inherently run a **5-Fold Stratified Cross-Validation**. During execution, it will generate and save the following artifacts to your working directory:
* Saved PyTorch model weights per fold (`model_foldX.pth`)
* Joblib dumps for scalers and PCA transforms
* Performance plots (Loss curves, Confusion Matrices, ROC, and Precision-Recall charts)

---

## Cross-Validation Results

The architecture achieves highly stable performance across all folds, demonstrating its robustness against overfitting.

* **Average Accuracy:** 96.2% ± 0.7%
* **Average AUC-ROC:** 0.984 ± 0.008

---

## Troubleshooting & Known Issues

### OOM Kernel Crash at Pipeline Completion
**Symptom:** The script runs perfectly through all 5 folds but the kernel crashes entirely when attempting to generate the final "Aggregate Final Plots."

**Cause:** This is a memory exhaustion issue. By appending all predictions, probabilities, and labels across all folds into large Python lists, and subsequently using `matplotlib` to render multi-layer charts without flushing the RAM, the system runs out of memory.

**Solution:** Modify the final plotting block to aggressively manage memory. Clear the figure canvases and run garbage collection. Here is how you should update the bottom of your script to prevent this:

```python
import gc

# Summarize CV results
accs = [metric[0] for metric in fold_results]
aucs = [metric[1] for metric in fold_results]

print("\n===== Cross-validation results =====")
print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"AUC:      {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# Extract data specifically from the final fold iteration
last_fold_acc, last_fold_auc, final_probs, final_preds, final_labels = fold_results[-1]

def generate_and_save_plot(plot_function, filename):
    plt.figure(figsize=(6, 5))
    plot_function()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()     # Clear the current figure
    plt.close()   # Close the window to free memory
    gc.collect()  # Force garbage collection

# 1. Confusion Matrix
def plot_cm():
    cm_matrix = confusion_matrix(final_labels, final_preds)
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Final Confusion Matrix (Last Fold)')

generate_and_save_plot(plot_cm, 'final_confusion_matrix.png')

# 2. ROC Curve
def plot_roc():
    false_pos_rate, true_pos_rate, _ = roc_curve(final_labels, final_probs)
    plt.plot(false_pos_rate, true_pos_rate, label=f'ROC Curve (AUC = {last_fold_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Final ROC Curve (Last Fold)')
    plt.legend()
    plt.grid(True)

generate_and_save_plot(plot_roc, 'final_roc_curve.png')

print("All charts generated and memory cleared successfully.")
```
