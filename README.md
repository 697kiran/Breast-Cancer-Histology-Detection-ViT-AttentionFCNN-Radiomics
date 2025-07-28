# Breast Cancer Detection Using Histological Images: A Hybrid ViT-AttentionFCNN-Radiomics Approach

This repository presents a novel automated system for breast cancer detection from histological images, leveraging a hybrid approach that integrates radiomic features with Vision Transformers (ViTs) and an attention-based Fully Connected Neural Network (AttentionFCNN). Our method aims to enhance the accuracy, consistency, and efficiency of breast cancer diagnosis, ultimately improving patient outcomes.

## Project Description

Traditional breast cancer diagnosis depends on histopathological examination, where pathologists manually analyze tissue samples under a microscope to identify abnormalities. While effective, this process is labor-intensive, time-consuming, and susceptible to human error—challenges that grow as case volumes increase.To address these issues, we've developed an innovative automated system that integrates radiomic features with Vision Transformers, or ViTs, to classify histological images as either benign or malignant.

We trained and tested our model on the BreakHis dataset, which includes 9,109 images across four magnification levels: 40X, 100X, 200X, and 400X.Our approach achieved an outstanding accuracy of 99.57% and an AUC-ROC score of 0.99, surpassing traditional convolutional neural network (CNN) methods by 9%.This system promises to support pathologists by enhancing diagnostic accuracy, consistency, and efficiency, ultimately improving patient outcomes.

## Features

  * **Hybrid Architecture:** Fuses radiomic features (shape, intensity, and texture) with Vision Transformers and an AttentionFCNN for comprehensive image analysis.
  * **High Accuracy:** Achieves 99.57% accuracy and 0.99 AUC-ROC on the BreakHis dataset.
  * **Efficiency:** Operates 3.8 times faster than traditional CNNs like ResNet-50, supporting real-time applications.
  * **Robust Feature Selection:** Utilizes a Random Forest Classifier to select the top 10 most predictive texture features , and Principal Component Analysis (PCA) for dimensionality reduction of shape and intensity features.
  * **Advanced Preprocessing:** Includes stain normalization to standardize color distribution and image patching (to $224 \\times 224$ pixels) for optimal ViT input.
  * **Attention Mechanism:** The AttentionFCNN enhances focus on critical image regions, crucial for subtle cancer signs.
  * **Generalizability:** Designed to be adaptable for multi-class classification and other cancer datasets (e.g., lung or prostate cancer) in future work.

## Dataset

The study utilizes the **BreakHis dataset**.

  * **Total Images:** 9,109 histopathological images of breast tumors.
  * **Labels:** Labeled as either benign (2,480 images) or malignant (5,429 images), reflecting real-world prevalence.
  * **Magnification Levels:** Organized across four magnification levels—40X, 100X, 200X, and 400X—with approximately equal distribution.
  * **Data Split:** Stratified split of 75% for training (6,832 images), 10% for validation (910 images), and 15% for testing (1,367 images) to maintain class balance across all subsets.

## Methodology

### 1\. Preprocessing

  * **Stain Normalization:** Performed to standardize color distribution across all images to a target color profile of [200, 165, 215] in RGB values, addressing variations due to staining techniques or scanner settings.
  * **Image Patching:** Large images (typically $700 \\times 460$ pixels) are divided into smaller $224 \\times 224$ pixel patches, the input size required by our Vision Transformer, with some overlap to preserve context.

### 2\. Radiomic Feature Extraction

  * **Categories:** Features fall into three categories: shape (e.g., area and perimeter), intensity (e.g., mean intensity and entropy), and texture features (derived from matrices like GLCM and NGTDM).
  * **Texture Feature Selection:** A Random Forest classifier is used to rank and select the top 10 most predictive texture features, such as GLCM contrast, which highlights tissue differences.
  * **Shape and Intensity Feature Reduction:** Principal Component Analysis (PCA) reduces these features to 40 components that retain 95% of the variance.

### 3\. Model Architecture

The heart of our system is a hybrid architecture that seamlessly integrates radiomic features with a Vision Transformer and an AttentionFCNN.

  * **Vision Transformer (ViT):** We started with a pre-trained `ViT-base-patch16-224` model, which processes the 224x224 image patches and generates a 768-dimensional CLS token—a compact representation of global image features.
  * **Feature Fusion:** The 10 selected texture features are passed through a linear layer to project them into the same 768-dimensional space, aligning them with the CLS token. These two vectors are then fused via another linear layer, producing a single 768-dimensional feature vector.
  * **Concatenation:** Next, we concatenate this fused vector with the 40 PCA components from the radiomic features, creating an 808-dimensional input.
  * **AttentionFCNN:** This input feeds into a fully connected neural network (FCNN) with two hidden layers—512 neurons and 256 neurons, respectively—followed by a dropout layer with a 0.6 rate to prevent overfitting.The attention mechanism allows the model to focus on the most relevant areas.
  * **Output Layer:** Finally, the output layer produces logits for the two classes: benign and malignant.

### 4\. Experimental Setup

  * **Optimizer:** AdamW optimizer, set at a learning rate of 1e-3 and a weight decay of 0.01 to regularize the model's parameters.
  * **Loss Function:** Cross-entropy, ideal for binary classification tasks.
  * **Batch Size:** 512, balancing memory efficiency and gradient stability.
  * **Epochs:** Trained for 10 epochs—sufficient for convergence without excessive computation.
  * **Backbone Freezing:** The ViT backbone was frozen, meaning its pre-trained weights remained unchanged, and only the custom layers we added were trained.This capitalizes on the ViT's general knowledge while tailoring the model to our specific histopathology task.

## Results

Our model achieved significant performance improvements:

  * **Accuracy:** 99.57% on the test set.
  * **AUC-ROC:** 0.99 on the test set.
  * **Performance Comparison:** Dramatically outperforms traditional CNN-based methods, boosting accuracy from approximately 90% to 99.57%.
  * **Inference Speed:** 3.8 times faster than ResNet-50.
  * **Consistency Across Magnifications:** F1-scores ranged from 0.927 at 40X to 0.950 at 400X, demonstrating consistent reliability across all levels.
  * **Minimal False Negatives:** The confusion matrix for the test set revealed minimal false negatives, which is crucial in medicine to avoid missing cancer cases.

## Why Our Approach Works

Our success stems from the synergistic blending of several key elements:

  * **Radiomic Features:** Provide biologically relevant insights into tumor characteristics that deep learning alone might miss.Texture features, specifically, capture tissue heterogeneity, a hallmark of malignancy, providing extra detail beyond other radiomics features.
  * **Vision Transformers (ViTs):** Capture global spatial patterns that CNNs often miss, understanding complex tissue structures by seeing the bigger picture.
  * **AttentionFCNN:** Combines the spatial strengths of convolutional networks with attention's ability to zero in on critical image regions, vital for accurate detection where cancer signs can be subtle.
  * **Hybrid Model:** This novel integration leverages the strengths of both quantitative feature analysis and transformer-based deep learning, delivering a highly effective solution.
  * **Random Forest Classifier:** Used for its robustness and ability to handle high-dimensional data from radiomics, as well as its capacity to rank feature importance, guiding us to the best predictors.
  * **Hyperparameter Tuning:** Ensures our model performs at its peak by adjusting settings like the number of trees and maximum depth for the Random Forest, and learning rate and dropout rates for the FCNN, avoiding overfitting and maximizing accuracy on unseen data.

## Future Work

We plan to expand this model's capabilities in the following areas:

  * **Multi-class Classification:** Extend the model to distinguish between subtypes of benign and malignant tumors for more granular diagnoses.
  * **Generalizability Testing:** Evaluate the model's performance on other cancer datasets, such as lung or prostate cancer, to broaden its impact.
  * **Attention-based Interpretability:** Further illuminate the model's decision-making process to boost trust and utility in clinical settings.

### Prerequisites

  * Python 3.x
  * PyTorch
  * Scikit-learn
  * Pytorch-Lightning (or similar deep learning framework)
  * `pyradiomics` (for radiomic feature extraction)

### Installation

```bash
git clone https://github.com/yourusername/Breast-Cancer-Histology-Detection-ViT-AttentionFCNN-Radiomics.git
cd Breast-Cancer-Histology-Detection-ViT-AttentionFCNN-Radiomics
pip install -r requirements.txt
```

## Contributing

We welcome contributions\! Please feel free to open issues or submit pull requests.

## License

* MIT, Apache 2.0, etc.
