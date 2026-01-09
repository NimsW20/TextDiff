# TextDiff
# [Enhancing Label-efficient Medical Image Segmentation with Text-guided Diffusion Models (MICCAI 2024, Early Accept)](https://arxiv.org/pdf/2407.05323)

[[Paper]([[Paper](https://link.springer.com/chapter/10.1007%2F978-3-030-87231-1_30)][[Code](https://github.com/chunmeifeng/T2Net)]
)][[Code](https://github.com/chunmeifeng/TextDiff)]

TextDiff is a multi-modal framework that repurposes frozen diffusion models for medical image segmentation. Instead of training a massive U-Net from scratch (which requires thousands of labeled images), this project "wiretaps" the intermediate features of a pre-trained Denoising Diffusion Probabilistic Model (DDPM) and fuses them with clinical text embeddings to generate segmentation masks.

This approach addresses the Label Bottleneck in medical AI, achieving state-of-the-art performance with minimal supervision.

## Model Architecture

The system consists of three main modules:
![Model Architecture](images/architecture.png)
1. The Frozen Teacher (Feature Extractor)
  - Backbone: Unconditional DDPM trained on ImageNet (Architecture defined in unet.py).
  - Mechanism: The input image is corrupted with noise at three specific timesteps ($t=50, 150, 250$) using a fixed variance schedule.
  - Feature Extraction: We extract intermediate feature maps from the U-Net decoder at blocks 4, 6, 8, and 12. This captures multi-scale representations ranging from coarse semantic shapes ($32 \times 32$) to fine textures ($256 \times 256$).
    
2. The Text Encoder
  - Backbone: Clinical BioBERT (Frozen).
  - Function: Converts unstructured medical reports (e.g., "dense tumor region with irregular boundaries") into rich semantic embeddings ($768$-dimensional vectors).
    
3. The Pixel Classifier (Trainable Student)
  - Input: Multi-scale noisy image features + Text embeddings.
  - Fusion: A Cross-Modal Attention mechanism aligns image pixels with relevant text descriptions.
  - Decoding: Aggregates features via bilinear upsampling and decodes them into a binary mask using a lightweight CNN.

## Reproduction Results

I successfully reproduced the TextDiff paper results on the **MoNuSeg** dataset. The model significantly outperforms traditional multi-modal transformers like GLORIA and LViT.

| Model | Method | Dice Score (%) | IoU (%) |
| :--- | :--- | :--- | :--- |
| **LViT** | Vision Transformer | 57.95 | - |
| **GLORIA** | Contrastive Learning | 66.38 | - |
| **TextDiff (Original)** | Diffusion + Text | 78.67 | 64.98 |
| **TextDiff (Ours)** | **Reproduction** | **78.69** | **64.97** |

### 🔍 Ablation Study
To validate the architecture, I performed an ablation study removing key components.

| Variation | Description | Dice (%) | IoU (%) | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **$\zeta_1$** | No Text (Image Only) | 77.93 | 64.05 | Performance drops without semantic guidance. |
| **$\zeta_2$** | Text w/o Attention (Concat) | 76.01 | 62.30 | **Surprising Result:** Simple concatenation adds noise, hurting performance. |
| **Ours** | **Full Attention Mechanism** | **78.69** | **64.97** | Cross-Attention is critical for effective fusion. |

> **Insight:** The $\zeta_2$ experiment proves that simply "adding text" is not enough. Without the Attention mechanism to spatially align the text with specific pixels, the text embeddings act as noise rather than guidance.

---

## Novel Extensions

To further improve the model's robustness and scale-awareness, I implemented two major architectural extensions:

### 1. Hierarchical Multi-Scale Attention
* **Problem:** The original model uses a *shared* attention block for all feature scales. This forces one set of weights to understand both abstract global shapes (deep layers) and fine edges (shallow layers).
* **Solution:** I implemented **Independent Attention Modules** for each decoder layer.
* **Impact:** Allows the model to learn **scale-specific representations**, improving boundary definition for variable-sized tumors.

### 2. Contrastive Alignment Loss
* **Problem:** The alignment between image and text is only learned implicitly via the segmentation mask (Dice Loss).
* **Solution:** Added an explicit **Cosine Similarity Loss** (inspired by CLIP).
* **Equation:** $L_{total} = L_{Dice} + \lambda (1 - \text{CosineSim}(I_{global}, T_{text}))$
* **Impact:** Forces the global image embedding to mathematically resemble the text embedding in vector space, creating more robust features even before segmentation occurs.

---

## Installation & Usage

### 1. Requirements
```bash
pip install torch torchvision numpy
pip install guided-diffusion
```

## Data Preparation

1. Add the monuseg dataset from HUANGLIZI/LViT
2. Add the clinical biobert model from emilyalsentzer/Bio_ClinicalBERT
3. Add the diffusion model (Image encoder) checkpoints from openai
a pre-trained Diffusion model (trained by OpenAI) to act as the "Image Encoder."

## Training 
To train the model with both extensions enabled:
```bash
python train.py --exp ./config/monuseg.json --batch_size 4
```
> Note: Flags for extensions (use_hierarchical, use_contrastive) are set in the pixel_classifier initialization within train.py.
