# Multimodal Parkinson’s Disease Detection Using Video Swin, AST, Cross-Attention and XGBoost

## Overview

This repository contains **Module 2** of a multimodal deep-learning framework for Parkinson’s Disease (PD) detection from video and audio information.

The objective of this module is to investigate whether complementary information contained in **visual movement/video representations** and **speech/audio representations** can be effectively combined for automated Parkinson’s Disease classification.

The proposed framework uses:

* **Video Swin 3D-Tiny** for spatiotemporal video representation learning
* **Audio Spectrogram Transformer (AST)** for audio representation learning
* **Bidirectional Cross-Attention** for multimodal interaction and feature fusion
* **Softmax classification** as the conventional baseline
* **XGBoost classification** as the optimized downstream classifier
* **Optuna** for hyperparameter optimization
* **SHAP / TreeSHAP** for model explainability
* Statistical analysis and ablation experiments for evaluating robustness and component contributions

A key methodological decision in this module is that **only the YouTube dataset is used for model development**. The Turning dataset is maintained separately and is not incorporated into model training, validation, hyperparameter optimization, or model selection.

---

# Research Objective

Parkinson's Disease can manifest through multiple observable characteristics, including changes in movement, speech, articulation, rhythm, and motor coordination.

A single modality may therefore provide incomplete information.

This module investigates a multimodal learning strategy in which:

```text
Video
  │
  ▼
Video Swin 3D-Tiny
  │
  ▼
Visual Spatiotemporal Tokens
  │
  ├───────────────┐
  │               │
  │               ▼
  │        Bidirectional
  │        Cross-Attention
  │               ▲
  │               │
  ▼               │
Audio → AST ──────┘
  │
  ▼
Audio Tokens
  │
  ▼
Fused Representation
  │
  ├──────────────► Softmax Classifier
  │
  └──────────────► XGBoost Classifier
```

The central research question is:

> **Can jointly learned visual and audio representations, combined through cross-modal attention, provide a stronger representation for Parkinson’s Disease classification than conventional single-modality or simple feature-concatenation approaches?**

---

# Dataset Description

## Raw Dataset Organization

The raw data are organized under:

```text
ParkinsonDataset/
│
├── Turning dataset/
│
└── Youtube dataset/
    ├── Negative/
    └── Positive/
```

The project contains two distinct video sources:

1. **YouTube Dataset**
2. **Turning Dataset**

Although both are related to Parkinson's Disease assessment, they have fundamentally different characteristics and are therefore treated differently in the experimental methodology.

---

# 1. YouTube Dataset

The YouTube dataset is the **primary dataset for Module 2 model development**.

It contains videos collected from YouTube and organized into two diagnostic classes:

```text
Youtube dataset/
├── Negative/
└── Positive/
```

The dataset contains both:

* Parkinson's Disease positive videos
* Control / negative videos

Unlike the Turning dataset, the YouTube videos contain **speech-oriented content**, making them suitable for multimodal video + audio learning.

The YouTube dataset contains:

* **117 videos in total**
* Positive class: **66 videos**
* Negative class: **51 videos**

All videos used in the final preprocessing pipeline were verified for readability and audio availability.

### Why YouTube is used for model development

The multimodal architecture requires both:

* visual information
* audio information

The YouTube videos provide both modalities in a common video sample.

Therefore, the complete multimodal pipeline can be trained using:

```text
YouTube Video
      │
      ├── Video frames
      │
      └── Audio signal
```

This enables joint learning between the visual and audio branches.

---

# 2. Turning Dataset

The Turning dataset consists of Parkinson's Disease-related turning/movement videos.

The important characteristic of this dataset is that it represents a **different type of clinical behavior** from the YouTube videos.

The Turning videos primarily contain **movement information** and do not provide the same usable speech/audio modality required by the multimodal architecture.

In addition, the Turning dataset does not provide an appropriate collection of true negative samples for conventional supervised training.

Therefore, the Turning dataset is **not mixed with the YouTube dataset for model development**.

It is maintained separately as an **external validation dataset**.

The experimental methodology follows:

```text
                    MODEL DEVELOPMENT
                           │
                           ▼
                    YouTube Dataset
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
            Train          Val         Test
              │            │            │
              └────────────┴────────────┘
                           │
                           ▼
                    Final Model
                           │
                           ▼
                 External Validation
                           │
                           ▼
                   Turning Dataset
```

This separation prevents the Turning dataset from influencing:

* model training
* hyperparameter optimization
* validation-based checkpoint selection
* ablation decisions
* model selection

This provides a more independent evaluation setting.

---

# Dataset Usage Strategy

| Dataset | Training | Validation | Testing | Hyperparameter Optimization | External Validation |
| ------- | -------: | ---------: | ------: | --------------------------: | ------------------: |
| YouTube |        ✓ |          ✓ |       ✓ |                           ✓ |                   — |
| Turning |        — |          — |       — |                           — |                   ✓ |

The two datasets are therefore **not simply merged into one training dataset**.

This separation is particularly important because the datasets differ in:

* video content
* recording conditions
* task characteristics
* modality availability
* subject/background characteristics
* clinical behavior being represented

Using them independently helps reduce the possibility that the model learns dataset-specific characteristics instead of disease-related patterns.

---

# Data Augmentation

The original Turning dataset did not contain an adequate set of negative/control turning samples.

Because artificially assigning negative labels to transformed positive samples would not create genuine negative clinical examples, such samples are **not treated as valid biological negative cases** for the multimodal training framework.

The YouTube dataset therefore provides the positive and negative classes required for supervised model development.

Augmentation was explored/generated in the broader dataset-processing workflow to address data limitations, particularly the lack of negative turning examples.

However, the generated Turning samples are **not used as genuine negative training labels in Module 2**.

The final Module 2 development strategy remains centered on the original labeled YouTube dataset.

---

# Dataset Splitting

The YouTube dataset is divided into independent:

* Training set
* Validation set
* Test set

The current split organization is:

```text
Module 2/
└── 02_splits/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

The split contains:

```text
Training:
81 videos
46 Positive
35 Negative

Validation:
18 videos
10 Positive
8 Negative

Testing:
18 videos
10 Positive
8 Negative
```

The validation set is used during model development for model selection and hyperparameter-related decisions.

The test set is kept separate for final YouTube evaluation.

---

# Data Metadata

Dataset metadata are maintained in:

```text
Module 2/
└── 00_dataset_analysis/
    └── youtube_complete_metadata.csv
```

The metadata include information such as:

* Video identifier
* Filename
* Path
* Class label
* Dataset source
* Video readability
* FPS
* Frame count
* Duration
* Width
* Height
* Audio availability
* Audio codec
* Audio sample rate
* Audio channels
* Audio duration

This metadata layer provides a reproducible record of the raw dataset before feature extraction.

---

# Preprocessing

The multimodal preprocessing pipeline converts each YouTube video into two synchronized feature sources:

```text
Raw YouTube Video
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
Video Processing           Audio Processing
       │                         │
       ▼                         ▼
32 RGB Frames              Audio Waveform
224 × 224                  16 kHz
       │                         │
       ▼                         ▼
Video Tensor               Mel Spectrogram
       │                         │
       ▼                         ▼
Video Swin                  AST
       │                         │
       └──────────┬──────────────┘
                  ▼
          Cross-Attention
                  │
                  ▼
         Fused Representation
```

---

# Video Preprocessing

Each video is converted into a fixed-length sequence of:

```text
32 frames
```

Each frame is resized to:

```text
224 × 224 pixels
```

The resulting video tensor follows the structure:

```text
[B, 3, 32, 224, 224]
```

where:

* `B` = batch size
* `3` = RGB channels
* `32` = temporal frames
* `224 × 224` = spatial resolution

ImageNet normalization is applied before the video is passed to Video Swin.

The fixed temporal sampling provides a consistent input format for the 3D video transformer.

---

# Audio Preprocessing

The audio stream is converted into a standardized waveform representation.

The preprocessing configuration includes:

```text
Sample rate:       16,000 Hz
Channels:          Mono
Duration:          10 seconds
Samples:           160,000
Mel bins:          128
FFT size:          1024
Hop length:        512
```

The processed audio is transformed into a log-mel spectrogram representation suitable for the Audio Spectrogram Transformer.

The resulting audio representation is processed independently from the visual branch before multimodal fusion.

---

# Preprocessing Configuration

The reproducible preprocessing configuration is stored in:

```text
Module 2/
└── 01_preprocessing/
    └── preprocessing_config.json
```

This separates preprocessing decisions from model implementation and makes it possible to reproduce the feature-extraction pipeline.

---

# Feature Extraction

A major component of the pipeline is the conversion of raw modalities into high-dimensional transformer representations.

Instead of directly feeding raw frames and raw audio into the final classifier, each modality is passed through a specialized pretrained transformer.

```text
Video Frames
     │
     ▼
Video Swin 3D-Tiny
     │
     ▼
Video Tokens
     │
     │
     │          Audio
     │            │
     │            ▼
     │           AST
     │            │
     │            ▼
     │       Audio Tokens
     │            │
     └──────┬─────┘
            ▼
     Cross-Attention
            │
            ▼
    Fused Representation
```

The extracted representations are cached to disk.

This allows later stages such as:

* statistical analysis
* ablation experiments
* SHAP analysis
* classifier comparison

to reuse the extracted features without repeatedly executing the computationally expensive Video Swin and AST encoders.

---

# Cached Feature Representation

The cached features are stored under:

```text
Module 2/
└── 04_step2_optuna_optimized/
    └── cached_features/
        ├── train_features.pt
        ├── validation_features.pt
        └── test_features.pt
```

Each feature file contains:

```text
video_tokens
audio_tokens
labels
```

The extracted token dimensions are approximately:

```text
Video:
[B, 784, 768]

Audio:
[B, 1214, 768]
```

The token dimensions preserve the transformer-level representation before multimodal fusion.

---

# Model Architecture

## Complete Architecture

The proposed multimodal architecture is:

```text
                         INPUT VIDEO
                              │
              ┌───────────────┴────────────────┐
              │                                │
              ▼                                ▼
       Video Frames                         Audio
              │                                │
              ▼                                ▼
    Video Swin 3D-Tiny               Mel Spectrogram
              │                                │
              ▼                                ▼
      Video Tokens                            AST
       784 × 768                               │
              │                                ▼
              │                         Audio Tokens
              │                          1214 × 768
              │                                │
              └──────────────┬─────────────────┘
                             ▼
                Bidirectional Cross-Attention
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
        Video-aware Audio           Audio-aware Video
          Representation              Representation
               │                           │
               └─────────────┬─────────────┘
                             ▼
                       Mean Pooling
                             │
                             ▼
                    Multimodal Fusion
                             │
                    768-dimensional
                     representation
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
        Softmax Classifier             XGBoost
        Baseline Model             Optimized Model
```

---

# Video Branch — Video Swin 3D-Tiny

The visual branch uses **Video Swin 3D-Tiny**.

Video Swin extends the transformer architecture to spatiotemporal video data.

Instead of treating every frame independently, the model learns relationships across:

* spatial regions
* temporal locations
* neighboring video patches

This is useful for movement-related information because Parkinson's Disease manifestations may be expressed through temporal patterns rather than isolated frames.

The video encoder produces:

```text
Video tokens:
784 × 768
```

These tokens preserve information learned from the spatiotemporal video sequence.

---

# Audio Branch — Audio Spectrogram Transformer

The audio branch uses the **Audio Spectrogram Transformer (AST)**.

The audio waveform is transformed into a spectrogram representation and processed using a transformer architecture.

AST enables the model to learn high-level patterns from the audio signal rather than relying solely on manually engineered acoustic features.

The audio encoder produces:

```text
Audio tokens:
1214 × 768
```

The resulting representation can capture acoustic information present in the speech-oriented YouTube videos.

---

# Why Separate Video and Audio Encoders?

Video and audio are fundamentally different data types.

A single encoder is therefore not directly appropriate for both modalities.

The architecture first learns modality-specific representations:

```text
Video → Video Swin → Visual Representation

Audio → AST → Acoustic Representation
```

The two representations are only combined after modality-specific encoding.

This allows each encoder to specialize in its own input modality before multimodal interaction occurs.

---

# Bidirectional Cross-Attention

Simple concatenation treats two modalities as independent feature vectors.

The proposed architecture instead uses **bidirectional cross-attention**.

The fusion module projects the 768-dimensional modality representations into a shared fusion space.

Current configuration:

```text
Input dimension:       768
Fusion dimension:     384
Attention heads:        4
Dropout:              0.4
```

Two attention directions are performed:

```text
Video → Audio
Audio → Video
```

Conceptually:

```text
Video Tokens ────────────────┐
                             │
                             ▼
                      Video-to-Audio
                       Cross-Attention
                             │
                             ▼
                      Audio-aware Video


Audio Tokens ────────────────┐
                             │
                             ▼
                      Audio-to-Video
                       Cross-Attention
                             │
                             ▼
                      Video-aware Audio
```

This enables information from one modality to influence the representation of the other modality.

---

# Fusion Process

After bidirectional attention, the resulting representations are refined through:

* residual connections
* layer normalization
* feed-forward transformations
* mean pooling

The pooled representations are concatenated:

```text
Video representation
        │
        ├──── 384 dimensions
        │
        ▼
     CONCAT
        ▲
        │
        ├──── 384 dimensions
        │
Audio representation
```

producing:

```text
384 + 384 = 768 dimensions
```

The final fused representation is therefore:

```text
768-dimensional multimodal feature vector
```

This vector becomes the input to the downstream classifier.

---

# Baseline Model — Softmax

The first classifier uses a conventional neural classification layer.

```text
Cross-Attention Fusion
          │
          ▼
   768-dimensional
    representation
          │
          ▼
     Linear Layer
          │
          ▼
      Softmax
          │
          ▼
   Control / PD
```

This configuration acts as the **unoptimized neural baseline**.

The purpose of the baseline is to establish the performance of the multimodal architecture before replacing the final classifier with an optimized tree-based model.

---

# Optimized Model — XGBoost

The optimized framework replaces the final Softmax classifier with **XGBoost**.

```text
Video Swin
     │
     ▼
Video Tokens
     │
     │
     ├──── Cross-Attention ────┐
     │                         │
     │                         ▼
     │                  768-D Fusion
     │                         │
     │                         ▼
     │                       XGBoost
     │                         │
     │                         ▼
     │                   Control / PD
     │
AST
 │
 ▼
Audio Tokens
```

The deep learning encoders and cross-attention module generate the multimodal representation.

XGBoost then operates on this learned representation as the final classifier.

---

# Why XGBoost?

The purpose of using XGBoost is to investigate whether a strong nonlinear tabular classifier can exploit the learned multimodal representation more effectively than a simple Softmax classification layer.

The deep architecture performs:

```text
Representation Learning
```

while XGBoost performs:

```text
Nonlinear Classification
```

This creates a hybrid architecture:

```text
Deep Multimodal Feature Learning
              +
      Gradient-Boosted Trees
```

---

# Hyperparameter Optimization

Optuna is used for automated hyperparameter optimization.

The optimization process explores parameters associated with both:

### Cross-Attention Fusion

* Fusion dimensionality
* Number of attention heads
* Dropout
* Learning rate
* Weight decay
* Label smoothing

### XGBoost

* Maximum tree depth
* Learning rate
* Number of estimators
* Minimum child weight
* Subsampling
* Feature subsampling
* L1 regularization
* L2 regularization

The optimized configuration is selected using the validation data rather than the test set.

The optimization workflow is:

```text
Training Data
     │
     ▼
Candidate Configuration
     │
     ▼
Train Fusion Model
     │
     ▼
Validation Evaluation
     │
     ▼
Optuna Trial
     │
     ├──── Better → Keep
     │
     └──── Worse → Reject
     │
     ▼
Best Configuration
     │
     ▼
Final Optimized Model
```

The test set is reserved for final evaluation.

---

# Step 2 Experimental Workflow

The optimized pipeline is organized into the following stages:

```text
Cached Video Tokens
        +
Cached Audio Tokens
        │
        ▼
Cross-Attention Fusion
        │
        ▼
Fused Representation
        │
        ├───────────────┐
        │               │
        ▼               ▼
     Softmax          XGBoost
     Baseline        Optimized
        │               │
        └───────┬───────┘
                ▼
       Comparative Evaluation
```

The use of cached representations makes subsequent experiments substantially more efficient.

---

# Evaluation Metrics

The framework evaluates the classifiers using multiple complementary metrics.

## Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

---

## Precision

Measures how many samples predicted as Parkinson's Disease are actually positive.

```text
Precision =
TP / (TP + FP)
```

---

## Recall / Sensitivity

Measures how many actual Parkinson's Disease cases are correctly detected.

```text
Recall =
TP / (TP + FN)
```

Recall is particularly important in disease-screening scenarios because false negatives represent missed disease cases.

---

## F1 Score

The harmonic mean of precision and recall.

```text
F1 =
2 × Precision × Recall
/
(Precision + Recall)
```

---

## False Positive Rate

Measures the proportion of control samples incorrectly classified as Parkinson's Disease.

```text
FPR =
FP / (FP + TN)
```

---

## AUROC

The Area Under the Receiver Operating Characteristic curve measures discrimination across classification thresholds.

The ROC curve examines:

```text
True Positive Rate
        vs.
False Positive Rate
```

AUROC is reported alongside threshold-dependent metrics to provide a broader view of classifier discrimination.

---

## Loss

Classification loss is recorded for the applicable neural training/evaluation stages.

Loss is monitored during model training and validation to identify convergence and potential overfitting.

---

## Computation Time

The framework also records computation time for:

* training
* validation
* testing

This provides information about computational efficiency in addition to predictive performance.

---

# Training Monitoring

Training is monitored using both training and validation metrics.

A typical training workflow is:

```text
Epoch
 │
 ├── Training
 │     ├── Loss
 │     ├── Accuracy
 │     └── AUROC
 │
 └── Validation
       ├── Loss
       ├── Accuracy
       └── AUROC
              │
              ▼
       Best Model Selection
```

Validation AUROC is used during the relevant model-selection process.

Training and validation curves can also be inspected for evidence of overfitting.

---

# Statistical Analysis

A separate statistical-analysis stage is included to determine whether differences between optimized and unoptimized models are consistent across repeated experimental runs.

A single train/test run cannot adequately establish statistical robustness.

Therefore, the statistical analysis uses multiple independent runs with controlled random seeds.

The general design is:

```text
Seed 42 ──┐
Seed 43 ──┤
Seed 44 ──┤──► Optimized Model
Seed 45 ──┤
Seed 46 ──┘

Seed 42 ──┐
Seed 43 ──┤
Seed 44 ──┤──► Unoptimized Model
Seed 45 ──┤
Seed 46 ──┘
```

The principal summary statistics include:

* Mean Accuracy
* Standard Deviation of Accuracy
* Mean AUROC
* Standard Deviation of AUROC

The analysis also examines:

* Precision
* Recall
* F1
* FPR
* confidence intervals
* paired statistical comparisons

---

# Paired Statistical Testing

The optimized and unoptimized models are evaluated using matched random seeds.

A paired Student's t-test is used to compare the corresponding performance values.

For example:

```text
Optimized Seed 42
        vs.
Unoptimized Seed 42

Optimized Seed 43
        vs.
Unoptimized Seed 43

...
```

This pairing controls for variation associated with the experimental seed.

The significance level is:

```text
α = 0.05
```

The statistical analysis is intended to determine whether observed performance differences are likely to represent consistent differences across repeated experiments rather than variation from a single run.

---

# Ablation Study

An ablation study evaluates the contribution of the principal components of the multimodal architecture.

The study includes five configurations.

---

## A1 — Without Video

Only the audio branch is retained.

```text
Audio
  │
  ▼
AST
  │
  ▼
Audio Representation
  │
  ▼
XGBoost
```

This evaluates the contribution of audio information independently.

Because only one modality is available, cross-modal attention is not applicable.

---

## A2 — Without Audio

Only the visual branch is retained.

```text
Video
  │
  ▼
Video Swin
  │
  ▼
Video Representation
  │
  ▼
XGBoost
```

This evaluates the contribution of visual information independently.

Again, cross-modal attention is not applicable because there is no second modality.

---

## A3 — Without Cross-Attention

Both modalities are retained, but the cross-attention mechanism is removed.

```text
Video → Video Swin ──┐
                     ├──► Concatenation ──► XGBoost
Audio → AST ─────────┘
```

This evaluates whether explicit cross-modal interaction provides an advantage over simple feature concatenation.

---

## A4 — Complete Hybrid + Softmax

This represents the complete multimodal architecture using the conventional classifier.

```text
Video → Video Swin ──┐
                     ├──► Cross-Attention
Audio → AST ─────────┘
                            │
                            ▼
                         Softmax
```

---

## A5 — Complete Hybrid + XGBoost

This represents the complete optimized architecture.

```text
Video → Video Swin ──┐
                     ├──► Cross-Attention
Audio → AST ─────────┘
                            │
                            ▼
                      Fused Features
                            │
                            ▼
                         XGBoost
```

All ablation configurations are evaluated using the same core evaluation metrics:

* Accuracy
* Precision
* Recall
* F1
* FPR
* AUROC
* Loss
* Computation time

This allows the contribution of each architectural component to be examined systematically.

---

# Explainability — SHAP

Model explainability is performed on the final optimized XGBoost classifier.

The objective is to understand:

1. Which fused features contribute most to classification?
2. Which features push predictions toward Parkinson's Disease?
3. Which features push predictions toward the control class?
4. How are contributions distributed across the video and audio portions of the fused representation?
5. Why were representative samples classified as PD or control?

---

# TreeSHAP

Because the final classifier is XGBoost, **TreeSHAP** is used to calculate SHAP values.

The input to TreeSHAP is the final fused representation:

```text
Video Tokens
      +
Audio Tokens
      │
      ▼
Bidirectional Cross-Attention
      │
      ▼
768-dimensional fused representation
      │
      ▼
XGBoost
      │
      ▼
TreeSHAP
```

The SHAP analysis therefore explains the behavior of the **final XGBoost classifier over the learned fused representation**.

---

# SHAP Feature Space

The fused representation contains 768 features.

The feature naming convention separates the two halves of the representation:

```text
Video_Swin_Fused_001
Video_Swin_Fused_002
...
Video_Swin_Fused_384

AST_Fused_001
AST_Fused_002
...
AST_Fused_384
```

These names identify the origin of each portion of the fused representation.

They should not be interpreted as directly corresponding to individual pixels, video frames, words, or audio frequencies.

---

# Global SHAP Analysis

Global SHAP analysis identifies the features with the greatest overall contribution to the classifier.

Outputs include:

* Global SHAP feature importance
* SHAP bar summary
* SHAP beeswarm plot
* Feature direction analysis
* Feature ranking tables

The global importance calculation is based on the magnitude of SHAP values.

A high absolute SHAP value indicates that a feature has a strong influence on the model prediction.

---

# SHAP Direction Analysis

SHAP values also provide information about the direction of a feature's contribution.

Conceptually:

```text
Positive SHAP
      │
      ▼
Pushes prediction toward PD

Negative SHAP
      │
      ▼
Pushes prediction away from PD
```

This makes SHAP more informative than feature importance alone.

Feature importance answers:

> Which features matter?

SHAP direction additionally helps answer:

> In which direction does the feature influence the prediction?

---

# SHAP Beeswarm Plot

The beeswarm visualization combines:

* feature importance
* individual sample contributions
* feature value information
* direction of influence

It provides a global view of how the most influential fused features behave across the test samples.

---

# Local SHAP Explanations

Local explanations are generated for representative samples, including:

* correctly classified Parkinson's Disease samples
* correctly classified control samples
* selected misclassified samples

For each sample, SHAP values show which features contributed to the individual prediction.

The analysis can be represented as:

```text
Sample
  │
  ▼
Fused Representation
  │
  ▼
XGBoost
  │
  ▼
Prediction
  │
  ▼
SHAP Explanation
  │
  ├── Features supporting PD
  │
  └── Features opposing PD
```

---

# SHAP Force Plots

Interactive force plots are generated for representative samples.

These plots visualize the local contributions that move the prediction away from the model's baseline expectation toward the final prediction.

They are useful for examining individual predictions rather than overall model behavior.

---

# Modality-Attributed SHAP

The fused feature representation contains two conceptual groups:

```text
384 Video-related fused features
+
384 Audio-related fused features
```

SHAP magnitudes can therefore be aggregated separately to estimate the relative contribution of the two modality-associated feature groups.

The resulting analysis can report:

```text
Video-attributed SHAP contribution
Audio-attributed SHAP contribution
```

This should be interpreted as **modality-attributed contribution within the fused representation**, rather than as a causal measurement of modality importance.

Because cross-attention allows the modalities to interact, the final fused representation is not a collection of completely independent video and audio features.

---

# Feature-Space Video Visualization

The SHAP pipeline can also generate feature-space visualizations for the video branch.

These visualizations help inspect how the video representation changes across the spatial-temporal token structure.

However, the TreeSHAP values of the final XGBoost classifier operate on the **768-dimensional fused representation**.

Therefore, they do not provide a direct pixel-level explanation.

The feature-space visualization should not be described as a direct clinical pixel-level heatmap unless an explicit mapping method is implemented between the fused features and original video pixels/frames.

---

# Explainability Pipeline

The complete explainability workflow is:

```text
Cached Test Video Tokens
          +
Cached Test Audio Tokens
          │
          ▼
Cross-Attention Fusion
          │
          ▼
768-D Fused Representation
          │
          ▼
Final XGBoost
          │
          ▼
       TreeSHAP
          │
     ┌────┼─────────────┐
     │    │             │
     ▼    ▼             ▼
 Global Local       Modality
 SHAP   SHAP        Attribution
     │    │             │
     ▼    ▼             ▼
 Bar   Force       Video / Audio
Beeswarm Plots      Contributions
```

---

# Reproducibility

The project is designed so that expensive feature extraction does not need to be repeated for every downstream experiment.

The workflow separates:

```text
Raw Data
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
Cached Representations
    ↓
Model Training
    ↓
Optimization
    ↓
Statistical Analysis
    ↓
Ablation
    ↓
Explainability
```

The use of fixed preprocessing configurations, explicit dataset splits, controlled random seeds, saved model checkpoints, and cached features supports reproducibility.


---

# Experimental Pipeline

The complete Module 2 workflow can be summarized as:

```text
                    RAW DATA
                       │
                       ▼
              Dataset Verification
                       │
                       ▼
             YouTube Dataset Only
                       │
                       ▼
                 Preprocessing
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
      Video Frames                Audio
          │                         │
          ▼                         ▼
  Video Swin 3D-Tiny               AST
          │                         │
          ▼                         ▼
   Video Tokens                Audio Tokens
          │                         │
          └────────────┬────────────┘
                       ▼
            Bidirectional
             Cross-Attention
                       │
                       ▼
            768-D Fused Features
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Softmax           XGBoost
          Baseline          Optimized
              │                 │
              └────────┬────────┘
                       ▼
              Test Evaluation
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
    Statistics      Ablation         SHAP
         │             │              │
         └─────────────┴──────────────┘
                       │
                       ▼
             External Validation
                (Turning)
```

---

# Methodological Considerations

## Dataset Separation

The YouTube and Turning datasets are intentionally separated.

The YouTube dataset is used to develop the multimodal classifier, while the Turning dataset is retained as an external validation source.

This avoids allowing the external dataset to influence model development.

---

## Modality Availability

The complete architecture requires both video and audio.

The Turning dataset does not provide the same audio information as the YouTube dataset.

Therefore, the multimodal model should not be evaluated on Turning by artificially inserting zero-valued or fabricated audio.

A missing modality is fundamentally different from an observed modality containing no signal.

---

## Avoiding Artificial Negative Labels

Synthetic transformations of positive movement videos do not inherently create clinically valid control examples.

Therefore, artificially transformed positive Turning videos should not be interpreted as genuine negative subjects.

The YouTube dataset provides the labeled positive and negative samples required for the primary supervised multimodal experiment.

---

# Limitations

Several limitations should be considered when interpreting the framework.

### Dataset Size

The YouTube dataset is relatively small compared with large-scale video and audio datasets.

This increases the risk of model overfitting and makes repeated statistical evaluation important.

### Dataset Heterogeneity

YouTube videos can vary substantially in:

* camera position
* lighting
* background
* video quality
* speaking style
* recording environment

These factors may introduce nuisance variation.

### Multimodal Availability

The final multimodal architecture assumes that both video and audio are available.

This limits direct application to datasets containing only movement videos.

### External Dataset Modality Mismatch

The Turning dataset represents a different task and modality configuration from the YouTube dataset.

Consequently, it cannot simply be inserted into the multimodal pipeline without designing an appropriate video-only evaluation pathway.

### SHAP Interpretability

TreeSHAP explains the XGBoost classifier in the learned feature space.

It does not automatically establish a direct causal or pixel-level explanation of Parkinson's Disease.

Modality-level SHAP aggregation should therefore be interpreted as an attribution analysis of the learned representation rather than causal evidence.

### Small Test Set

The YouTube test split is relatively small.

Consequently, individual predictions can cause noticeable changes in percentage-based metrics.

This is another reason for reporting repeated-run statistics rather than relying exclusively on a single test run.

---

# Technologies Used

## Deep Learning

* PyTorch
* Torchvision
* Hugging Face Transformers

## Video Representation

* Video Swin 3D-Tiny

## Audio Representation

* Audio Spectrogram Transformer (AST)

## Multimodal Learning

* Multi-Head Cross-Attention
* Layer Normalization
* Feed-Forward Networks
* Feature Fusion

## Machine Learning

* XGBoost
* Softmax classification

## Optimization

* Optuna

## Explainability

* SHAP
* TreeSHAP

## Data Processing

* NumPy
* Pandas
* Torch tensors
* Audio/spectrogram preprocessing utilities

## Environment

* Google Colab
* CUDA-enabled GPU

---

# Reproducibility Checklist

To reproduce the Module 2 pipeline:

1. Verify the raw YouTube dataset.
2. Generate or verify dataset metadata.
3. Apply the fixed preprocessing configuration.
4. Generate the YouTube train/validation/test splits.
5. Extract Video Swin representations.
6. Extract AST representations.
7. Cache the modality-specific features.
8. Train the bidirectional cross-attention fusion model.
9. Evaluate the Softmax baseline.
10. Optimize the fusion and XGBoost parameters using Optuna.
11. Evaluate the optimized XGBoost model.
12. Repeat experiments across controlled random seeds for statistical analysis.
13. Run the ablation configurations.
14. Apply TreeSHAP to the final optimized XGBoost representation.
15. Preserve the Turning dataset for external validation rather than model development.

---

# Summary

Module 2 implements a multimodal Parkinson's Disease detection framework based on learned visual and audio representations.

The primary pipeline is:

```text
YouTube Video
     │
     ├──────────────► Video Swin 3D-Tiny
     │                         │
     │                         ▼
     │                   Video Tokens
     │
     └──────────────► AST
                               │
                               ▼
                         Audio Tokens
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
          Video-to-Audio              Audio-to-Video
          Cross-Attention             Cross-Attention
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                     Multimodal Fusion
                               │
                     768-D Representation
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
              Softmax                   XGBoost
              Baseline                 Optimized
```

The framework is complemented by:

* controlled dataset splitting
* cached feature extraction
* hyperparameter optimization
* repeated-run statistical analysis
* systematic ablation experiments
* global and local SHAP explainability

The overall experimental design keeps the **YouTube dataset as the sole source for multimodal model development**, while preserving the **Turning dataset as an independent external validation resource**.

This separation, together with cached representations, controlled experiments, ablation analysis, statistical testing, and explainability, provides a structured research framework for investigating multimodal Parkinson's Disease detection.
