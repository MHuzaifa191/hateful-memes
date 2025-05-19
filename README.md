# Introduction

This repository contains code for detecting hateful memes using multimodal deep learning approaches. The project implements various fusion strategies to combine text and image features for effective classification of hateful content in memes.

## Overview

Hateful memes present a unique challenge for content moderation systems as they combine visual and textual elements in ways that may not be hateful when considered separately. This project explores different neural network architectures to address this multimodal classification problem.

## Running Instructions

Each folder contains code for a different model. You need to create a kaggle/colab notebook and first install the requirements using this command:

```
!pip install transformers datasets torchvision wordcloud matplotlib tensorboard gdown
```
Then load the dataset on kaggle. It is already available there.

```
facebook_hateful_meme_dataset_path = kagglehub.dataset_download('parthplc/facebook-hateful-meme-dataset')
```

Then copy each file on a notebook cell and run it. Copy files onto cells in this order:

- dataset_implementation.py
- text_image_models.py
- evaluation_kaggle.py

## Results

I have included results in `Images` and `Tensorboard Logs` folders.


## Models Implemented

### Late Fusion Models
- **LSTM + CNN**: Text features are extracted using LSTM networks while image features are extracted using CNNs. Features are concatenated before final classification.
- **BERT + ResNet**: Leverages pre-trained BERT for text and ResNet for images, with late fusion of features.

### Early Fusion Models
- **LSTM + CNN with Early Fusion**: Text and image features are combined at an earlier stage in the network.
- **BERT + ResNet with Early Fusion**: Combines BERT text embeddings with ResNet image features at an early stage of the network.

## Dataset

The models are trained and evaluated on the Facebook Hateful Memes Challenge dataset, which contains memes labeled as either hateful or non-hateful.

## Key Features

- Multimodal fusion strategies (early and late fusion)
- Data augmentation techniques
- Class-weighted loss functions to handle imbalanced data
- Comprehensive evaluation metrics (AUROC, F1, precision, recall)
- TensorBoard integration for visualization
- Early stopping to prevent overfitting
- Learning rate scheduling

## Results

The models achieve varying performance on the validation set:
- Late Fusion LSTM+CNN: AUROC ~0.58-0.60
- Early Fusion models show comparable performance

## Usage

The code is designed to run in a Kaggle environment with the Facebook Hateful Memes dataset.

```python
# Example usage
python evaluation_kaggle.py
```

## Requirements

- PyTorch
- torchvision
- transformers (for BERT models)
- scikit-learn
- matplotlib
- tensorboard
- numpy
- pandas

## Future Work

- Implement attention mechanisms for better feature fusion
- Explore more sophisticated data augmentation techniques
- Experiment with ensemble methods
- Incorporate visual-linguistic pre-trained models like CLIP

## License

This project is available under the MIT License.
