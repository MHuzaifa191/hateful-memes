# @title Dataset Implementation

import os
import torch
import json
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import WeightedRandomSampler

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

class HatefulMemesDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, text_processor=None, 
                 max_length=128, augment=False):
        """
        Custom PyTorch Dataset for the Hateful Memes dataset
        
        Args:
            data_dir: Directory containing the dataset files
            split: 'train', 'dev', or 'test'
            transform: Image transformations
            text_processor: Text preprocessing function or tokenizer
            max_length: Maximum sequence length for text
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.text_processor = text_processor
        self.max_length = max_length
        self.augment = augment
        
        # Load annotations
        json_file = os.path.join(data_dir, f"{split}.jsonl")
        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
        # Set up default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        # Set up default text processor if none provided
        if self.text_processor is None:
            self.text_processor = BertTokenizer.from_pretrained('bert-base-uncased')
            
        # Setup for text augmentation
        self.stop_words = set(stopwords.words('english'))
        
        # Calculate class weights for sampling
        if split == 'train':
            labels = [item['label'] for item in self.data]
            class_counts = Counter(labels)
            total_samples = len(labels)
            self.class_weights = {
                0: total_samples / (2 * class_counts[0]),
                1: total_samples / (2 * class_counts[1])
            }
            self.sample_weights = [self.class_weights[label] for label in labels]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, item['img'])
        image = Image.open(img_path).convert('RGB')
        
        # Get text and label
        text = item['text']
        label = torch.tensor(item['label'], dtype=torch.float32)
        
        # Apply augmentations if needed
        if self.augment and item['label'] == 1:  # Apply augmentation to minority class (assuming hateful is minority)
            # Image augmentation
            aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                self.transform
            ])
            image = aug_transform(image)
            
            # Text augmentation - simple synonym replacement (could be enhanced)
            text = self.augment_text(text)
        else:
            image = self.transform(image)
        
        # Process text based on the processor type
        if isinstance(self.text_processor, BertTokenizer):
            # For BERT
            text_encoded = self.text_processor(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'image': image,
                'input_ids': text_encoded['input_ids'].squeeze(),
                'attention_mask': text_encoded['attention_mask'].squeeze(),
                'text': text,  # Keep original text for visualization
                'label': label,
                'id': item['id']
            }
        else:
            # For LSTM or other text processors
            tokens = self.text_processor(text)
            return {
                'image': image,
                'text_tokens': tokens,
                'text': text,  # Keep original text for visualization
                'label': label,
                'id': item['id']
            }
    
    def augment_text(self, text):
        """Simple text augmentation by replacing random words with synonyms"""
        # This is a placeholder - a more sophisticated approach would use WordNet
        words = text.split()
        if len(words) <= 3:
            return text  # Don't augment very short texts
            
        # Randomly choose 1-2 words to modify
        num_to_change = min(2, max(1, int(len(words) * 0.2)))
        indices = random.sample(range(len(words)), num_to_change)
        
        for idx in indices:
            # Simple character swap - in practice, use a synonym library
            word = words[idx]
            if len(word) > 3:
                i, j = sorted(random.sample(range(1, len(word)-1), 2))
                chars = list(word)
                chars[i], chars[j] = chars[j], chars[i]
                words[idx] = ''.join(chars)
                
        return ' '.join(words)

    def get_sampler(self):
        """Returns a weighted sampler to handle class imbalance"""
        if hasattr(self, 'sample_weights'):
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None

# Functions for text preprocessing

def preprocess_text_for_lstm(text):
    """Preprocess text for LSTM model"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Data preprocessing and visualization functions

def analyze_class_distribution(dataset):
    """Analyze and visualize class distribution"""
    labels = [item['label'] for item in dataset.data]
    label_counts = Counter(labels)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Non-Hateful (0)', 'Hateful (1)'], [label_counts[0], label_counts[1]])
    plt.title('Class Distribution in Dataset')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    plt.close()
    
    print(f"Class distribution: Non-Hateful={label_counts[0]}, Hateful={label_counts[1]}")
    print(f"Class imbalance ratio: {label_counts[0]/label_counts[1]:.2f}:1")
    
    return label_counts

def generate_word_cloud(dataset):
    """Generate a word cloud of important words based on TF-IDF"""
    # Extract all text from dataset
    all_texts = [item['text'] for item in dataset.data]
    
    # Separate hateful and non-hateful texts
    hateful_texts = [item['text'] for item in dataset.data if item['label'] == 1]
    non_hateful_texts = [item['text'] for item in dataset.data if item['label'] == 0]
    
    # Calculate TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = tfidf.fit_transform(all_texts)
    
    # Get feature names and TF-IDF scores
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    
    # Create a word cloud of high TF-IDF terms
    word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
    
    # Generate word cloud for all texts
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Important Terms (High TF-IDF)')
    plt.savefig('wordcloud_all.png')
    plt.close()
    
    # Generate word clouds for hateful and non-hateful separately
    if hateful_texts:
        tfidf_hateful = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix_hateful = tfidf_hateful.fit_transform(hateful_texts)
        feature_names_hateful = tfidf_hateful.get_feature_names_out()
        tfidf_scores_hateful = tfidf_matrix_hateful.sum(axis=0).A1
        word_scores_hateful = {feature_names_hateful[i]: tfidf_scores_hateful[i] for i in range(len(feature_names_hateful))}
        
        wordcloud_hateful = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores_hateful)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_hateful, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Important Terms in Hateful Memes')
        plt.savefig('wordcloud_hateful.png')
        plt.close()
    
    if non_hateful_texts:
        tfidf_non_hateful = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix_non_hateful = tfidf_non_hateful.fit_transform(non_hateful_texts)
        feature_names_non_hateful = tfidf_non_hateful.get_feature_names_out()
        tfidf_scores_non_hateful = tfidf_matrix_non_hateful.sum(axis=0).A1
        word_scores_non_hateful = {feature_names_non_hateful[i]: tfidf_scores_non_hateful[i] for i in range(len(feature_names_non_hateful))}
        
        wordcloud_non_hateful = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores_non_hateful)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_non_hateful, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Important Terms in Non-Hateful Memes')
        plt.savefig('wordcloud_non_hateful.png')
        plt.close()

def visualize_sample_memes(dataset, num_samples=5):
    """Visualize sample memes from the dataset"""
    # Get indices of hateful and non-hateful memes
    hateful_indices = [i for i, item in enumerate(dataset.data) if item['label'] == 1]
    non_hateful_indices = [i for i, item in enumerate(dataset.data) if item['label'] == 0]
    
    # Sample from each class
    sampled_hateful = random.sample(hateful_indices, min(num_samples, len(hateful_indices)))
    sampled_non_hateful = random.sample(non_hateful_indices, min(num_samples, len(non_hateful_indices)))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    # Plot hateful memes
    for i, idx in enumerate(sampled_hateful):
        item = dataset.data[idx]
        img_path = os.path.join(dataset.data_dir, item['img'])
        image = Image.open(img_path).convert('RGB')
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Hateful: {item['text']}", fontsize=8)
        axes[0, i].axis('off')
    
    # Plot non-hateful memes
    for i, idx in enumerate(sampled_non_hateful):
        item = dataset.data[idx]
        img_path = os.path.join(dataset.data_dir, item['img'])
        image = Image.open(img_path).convert('RGB')
        axes[1, i].imshow(image)
        axes[1, i].set_title(f"Non-Hateful: {item['text']}", fontsize=8)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_memes.png')
    plt.close()

# Example usage:
if __name__ == "__main__":
    data_dir = "/kaggle/input/facebook-hateful-meme-dataset/data"
    
    # Create dataset
    train_dataset = HatefulMemesDataset(
        data_dir=data_dir,
        split='train',
        augment=True
    )
    
    # Create data loader with weighted sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_dataset.get_sampler(),  # Use weighted sampler
        num_workers=4
    )
    
    # Add more detailed analysis
    analyze_class_distribution(train_dataset)
    print("\nData Loading Statistics:")
    print(f"Total samples: {len(train_dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Effective samples per epoch: {len(train_loader) * train_loader.batch_size}")
    
    # Verify augmentation
    if train_dataset.augment:
        print("\nAugmentation Verification:")
        sample_idx = next(iter(train_dataset.get_sampler()))
        original_item = train_dataset.data[sample_idx]
        augmented_item = train_dataset[sample_idx]
        print(f"Original text: {original_item['text']}")
        print(f"Augmented text: {augmented_item['text']}")
    
    # Analyze and visualize the dataset
    generate_word_cloud(train_dataset)
    visualize_sample_memes(train_dataset)
    
    # Print a sample batch
    batch = next(iter(train_loader))
    print(f"Batch size: {len(batch['image'])}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels: {batch['label']}")




