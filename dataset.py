"""
Multimodal Product Classification - Dataset & Preprocessing Pipeline
=====================================================================
Handles:
1. Multimodal Text Preprocessing (combining display name + description)
2. Vocabulary construction with frequency filtering & padding/truncation
3. Vision Transforms with Data Augmentation (Train) & Resizing (Val/Test)
4. Stratified Train/Val split safeguarding rare classes (for Balanced Accuracy)
5. Smoothed Class Weights calculation to optimize Balanced Accuracy
"""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


def clean_text(text):
    """
    Cleans and standardizes raw text:
    - Lowercase
    - Strip punctuation and abnormal symbols
    - Collapse extra whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Vocabulary:
    """
    Builds and manages word vocabulary from text corpus.
    <PAD>: 0
    <UNK>: 1
    """
    def __init__(self, max_vocab_size=10000, min_freq=1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            tokens = clean_text(t).split()
            counter.update(tokens)
        
        # Filter by min_freq and take top words
        filtered = [(w, c) for w, c in counter.most_common() if c >= self.min_freq]
        top_words = filtered[: (self.max_vocab_size - 2)]
        
        for idx, (w, _) in enumerate(top_words, start=2):
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, text, max_len=60):
        tokens = clean_text(text).split()
        indices = [self.word2idx.get(w, 1) for w in tokens]
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.word2idx)


class MultimodalDataset(Dataset):
    """
    Dual-modality dataset loading both image and tokenized text sequence.
    """
    def __init__(self, df, img_dir, vocab, max_text_len=60, transform=None, is_test=False, cat2idx=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.transform = transform
        self.is_test = is_test
        self.cat2idx = cat2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # 1. Load Image
        img_filename = str(row['image'])
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback for corrupted/missing file: blank neutral image
            image = Image.new('RGB', (128, 128), color=(128, 128, 128))
            
        if self.transform:
            image = self.transform(image)
            
        # 2. Text Modality (combine display name and description)
        display_name = str(row.get('display name', ''))
        description = str(row.get('description', ''))
        combined_text = f"{display_name} {description}".strip()
        text_tensor = self.vocab.encode(combined_text, max_len=self.max_text_len)
        
        # 3. Output
        if self.is_test:
            item_id = int(row['id'])
            return image, text_tensor, item_id
        else:
            label_str = row['category']
            label_idx = self.cat2idx[label_str] if self.cat2idx is not None else 0
            return image, text_tensor, torch.tensor(label_idx, dtype=torch.long)


def get_vision_transforms(image_size=(128, 128)):
    """
    Returns image transformation pipelines with data augmentation for training,
    and deterministic resizing for validation and testing.
    """
    # ImageNet mean & std for transfer learning normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.15), int(image_size[1] * 1.15))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


def compute_balanced_class_weights(df, cat2idx, smoothing=0.35):
    """
    Computes smooth inverse class frequency weights to directly maximize Balanced Accuracy.
    
    Formula: weight_c = (max_count / count_c) ** smoothing
    Smoothing factor (e.g. 0.35) prevents 1-sample classes from having 5000x weight,
    which would otherwise destabilize gradient descent.
    """
    counts = df['category'].value_counts()
    num_classes = len(cat2idx)
    weights = np.ones(num_classes, dtype=np.float32)
    max_count = counts.max()
    
    for cat, idx in cat2idx.items():
        c = counts.get(cat, 1)
        w = (max_count / max(c, 1)) ** smoothing
        weights[idx] = w
        
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_stratified_train_val_split(df, test_size=0.15, random_state=42):
    """
    Safely splits dataset ensuring:
    - Classes with >= 2 instances are stratified across train and val.
    - Single-instance classes are placed in train so the model has exposure to them.
    """
    counts = df['category'].value_counts()
    multi_instance_classes = counts[counts >= 2].index
    single_instance_classes = counts[counts == 1].index

    df_multi = df[df['category'].isin(multi_instance_classes)]
    df_single = df[df['category'].isin(single_instance_classes)]

    split_res = train_test_split(
        df_multi,
        test_size=test_size,
        stratify=df_multi['category'],
        random_state=random_state
    )
    train_multi = pd.DataFrame(split_res[0])
    val_multi = pd.DataFrame(split_res[1])

    # Combine single instances into training set
    train_df = pd.concat([train_multi, df_single], ignore_index=True)
    val_df = val_multi.reset_index(drop=True)

    print(f"Dataset Split: {len(train_df)} train samples, {len(val_df)} validation samples.")
    return train_df, val_df
