"""
Multimodal Product Classification - Training & Inference Pipeline
==================================================================
Runs complete training loop, computes Balanced Accuracy, saves checkpoints,
plots loss/accuracy curves, and exports Kaggle submission.csv.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DualBranchMultimodalModel, get_model_summary_and_check_cap
from dataset import (
    Vocabulary,
    MultimodalDataset,
    get_vision_transforms,
    compute_balanced_class_weights,
    get_stratified_train_val_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Product Classification")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet18", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_text_len", type=int, default=60)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    return parser.parse_args()


def resolve_paths(args):
    # Candidate paths for Kaggle and Local setups
    kaggle_data_dir = "/kaggle/input/competitions/bcs-714-a-multimodal-fusion/data/data/"
    kaggle_train_csv = "/kaggle/input/competitions/bcs-714-a-multimodal-fusion/train.csv"
    kaggle_test_csv = "/kaggle/input/competitions/bcs-714-a-multimodal-fusion/test.csv"

    local_data_dir = r"C:\Users\dheem\Downloads\bcs-714-a-multimodal-fusion\data\data"
    local_train_csv = r"C:\Users\dheem\Downloads\bcs-714-a-multimodal-fusion\train.csv"
    local_test_csv = r"C:\Users\dheem\Downloads\bcs-714-a-multimodal-fusion\test.csv"

    if args.data_dir and os.path.exists(args.data_dir):
        data_dir = args.data_dir
    elif os.path.exists(kaggle_data_dir):
        data_dir = kaggle_data_dir
    elif os.path.exists(local_data_dir):
        data_dir = local_data_dir
    else:
        # Fallback to local data folder
        data_dir = r"data/data"

    if args.train_csv and os.path.exists(args.train_csv):
        train_csv = args.train_csv
    elif os.path.exists(kaggle_train_csv):
        train_csv = kaggle_train_csv
    elif os.path.exists(local_train_csv):
        train_csv = local_train_csv
    else:
        train_csv = "train.csv"

    if args.test_csv and os.path.exists(args.test_csv):
        test_csv = args.test_csv
    elif os.path.exists(kaggle_test_csv):
        test_csv = kaggle_test_csv
    elif os.path.exists(local_test_csv):
        test_csv = local_test_csv
    else:
        test_csv = "test.csv"

    print(f"Paths Resolved:\n  Data Dir:  {data_dir}\n  Train CSV: {train_csv}\n  Test CSV:  {test_csv}")
    return data_dir, train_csv, test_csv


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, texts, labels in tqdm(loader, desc="Training", leave=False):
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for RNN stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_bacc = balanced_accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_bacc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, texts, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_bacc = balanced_accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_bacc


def run_inference(model, test_loader, idx2cat, device, output_path="submission.csv"):
    print("\nRunning Inference on Test Set...")
    model.eval()
    predictions = []
    item_ids = []

    with torch.no_grad():
        for images, texts, ids in tqdm(test_loader, desc="Predicting Test"):
            images, texts = images.to(device), texts.to(device)
            outputs = model(images, texts)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for p in preds:
                predictions.append(idx2cat[p])
            item_ids.extend(ids.numpy())

    submission_df = pd.DataFrame({
        "id": item_ids,
        "category": predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Submission exported to: {output_path} ({len(submission_df)} rows)")
    return submission_df


def plot_curves(history, output_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # 1. Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s-", color="#d62728", label="Val Loss", linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    
    # 2. Balanced Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_bacc"]], "o-", color="#2ca02c", label="Train Balanced Acc (%)", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_bacc"]], "s-", color="#ff7f0e", label="Val Balanced Acc (%)", linewidth=2)
    plt.title("Training vs Validation Balanced Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Balanced Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Training curves saved to: {output_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    data_dir, train_csv_path, test_csv_path = resolve_paths(args)

    # 1. Load Data
    print("\n[Step 1/6] Loading CSV Data...")
    train_raw = pd.read_csv(train_csv_path)
    test_raw = pd.read_csv(test_csv_path)

    # 2. Create Label Encoders
    unique_cats = sorted(train_raw["category"].unique())
    num_classes = len(unique_cats)
    cat2idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    idx2cat = {idx: cat for cat, idx in cat2idx.items()}
    print(f"Detected {num_classes} unique target categories.")

    # 3. Build Vocabulary
    print("\n[Step 2/6] Building Text Vocabulary from combined text...")
    vocab = Vocabulary(max_vocab_size=args.vocab_size)
    combined_train_texts = (
        train_raw["display name"].fillna("").astype(str) + " " + train_raw["description"].fillna("").astype(str)
    )
    vocab.fit(combined_train_texts)
    print(f"Vocabulary Size: {len(vocab):,} unique tokens.")

    # 4. Stratified Split & Datasets
    print("\n[Step 3/6] Setting up DataLoaders with Data Augmentation...")
    train_df, val_df = get_stratified_train_val_split(train_raw, test_size=0.15, random_state=42)
    train_transform, eval_transform = get_vision_transforms(image_size=(args.image_size, args.image_size))

    train_dataset = MultimodalDataset(train_df, data_dir, vocab, max_text_len=args.max_text_len, transform=train_transform, cat2idx=cat2idx)
    val_dataset = MultimodalDataset(val_df, data_dir, vocab, max_text_len=args.max_text_len, transform=eval_transform, cat2idx=cat2idx)
    test_dataset = MultimodalDataset(test_raw, data_dir, vocab, max_text_len=args.max_text_len, transform=eval_transform, is_test=True)

    num_workers = 2 if sys.platform != "win32" else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 5. Initialize Model & Verify Parameter Limits
    print("\n[Step 4/6] Initializing Dual-Branch Multimodal Model...")
    model = DualBranchMultimodalModel(
        num_classes=num_classes,
        vocab_size=len(vocab),
        vision_backbone=args.backbone,
        pretrained_vision=True,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        fusion_hidden_dim=512,
        dropout_rate=0.3
    ).to(device)

    # Strictly verify < 15M parameters
    get_model_summary_and_check_cap(model, max_param_cap=15_000_000)

    # 6. Loss & Optimization Setup
    class_weights = compute_balanced_class_weights(train_df, cat2idx, smoothing=0.35).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 7. Training Loop
    print("\n[Step 5/6] Starting Training with Balanced Accuracy Optimization...")
    history = {"train_loss": [], "val_loss": [], "train_bacc": [], "val_bacc": []}
    best_val_bacc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_multimodal_model.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")
        t_loss, t_bacc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_bacc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_bacc"].append(t_bacc)
        history["val_bacc"].append(v_bacc)

        print(f"Epoch {epoch:02d} Summary:")
        print(f"  Train Loss: {t_loss:.4f} | Train Balanced Acc: {t_bacc*100:.2f}%")
        print(f"  Val Loss:   {v_loss:.4f} | Val Balanced Acc:   {v_bacc*100:.2f}%")

        if v_bacc > best_val_bacc:
            best_val_bacc = v_bacc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Model checkpoint updated! (Best Val Balanced Acc: {best_val_bacc*100:.2f}%)")

    # Plot & Save Curves
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_curves(history, output_path=curves_path)

    # 8. Inference with Best Model
    print("\n[Step 6/6] Generating Final Test Predictions...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best checkpoint from: {best_model_path}")

    sub_path = os.path.join(args.output_dir, "submission.csv")
    run_inference(model, test_loader, idx2cat, device, output_path=sub_path)
    print("\nPipeline Complete! Model and predictions ready for submission.")


if __name__ == "__main__":
    main()
