# BCS714A Multimodal Product Classification

A dual-branch multimodal deep learning model combining Computer Vision (CNN) and Natural Language Processing (Bi-RNN) to classify e-commerce products across 140 categories.

---

## 🎯 Competition & Challenge Constraints

- **Competition:** [BCS714A Multimodal Fusion on Kaggle](https://www.kaggle.com/t/08d5b24fd37f413c99c47d5fe82468bc)
- **Evaluation Metric:** **Balanced Accuracy** (macro-average recall across all 140 product categories)
- **Strict Parameter Cap:** Total parameters must be strictly **< 15,000,000 (< 15 Million)**
- **No Pretrained VLMs:** Pre-trained Vision-Language Models (CLIP, BLIP, etc.) are strictly prohibited.
- **Architectural Rules:**
  - **Vision Branch:** Lightweight CNN backbone (`MobileNetV2` with 1,280 features or `ResNet-18` with 512 features).
  - **NLP Branch:** Word Embeddings + Bidirectional Recurrent Neural Network (`GRU` / `LSTM`) with temporal pooling (256 features).
  - **Fusion Layer:** Concatenation (`torch.cat([img_feat, text_feat], dim=1)`) followed by an MLP classifier with Dropout and BatchNorm.
  - **Imbalance Handling:** Balanced class weighting applied to Cross-Entropy Loss.

---

## 📁 Repository Structure

```
.
├── dataset.py                               # Text tokenizer, Vocabulary builder & MultimodalDataset loader
├── model.py                                 # Dual-branch multimodal architecture & parameter check audit
├── train.py                                 # Complete training loop, Balanced Accuracy evaluation & submission generator
├── generate_curves.py                       # Plotting script for loss and accuracy progression
├── generate_diagrams.py                     # Visual architecture block diagram generator
├── build_notebook.py                        # Utility script to compile scripts into Jupyter notebooks
├── multimodal_product_classification.ipynb  # Self-contained Jupyter notebook for Kaggle / Google Colab
├── starter_notebook.ipynb                   # Starter notebook version
├── architecture_diagram.png                 # Diagram illustrating the dual-branch fusion pipeline
├── training_curves.png                      # Visualized loss & Balanced Accuracy learning curves
├── .gitignore                               # Git ignore rules for checkpoints, data, and cache
└── README.md                                # Project documentation
```

---

## 🚀 Quickstart for Team Members

### 1. Clone the Repository
```bash
git clone https://github.com/Dheemanth07/multimodal-product-classification.git
cd multimodal-product-classification
```

### 2. Install Dependencies
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow tqdm
```

### 3. Run Training Locally
```bash
python train.py --data_dir ./data --epochs 10 --batch_size 32
```

### 4. Running on Kaggle / Google Colab
Upload and open `multimodal_product_classification.ipynb` on Kaggle or Colab, enable GPU acceleration (T4 GPU), and execute all cells.
The final cell automatically exports `submission.csv` ready for leaderboard scoring.
