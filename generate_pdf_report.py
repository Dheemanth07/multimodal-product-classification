"""
Script to generate the official 3-Page Technical Report PDF for:
BCS714A Multimodal Product Classification (Assignment 2)
Adheres strictly to the university rubric and deliverable guidelines (Exactly 3 Pages).
Features:
- Accurate team credentials: Dream Team (Dheemanth D, Sai Sarvesh, M Prajwal, Priyanshu)
- Balanced layout with professional breathing room and legible typography
- Crystal-clear un-squished diagrams with 1:1 true aspect ratios
- Complete tensor tracing and anti-overfitting / anti-underfitting analysis
- Precise academic terminology (Weighted Cross-Entropy Loss, Balanced Accuracy Metric, Parameter Constraint & Budget)
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(44, letter[1] - 28, "Multimodal Product Classification | Technical Engineering Report")
            self.drawRightString(letter[0] - 44, letter[1] - 28, "Team: Dream Team | Activity-Based Learning (ABL)")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(44, letter[1] - 32, letter[0] - 44, letter[1] - 32)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(44, 34, letter[0] - 44, 34)
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(44, 23, "Department of Computer Science & Engineering | Multimodal Classification Submission")
        self.drawRightString(letter[0] - 44, 23, page_str)
        self.restoreState()


def create_technical_report(output_filename="Technical_Report_Multimodal_Product_Classification.pdf"):
    # Margins: 44pt left/right, 34pt top, 38pt bottom -> printable: 524pt x 720pt
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=44,
        rightMargin=44,
        topMargin=34,
        bottomMargin=38
    )

    # Styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=14.5,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='Helvetica',
        fontSize=8.8,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        fontName='Helvetica-Bold',
        fontSize=9.8,
        leading=12.5,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=6,
        spaceAfter=3.0,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        fontName='Helvetica',
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=3.0
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        fontName='Helvetica',
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=3.8
    )

    ablation_bullet_style = ParagraphStyle(
        'AblationBullet',
        fontName='Helvetica',
        fontSize=7.6,
        leading=10.3,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5.2
    )

    caption_style = ParagraphStyle(
        'DocCaption',
        fontName='Helvetica-Oblique',
        fontSize=6.9,
        leading=8.8,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=3.5
    )

    table_cell = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=7.1,
        leading=9.0,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.1,
        leading=9.0,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # =========================================================================
    # PAGE 1: HEADER, METADATA, ARCHITECTURE & PARAMETER BUDGET AUDIT
    # =========================================================================
    story.append(Paragraph("Multimodal Product Classification", title_style))
    story.append(Paragraph("Dual-Branch Multimodal Fusion Network (Vision + NLP) | Technical Engineering Report", subtitle_style))
    
    # Metadata Box
    meta_data = [
        [
            Paragraph("<b>Kaggle Team Name:</b> Dream Team", table_cell),
            Paragraph("<b>Course:</b> BCS714A - Deep Learning (ABL)", table_cell)
        ],
        [
            Paragraph("<b>Team Leader:</b> Dheemanth D", table_cell),
            Paragraph("<b>Faculty In-Charge:</b> Dr. Aravinda S Rao, CSE", table_cell)
        ],
        [
            Paragraph("<b>Team Members:</b> Sai Sarvesh, M Prajwal, Priyanshu", table_cell),
            Paragraph("<b>Evaluation Metric:</b> Balanced Accuracy (Macro Recall)", table_cell)
        ],
        [
            Paragraph("<b>Parameter Constraint:</b> &lt; 15,000,000 (No VLMs Allowed)", table_cell),
            Paragraph("<b>Exact Model Parameters:</b> 4,858,252 (Headroom: 10,141,748)", table_cell_bold)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))

    # Section 1: Problem Overview
    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(Paragraph(
        "E-commerce catalog classification requires categorizing products across <b>140 fine-grained, imbalanced categories</b>. "
        "Unimodal approaches suffer severely: visual representations fail when products share similar packaging or silhouettes (e.g., book genres "
        "or technical cables), while textual representations fail when titles omit visual attributes (e.g., apparel color, texture, or pattern). "
        "This project implements an end-to-end <b>Dual-Branch Multimodal Neural Network</b> fusing Computer Vision (CNN) and Natural Language "
        "Processing (Bi-GRU) to achieve robust cross-modal representations. Trained using <b>Weighted Cross-Entropy Loss</b> and evaluated via "
        "<b>Balanced Accuracy</b> (macro-average recall), the model satisfies the strict <b>&lt; 15,000,000 parameter constraint</b> without relying on heavy Vision-Language Models (VLMs).",
        body_style
    ))

    # Section 2: Dual-Branch Architecture
    story.append(Spacer(1, 4))
    story.append(Paragraph("2. Dual-Branch Multimodal Network Architecture", h1_style))
    story.append(Paragraph(
        "The architecture decouples representation learning into two specialized branches before joining in a late-fusion classification head:",
        body_style
    ))
    
    arch_bullets = [
        "<b>Vision Branch (Module 3):</b> Processes 128x128x3 RGB images via a MobileNetV2 feature extractor (17 inverted residual blocks). "
        "Adaptive Average Pooling condenses spatial feature maps into a 1,280-dimensional <b>Global Visual Feature Vector</b> [B, 1280].",
        "<b>NLP Branch (Modules 4 & 5):</b> Combines product 'display name' and 'description'. A trainable embedding layer (10,000 vocab x 128 dim) "
        "feeds a 2-layer Bidirectional GRU (hidden_dim=128). A <b>Mask-Aware Temporal Average Pooling</b> layer eliminates padding token dilution, "
        "yielding a length-normalized 256-dimensional textual feature vector [B, 256].",
        "<b>Late Fusion & Classifier:</b> Manual concatenation <code>torch.cat([img, text], dim=1)</code> yields a 1,536-dimensional joint embedding, "
        "fed to an MLP (Linear(1536, 512) -&gt; BatchNorm1d -&gt; ReLU -&gt; Dropout(0.3) -&gt; Linear(512, 140)) to generate 140 class prediction logits, "
        "optimized via Weighted Cross-Entropy Loss with smoothed inverse class frequency weights."
    ]
    for b in arch_bullets:
        story.append(Paragraph(f"&bull; {b}", bullet_style))

    # Architecture Diagram (vertically extended layout for optimal clarity)
    if os.path.exists("architecture_diagram.png"):
        img_arch = Image("architecture_diagram.png", width=514, height=342)
        story.append(Spacer(1, 4))
        story.append(img_arch)
        story.append(Paragraph("Figure 1: End-to-End Dual-Branch Multimodal Architecture with Layer-wise Tensor Dimensions and Parameter Budget", caption_style))

    # =========================================================================
    # PAGE 2: PARAMETER BUDGET AUDIT, TRAINING DYNAMICS & HYPERPARAMETERS
    # =========================================================================
    story.append(PageBreak())

    # Section 3: Parameter Audit Table
    story.append(Paragraph("3. Parameter Constraint & Exact Parameter Budget Audit (&lt; 15M Constraint)", h1_style))
    param_data = [
        [Paragraph("<b>Subsystem / Layer Module</b>", table_cell_bold), Paragraph("<b>Layer Description & Tensor Shape</b>", table_cell_bold), Paragraph("<b>Exact Parameters</b>", table_cell_bold), Paragraph("<b>% Budget</b>", table_cell_bold)],
        [Paragraph("Vision Branch (MobileNetV2)", table_cell), Paragraph("Conv2d & InvertedResiduals ([B, 3, 128, 128] -&gt; [B, 1280])", table_cell), Paragraph("2,223,872", table_cell), Paragraph("14.8%", table_cell)],
        [Paragraph("NLP Word Embedding", table_cell), Paragraph("Trainable nn.Embedding(10000, 128) ([B, 60] -&gt; [B, 60, 128])", table_cell), Paragraph("1,280,000", table_cell), Paragraph("8.5%", table_cell)],
        [Paragraph("NLP Recurrent Backbone", table_cell), Paragraph("2-Layer Bidirectional GRU (hidden=128) ([B, 60, 128] -&gt; [B, 256])", table_cell), Paragraph("494,592", table_cell), Paragraph("3.3%", table_cell)],
        [Paragraph("Fusion Classifier Head", table_cell), Paragraph("Linear(1536, 512) + BatchNorm1d(512) + Linear(512, 140)", table_cell), Paragraph("859,788", table_cell), Paragraph("5.7%", table_cell)],
        [Paragraph("<b>Total Model Parameters</b>", table_cell_bold), Paragraph("<b>Entire Multimodal Network (All Trainable)</b>", table_cell_bold), Paragraph("<b>4,858,252</b>", table_cell_bold), Paragraph("<b>32.4%</b>", table_cell_bold)],
        [Paragraph("<b>Parameter Constraint</b>", table_cell_bold), Paragraph("<b>Competition Ceiling (assert total &lt; 15,000,000)</b>", table_cell_bold), Paragraph("<b>15,000,000</b>", table_cell_bold), Paragraph("<b>PASSED</b>", table_cell_bold)]
    ]
    param_table = Table(param_data, colWidths=[120, 240, 84, 80])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#f1f5f9")),
        ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor("#dcfce7")),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 8))

    # Section 4: Training Dynamics & Empirical Performance
    story.append(Paragraph("4. Training Dynamics & Empirical Performance", h1_style))
    story.append(Paragraph(
        "To optimize for the competition's macro-average <b>Balanced Accuracy</b> metric across 140 imbalanced categories, the network was trained "
        "using <b>Smoothed Inverse Class Frequency Weighting</b> in Cross-Entropy Loss: <code>w_c = (max_count / count_c)^0.35</code>. "
        "This balances gradient updates between head and tail classes without introducing destabilizing gradient spikes. "
        "Optimization was performed using AdamW with decoupled weight decay (1e-4) and a Cosine Annealing learning rate schedule.",
        body_style
    ))

    # Training Curves Figure
    if os.path.exists("training_curves.png"):
        img_curves = Image("training_curves.png", width=514, height=154)
        story.append(Spacer(1, 2))
        story.append(img_curves)
        story.append(Paragraph("Figure 2: Empirical Training vs. Validation Loss (left) and Balanced Accuracy (right) over 5 Epochs", caption_style))

    # Epoch Metrics Table
    metrics_data = [
        [Paragraph("<b>Epoch</b>", table_cell_bold), Paragraph("<b>Train Loss (Weighted CE)</b>", table_cell_bold), Paragraph("<b>Val Loss (Weighted CE)</b>", table_cell_bold), Paragraph("<b>Train Balanced Acc (%)</b>", table_cell_bold), Paragraph("<b>Val Balanced Acc (%)</b>", table_cell_bold), Paragraph("<b>Effective LR</b>", table_cell_bold)],
        [Paragraph("Epoch 1", table_cell), Paragraph("1.119", table_cell), Paragraph("0.490", table_cell), Paragraph("38.4%", table_cell), Paragraph("64.6%", table_cell), Paragraph("1.00e-3", table_cell)],
        [Paragraph("Epoch 2", table_cell), Paragraph("0.371", table_cell), Paragraph("0.211", table_cell), Paragraph("65.0%", table_cell), Paragraph("83.4%", table_cell), Paragraph("9.05e-4", table_cell)],
        [Paragraph("Epoch 3", table_cell), Paragraph("0.182", table_cell), Paragraph("0.192", table_cell), Paragraph("80.8%", table_cell), Paragraph("86.2%", table_cell), Paragraph("6.55e-4", table_cell)],
        [Paragraph("Epoch 4", table_cell), Paragraph("0.108", table_cell), Paragraph("0.142", table_cell), Paragraph("88.8%", table_cell), Paragraph("89.5%", table_cell), Paragraph("3.45e-4", table_cell)],
        [Paragraph("Epoch 5", table_cell), Paragraph("0.070", table_cell), Paragraph("0.132", table_cell), Paragraph("92.8%", table_cell), Paragraph("<b>90.1%</b>", table_cell_bold), Paragraph("9.55e-5", table_cell)],
    ]
    metrics_table = Table(metrics_data, colWidths=[65, 100, 100, 100, 95, 64])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#fef9c3")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
    ]))
    story.append(Spacer(1, 4))
    story.append(metrics_table)
    story.append(Spacer(1, 10))

    # Section 5: Hyperparameters Table
    story.append(Paragraph("5. Hyperparameter Choices & Technical Rationale", h1_style))
    
    hp_data = [
        [Paragraph("<b>Hyperparameter</b>", table_cell_bold), Paragraph("<b>Configured Value</b>", table_cell_bold), Paragraph("<b>Technical Justification & Operational Rationale</b>", table_cell_bold)],
        [
            Paragraph("Learning Rate & Scheduler", table_cell),
            Paragraph("1e-3 (Cosine Annealing)", table_cell),
            Paragraph("Initial LR of 0.001 facilitates rapid multimodal representation alignment; Cosine Annealing gradually reduces the learning rate throughout the five-epoch training schedule.", table_cell)
        ],
        [
            Paragraph("Optimizer & Weight Decay", table_cell),
            Paragraph("AdamW (decay=1e-4)", table_cell),
            Paragraph("Decoupled weight decay provides superior regularization over standard Adam, preventing over-parameterized recurrent weights from overfitting.", table_cell)
        ],
        [
            Paragraph("Batch Size & Stability", table_cell),
            Paragraph("64 (drop_last=True)", table_cell),
            Paragraph("Provides stable gradient estimation across 140 classes within 16GB GPU VRAM. drop_last=True prevents BatchNorm1d batch-size-1 runtime crashes.", table_cell)
        ],
        [
            Paragraph("Image Resolution & Augmentation", table_cell),
            Paragraph("128 x 128 (Augmented)", table_cell),
            Paragraph("Optimal compromise between fine spatial feature extraction and training throughput. RandomCrop, HorizontalFlip, and ColorJitter prevent visual memorization.", table_cell)
        ],
        [
            Paragraph("Text Sequence Length", table_cell),
            Paragraph("Max Len = 60 tokens", table_cell),
            Paragraph("A maximum sequence length of 60 tokens limits recurrent computation and memory usage while retaining a fixed input shape for efficient batch processing.", table_cell)
        ],
        [
            Paragraph("Class Weight Smoothing", table_cell),
            Paragraph("Power alpha = 0.35", table_cell),
            Paragraph("The exponent 0.35 smooths the inverse-frequency weighting, preventing excessively large weights for rare classes and stabilizing stochastic gradient descent.", table_cell)
        ],
        [
            Paragraph("Regularization", table_cell),
            Paragraph("Dropout=0.3 + GradClip=5.0", table_cell),
            Paragraph("Dropout prevents late-fusion co-adaptation; gradient norm clipping at 5.0 prevents exploding gradients in the unrolled recurrent GRU.", table_cell)
        ]
    ]
    hp_table = Table(hp_data, colWidths=[120, 110, 294])
    hp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
    ]))
    story.append(hp_table)

    # =========================================================================
    # PAGE 3: FAILED EXPERIMENTS, ABLATIONS & VIVA DEFENSE PREPARATION
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("6. Architectural Ablations & Lessons from Failed Experiments", h1_style))
    story.append(Paragraph(
        "Design iterations and empirical component ablations guided the architectural selection under competition constraints:",
        body_style
    ))

    ablations = [
        "<b>Class-Imbalance Objective & Loss Weighting:</b> Training with unweighted Cross-Entropy optimizes for bulk sample accuracy, causing the network to favor dominant head categories while collapsing on rare tail classes. Introducing smoothed inverse class frequency weights ensured balanced gradient propagation across all 140 classes, directly addressing the macro-average Balanced Accuracy competition metric.",
        
        "<b>Multimodal Complementarity vs. Unimodal Limitations:</b> Architectural analysis of unimodal baselines highlights distinct failure modes: visual representations alone struggle with visually indistinguishable items sharing standardized packaging (e.g., book genres, electronics cables), whereas textual representations alone fail when product titles omit sensory attributes (e.g., apparel color, pattern, texture). Late-fusion concatenation allows both modalities to contribute complementary representations.",
        
        "<b>Mask-Aware Temporal Average Pooling:</b> Relying solely on the final hidden state of the bidirectional GRU exposes recurrent representations to sequence padding artifacts. Mask-aware temporal average pooling computes length-normalized representations strictly over valid non-padded tokens, preventing zero-padding tokens from diluting the 256-dimensional textual feature vector.",
        
        "<b>Backbone Selection & Parameter Constraint Compliance:</b> Standard heavy convolutional networks (e.g., ResNet-50 with ~25.6M parameters) violate the strict &lt; 15,000,000 parameter competition ceiling. MobileNetV2 with depthwise separable convolutions was chosen to provide high-capacity feature extraction at 2.22M parameters, ensuring the entire dual-branch system (4.86M parameters) stays comfortably within the budget with over 67% headroom."
    ]
    for ab in ablations:
        story.append(Paragraph(f"&bull; {ab}", ablation_bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Viva Voce Technical Defense: Tensor Dimension Tracing", h1_style))
    story.append(Paragraph(
        "Complete step-by-step tensor dimension trace to support the individual live code defense and layer modification questions:",
        body_style
    ))

    # Column widths: 32 + 128 + 92 + 82 + 190 = 524 pt (exact fit, no header wrapping on 'Step')
    viva_data = [
        [Paragraph("<b>Step</b>", table_cell_bold), Paragraph("<b>Operation / Code Layer</b>", table_cell_bold), Paragraph("<b>Input Shape</b>", table_cell_bold), Paragraph("<b>Output Shape</b>", table_cell_bold), Paragraph("<b>Theoretical & Mathematical Purpose</b>", table_cell_bold)],
        [
            Paragraph("1", table_cell), Paragraph("Vision Normalization", table_cell),
            Paragraph("Raw PIL Images", table_cell), Paragraph("[B, 3, 128, 128]", table_cell),
            Paragraph("RGB normalized by ImageNet mean & std", table_cell)
        ],
        [
            Paragraph("2", table_cell), Paragraph("MobileNetV2 Features", table_cell),
            Paragraph("[B, 3, 128, 128]", table_cell), Paragraph("[B, 1280, 4, 4]", table_cell),
            Paragraph("Depthwise separable convolutional feature extraction", table_cell)
        ],
        [
            Paragraph("3", table_cell), Paragraph("AdaptiveAvgPool2d + Flatten", table_cell),
            Paragraph("[B, 1280, 4, 4]", table_cell), Paragraph("[B, 1280]", table_cell),
            Paragraph("Spatial pooling to Global Visual Feature Vector", table_cell)
        ],
        [
            Paragraph("4", table_cell), Paragraph("Word Embedding Layer", table_cell),
            Paragraph("[B, 60] (token IDs)", table_cell), Paragraph("[B, 60, 128]", table_cell),
            Paragraph("Dense lookup mapping for 10K vocabulary tokens", table_cell)
        ],
        [
            Paragraph("5", table_cell), Paragraph("2-Layer Bidirectional GRU", table_cell),
            Paragraph("[B, 60, 128]", table_cell), Paragraph("[B, 60, 256]", table_cell),
            Paragraph("Bidirectional sequence context modeling (2 * 128)", table_cell)
        ],
        [
            Paragraph("6", table_cell), Paragraph("Mask-Aware Temporal Pool", table_cell),
            Paragraph("[B, 60, 256]", table_cell), Paragraph("[B, 256]", table_cell),
            Paragraph("Mask-Aware Temporal Average Pooling across tokens", table_cell)
        ],
        [
            Paragraph("7", table_cell), Paragraph("Late Multimodal Concatenation", table_cell),
            Paragraph("[B, 1280] + [B, 256]", table_cell), Paragraph("[B, 1536]", table_cell),
            Paragraph("Feature fusion: torch.cat((img_feat, text_feat), dim=1)", table_cell)
        ],
        [
            Paragraph("8", table_cell), Paragraph("Classifier Linear 1 + BN + ReLU", table_cell),
            Paragraph("[B, 1536]", table_cell), Paragraph("[B, 512]", table_cell),
            Paragraph("Multimodal projection, batch stabilization and activation", table_cell)
        ],
        [
            Paragraph("9", table_cell), Paragraph("Dropout(0.3) + Linear 2", table_cell),
            Paragraph("[B, 512]", table_cell), Paragraph("[B, 140]", table_cell),
            Paragraph("Regularization and 140 class logits for Weighted Cross-Entropy", table_cell)
        ]
    ]
    viva_table = Table(viva_data, colWidths=[32, 128, 92, 82, 190])
    viva_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor("#f0fdf4")),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#fefce8")),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(viva_table)
    story.append(Spacer(1, 10))

    # Section 8: Model Generalization & Safeguards
    story.append(Paragraph("8. Generalization Safeguards: Overfitting & Underfitting Prevention", h1_style))
    story.append(Paragraph(
        "To reduce overfitting risk and improve generalization, the pipeline employs systematic defensive mechanisms:",
        body_style
    ))
    story.append(Paragraph(
        "&bull; <b>1. Underfitting Prevention:</b> Pre-trained MobileNetV2 weights bootstrap visual features; a vocabulary capped at 10,000 tokens is constructed from the most frequent training-corpus words; and a Cosine Annealing schedule allows rapid initial convergence.",
        ablation_bullet_style
    ))
    story.append(Paragraph(
        "&bull; <b>2. Overfitting Prevention:</b> Training data augmentation (RandomCrop, HorizontalFlip, ColorJitter) prevents visual memorization; multi-stage Dropout (0.2 in RNN, 0.3 in classifier) prevents co-adaptation; AdamW weight decay (1e-4) penalizes excessive weights; and test predictions are strictly generated from the <b>best validation Balanced Accuracy checkpoint</b> (90.09%) rather than the final epoch.",
        ablation_bullet_style
    ))
    story.append(Spacer(1, 10))

    # Section 9: Kaggle Submission Verification & Integrity Checks
    story.append(Paragraph("9. Kaggle Submission Verification & Leaderboard Results", h1_style))
    story.append(Paragraph(
        "Final test predictions were exported to <code>submission.csv</code> with strict compliance: exactly 8,832 test rows matching test.csv, "
        "exactly 2 columns (<code>id</code>, <code>category</code>), zero null/NaN predictions, and validated against the competition sample submission schema. "
        "Upon submission to the official Kaggle evaluation server, the pipeline achieved a <b>Public Leaderboard Balanced Accuracy of 0.94999 (Rank #1)</b>, "
        "confirming that the dual-branch representations generalize robustly to unseen test products without overfitting or class collapse.",
        body_style
    ))

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Technical Report PDF successfully generated: {output_filename}")


if __name__ == "__main__":
    create_technical_report()
