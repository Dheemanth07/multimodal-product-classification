"""
Generates an ultra-clean, publication-grade architectural system diagram for
BCS714A Multimodal Product Classification.

Design Highlights:
- Vertically extended layout for maximum clarity and legible typography
- Perfectly aligned 7-column pipeline grid with balanced aspect ratio
- Symmetrical branch routing into the late fusion node
- Crystal-clear tensor shapes at every transition
- Technically accurate Loss (Weighted Cross-Entropy) and Metric (Balanced Accuracy)
- Structured parameter budget badge with exact parameter counts (4,858,252 < 15,000,000)
- Professional academic styling, zero AI clip-art/fluff
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_architecture_diagram(output_path="architecture_diagram.png"):
    # Figsize (18.5, 12.0) provides generous vertical height and clarity
    fig, ax = plt.subplots(figsize=(18.5, 12.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # =========================================================================
    # 1. HEADER & METADATA
    # =========================================================================
    ax.text(9.25, 11.52, "Dual-Branch Multimodal Fusion Architecture", 
            ha="center", va="center", fontsize=17.5, fontweight="bold", color="#0f172a")
    ax.text(9.25, 11.10, "Computer Vision (CNN: MobileNetV2) + Natural Language Processing (Bi-GRU) Late Fusion Network", 
            ha="center", va="center", fontsize=11.2, fontweight="medium", color="#334155")
    ax.text(9.25, 10.74, "Parameter Constraint: < 15,000,000 Parameters  |  Evaluation Metric: Balanced Accuracy (Macro Recall)", 
            ha="center", va="center", fontsize=9.8, fontstyle="italic", color="#64748b")

    # Dividing rule below header
    ax.plot([0.6, 17.9], [10.48, 10.48], color="#e2e8f0", lw=1.2)

    # =========================================================================
    # 2. HELPER FUNCTIONS
    # =========================================================================
    def draw_card(x, y, w, h, header, subtext, tensor_text, bg_color, border_color, tag_bg=None):
        """Draws a card with rounded corners, centered text, and tensor dimension tag."""
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=3
        )
        ax.add_patch(box)

        # Header title
        ax.text(x + w / 2, y + h * 0.78, header, ha="center", va="center", 
                fontsize=10.5, fontweight="bold", color="#0f172a", zorder=4)

        # Subtitle / description
        if subtext:
            ax.text(x + w / 2, y + h * 0.48, subtext, ha="center", va="center", 
                    fontsize=8.6, color="#334155", linespacing=1.30, zorder=4)

        # Tensor shape pill badge
        if tensor_text:
            pill_w = min(w * 0.88, 1.95)
            pill_h = 0.36
            pill_x = x + (w - pill_w) / 2
            pill_y = y + h * 0.10
            pill = patches.FancyBboxPatch(
                (pill_x, pill_y), pill_w, pill_h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#ffffff" if tag_bg is None else tag_bg,
                edgecolor=border_color, linewidth=1.0, zorder=4
            )
            ax.add_patch(pill)
            ax.text(pill_x + pill_w / 2, pill_y + pill_h / 2, tensor_text, 
                    ha="center", va="center", fontsize=8.2, fontweight="bold", 
                    color=border_color, family="monospace", zorder=5)

    def draw_arrow(x1, y1, x2, y2, color="#475569", lw=1.8):
        """Draws a crisp connector arrow."""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14),
            zorder=2
        )

    # =========================================================================
    # 3. CONTAINER BACKGROUND REGIONS
    # =========================================================================
    # Vision Branch Container (Top Left)
    v_cont = patches.FancyBboxPatch(
        (0.6, 6.30), 9.35, 3.95,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#f0f7ff", edgecolor="#93c5fd", linewidth=1.1, linestyle="--", zorder=1
    )
    ax.add_patch(v_cont)
    ax.text(0.85, 9.88, "VISION BRANCH (Module 3)", fontsize=9.4, fontweight="bold", color="#1d4ed8", zorder=2)

    # NLP Branch Container (Bottom Left)
    n_cont = patches.FancyBboxPatch(
        (0.6, 1.85), 9.35, 3.95,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#fef6f6", edgecolor="#fca5a5", linewidth=1.1, linestyle="--", zorder=1
    )
    ax.add_patch(n_cont)
    ax.text(0.85, 5.43, "NLP BRANCH (Modules 4 & 5)", fontsize=9.4, fontweight="bold", color="#b91c1c", zorder=2)

    # Fusion & Classification Container (Right)
    f_cont = patches.FancyBboxPatch(
        (10.25, 1.85), 7.65, 8.40,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#f6fbf8", edgecolor="#86efac", linewidth=1.1, linestyle="--", zorder=1
    )
    ax.add_patch(f_cont)
    ax.text(10.50, 9.88, "MULTIMODAL FUSION & CLASSIFICATION HEAD", fontsize=9.4, fontweight="bold", color="#15803d", zorder=2)

    # =========================================================================
    # 4. VISION BRANCH NODES (Center y = 8.00)
    # =========================================================================
    y_v = 8.00
    h_card = 2.50
    y_vc = y_v - h_card / 2

    # V1: Product Image Input
    draw_card(0.85, y_vc, 1.90, h_card, 
              "Product Image", "RGB Image Tensor\n(128x128 Resolution)", "[B, 3, 128, 128]", 
              "#ffffff", "#2563eb")

    draw_arrow(2.75, y_v, 3.10, y_v, color="#2563eb")

    # V2: MobileNetV2 Backbone
    draw_card(3.10, y_vc, 2.20, h_card, 
              "Vision Backbone", "MobileNetV2 Features\n17 Inverted Residuals", "[B, 1280, 4, 4]", 
              "#ffffff", "#1d4ed8")

    draw_arrow(5.30, y_v, 5.65, y_v, color="#1d4ed8")

    # V3: Spatial Reduction
    draw_card(5.65, y_vc, 1.90, h_card, 
              "Spatial Pooling", "AdaptiveAvgPool2d\n+ Flatten Layer", "[B, 1280, 1, 1]", 
              "#ffffff", "#1e40af")

    draw_arrow(7.55, y_v, 7.90, y_v, color="#1e40af")

    # V4: Global Visual Feature Vector
    draw_card(7.90, y_vc, 1.80, h_card, 
              "Image Vector", "Global Visual\nFeature Vector", "[B, 1280]", 
              "#eff6ff", "#1e3a8a", tag_bg="#dbeafe")

    # =========================================================================
    # 5. NLP BRANCH NODES (Center y = 3.55)
    # =========================================================================
    y_n = 3.55
    y_nc = y_n - h_card / 2

    # N1: Product Text Input
    draw_card(0.85, y_nc, 1.90, h_card, 
              "Product Text", "Title + Description\nCleaned & Tokenized", "[B, 60]", 
              "#ffffff", "#dc2626")

    draw_arrow(2.75, y_n, 3.10, y_n, color="#dc2626")

    # N2: Word Embedding
    draw_card(3.10, y_nc, 2.20, h_card, 
              "Word Embedding", "Trainable Lookup\n10,000 Vocab x 128 Dim", "[B, 60, 128]", 
              "#ffffff", "#b91c1c")

    draw_arrow(5.30, y_n, 5.65, y_n, color="#b91c1c")

    # N3: 2-Layer Bi-GRU
    draw_card(5.65, y_nc, 1.90, h_card, 
              "Bidirectional GRU", "2 Layers, Hidden=128\nDropout=0.2 (2x128)", "[B, 60, 256]", 
              "#ffffff", "#991b1b")

    draw_arrow(7.55, y_n, 7.90, y_n, color="#991b1b")

    # N4: Mask-Aware Temporal Average Pooling
    draw_card(7.90, y_nc, 1.80, h_card, 
              "Text Vector", "Mask-Aware Temporal\nAverage Pooling", "[B, 256]", 
              "#fef2f2", "#7f1d1d", tag_bg="#fee2e2")

    # =========================================================================
    # 6. MULTIMODAL LATE FUSION & CLASSIFIER (Center y = 5.775)
    # =========================================================================
    y_f = (y_v + y_n) / 2  # Exactly 5.775: perfect mathematical symmetry

    # Symmetrical routing arrows from V4 and N4 into Late Fusion
    ax.annotate(
        "", xy=(10.55, y_f + 0.70), xytext=(9.70, y_v),
        arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=2.2, mutation_scale=16),
        zorder=2
    )
    ax.text(10.05, 7.25, "1,280 features", fontsize=8.4, fontweight="bold", color="#0369a1", rotation=-34, zorder=5)

    ax.annotate(
        "", xy=(10.55, y_f - 0.70), xytext=(9.70, y_n),
        arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=2.2, mutation_scale=16),
        zorder=2
    )
    ax.text(10.05, 4.30, "256 features", fontsize=8.4, fontweight="bold", color="#0369a1", rotation=34, zorder=5)

    # F1: Late Fusion Node (Concatenation)
    h_fuse = 3.20
    draw_card(10.55, y_f - h_fuse / 2, 2.05, h_fuse, 
              "Late Fusion", "torch.cat(dim=1)\n(1280 + 256)\nMultimodal Joint", "[B, 1536]", 
              "#ecfdf5", "#059669", tag_bg="#d1fae5")

    draw_arrow(12.60, y_f, 12.95, y_f, color="#059669", lw=2.0)

    # F2: Dense Feature Projection & Regularization
    h_proj = 3.50
    draw_card(12.95, y_f - h_proj / 2, 2.25, h_proj, 
              "Dense Projection", "Linear(1536 -> 512)\nBatchNorm1d + ReLU\nDropout(p = 0.3)", "[B, 512]", 
              "#ffffff", "#047857")

    draw_arrow(15.20, y_f, 15.55, y_f, color="#047857", lw=2.0)

    # F3: Final Output Classification Head
    draw_card(15.55, y_f - h_proj / 2, 2.15, h_proj, 
              "Classification Head", "Linear(512 -> 140)\nWeighted Cross-Entropy\nEval: Balanced Accuracy", "[B, 140 Logits]", 
              "#fefce8", "#b45309", tag_bg="#fef08a")

    # =========================================================================
    # 7. BOTTOM AUDIT & SPECIFICATION PANEL (EXACT PARAMETER BUDGET)
    # =========================================================================
    audit_bg = patches.FancyBboxPatch(
        (0.6, 0.25), 17.30, 1.30,
        boxstyle="round,pad=0.08,rounding_size=0.10",
        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.2, zorder=2
    )
    ax.add_patch(audit_bg)

    specs = [
        ("Vision Backbone", "MobileNetV2 Feature Extractor\nExact Parameters: 2,223,872 (45.8%)", "#1e40af"),
        ("NLP Backbone", "Embed + 2-Layer Bi-GRU\nExact Parameters: 1,774,592 (36.5%)", "#b91c1c"),
        ("Fusion Head", "Dense MLP (1536 -> 512 -> 140)\nExact Parameters: 859,788 (17.7%)", "#047857"),
        ("Parameter Budget", "Total Parameters: 4,858,252 (Exact)\nConstraint: < 15M  [Headroom: 10,141,748]", "#b45309"),
    ]

    col_w = 4.25
    for i, (title, detail, border_c) in enumerate(specs):
        bx = 0.75 + i * col_w
        by = 0.35
        badge = patches.FancyBboxPatch(
            (bx, by), 4.08, 1.08,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor="#f8fafc", edgecolor=border_c, linewidth=1.0, zorder=3
        )
        ax.add_patch(badge)
        ax.text(bx + 2.04, by + 0.73, title, ha="center", va="center", 
                fontsize=9.0, fontweight="bold", color=border_c, zorder=4)
        ax.text(bx + 2.04, by + 0.35, detail, ha="center", va="center", 
                fontsize=7.6, color="#334155", linespacing=1.28, family="sans-serif", zorder=4)

    # Coordinate limits
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, 12.0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Extended architecture diagram exported to: {output_path}")


if __name__ == "__main__":
    draw_architecture_diagram("architecture_diagram.png")
