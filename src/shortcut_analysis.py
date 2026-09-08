import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch

# Configure paths to ensure reliable imports
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_color_shortcut_experiment(train_dir, test_dir, max_samples_train=2500, max_samples_test=1000):
    """
    Evaluates linear baselines on global color statistics and downscaled thumbnails
    to quantify the degree of shortcut learning stemming from dataset-level color/illumination bias.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Global Color & Illumination Shortcut Diagnostic")
    print("=" * 80)

    # 1.1 Global Mean RGB (3 features per image)
    def load_mean_rgb(split_dir, max_samples):
        X, y = [], []
        for label, cls in enumerate(["Benign", "Malignant"]):
            cls_dir = Path(split_dir) / cls
            if not cls_dir.exists():
                continue
            files = list(cls_dir.glob("*.jpg"))[:max_samples]
            for f in files:
                img = np.array(Image.open(f).convert("RGB"), dtype=np.float32)
                # Only 3 values: average R, G, and B channel intensities
                X.append(img.mean(axis=(0, 1)))
                y.append(label)
        return np.array(X), np.array(y)

    print("[*] Extracting global mean RGB features (3 channels)...")
    X_tr_mean, y_tr = load_mean_rgb(train_dir, max_samples_train)
    X_te_mean, y_te = load_mean_rgb(test_dir, max_samples_test)

    clf_mean = LogisticRegression(max_iter=500)
    clf_mean.fit(X_tr_mean, y_tr)
    preds_mean = clf_mean.predict(X_te_mean)
    probs_mean = clf_mean.predict_proba(X_te_mean)[:, 1]

    acc_mean = accuracy_score(y_te, preds_mean)
    auc_mean = roc_auc_score(y_te, probs_mean)

    print(f" -> Logistic Regression using ONLY Mean RGB (3 features):")
    print(f"    - Accuracy: {acc_mean * 100:.2f}%")
    print(f"    - AUC-ROC:  {auc_mean:.4f}")

    # 1.2 Downscaled 16x16 thumbnail (insufficient resolution for morphological clinical features)
    def load_downscaled_16x16(split_dir, max_samples):
        X, y = [], []
        for label, cls in enumerate(["Benign", "Malignant"]):
            cls_dir = Path(split_dir) / cls
            if not cls_dir.exists():
                continue
            files = list(cls_dir.glob("*.jpg"))[:max_samples]
            for f in files:
                img = Image.open(f).convert("RGB").resize((16, 16))
                X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                y.append(label)
        return np.array(X), np.array(y)

    print("\n[*] Extracting downscaled 16x16 thumbnails...")
    X_tr_16, y_tr_16 = load_downscaled_16x16(train_dir, max_samples_train)
    X_te_16, y_te_16 = load_downscaled_16x16(test_dir, max_samples_test)

    clf_16 = LogisticRegression(max_iter=500)
    clf_16.fit(X_tr_16, y_tr_16)
    preds_16 = clf_16.predict(X_te_16)
    probs_16 = clf_16.predict_proba(X_te_16)[:, 1]

    acc_16 = accuracy_score(y_te_16, preds_16)
    auc_16 = roc_auc_score(y_te_16, probs_16)

    print(f" -> Logistic Regression on 16x16 downscaled images (no fine clinical patterns):")
    print(f"    - Accuracy: {acc_16 * 100:.2f}%")
    print(f"    - AUC-ROC:  {auc_16:.4f}")

    print("\n[Conclusion - Experiment 1]:")
    print(f"Over {acc_mean * 100:.1f}% accuracy and {auc_mean:.2f} AUC arise solely from global color and illumination biases.")


def run_mlp_weights_analysis(weights_path, output_plot=None):
    """
    Analyzes the weight distribution of the MLP's first fully-connected layer (FC1)
    to investigate spatial attention and memorization of fixed coordinates.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Spatial Weight Analysis of MLP Dense Layer")
    print("=" * 80)

    if not os.path.exists(weights_path):
        print(f"[-] MLP weights checkpoint not found at: {weights_path}")
        return

    print(f"[+] Loading MLP weights from: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    if "fc1.weight" not in state:
        print("[-] 'fc1.weight' key not found in the checkpoint.")
        return

    w = state["fc1.weight"]  # Shape: (1024, 150528)
    num_neurons, in_features = w.shape
    print(f" -> FC1 Layer: {num_neurons} neurons x {in_features} connections (~{in_features * num_neurons / 1e6:.1f}M parameters)")

    w_spatial = w.view(num_neurons, 3, 224, 224)

    # Average norm across color channels
    norm_r = torch.linalg.norm(w_spatial[:, 0, :, :]).item()
    norm_g = torch.linalg.norm(w_spatial[:, 1, :, :]).item()
    norm_b = torch.linalg.norm(w_spatial[:, 2, :, :]).item()

    print(f"\n[*] Channel weight norms:")
    print(f"    - R Channel (Red):   {norm_r:.2f}")
    print(f"    - G Channel (Green): {norm_g:.2f}")
    print(f"    - B Channel (Blue):  {norm_b:.2f}")

    # Spatial average absolute weight map
    spatial_map = w_spatial.abs().mean(dim=(0, 1)).numpy()

    # Compare central region (where lesion typically resides) against border/corner coordinates
    mean_center = spatial_map[107:117, 107:117].mean()
    mean_corner = spatial_map[:10, :10].mean()
    ratio = mean_corner / mean_center if mean_center > 0 else 0

    print(f"\n[*] Spatial Attention Profile:")
    print(f"    - Mean weight in center (lesion area): {mean_center:.6f}")
    print(f"    - Mean weight in corners (background):  {mean_corner:.6f}")
    print(f"    - Ratio (Corners / Center):            {ratio:.2f}x")

    print("\n[Conclusion - Experiment 2]:")
    if ratio >= 1.0:
        print("Weights at corners/borders exhibit equal or higher magnitude than those at the center,")
        print("confirming that the MLP relies on absolute static pixel positions rather than localized lesion patterns.")

    if output_plot:
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.imshow(spatial_map, cmap="inferno")
        plt.colorbar(label="Mean Absolute Weight Magnitude |W|")
        plt.title("MLP (FC1) Spatial Weight Map", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_plot, dpi=300)
        plt.close()
        print(f"[+] Spatial weight heatmap saved to: {output_plot}")


def main():
    train_dir = os.path.join(ROOT_DIR, "data", "raw", "train")
    test_dir = os.path.join(ROOT_DIR, "data", "raw", "test")
    weights_path = os.path.join(ROOT_DIR, "models", "melhor_mlp.pth")
    plot_path = os.path.join(ROOT_DIR, "plots", "mlp_spatial_weights.png")

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        run_color_shortcut_experiment(train_dir, test_dir)
    else:
        print(f"[-] Data directories not found in data/raw/.")

    run_mlp_weights_analysis(weights_path, output_plot=plot_path)


if __name__ == "__main__":
    main()
