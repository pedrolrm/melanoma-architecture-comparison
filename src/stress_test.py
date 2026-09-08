import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure paths to ensure reliable imports
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.mlp import MLP
from models.vgg import MelalomaVGG16


class CenterMask:
    """Masks the central region of the image (blacks out the lesion, leaving only skin/background)."""
    def __init__(self, y_min=56, y_max=168, x_min=56, x_max=168):
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, img):
        arr = np.array(img).copy()
        arr[self.y_min:self.y_max, self.x_min:self.x_max] = 0
        return Image.fromarray(arr)


class BorderMask:
    """Masks the image borders (blacks out background, leaving only the central lesion)."""
    def __init__(self, border_size=40):
        self.border_size = border_size

    def __call__(self, img):
        arr = np.array(img).copy()
        b = self.border_size
        arr[:b, :] = 0
        arr[-b:, :] = 0
        arr[:, :b] = 0
        arr[:, -b:] = 0
        return Image.fromarray(arr)


def get_stress_scenarios():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    scenarios = {
        "1. Original": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm,
        ]),
        "2. Translation (10% Shift)": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            norm,
        ]),
        "3. Rotation 90°": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            norm,
        ]),
        "4. Rotation 180°": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor(),
            norm,
        ]),
        "5. Horizontal Flip": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm,
        ]),
        "6. Vertical Flip": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            norm,
        ]),
        "7. Lesion Masked (Background Only)": transforms.Compose([
            transforms.Resize((224, 224)),
            CenterMask(),
            transforms.ToTensor(),
            norm,
        ]),
        "8. Background Masked (Lesion Only)": transforms.Compose([
            transforms.Resize((224, 224)),
            BorderMask(),
            transforms.ToTensor(),
            norm,
        ]),
    }
    return scenarios


def evaluate_scenario(model, loader, device):
    y_true, y_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds_prob = model(images)
            y_true.extend(labels.cpu().numpy().flatten().tolist())
            y_probs.extend(preds_prob.squeeze().cpu().numpy().flatten().tolist())

    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    y_preds = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_preds)
    try:
        auc_val = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_val = 0.5

    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = recall_score(y_true, y_preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)

    return {
        "acc": acc,
        "auc": auc_val,
        "sens": sens,
        "spec": spec,
        "prec": prec,
        "f1": f1,
    }


def run_stress_tests(test_dir, models_info, output_plot=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {device}")

    loaded_models = []
    for name, model_instance, weights_path in models_info:
        if not os.path.exists(weights_path):
            print(f"[-] Warning: Weights for '{name}' at {weights_path} not found. Skipping...")
            continue
        print(f"[+] Loading model '{name}' from {weights_path}...")
        model_instance.load_state_dict(torch.load(weights_path, map_location=device))
        model_instance.to(device)
        model_instance.eval()
        loaded_models.append((name, model_instance))

    if not loaded_models:
        print("[-] No models could be loaded. Aborting.")
        return {}

    scenarios = get_stress_scenarios()
    results = {m_name: {} for m_name, _ in loaded_models}

    print("\n" + "=" * 98)
    print(f"{'Scenario':<34} | {'Model':<22} | {'Accuracy':<10} | {'AUC':<10} | {'Sens.':<8} | {'Spec.':<8}")
    print("=" * 98)

    for sc_name, transform_fn in scenarios.items():
        # Fix seed to ensure identical random transformations across models
        torch.manual_seed(42)
        np.random.seed(42)

        dataset = datasets.ImageFolder(root=test_dir, transform=transform_fn)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

        for m_name, model in loaded_models:
            metrics = evaluate_scenario(model, loader, device)
            results[m_name][sc_name] = metrics
            print(
                f"{sc_name:<34} | {m_name:<22} | {metrics['acc']:.4f}     | {metrics['auc']:.4f}     | "
                f"{metrics['sens']:.4f}   | {metrics['spec']:.4f}"
            )
        print("-" * 98)

    if output_plot:
        plot_stress_comparison(results, output_plot)

    return results


def plot_stress_comparison(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = list(results.keys())
    if not models:
        return

    scenarios = list(results[models[0]].keys())
    x = np.arange(len(scenarios))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, m_name in enumerate(models):
        accs = [results[m_name][sc]["acc"] * 100 for sc in scenarios]
        offset = (idx - len(models) / 2 + 0.5) * width
        rects = ax.bar(x + offset, accs, width, label=m_name)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Spatial Robustness & Perturbation Stress Test", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Robustness comparison plot saved to: {save_path}")


def main():
    test_dir = os.path.join(ROOT_DIR, "data", "raw", "test")
    if not os.path.exists(test_dir):
        print(f"[-] Test directory not found: {test_dir}")
        return

    models_info = [
        ("MLP", MLP(), os.path.join(ROOT_DIR, "models", "melhor_mlp.pth")),
        ("VGG-16 (Scratch)", MelalomaVGG16(pretrained=False), os.path.join(ROOT_DIR, "models", "melhor_vgg_scratch.pth")),
        ("VGG-16 (Pretrained)", MelalomaVGG16(pretrained=True), os.path.join(ROOT_DIR, "models", "melhor_vgg_pretrained.pth")),
    ]

    plot_path = os.path.join(ROOT_DIR, "plots", "stress_test_comparison.png")
    run_stress_tests(test_dir, models_info, output_plot=plot_path)


if __name__ == "__main__":
    main()
