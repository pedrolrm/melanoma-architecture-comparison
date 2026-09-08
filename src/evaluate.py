import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Suporta execução tanto da raiz do repositório quanto de dentro de src/
try:
    from models.mlp import MLP
    from models.vgg import MelalomaVGG16
except ImportError:
    from mlp import MLP
    from vgg import MelalomaVGG16

def evaluate_models(models_info, test_loader, device):
    results = {}

    for name, model_instance, weights_path in models_info:
        if not os.path.exists(weights_path):
            print(f"[-] Warning: Weights for '{name}' at {weights_path} were not found. Skipping...")
            continue

        print(f"[+] Evaluating: {name}...")
        model_instance.load_state_dict(torch.load(weights_path, map_location=device))
        model_instance.to(device)
        model_instance.eval()

        y_true = []
        y_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                preds_prob = model_instance(images)

                y_true.extend(labels.cpu().numpy().flatten().tolist())
                y_probs.extend(preds_prob.squeeze().cpu().numpy().flatten().tolist())

        y_true = np.array(y_true, dtype=int)
        y_probs = np.array(y_probs, dtype=float)
        y_preds = (y_probs >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_preds)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            "y_true": y_true,
            "y_probs": y_probs,
            "y_preds": y_preds,
            "acc": accuracy_score(y_true, y_preds),
            "sens": recall_score(y_true, y_preds, pos_label=1, zero_division=0),
            "spec": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "prec": precision_score(y_true, y_preds, zero_division=0),
            "f1": f1_score(y_true, y_preds, zero_division=0),
            "auc": roc_auc_score(y_true, y_probs),
            "cm": cm
        }

    return results

def print_metrics_table(results):
    print("\n" + "=" * 80)
    print(f"{'Model':<22} | {'Acc':<7} | {'Sens':<7} | {'Spec':<7} | {'Prec':<7} | {'F1':<7} | {'AUC':<7}")
    print("=" * 80)
    for name, metrics in results.items():
        print(f"{name:<22} | {metrics['acc']:.4f}  | {metrics['sens']:.4f}  | {metrics['spec']:.4f}  | {metrics['prec']:.4f}  | {metrics['f1']:.4f}  | {metrics['auc']:.4f}")
    print("=" * 80 + "\n")

def plot_all_figures(results, class_names):
    os.makedirs('plots', exist_ok=True)
    num_models = len(results)
    if num_models == 0:
        print("No results to plot.")
        return

    display_classes = [c.capitalize() for c in class_names]

    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))
    if num_models == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, results.items()):
        cm = metrics["cm"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=display_classes, yticklabels=display_classes)
        ax.set_title(f'Confusion Matrix\n{name}', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig('plots/matrizes_confusao.png', dpi=300)
    plt.show()

    # 2. Comparative ROC Curves
    plt.figure(figsize=(8, 6))
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["y_probs"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Comparative ROC Curves', fontsize=13)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/curvas_roc.png', dpi=300)
    plt.show()

    # 3. Comparative Bar Chart of Metrics
    metric_keys = ['acc', 'sens', 'spec', 'prec', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC']
    
    x = np.arange(len(metric_keys))
    width = 0.8 / num_models

    plt.figure(figsize=(10, 6))
    for i, (name, metrics) in enumerate(results.items()):
        values = [metrics[k] for k in metric_keys]
        plt.bar(x + (i - num_models / 2 + 0.5) * width, values, width, label=name)

    plt.xticks(x, metric_labels, fontsize=11)
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=11)
    plt.title('Overall Performance Comparison on Test Set', fontsize=13)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/comparativo_metricas.png', dpi=300)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}")

    transformacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    caminho_teste = os.path.join('data', 'raw', 'test')
    dataset_teste = datasets.ImageFolder(root=caminho_teste, transform=transformacao)
    test_loader = DataLoader(dataset_teste, batch_size=32, shuffle=False, num_workers=2)

    class_names = [k for k, _ in sorted(dataset_teste.class_to_idx.items(), key=lambda item: item[1])]

    models_info = [
        ("MLP", MLP(), "models/melhor_mlp.pth"),
        ("VGG-16 (Scratch)", MelalomaVGG16(pretrained=False), "models/melhor_vgg_scratch.pth"),
        ("VGG-16 (Pretrained)", MelalomaVGG16(pretrained=True), "models/melhor_vgg_pretrained.pth")
    ]

    results = evaluate_models(models_info, test_loader, device)
    print_metrics_table(results)
    plot_all_figures(results, class_names)

if __name__ == '__main__':
    main()