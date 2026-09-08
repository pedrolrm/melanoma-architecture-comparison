import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.mlp import MLP
from models.vgg import MelalomaVGG16

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_single_model(model_name, model, train_loader, val_loader, device, save_path, epochs=50, patience=5, lr=1e-4):
    print(f"\n==========================================")
    print(f"Iniciando treino: {model_name}")
    print(f"==========================================")
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    usa_cuda = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=usa_cuda)

    melhor_val_loss = float('inf')
    epochs_sem_melhora = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        acertos_treino = 0

        for imagens, labels in train_loader:
            imagens = imagens.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=usa_cuda):
                preds_prob = model(imagens)
                loss = criterion(preds_prob, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imagens.size(0)
            preds = (preds_prob >= 0.5).float()
            acertos_treino += (preds == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = acertos_treino / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        acertos_val = 0

        with torch.no_grad():
            for imagens, labels in val_loader:
                imagens = imagens.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                with autocast(device_type=device.type, enabled=usa_cuda):
                    preds_prob = model(imagens)
                    loss = criterion(preds_prob, labels)

                val_loss += loss.item() * imagens.size(0)
                preds = (preds_prob >= 0.5).float()
                acertos_val += (preds == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = acertos_val / len(val_loader.dataset)

        print(f"[{model_name}] Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < melhor_val_loss:
            melhor_val_loss = val_loss
            epochs_sem_melhora = 0
            torch.save(model.state_dict(), save_path)
            print(f" -> Modelo salvo em: {save_path}")
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= patience:
                print(f"Early stopping ativado na época {epoch+1}.")
                break

def main():
    seed = 42
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo detectado: {device}")

    batch_size = 32
    num_workers = 2
    epochs = 50
    patience = 5
    learning_rate = 1e-4

    transformacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    caminho_dados_treino = os.path.join('data', 'raw', 'train')
    dataset_completo = datasets.ImageFolder(root=caminho_dados_treino, transform=transformacao)

    g = torch.Generator()
    g.manual_seed(seed)

    tamanho_val = int(0.1 * len(dataset_completo))
    tamanho_treino = len(dataset_completo) - tamanho_val
    train_dataset, val_dataset = random_split(dataset_completo, [tamanho_treino, tamanho_val], generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    configs = [
        ("MLP", MLP, {}, "models/melhor_mlp.pth"),
        ("VGG16_Scratch", MelalomaVGG16, {"pretrained": False}, "models/melhor_vgg_scratch.pth"),
        ("VGG16_Pretrained", MelalomaVGG16, {"pretrained": True}, "models/melhor_vgg_pretrained.pth")
    ]

    for model_name, model_class, model_kwargs, save_path in configs:
        set_seed(seed)
        model_instance = model_class(**model_kwargs)
        train_single_model(
            model_name=model_name,
            model=model_instance,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_path=save_path,
            epochs=epochs,
            patience=patience,
            lr=learning_rate
        )

if __name__ == '__main__':
    main()