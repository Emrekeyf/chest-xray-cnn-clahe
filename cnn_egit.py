
from pathlib import Path
import csv
import random
import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Genel ayarlar
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hazir_veriseti"
OUT = ROOT / "cnn_sonuclar"
OUT.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 15
BATCH_SIZE = 32
LR = 0.0005
SIZE = 64
CLASSES = ["COVID", "Normal"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Kullanilan cihaz:", DEVICE)


# Veri yukleme ve goruntu isleme
class XRayDataset(Dataset):
    def __init__(self, split, clahe=False):
        self.items = []
        self.clahe = clahe

        for label, cls in enumerate(CLASSES):
            folder = DATA / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Veri klasoru bulunamadi: {folder}\n"
                    "Once veri_hazirla.py komutunu calistirin veya hazir_veriseti klasorunu yerlestirin."
                )

            for path in sorted(folder.glob("*.png")):
                self.items.append((path, label))

        if not self.items:
            raise RuntimeError(
                f"Goruntu bulunamadi: {split}"
            )

        self.processor = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]

        # Windows'ta Turkce karakter iceren yollarda cv2.imread hata verebilir.
        # numpy.fromfile + cv2.imdecode Unicode dosya yollarini destekler.
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError, cv2.error) as exc:
            raise RuntimeError(f"Goruntu okunamadi: {path}") from exc

        if img is None:
            raise RuntimeError(f"Goruntu okunamadi: {path}")

        img = cv2.resize(img, (SIZE, SIZE))

        if self.clahe:
            img = self.processor.apply(img)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        tensor = torch.from_numpy(
            img.copy()
        ).unsqueeze(0)

        return tensor, label


# CNN mimarisi
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Linear(
            16 * 16 * 16, 2
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            predictions.extend(pred.cpu().tolist())
            targets.extend(y.cpu().tolist())

    return (
        total_loss / total,
        correct / total,
        targets,
        predictions
    )


def run_experiment(name, use_clahe):
    print(f"\n===== {name} =====")

    # Her deneyde ayni baslangic
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_ds = XRayDataset("train", use_clahe)
    val_ds = XRayDataset("val", use_clahe)
    test_ds = XRayDataset("test", use_clahe)

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion
        )

        history.append([
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc
        ])

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_state)

    torch.save(
        best_state,
        OUT / f"{name}_model.pth"
    )

    with open(
        OUT / f"{name}_history.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ])
        writer.writerows(history)

    # Test sadece egitim bittikten sonra
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion
    )

    tp = sum(
        a == 0 and b == 0
        for a, b in zip(y_true, y_pred)
    )
    fn = sum(
        a == 0 and b == 1
        for a, b in zip(y_true, y_pred)
    )
    fp = sum(
        a == 1 and b == 0
        for a, b in zip(y_true, y_pred)
    )
    tn = sum(
        a == 1 and b == 1
        for a, b in zip(y_true, y_pred)
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    with open(
        OUT / f"{name}_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ["accuracy", test_acc],
            ["precision_COVID", precision],
            ["recall_COVID", recall],
            ["f1_COVID", f1],
            ["TP", tp],
            ["FN", fn],
            ["FP", fp],
            ["TN", tn],
            ["test_loss", test_loss]
        ])

    print(f"\n{name} TEST SONUCLARI")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")

    return history


if __name__ == "__main__":
    raw_history = run_experiment("HAM", False)
    clahe_history = run_experiment("CLAHE", True)

    # Grafikler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, history in [
        ("HAM", raw_history),
        ("CLAHE", clahe_history)
    ]:
        epochs = [row[0] for row in history]
        train_acc = [row[2] for row in history]
        val_acc = [row[4] for row in history]
        val_loss = [row[3] for row in history]

        axes[0].plot(
            epochs, val_acc, label=name
        )
        axes[1].plot(
            epochs, val_loss, label=name
        )

    axes[0].set_title("Dogrulama Dogrulugu")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")

    axes[1].set_title("Dogrulama Kaybi")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        OUT / "karsilastirma.png",
        dpi=200
    )
    plt.close(fig)

    print("\nTum deneyler tamamlandi!")
    print("Sonuc klasoru:", OUT)
