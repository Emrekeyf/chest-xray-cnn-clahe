
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

# Proje ve sonuc klasoru
ROOT = Path(__file__).resolve().parent
SONUC = ROOT / "cnn_sonuclar"


def metrik_oku(dosya):
    with open(dosya, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            row["metric"]: float(row["value"])
            for row in reader
        }


def confusion_matrix_olustur(model_adi):
    dosya = SONUC / f"{model_adi}_metrics.csv"
    if not dosya.exists():
        raise FileNotFoundError(
            f"Sonuc dosyasi bulunamadi: {dosya}\n"
            "Once cnn_egit.py dosyasini calistirin."
        )
    metrikler = metrik_oku(dosya)

    tp = int(metrikler["TP"])
    fn = int(metrikler["FN"])
    fp = int(metrikler["FP"])
    tn = int(metrikler["TN"])

    matris = np.array([
        [tp, fn],
        [fp, tn]
    ])

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(matris, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["COVID", "Normal"])
    ax.set_yticklabels(["COVID", "Normal"])

    ax.set_xlabel("Tahmin Edilen Sinif")
    ax.set_ylabel("Gercek Sinif")
    ax.set_title(f"{model_adi} CNN - Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                str(matris[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                color=(
                    "white"
                    if matris[i, j] > matris.max() / 2
                    else "black"
                )
            )

    fig.tight_layout()

    kayit = SONUC / f"{model_adi}_confusion_matrix.png"
    fig.savefig(kayit, dpi=300)
    plt.close(fig)

    print(f"Olusturuldu: {kayit.name}")


if __name__ == "__main__":
    confusion_matrix_olustur("HAM")
    confusion_matrix_olustur("CLAHE")
    print("\nConfusion Matrix gorselleri hazir!")
