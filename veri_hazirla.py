"""COVID / Normal goruntulerinden sabit tohumla train/val/test ayirimi yapar.

Kaynak: COVID-19_Radiography_Dataset/{COVID,Normal}/images/*.png
Hedef : hazir_veriseti/{train,val,test}/{COVID,Normal}/*.png

Mevcut bir hazir_veriseti klasorunu degistirmemek icin guvenlik kontrolu vardir.
"""

from pathlib import Path
import argparse
import random
import shutil

ANA_KLASOR = Path(__file__).resolve().parent
KAYNAK = ANA_KLASOR / "COVID-19_Radiography_Dataset"
HEDEF = ANA_KLASOR / "hazir_veriseti"
SINIFLAR = ["COVID", "Normal"]
GORUNTU_SAYISI = 3616
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="COVID/Normal veri bolme")
    parser.add_argument("--images-per-class", type=int, default=GORUNTU_SAYISI,
                        help="Sinif basina kullanilacak goruntu sayisi (varsayilan: 3616)")
    args = parser.parse_args()

    if args.images_per_class < 3:
        parser.error("En az 3 goruntu/sinif gereklidir.")

    # Daha once hazirlanan veriler sessizce karismasin veya degismesin.
    if HEDEF.exists() and any(HEDEF.iterdir()):
        raise FileExistsError(
            f"Hedef klasor zaten dolu: {HEDEF}\n"
            "Eski bolunmeyi koruyun. Yeni bolunme icin klasoru manuel olarak "
            "yedekleyip farkli bir adla tasiyin."
        )

    random.seed(SEED)
    plan = []
    for sinif in SINIFLAR:
        kaynak_klasor = KAYNAK / sinif / "images"
        if not kaynak_klasor.is_dir():
            raise FileNotFoundError(
                f"Veri kaynagi bulunamadi: {kaynak_klasor}"
            )
        goruntuler = sorted(kaynak_klasor.glob("*.png"))
        if len(goruntuler) < args.images_per_class:
            raise ValueError(
                f"{sinif}: Yeterli goruntu yok! Bulunan: {len(goruntuler)}, "
                f"istenen: {args.images_per_class}"
            )

        # Orijinal kodun secim sirasi ve tohum degeri korunuyor.
        random.shuffle(goruntuler)
        secilen = goruntuler[:args.images_per_class]
        egitim_sayisi = int(args.images_per_class * 0.70)
        dogrulama_sayisi = int(args.images_per_class * 0.15)
        bolumler = {
            "train": secilen[:egitim_sayisi],
            "val": secilen[egitim_sayisi:egitim_sayisi + dogrulama_sayisi],
            "test": secilen[egitim_sayisi + dogrulama_sayisi:]
        }
        plan.append((sinif, bolumler))

    for sinif, bolumler in plan:
        for bolum, dosyalar in bolumler.items():
            hedef_klasor = HEDEF / bolum / sinif
            hedef_klasor.mkdir(parents=True, exist_ok=True)
            for dosya in dosyalar:
                shutil.copy2(dosya, hedef_klasor / dosya.name)
            print(f"{bolum} / {sinif}: {len(dosyalar)} goruntu")
    print("\nVeri hazirlama tamamlandi!")


if __name__ == "__main__":
    main()
