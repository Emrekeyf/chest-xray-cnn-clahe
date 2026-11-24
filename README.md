# Akciğer Röntgenlerinde CNN ile COVID-19 Sınıflandırması: Ham Görüntü ve CLAHE Karşılaştırması

**Python · PyTorch · OpenCV · CNN · CLAHE**

Bu proje, Bilgisayar Mühendisliği yüksek lisansındaki uygulamalı Derin Öğrenme çalışmasını GitHub üzerinde sunmak için düzenlenmiştir. Göğüs röntgenlerinin **COVID-19 / Normal** olarak ikili sınıflandırılması için küçük bir CNN kullanılır. Aynı mimari **ham görüntüler** ve **CLAHE (Contrast Limited Adaptive Histogram Equalization)** uygulanmış görüntülerle ayrı ayrı eğitilerek karşılaştırılır.

> **Kapsam:** Eğitim ve araştırma amaçlı bir yazılım deneyi. Klinik tanı aracı değildir. Haricî veri kümesinde, farklı merkezlerde veya hasta bazında genellenebilirlik bu depoda kanıtlanmamıştır.

## Dosyalar

| Dosya | Amaç |
| --- | --- |
| `veri_hazirla.py` | Ham veri kümesinden dengeli COVID/Normal eğitim, doğrulama ve test ayrımı üretir. |
| `cnn_egit.py` | Ham ve CLAHE olmak üzere iki ayrı CNN deneyini eğitir ve test eder; metrikler ve grafik oluşturur. |
| `confusion_matrix.py` | `cnn_sonuclar` altındaki metriklerden iki karışıklık matrisi görseli üretir. |
| `requirements.txt` | Gerekli Python paketleri. |
| `.gitignore` | Verilerin, ağırlıkların ve deney çıktılarının yanlışlıkla GitHub'a eklenmesini önler. |

## Veri seti

Kullanılan kaynak: **COVID-19 Radiography Database** (COVID ve Normal sınıfları). Veri seti bu depoya yüklenmemiştir. Veri kaynağının açıklamalarını ve kullanım koşullarını veri sağlayıcısından ayrıca kontrol edin.

```text
Derin_Ogrenme_GitHub/
├── COVID-19_Radiography_Dataset/    # GitHub disinda tutulur
│   ├── COVID/images/*.png
│   └── Normal/images/*.png
├── cnn_egit.py
├── confusion_matrix.py
├── veri_hazirla.py
├── requirements.txt
├── .gitignore
└── README.md
```

`python veri_hazirla.py` çalıştırılınca varsayılan olarak her sınıftan **3.616** görüntü seçilir ve ayrı klasörlere kopyalanır:

| Bölüm | COVID | Normal | Toplam |
| --- | ---: | ---: | ---: |
| Eğitim (`train`) | 2.531 | 2.531 | 5.062 |
| Doğrulama (`val`) | 542 | 542 | 1.084 |
| Test (`test`) | 543 | 543 | 1.086 |
| **Toplam** | **3.616** | **3.616** | **7.232** |

- Rastgelelik tohumu `42` olarak sabitlenir; sınıflar ayrı ayrı seçilir.
- Bölme **görüntü düzeyindedir**; aynı hastaya ait görüntülerin farklı bölümlere düşmediği garanti edilmez. Veri tekrarları, hasta kimlikleri ve olası veri sızıntıları ayrıca kontrol edilmelidir.
- `hazir_veriseti` zaten doluysa kod **üzerine yazmaz**. Önceki deney bölünmesini korumak için bu bilinçli bir güvenlik önlemidir. Eski `hazir_veriseti` klasörünüz varsa onu kullanabilirsiniz.

## Kurulum (Windows, CMD)

Python **3.11** önerilir. Özellikle CUDA kullanacaksanız PyTorch'u sisteminize uygun biçimde [resmî PyTorch sitesinden](https://pytorch.org/get-started/locally/) kurun.

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Hazır veri bölünmesi yoksa ham veri setini belirtilen klasöre yerleştirin ve:

```cmd
.venv\Scripts\python.exe veri_hazirla.py
```

**Eğitim:**

```cmd
.venv\Scripts\python.exe cnn_egit.py
```

**Karışıklık matrisleri:**

```cmd
.venv\Scripts\python.exe confusion_matrix.py
```

## Model ve deney kurulumu

- Giriş: **64 × 64**, gri tonlu, piksel aralığı `[-1, 1]` olarak normalize edilmiş görüntüler.
- CNN: `Conv(1→8) → ReLU → MaxPool → Conv(8→16) → ReLU → MaxPool → Flatten → Linear(4096→2)`.
- Eğitim: **15 epoch**, batch size **32**, Adam, öğrenme oranı **0.0005**, `CrossEntropyLoss`.
- Deneyler: `HAM` (CLAHE yok) ve `CLAHE` (`clipLimit=2.0`, `tileGridSize=(8,8)`). CLAHE, 64×64'e yeniden boyutlandırmadan sonra uygulanır.
- İki deney aynı başlangıç tohumu kullanılarak ayrı ayrı eğitilir. En düşük **doğrulama kaybı** sağlayan model saklanır ve **test kümesi** eğitim sonunda yalnızca değerlendirme için kullanılır.
- Sonuçlar: accuracy, COVID için precision/recall/F1, TP/FN/FP/TN, loss, epoch geçmişi ve karşılaştırma grafiği.

```text
cnn_sonuclar/                   # GitHub'a eklenmez
├── HAM_model.pth
├── HAM_history.csv
├── HAM_metrics.csv
├── CLAHE_model.pth
├── CLAHE_history.csv
├── CLAHE_metrics.csv
├── karsilastirma.png
├── HAM_confusion_matrix.png
└── CLAHE_confusion_matrix.png
```

**Karışıklık matrisi:** Satırlar gerçek sınıf, sütunlar tahmin sınıfıdır; sıra `COVID`, `Normal` biçimindedir.

## Deney sonuçları ve bilimsel sınırlılıklar

Bu GitHub paketine **orijinal deneylere ait doğrulanmış CSV dosyaları eklenmemiştir**. Bu nedenle burada belirli bir accuracy veya F1 sonucu iddia edilmiyor. Çalıştırıldığında kendi veri bölünmenizle yeni sonuçlar üretilecektir; geçmiş deneyle birebir aynı sayısal sonuçlar elde edilmeyebilir.

Bu kod, bir bilimsel makaledeki hazır ağların birebir yeniden üretimi değildir: iki katmanlı basit bir CNN ile **ham/CLAHE karşılaştırması** yapılır. Röntgen görüntülerinden otomatik hastalık sınıflandırması klinik karar verme yerine kullanılmamalıdır.

## Paylaşım notu

Tıbbi görüntüler, veri kümesi, model ağırlıkları ve sonuç dosyaları `.gitignore` ile hariç tutulmuştur. Bu dosyalar yerel bilgisayarda kalır. Herhangi bir örnek görselin yayımlanması için kaynak lisansı ve varsa hasta mahremiyeti koşulları önceden kontrol edilmelidir.
