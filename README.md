# Türkçe Ses Sahteciliği Tespiti

Türkçe konuşmalarda gerçek insan sesini yapay/sentetik sesten ayırmak için
AASIST, RawNet2, RawGAT-ST ve XLS-R tabanlı deneyler içeren araştırma
deposudur.

> **Durum:** Araştırma ve geliştirme aşamasındadır. Sonuçlar üretim kullanımı
> için doğrulanmamıştır.

## Hızlı başlangıç

GitHub'dan yeni bir çalışma kopyası oluşturmak için:

```bash
git clone https://github.com/Sopwit/turkce-ses-sahteciligi-tespiti.git
cd turkce-ses-sahteciligi-tespiti
```

Kurulum ve eğitim komutlarını depo kökünden çalıştırın.

RedSprite çalışma alanında bu depo `01-kod/turkce-ses-sahteciligi-tespiti/`
altındadır. Yerel veri, ağırlık ve koşu konumları
[`project-layout.json`](project-layout.json) ile çözümlenir. Diskin proje
kökündeki `README.md` başvuru ve literatür klasörlerini gösterir. Tek başına
klonlanan depoda betikler `dataset/` ve `experiments/turkish/` yollarını kullanır.

### 1. Ortamı kurun

macOS Apple Silicon:

```bash
bash setup-macos.sh
source "$(python3 scripts/project_paths.py runtime)/.venv-aasist-py314/bin/activate"
```

Windows 10/11 x64 PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-windows.ps1 -Mode Setup
```

Windows’ta veri setleri de indirilecekse:

```powershell
.\setup-windows.ps1 -Mode All -MlaadLimit 20
```

Kurulum betikleri mevcut dosyaları silmez. macOS betiği Apple Silicon için
hazırlanmıştır; Windows betiği x64 içindir. Linux üzerinde Python 3.11 veya
daha yenisiyle ortamı kurun.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux'ta AASIST ile eğitim için başlangıç ağırlığını indirin. AASIST-L
deneyi için ikinci komut gerekir. RawNet2 ve RawGAT-ST eğitimleri başlangıç
ağırlığı gerektirmez.

```bash
weights_dir="$(python scripts/project_paths.py weights)"
mkdir -p "$weights_dir"
hf download SpeechAntiSpoofingBenchmarks/AASIST AASIST.pth --local-dir "$weights_dir"
curl -fL https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST-L.pth \
  -o "$weights_dir/AASIST-L.pth"
```

### 2. Veri setlerini hazırlayın

Windows `-Mode All` veri setlerini indirmeyi dener. Diğer platformlarda veri
sağlayıcılarının lisans ve erişim koşullarına uyarak veri setlerini
`python scripts/project_paths.py data` ile gösterilen dizine yerleştirin.
Sonra manifest üretin.

```bash
python scripts/create_turkish_manifest.py --extract-archives
```

Bu komut manifest dizininde `turkish_manifest.jsonl` üretir. Gerçek kayıtlar `label=1`,
sentetik kayıtlar `label=0` olarak işaretlenir. Her split’te iki sınıfın da
bulunması gerekir.

### 3. Modelleri doğrulayın

Bağımlılıkların kurulu olduğu ortamı etkinleştirdikten sonra:

```bash
python scripts/verify_models.py --device cpu
```

Apple Silicon için `--device mps`, NVIDIA için `--device cuda` kullanılabilir.
Bu kontrol RawNet2 ve RawGAT-ST dahil yerel model mimarilerinin ileri geçişini
doğrular; veri indirmez.

### 4. Türkçe eğitim pipeline’ını çalıştırın

Manifest hazırlama, model doğrulama ve eğitimi tek akışta çalıştırmak için:

```bash
python scripts/run_turkish_pipeline.py \
  --models AASIST \
  --epochs 1 \
  --batch-size 4
```

`--models all` ile `AASIST`, `AASIST-L`, `RawNet2`, `RawGAT-ST` ve `XLSR`
kollarını seçebilirsiniz. İlk denemelerde düşük epoch/batch değerleri
kullanın. Sadece manifest ve hazırlık için `--prepare-only`, model kontrolünü
atlamak için `--skip-verify` eklenebilir.

`--manifest` ile verilen mevcut dosya girdi olarak kullanılır. `--extract-archives`
veya `--holdout-generator` seçilirse pipeline yeni bir manifest üretir. Her eğitim
koşusu tarihli ve benzersiz bir çıktı dizini alır.

Pipeline seçenekleri:

```text
--device auto|cpu|cuda|mps
--phone-channel
--holdout-generator OmniVoice
--prepare-only
--skip-verify
--run-dir <yeni-cikti-dizini>
```

## Veri ve deney akışları

[Kanal deneyi protokolü ve AASIST-L ön sonuçları](docs/telefon-kanali-deneyi.md).

Telefon kanalı simülasyonu:

```bash
python scripts/create_phone_channel_manifest.py
```

Gerçek codec deneyi için seçilen codec’i destekleyen bir FFmpeg derlemesi gerekir.

```bash
python scripts/create_real_codec_manifest.py \
  --input-manifest "$(python scripts/project_paths.py manifests)/turkish_manifest.jsonl" \
  --output-manifest "$(python scripts/project_paths.py manifests)/turkish_opus_manifest.jsonl" \
  --output-dir "$(python scripts/project_paths.py channels)/real_codec_opus" \
  --codec opus
```

Common Voice TR dışa aktarma:

```bash
python scripts/download_commonvoice_tr.py --limit 1000
python scripts/download_commonvoice_tr.py --all
```

MLAAD Türkçe örneklemi:

```bash
python scripts/mlaad_tr_dilimli_indir.py --list
python scripts/mlaad_tr_dilimli_indir.py --model bark --limit 20
python scripts/mlaad_tr_dilimli_indir.py --all
```

XLS-R ince ayarı:

```bash
python scripts/train_xlsr_turkish.py \
  --manifest "$(python scripts/project_paths.py manifests)/turkish_manifest.jsonl" \
  --epochs 3
```

Tüm scriptlerin seçenekleri ve görevleri için
[scripts/README.md](scripts/README.md) dosyasına bakın.

## Dizin yapısı

```text
.
├── aasist-source/       # Model kodu ve config dosyaları
├── dataset/             # Veri kaynakları ve tek başına klonlama için varsayılan yol
├── docs/                # Plan, günlük ve referans belgeleri
├── experiments/         # Deney çıktı rehberi ve tek başına klonlama için varsayılan yol
├── scripts/             # Veri, eğitim, indirme ve doğrulama araçları
│   └── README.md        # Ayrıntılı komut kataloğu
├── setup-macos.sh       # macOS giriş betiği
├── setup-windows.ps1    # Windows giriş betiği
└── Makefile             # Repo kalite komutları
```

Veri setleri, model ağırlıkları, sanal ortamlar ve deney çıktıları Git’e
eklenmez. Lisans ve kullanım koşullarını her veri sağlayıcısı için ayrıca
kontrol edin.

## Lisans ve veri

Model kodunun ve üçüncü taraf bileşenlerin koşulları [LICENSE](LICENSE),
[NOTICE](NOTICE) ve [AASIST bildiriminde](aasist-source/NOTICE) yer alır.
Eski ASVspoof akışındaki t-DCF uygulamasının lisansı CC BY-NC-SA 4.0'dır.
Ses dosyaları, manifestler, model ağırlıkları ve başvuru evrakları RedSprite
çalışma alanında tutulur. Her veri ve ağırlık kaynağı için sağlayıcının
koşulları geçerlidir.

## Model ve veri sınırları

- Bu proje ASVspoof veri setlerini kullanmaz.
- `aasist-source/main.py` eski ASVspoof uyumlu akıştır; Türkçe deneylerde
  `scripts/train_turkish.py` veya pipeline kullanılmalıdır.
- RawNet2 ve RawGAT-ST için bu repoda Türkçe hazır ağırlık yoktur; Türkçe
  manifest ile yeniden eğitim gerekir.
- `AASIST.pth` ve `AASIST-L.pth` yalnızca kurulum betikleriyle sağlanan
  başlangıç ağırlıklarıdır; Türkçe performans garantisi vermez.

## Repo kontrolleri

```bash
make check
make compile
make test
```

`make check` repo yapısını, manifest JSONL biçimini ve Python sözdizimini
kontrol eder. CI da aynı kontrolleri çalıştırır.

## Belgeler

- [Script kataloğu](scripts/README.md)
- [Veri setleri](dataset/readme.md)
- [Deney çıktıları](experiments/README.md)
- [Proje belgeleri](docs/README.md)
- [Katkı rehberi](CONTRIBUTING.md)
