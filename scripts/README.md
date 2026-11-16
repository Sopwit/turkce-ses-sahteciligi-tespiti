# Script kataloğu

Tüm komutlar depo kökünden çalıştırılır. Python scriptleri için kurulumdan
sonra sanal ortamı etkinleştirin. Scriptler veri setlerini, model ağırlıklarını
ve deney çıktısını Git’e eklemez.

## Kurulum

| Script | Platform | Görev |
|---|---|---|
| [`../setup-macos.sh`](../setup-macos.sh) | macOS Apple Silicon | Tek komutla macOS ortam kurulumu |
| [`../setup-windows.ps1`](../setup-windows.ps1) | Windows 10/11 x64 | Tek komutla Windows ortam/veri kurulumu |
| [`2209A_AASIST_kurulum_Python314.sh`](2209A_AASIST_kurulum_Python314.sh) | macOS | Ayrıntılı Python, bağımlılık ve ağırlık kurulumu |
| [`TUBITAK_2209A_Windows_Tek_Script-2.ps1`](TUBITAK_2209A_Windows_Tek_Script-2.ps1) | Windows | Ayrıntılı kurulum ve veri seçenekleri |
| [`check_repository.py`](check_repository.py) | Tüm platformlar | Repo ve manifest kalite kontrolü |
| [`validate_manifest.py`](validate_manifest.py) | Tüm platformlar | Split, speaker leakage ve sınıf dengesi kontrolü |

```bash
python3 scripts/check_repository.py
```

## Veri hazırlama

| Script | Görev |
|---|---|
| [`create_turkish_manifest.py`](create_turkish_manifest.py) | Veri klasörlerinden Türkçe JSONL manifest üretir |
| [`download_commonvoice_tr.py`](download_commonvoice_tr.py) | Common Voice TR kayıtlarını WAV olarak dışa aktarır |
| [`mlaad_tr_dilimli_indir.py`](mlaad_tr_dilimli_indir.py) | MLAAD Türkçe klasörlerini sınırlı veya tam indirir |
| [`create_phone_channel_manifest.py`](create_phone_channel_manifest.py) | 8 kHz/G.711 telefon kanalı simülasyonu üretir |
| [`create_real_codec_manifest.py`](create_real_codec_manifest.py) | FFmpeg ile GSM/AMR-NB/Opus varyantı üretir |

Önerilen sıra:

```bash
python scripts/create_turkish_manifest.py --root "$(python scripts/project_paths.py data)" --extract-archives
python scripts/validate_manifest.py "$(python scripts/project_paths.py manifests)/turkish_manifest.jsonl"
python scripts/create_phone_channel_manifest.py
```

`create_turkish_manifest.py` seçenekleri:

```text
--root PATH
--output PATH
--extract-archives
--holdout-generator NAME
--group-disjoint
--exclude-source SOURCE
```

Tarihsel grup ayrımı için kaynak metadata ile manifest üretin. Kişi düzeyi doğrulama için yeni şemanın `speaker_roles` alanı gerekir.

```bash
python scripts/create_turkish_manifest.py \
  --group-disjoint \
  --output "$(python scripts/project_paths.py manifests)/turkish_speaker_disjoint_manifest.jsonl"
python scripts/validate_manifest.py \
  "$(python scripts/project_paths.py manifests)/turkish_speaker_disjoint_manifest.jsonl" \
  --require-speaker-id \
  --group-disjoint
```

Eski Common Voice 17 kopyası `client_id` içermediğinden bu grup ayrımından çıkarılabilir.

```bash
python scripts/create_turkish_manifest.py \
  --group-disjoint \
  --exclude-source commonvoice_17_tr_fixed \
  --output "$(python scripts/project_paths.py manifests)/turkish_speaker_disjoint_manifest.jsonl"
```

Tam veri için Common Voice’in `client_id` içeren kaynağını Mozilla Data
Collective’den indirip metadata ile yeniden dışa aktarın. `client_id` yoksa
dosya adına veya akustik benzerliğe dayalı sahte speaker ID üretmeyin.
Mevcut metadata’lı alt kümenin doğrulama sonucu yaklaşık 11.899 kayıt ve
split’ler arası sıfır speaker overlap’tir.

Common Voice ve MLAAD örnekleri:

```bash
python scripts/download_commonvoice_tr.py --limit 1000
python scripts/download_commonvoice_tr.py --all
python scripts/mlaad_tr_dilimli_indir.py --list
python scripts/mlaad_tr_dilimli_indir.py --model bark --limit 20
python scripts/mlaad_tr_dilimli_indir.py --all
```

Speaker-disjoint Common Voice için Mozilla Data Collective’den indirilen resmi
arşivi kullanın. Arşivde `clips/`, `train.tsv`, `dev.tsv` ve `test.tsv`
bulunmalıdır:

```bash
python scripts/download_commonvoice_tr.py \
  --source-dir /path/to/cv-corpus-tr \
  --output "$(python scripts/project_paths.py data)/commonvoice_17_tr_fixed" \
  --all \
  --require-speaker-id
python scripts/create_turkish_manifest.py \
  --group-disjoint \
  --output "$(python scripts/project_paths.py manifests)/turkish_speaker_disjoint_manifest.jsonl"
python scripts/validate_manifest.py \
  "$(python scripts/project_paths.py manifests)/turkish_speaker_disjoint_manifest.jsonl" \
  --require-speaker-id \
  --group-disjoint
```

Resmi arşiv `client_id` bilgisini korur ve exporter bunu `speaker_id` olarak
manifest’e taşır. Mevcut `ysdede/commonvoice_17_tr_fixed` HF kopyası bu
alanı içermediği için `--require-speaker-id` ile bilinçli olarak reddedilir.

Tam MLAAD indirmesi büyük olabilir. Hugging Face kimlik doğrulaması
gerekiyorsa token’ı script içine yazmadan `hf auth login` kullanın.
`gdown`, yalnızca `-IncludeASR` ile istenen opsiyonel TurkicASR arşivi için
kurulur; ana Türkçe sahtecilik akışı için gerekli değildir.

## Eğitim ve doğrulama

| Script | Görev |
|---|---|
| [`run_turkish_pipeline.py`](run_turkish_pipeline.py) | Manifest, doğrulama ve eğitim akışını birleştirir |
| [`train_turkish.py`](train_turkish.py) | Tek bir Türkçe modelini eğitir |
| [`metrics.py`](metrics.py) | EER, ROC-AUC, F1 ve sınıflandırma metriklerini hesaplar |
| [`train_xlsr_turkish.py`](train_xlsr_turkish.py) | XLS-R ince ayarı yapar |
| [`verify_models.py`](verify_models.py) | Yerel model ileri geçişini doğrular |

Pipeline:

```bash
python scripts/run_turkish_pipeline.py \
  --models AASIST \
  --extract-archives \
  --epochs 1 \
  --batch-size 4 \
  --device auto
```

Pipeline modelleri: `AASIST`, `AASIST-L`, `RawNet2`, `RawGAT-ST`, `XLSR`
ve `all`. Yararlı seçenekler:

```text
--manifest PATH
--phone-channel
--holdout-generator NAME
--prepare-only
--skip-verify
--device auto|cpu|cuda|mps
--epochs N
--batch-size N
```

Tek model:

```bash
python scripts/train_turkish.py \
  --manifest "$(python scripts/project_paths.py manifests)/turkish_manifest.jsonl" \
  --config aasist-source/config/Turkish_RawNet2.conf \
  --device auto
```

`--weights` başlangıç ağırlığı, `--output` deney klasörü, `--seed` deney
tekrarlanabilirliği için kullanılır. Test değerlendirmesi en iyi dev checkpoint
üzerinde yapılır. `metrics.json` içinde EER, ROC-AUC, F1, precision, recall,
balanced accuracy ve confusion matrix kaydedilir. Varsayılan çıktılar
`python scripts/project_paths.py runs` ile gösterilen dizindedir.

Model doğrulama:

```bash
python scripts/verify_models.py --model all --device cpu
python scripts/verify_models.py --model AASIST --device mps
```

## Platform notları

- macOS Apple Silicon: `auto` MPS’yi seçebilir; sorun olursa `cpu` kullanın.
- Windows NVIDIA: uyumlu NVIDIA sürücüsü ve CUDA uyumlu PyTorch kurulumu
  gerekir.
- Windows CPU: çalışır ancak eğitim yavaştır.
- Linux: platform kurulum betiği yoktur; manuel sanal ortam kurun.

Türkçe deneylerde eski ASVspoof akışı olan `aasist-source/main.py` yerine
`run_turkish_pipeline.py` veya `train_turkish.py` kullanın.

## AASIST-L telefon kanalı protokolü

`prepare_turkish_protocol_v2.py` kaynak manifestlerden bir envanter ve yalnız
train/dev içeren bir geliştirme manifesti yazar. Yeni dizin seçin. FLEURS,
geliştirme dışında tutulur. Geçmiş sonuç `P0`, orijinal AASIST-L ağırlığıyla
yapılan eğitim içermeyen değerlendirme `M0` olarak adlandırılır.

```bash
manifests="$(python scripts/project_paths.py manifests)"
weights="$(python scripts/project_paths.py weights)/AASIST-L.pth"
config=aasist-source/config/AASIST-L.conf

python scripts/prepare_turkish_protocol_v2.py \
  --inventory "$manifests/turkish_speaker_disjoint_manifest_commonvoice27.jsonl" \
  --development "$manifests/turkish_channel_benchmark_v1.jsonl" \
  --common-voice "$manifests/commonvoice_tr_manifest.jsonl" \
  --external "$manifests/fleurs_tr_harici_test_v1.jsonl" \
  --output-dir "$manifests/protokol-v2"
dev="$manifests/protokol-v2/turkish_channel_development_v2.jsonl"
python scripts/validate_manifest.py "$dev" --splits train dev --exclude-source FLEURS_TR
```

M1 ve M2 aynı geliştirme manifesti, tohum ve başlangıç ağırlıklarını kullanır.
M2 için `create_real_codec_manifest.py --split train` ile yeni hedeflerde
G.711 μ-law ve Opus eğitim görünümleri oluşturulur. Her koşu yeni çıktı dizini alır.

```bash
python scripts/train_turkish.py --manifest "$dev" --config "$config" \
  --weights "$weights" --output "$m1_run" --arm M1 --seed 2209 --device cuda
python scripts/train_turkish.py --manifest "$dev" --config "$config" \
  --weights "$weights" --output "$m2_run" --arm M2 --seed 2209 --device cuda \
  --channel-train "$g711_train" --channel-train "$opus_train"
```

Model seçimi sonrasında iki çağrıyı ayrı çalıştırın. `$heldout` yalnız bağımsız
protokolü onaylanan test kayıtlarını, `$g711_test` ve `$opus_test` aynı
test kimliklerinin yeni kodek kopyalarını gösterir. `$lock` ve `$result`
yeni çıktı dosyaları olmalıdır. M0 için model orijinal ağırlık dosyasıdır.

```bash
python scripts/evaluate_turkish_channels.py --freeze --arm M1 --manifest "$dev" \
  --config "$config" --weights "$m1_run/best.pth" --output "$lock"
python scripts/evaluate_turkish_channels.py --lock "$lock" --arm M1 --manifest "$heldout" \
  --config "$config" --weights "$m1_run/best.pth" \
  --g711 "$g711_test" --opus "$opus_test" --output "$result"
```

`validate_manifest.py --speaker-disjoint` her gereken rol için doğrulanmış
kişi kimliği arar. Mevcut V2 sentetik TTS adaylarının kişi düzeyi kökeni
eksik olduğundan tam konuşmacı bağımsızlığı onaylanmaz.

Common Voice 27.0 ham arşivini yeniden dışa aktarırken mevcut veriden ayrı
bir hedef seçin. `download_commonvoice_tr.py --source-dir ... --all
--require-speaker-id --provenance ...` kaynak MP3 ve cümle kimliklerinin
WAV yollarıyla eşlemesini ayrı bir JSONL dosyasına yazar. TSV başına
yazdırılan adetleri sağlayıcı kartıyla karşılaştırın.
