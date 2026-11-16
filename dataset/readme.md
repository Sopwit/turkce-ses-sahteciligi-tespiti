# Veri setleri

RedSprite çalışma alanındaki gerçek ve sentetik sesler `02-veri/islenmis/`,
manifestler `02-veri/manifestler/` altında tutulur. `project_paths.py` bu yolları
çalışma alanına göre çözümler. Büyük dosyalar Git’e eklenmez; yalnızca README
ve küçük metadata dosyaları sürümlenir.

## Veri kaynakları

| Klasör | Tür | Kullanım |
|---|---|---|
| `commonvoice_27_tr_fixed/` | Gerçek | RedSprite'taki konuşmacı ayrımlı deneyin insan konuşması kaynağı |
| `commonvoice_17_tr_fixed/` | Gerçek | Eski akışın insan konuşması kaynağı |
| `FLEURS_TR/` | Gerçek | İnsan konuşması ve değerlendirme |
| `MLAAD/` | Sentetik | TTS üreticileri arası çeşitlilik |
| `girenit-tts-tr-synthetic/` | Sentetik | Türkçe TTS örnekleri |
| `Synthetic_Turkish_TTS_Data/` | Sentetik | Parquet tabanlı TTS verisi |
| `turkic_languages_model/` | Yardımcı ASR | İsteğe bağlı, sahtecilik sınıfı değildir |

Windows’ta indirme akışı için:

```powershell
.\setup-windows.ps1 -Mode All -MlaadLimit 20
```

Manuel/ macOS akışında veri sağlayıcılarından dosyaları ilgili klasörlere
yerleştirin. Lisans, atıf, erişim ve yeniden dağıtım koşullarını ayrıca
inceleyin.

Speaker-disjoint deney için resmi Common Voice arşivini `clips/` ve TSV
metadata dosyalarıyla dışa aktarın:

```bash
python scripts/download_commonvoice_tr.py \
  --source-dir /path/to/cv-corpus-tr \
  --output "$(python scripts/project_paths.py data)/commonvoice_17_tr_fixed" \
  --all \
  --require-speaker-id
```

2026 Common Voice 27.0 Türkçe arşiviyle kullanılan final akış:

RedSprite'ta bu akıştan üretilen manifest hazırdır:
`turkish_speaker_disjoint_manifest_commonvoice27.jsonl`.
Genel `turkish_manifest.jsonl` de RedSprite'taki kaynaklar üzerinden yeniden
üretilmiştir. Aşağıdaki komutlar, Common Voice 27 ham arşivi
ham veri kökündeki `commonvoice_27_tr_source/` altına alındığında dışa aktarımı tekrarlar.

```bash
python scripts/download_commonvoice_tr.py \
  --source-dir "$(python scripts/project_paths.py raw)/commonvoice_27_tr_source" \
  --output "$(python scripts/project_paths.py data)/commonvoice_27_tr_fixed" \
  --all \
  --require-speaker-id

python scripts/create_turkish_manifest.py \
  --root "$(python scripts/project_paths.py data)" \
  --exclude-source commonvoice_17_tr_fixed \
  --speaker-disjoint \
  --output "$(python scripts/project_paths.py manifests)/turkish_speaker_disjoint_manifest_commonvoice27.jsonl"

python scripts/validate_manifest.py \
  "$(python scripts/project_paths.py manifests)/turkish_speaker_disjoint_manifest_commonvoice27.jsonl" \
  --require-speaker-id \
  --speaker-disjoint
```

Bu arşivden 64.808 kayıt WAV olarak dışa aktarılmıştır. Eski
`commonvoice_17_tr_fixed` kopyası `client_id` içermediği için final
speaker-disjoint manifestten çıkarılmıştır. Final manifest 76.707 kayıt içerir
ve splitler arasında tanımlı grup kimliği kesişimi yoktur. FLEURS için
gruplama ses örneği kimliğine dayanır. FLEURS dahil tüm kaynaklarda bağımsız
konuşmacı testi yapıldığı iddiası bu manifestle doğrulanamaz.

## Manifest üretimi

```bash
python scripts/create_turkish_manifest.py \
  --root "$(python scripts/project_paths.py data)" \
  --output "$(python scripts/project_paths.py manifests)/turkish_manifest.jsonl" \
  --extract-archives
```

Manifest alanları veri kaynağına göre `path`, `label`, `split` ve varsa
`generator`/metadata alanlarını içerir. Bu projede:

- `label=1`: gerçek insan sesi
- `label=0`: sentetik/yapay ses

Eğitimden önce manifestte `train`, `dev` ve `test` split’lerinin ve iki sınıfın
da bulunduğunu kontrol edin. Görülmemiş üretici deneyi için:

```bash
python scripts/create_turkish_manifest.py \
  --holdout-generator OmniVoice
```

## Kanal ve codec varyantları

Telefon kanalı simülasyonu:

```bash
python scripts/create_phone_channel_manifest.py
```

Gerçek codec varyantları için FFmpeg gerekir:

```bash
python scripts/create_real_codec_manifest.py \
  --input-manifest "$(python scripts/project_paths.py manifests)/turkish_manifest.jsonl" \
  --output-manifest "$(python scripts/project_paths.py manifests)/turkish_opus_manifest.jsonl" \
  --output-dir "$(python scripts/project_paths.py channels)/real_codec_opus" \
  --codec opus
```

Codec scripti encoder bulunamadığında sessizce başka codec’e dönmez; hatayı
gösterir.
