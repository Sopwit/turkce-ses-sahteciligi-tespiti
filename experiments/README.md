# Deney çıktıları

RedSprite'ta checkpoint'ler ve metrikler `03-deneyler/kosular/` altındadır.
Tek başına klonlanan depoda varsayılan konum `experiments/turkish/` olur.
Etkin dizini `python scripts/project_paths.py runs` gösterir. Pipeline her
çalıştırma için bu dizinde benzersiz bir alt klasör oluşturur ve
`run_summary.json` dosyasını oraya yazar.

`train_turkish.py` model klasörüne checkpoint ve `metrics.json` yazar.
`metrics.json` test loss/accuracy yanında EER, ROC-AUC, F1, precision, recall,
balanced accuracy ve confusion matrix içerir. Test değerlendirmesi en iyi dev
checkpoint ile yapılır.
`train_xlsr_turkish.py` de en iyi doğrulama checkpoint’iyle aynı test
metriklerini üretir.
`run_turkish_pipeline.py` çalıştırılan manifest, model ve seçenekleri
`run_summary.json` içinde kaydeder.

Deney paylaşırken manifest yolu/sürümü, config, model ağırlığı, epoch, batch
size, seed, cihaz ve metrikleri birlikte belirtin. Test kümesini eğitimde
kullanmayın; telefon kanalı ve görülmemiş üretici sonuçlarını temiz test
sonuçlarından ayrı raporlayın.

Eski `aasist-source/main.py` ASVspoof uyumlu akışa aittir ve çıktısını
`aasist-source/exp_result/` altında oluşturabilir. Türkçe deneylerde bunun
yerine `scripts/run_turkish_pipeline.py` kullanın.
