# Deney çıktıları

RedSprite'ta checkpoint'ler ve metrikler `03-deneyler/kosular/` altındadır.
Tek başına klonlanan depoda varsayılan konum `experiments/turkish/` olur.
Etkin dizini `python scripts/project_paths.py runs` gösterir. Pipeline her
çalıştırma için bu dizinde benzersiz bir alt klasör oluşturur ve
`run_summary.json` dosyasını oraya yazar.

`train_turkish.py` yalnız eğitim ve doğrulama kullanır. Koşu klasörüne
`best.pth` ve doğrulama geçmişini içeren `metrics.json` yazar. Saklı test
için doğrulamada eşik dondurulduktan sonra [ayrı değerlendirme komutu](../scripts/README.md#aasist-l-telefon-kanalı-protokolü)
çalıştırılır. Tarihsel XLS-R giriş noktası bu AASIST-L protokolünün dışındadır.
[Bir dönemlik kanal ön çalışması](../docs/telefon-kanali-deneyi.md) P0 olarak
adlandırılır. M0, yayımlanmış AASIST-L kontrol noktasını doğrudan kullanır.

`run_turkish_pipeline.py` çalıştırılan manifest, model ve seçenekleri
`run_summary.json` içinde kaydeder.

Deney paylaşırken manifest yolu/sürümü, config, model ağırlığı, epoch, batch
size, seed, cihaz ve metrikleri birlikte belirtin. Test kümesini eğitimde
kullanmayın; telefon kanalı ve görülmemiş üretici sonuçlarını temiz test
sonuçlarından ayrı raporlayın.

Eski `aasist-source/main.py` ASVspoof uyumlu akışa aittir ve çıktısını
`aasist-source/exp_result/` altında oluşturabilir. Türkçe deneylerde bunun
yerine `scripts/run_turkish_pipeline.py` kullanın.
