# Türkçe telefon kanalı deneyi

Bu ön çalışma, aynı Türkçe kayıtlar G.711 μ-law ve dar bant Opus ile yeniden kodlandığında AASIST-L’nin hata oranlarının nasıl değiştiğini ölçüyor. Gerçek sınıfı Common Voice Türkçe kayıtlarından, sentetik sınıfı MLAAD’ın Türkçe TTS kayıtlarından oluşturuldu.

## Deney düzeni

| Bölüm | Gerçek klip | Sentetik klip | Sentetik üretici ailesi |
| --- | ---: | ---: | --- |
| Eğitim | 1.098 | 1.098 | Freya, Bark, Glow-TTS |
| Doğrulama | 400 | 400 | OmniVoice |
| Test | 800 | 800 | VoxCPM, XTTS |

Örnekleme tohumu 2209. Klipler 2,5–8 saniye aralığında seçildi. Gerçek kayıtlarda Common Voice konuşmacı ayrımı korundu. Temiz testteki 1.600 klip, test kimlikleri ve sınıf etiketleri korunarak FFmpeg ile G.711 ve 12 kbit/s dar bant Opus koşullarına dönüştürüldü. Kanal dosyaları 8 kHz mono PCM olarak çözüldü ve ortak 16 kHz model girişine dönüştürüldü.

Yayımlanmış AASIST-L ağırlıklarıyla başlayan model bir dönem eğitildi. Temiz doğrulamada dengeli doğruluğu en yüksek yapan **0,9995589852333069** ham gerçek-ses skoru eşiği üç koşulda sabit tutuldu. Yanlış alarm ve kaçırma bu eşikte ölçüldü. EER ve ROC-AUC, her koşulun skor dağılımından hesaplandı.

## Ön çalışma sonuçları

| Koşul | EER | ROC-AUC | Gerçek seste yanlış alarm | Sentetik seste kaçırma |
| --- | ---: | ---: | ---: | ---: |
| Temiz | %29,750 | 0,7900 | %36,625 | %22,125 |
| G.711 μ-law | %36,375 | 0,7132 | %92,250 | %1,250 |
| Dar bant Opus | %36,750 | 0,7082 | %95,125 | %0,625 |

16 Kasım 2026 tarihli yerel tekrar değerlendirmesinde aynı kontrol noktasıyla skorlar yeniden üretildi. Sabit eşiğin kaynak doğrulama skorlarından yeniden seçilmesi ve üç koşulun metrikleri kayıtlı klip skorları üzerinden doğrulandı.

Aynı eşikte, harici gerçek-ses kontrollerinde yanlış alarm oranı FLEURS Türkçe resmî testindeki 743 klipte %72,54, Real-TurnTurk’teki 120 klipte %82,50 oldu. Her iki harici kontrol yalnız gerçek ses içeriyor.

Bulgular tek bir eğitim koşusuna, kontrollü kodek dönüşümlerine ve seçilen TTS ailelerine ait. Canlı şebeke, çağrı merkezi, görüşme prototipi ve ses dönüştürme performansı için ayrı deneyler gerekir.

İlgili kod [örneklem hazırlama](../scripts/prepare_turkish_benchmark.py), [kodek dönüşümü](../scripts/create_real_codec_manifest.py) ve [eşleştirilmiş kanal değerlendirmesi](../scripts/evaluate_turkish_channels.py) betiklerinde yer alıyor.
