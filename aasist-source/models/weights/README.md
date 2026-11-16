# Yerel model ağırlıkları

Bu klasörde AASIST ve AASIST-L önceden eğitilmiş ağırlıkları bulunur.
Ağırlıklar büyük ikili dosyalardır ve `.gitignore` nedeniyle Git'e eklenmez.

Kurulum scriptleri şu dosyaları indirir:

- `AASIST.pth`
- `AASIST-L.pth`

RawNet2 ve RawGAT-ST için Türkçe hazır ağırlık bulunmadığından bu klasöre
otomatik ağırlık indirilmez. Türkçe eğitim çıktıları `python scripts/project_paths.py runs` ile gösterilen dizinde
oluşturulur.

RedSprite çalışma alanında başlangıç ağırlıkları `03-deneyler/model-agirliklari/`
altındadır. Etkin klasörü `python scripts/project_paths.py weights` gösterir.
