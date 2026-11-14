# AI Əsaslı Tikinti Sahəsi Təhlükəsizlik və Dəbilqə (Kaska) Nəzarət Sistemi

Bu layihə inşaat və tikinti sahələrində işçilərin təhlükəsizlik qaydalarına (PPE) əməl etməsini, xüsusilə **kaska (dəbilqə)** taxıb-taxmadıqlarını real vaxt rejimində (RTSP kamera və ya yüklənən videolar üzərindən) yüksək dəqiqliklə müəyyən edən süni intellekt sistemidir.

## Əsas Xüsusiyyətlər
- **Yüksək Dəqiqlikli Hibrid AI Modeli**: YOLO11s (yüksək rezolyusiyada insan aşkarlanması) və dəqiqləşdirilmiş PPE modeli (`Hansung-Cho`).
- **Fiziki Kolorimetriya və Lüminans Filtrləməsi**: Adi papaqları, beretləri, qara papaqları və keçəl insanları tikinti kaskalarından fərqləndirir (Sarı, Ağ, Mavi, Narıncı sertifikatlı kaskalar).
- **İnteraktiv Veb İdarəetmə Paneli**:
  - Canlı video yayımı və aşkarlama (HUD)
  - Yeni video fayllarının sürükləyib-yüklənməsi (Drag & Drop)
  - IP Kamera / RTSP / Canlı yayım linki qoşulması
  - Real vaxt KPI statistikası (Aktiv işçilər, Kaskalılar, Qaydapozuntuları)
  - Qaydapozuntusu anında çəkilən avtomatik foto-sübutlar (Snapshot Gallery)
  - Qayda pozuntusu qeydiyyat jurnalı (Qayda pozan şəxslərin siyahısı)

## Quraşdırılma və İşə Salınma

```bash
git clone https://github.com/tural1227/ai-construction-safety.git
cd ai-construction-safety
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

Sistemə daxil olmaq üçün brauzerdə `http://localhost:5000` ünvanını açın.
