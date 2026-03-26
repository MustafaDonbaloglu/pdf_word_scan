# PDF İngilizce Kelime Çıkarıcı & Analiz Aracı

Bu proje, Veri Madenciliği dersi kapsamında geçmiş yıllardaki İngilizce sınav (ör. YDS) PDF'lerinden kelimeleri çıkarıp temizleyerek frekans analizini yapmayı ve gelecek yıllardaki kelime eğilimlerini tahmin edebilmek için temiz bir veri seti oluşturmayı amaçlamaktadır.

## Özellikler

- **PDF'ten Metin Çıkarımı:** `pdfplumber` (ve gerekirse `pdfminer`) kullanılarak PDF dosyalarından metin okunur.
- **İngilizce Kelime Filtresi:** NLTK (`stopwords`, `wordnet`, `words`) kullanılarak yalnızca anlamlı İngilizce kelimeler alınır; Türkçe kelimeler, özel isimler, roman rakamları ve tek harflik gürültüler elenir.
- **Lemmatization:** Kelimeler köklerine (lemma) indirgenerek (ör. "running" -> "run", "cats" -> "cat") sayılır.
- **Normalize Frekans (Relative Frequency):** Sınav uzunlukları değişebileceğinden, kelime sayıları "toplam kelime içindeki oranı" (relative_freq) olarak da hesaplanır. Bu, yıllar arası adil karşılaştırma sağlar.
- **Yıl ve Dönem Bazlı Çıktılar:** Her PDF'in yılı ve dönemi klasör isminden veya dosya adından otomatik çekilir, sonuçlar `outputs/YIL/DÖNEM/` altına CSV ve Excel olarak kaydedilir.
- **Veri Madenciliği İçin Birleştirilmiş Veri:** Tüm yılların verisi `outputs/_all/` klasörü altında hem "Long Format" hem de zaman serisi analizi için "Pivot Format" (yıllara göre dağılım) şeklinde Excel/CSV olarak sunulur.

## Kurulum ve Gereksinimler

Proje Python 3.9+ ile çalışmaktadır. Gerekli kütüphaneleri yüklemek için terminalde/komut satırında şu komutu çalıştırın:

```bash
pip install pandas nltk pdfplumber pdfminer.six openpyxl
```

*(Not: OCR özelliğinin çalışması gerekiyorsa sistemde `tesseract-ocr` ve `poppler` kurulu olmalı, ayrıca `pytesseract` ve `pdf2image` kütüphaneleri yüklenmelidir. Mevcut akışta standart PDF metni varsa OCR'a ihtiyaç duyulmaz).*

## Klasör Yapısı

Çalıştırmadan önce PDF dosyalarınızı `pdfler` klasörü içerisine yıl ve dönem hiyerarşisinde yerleştirmelisiniz. Örnek:

```
pdfscan/
├── pdfler/
│   ├── 2016/
│   │   ├── Yds_Bahar/
│   │   │   └── 2016_YDS_ILKBAHAR_INGILIZCE.pdf
│   │   └── Yds_Sonbahar/
│   │       └── 2016_YDS_Sonbahar_Ingilizce.pdf
│   └── 2017/
│       └── ...
├── script.py
└── README.md
```

## Nasıl Çalıştırılır?

1. Terminali projenin bulunduğu dizinde (`pdfscan/`) açın.
2. Aşağıdaki komutla betiği çalıştırın:

```bash
python script.py
```

İlk çalıştırmada script otomatik olarak gerekli NLTK veri setlerini (stopwords, wordnet, vb.) indirecektir.
İşlem tamamlandığında terminalde en çok geçen 20 kelime özet olarak yazdırılacak ve tüm sonuçlar `outputs/` klasöründe oluşturulacaktır.

## Çıktıların İncelenmesi (outputs Klasörü)

Script çalıştıktan sonra proje dizininde bir `outputs/` klasörü oluşur:

1. **`outputs/<Yıl>/<Dönem>/` Klasörleri:**
   - Her bir PDF için o yıl ve döneme ait kelime listesini içerir (`words.csv`, `words.xlsx`).
   - Sadece o dönemi incelemek isterseniz bu dosyaları kullanabilirsiniz.
2. **`outputs/_all/` Klasörü (Veri Madenciliği Hedefi İçin En Önemli Kısım):**
   - **`all_terms_long.csv` / `all_terms_long.xlsx`:** Tüm yılların verisinin alt alta birleştirildiği, makine öğrenmesi modelleri veya veri analizi araçları (Python/Pandas, R, SPSS vb.) için en uygun uzun formattır.
   - **Excel dosyasındaki "pivot" sekmesi:** Satırlarda kelimeler, sütunlarda yıllar (veya dönemler) yer alır. Hücrelerde ise o kelimenin o yılki *görece frekansı (relative_freq)* bulunur. Bu tablo, bir kelimenin yıllar içindeki trendini (artış/azalış) görselleştirmek ve gelecek tahmini (zaman serisi) yapmak için doğrudan kullanılabilir.

## Ekip Üyeleri

- Burak Emre Çakmak (220507022)
- İbrahim Sali Aktaş (220507075)
- Buğra Kaan Aras (220507051)
- Mustafa Donbaloğlu (220507059)

## Github Repo:

```
https://github.com/MustafaDonbaloglu/pdf_word_scan
```