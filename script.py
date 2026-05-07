"""
PDF kelime çıkarım ve çıktı üretim script'i (script.py)

Bu dosya, `pdfler/` klasörü altındaki PDF’leri tarar, metin çıkarımı yapar, metni temizleyip
İngilizce kelime listesi üretir ve sonuçları CSV/Excel olarak `outputs/` altına yazar.

Beklenen klasör yapısı (önerilen):
  pdfler/
    2020/
      termA/
        file1.pdf
        file2.pdf
      termB/
        ...

Çıktılar:
- outputs/<year>/<term>/words.csv + words.xlsx (kelime frekansları + relative_freq + total_tokens)
- outputs/_all/all_terms_long.csv + all_terms_long.xlsx (tüm year/term’lerin birleşik long formatı)

Metin çıkarım stratejisi (fallback’li):
1) pdfplumber (varsa) sayfa sayfa metin çekme
2) pdfminer (varsa) tek adım metin çekme
3) Eğer metin çok kısa kaldıysa ve OCR bağımlılıkları varsa: pdf2image + pytesseract ile OCR

Not: `app.py`, özellikle `outputs/_all/all_terms_long.csv` dosyasını okuyarak arayüz sağlar.
"""

from collections import Counter
import argparse
import os
from pathlib import Path
import re

import nltk
from nltk.corpus import stopwords, words as nltk_words, wordnet
from nltk.stem import WordNetLemmatizer

import pandas as pd

try:
    # PDF'ten metin çıkarmada en iyi sonuçlar için öncelikli tercih.
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    # pdfplumber yoksa/başarısızsa metin çıkarmada ikinci seçenek.
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

try:
    # OCR motoru (sadece görüntü tabanlı PDF’lerde devreye girer).
    import pytesseract
except Exception:
    pytesseract = None

try:
    # PDF sayfalarını görüntüye çevirme (OCR için gerekli).
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

def _ensure_nltk_resource(resource_path, package_name):
    # NLTK veri setleri kod çalışmadan önce sistemde bulunmalı.
    # resource_path: NLTK içinde aranacak yol (örn. corpora/stopwords)
    # package_name: eksikse indirilecek paket adı (örn. stopwords)
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name)


def ensure_nltk_data():
    # Metin işleme adımlarında kullanılan NLTK veri setlerini garanti altına alır.
    # - stopwords: yaygın İngilizce durak kelimeleri
    # - wordnet: lemmatization ve synset kontrolü
    # - words: İngilizce sözlük listesi
    _ensure_nltk_resource("corpora/stopwords", "stopwords")
    _ensure_nltk_resource("corpora/wordnet", "wordnet")
    _ensure_nltk_resource("corpora/words", "words")

# ----------------------------
# PDF'ten metin çekme
# ----------------------------
def extract_text_from_pdf(pdf_path):
    # PDF’ten metin çıkarma:
    # - Önce doğrudan metin katmanı okumayı dener (pdfplumber/pdfminer).
    # - Eğer çıkan metin çok kısa ise (örn. taranmış görüntü PDF) OCR’e düşer.
    # - Hata durumunda OCR mümkünse yine denenir; değilse hata yukarı fırlatılır.
    text = ""

    try:
        if pdfplumber is not None:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif pdfminer_extract_text is not None:
            text = pdfminer_extract_text(pdf_path) or ""

        # Metin "çok kısa" kaldıysa büyük ihtimalle PDF görüntü tabanlıdır; OCR dene.
        if (
            len(text.strip()) < 100
            and convert_from_path is not None
            and pytesseract is not None
        ):
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img) or ""

    except Exception as e:
        # Doğrudan metin çıkarımı başarısız olsa bile OCR yolu varsa son bir şans.
        if convert_from_path is not None and pytesseract is not None:
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img) or ""
        else:
            raise e

    return text


# ----------------------------
# Metin temizleme ve kelime çıkarma
# ----------------------------
def _build_english_vocab():
    # NLTK words corpus’unu kullanarak (lower-case) bir İngilizce kelime sözlüğü üretir.
    # Bu set, hızlı üyelik kontrolü için kullanılır.
    return set(w.lower() for w in nltk_words.words())


def _is_english_word(word, english_vocab):
    # “Bu token İngilizce kelime mi?” filtresi.
    # 1) Roma rakamları gibi false-positive’leri ele
    # 2) Çok kısa kelimelerde (<=2) kontrollü istisna listesi kullan
    # 3) Sözlükte varsa kabul et; değilse WordNet synset var mı bak (daha esnek)
    if re.fullmatch(r"[ivxlcdm]+", word) and len(word) <= 6:
        return False

    if len(word) <= 2:
        return word in {"go", "us", "we"}

    if word in english_vocab:
        return True
    return bool(wordnet.synsets(word))


def process_text(text, english_vocab):
    # Ham metinden temizlenmiş kelime listesi üretir.
    # Adımlar:
    # - Regex tokenization: sadece [a-zA-Z] kelimeleri
    # - Stopword temizliği
    # - Lemmatization (WordNetLemmatizer)
    # - İngilizce filtresi (_is_english_word)
    tokens = re.findall(r"\b[a-zA-Z]+\b", (text or "").lower())

    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    lemmas = [w for w in lemmas if _is_english_word(w, english_vocab)]
    return lemmas


# ----------------------------
# Ana işlem
# ----------------------------
def analyze_pdf(pdf_path, year, term, english_vocab):
    # Tek bir PDF dosyasını analiz eder ve kelime frekans DataFrame’i üretir.
    # Bu fonksiyon tekil kullanım için uygun; klasör bazlı toplu işlemde `process_all_pdfs` kullanılır.
    pdf_path = str(pdf_path)

    text = extract_text_from_pdf(pdf_path)

    words = process_text(text, english_vocab)

    word_counts = Counter(words)

    df = pd.DataFrame(word_counts.items(), columns=["word", "frequency"])
    df["year"] = year
    df["term"] = term
    df["source_pdf"] = os.path.basename(pdf_path)
    df = df.sort_values(["frequency", "word"], ascending=[False, True], ignore_index=True)

    return df


# ---------------a-------------
# KLASÖRDEKİ TÜM PDFLERİ İŞLE
# ----------------------------
def parse_pdf_context(pdf_path, root_dir):
    # PDF dosya yolundan bağlam (year, term) çıkarmaya çalışır.
    # `root_dir` altında göreli yol beklenir:
    #   <year>/<term>/dosya.pdf
    # Uymuyorsa "unknown" döner.
    pdf_path = Path(pdf_path)
    root_dir = Path(root_dir)
    try:
        rel = pdf_path.relative_to(root_dir)
        parts = rel.parts
    except Exception:
        parts = ()

    year = parts[0] if len(parts) >= 1 else "unknown"
    term = parts[1] if len(parts) >= 2 else "unknown"
    return year, term


def save_outputs(df, out_dir, base_name="words"):
    # Bir term’in sonucunu hem CSV hem Excel olarak kaydeder.
    # - UTF-8 SIG: Excel’de Türkçe karakterlerin daha sorunsuz açılması için.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{base_name}.csv"
    xlsx_path = out_dir / f"{base_name}.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="words")


def _latest_mtime(paths):
    # Verilen Path listesindeki en güncel (latest) değiştirilme zamanını döndürür.
    latest = 0.0
    for p in paths:
        try:
            latest = max(latest, p.stat().st_mtime)
        except Exception:
            continue
    return latest


def _outputs_mtime(out_dir):
    # Mevcut çıktı dosyalarının (words.csv + words.xlsx) “en eski” mtime’ını döndürür.
    # min(...) kullanma nedeni: iki dosyadan biri daha eskiyse çıktı seti güncel sayılmasın.
    out_dir = Path(out_dir)
    csv_path = out_dir / "words.csv"
    xlsx_path = out_dir / "words.xlsx"
    if not csv_path.exists() or not xlsx_path.exists():
        return None
    return min(csv_path.stat().st_mtime, xlsx_path.stat().st_mtime)


def _read_existing_source_pdfs(out_dir):
    # Daha önce üretilmiş words.csv içinden source_pdf alanını okuyup set’e çevirir.
    # source_pdf birden fazla PDF’i ", " ile birleştirebildiği için split edilir.
    out_dir = Path(out_dir)
    csv_path = out_dir / "words.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["source_pdf"], nrows=1)
        if df.empty:
            return set()
        value = str(df.iloc[0]["source_pdf"] or "")
        if not value.strip():
            return set()
        return set([v.strip() for v in value.split(",") if v.strip()])
    except Exception:
        return None


def _should_process_term(out_dir, pdf_paths, only_new):
    # only_new=True ise gereksiz yeniden hesaplamayı azaltmak için term bazında karar verir.
    # Yeniden üretme koşulları:
    # - Çıktı dosyaları yoksa
    # - PDF listesi değişmişse (dosya adı seti farklıysa)
    # - PDF’lerden herhangi biri çıktıdan daha yeni ise
    if not only_new:
        return True

    pdf_paths = list(pdf_paths)
    if not pdf_paths:
        return False

    out_mtime = _outputs_mtime(out_dir)
    if out_mtime is None:
        return True

    current_names = set(p.name for p in pdf_paths)
    existing_names = _read_existing_source_pdfs(out_dir)
    if existing_names is None:
        return True
    if existing_names != current_names:
        return True

    latest_pdf_mtime = _latest_mtime(pdf_paths)
    return latest_pdf_mtime > out_mtime


def process_all_pdfs(folder_path, outputs_root="outputs", only_new=False):
    # Belirtilen klasör altında tüm PDF’leri bulur, (year, term) bazında gruplar ve işler.
    # Her grup için:
    # - Tüm PDF’lerin token’ları bir Counter’da toplanır
    # - total_tokens ve relative_freq hesaplanır
    # - outputs/<year>/<term>/words.(csv|xlsx) yazılır
    # Ayrıca tüm grupların birleşimi long format olarak döndürülür.
    folder_path = Path(folder_path)
    outputs_root = Path(outputs_root)

    ensure_nltk_data()
    english_vocab = _build_english_vocab()

    # PDF’leri year/term anahtarına göre gruplayarak “aynı klasördeki PDF’leri birlikte” ele al.
    term_pdfs = {}
    for pdf_path in sorted(folder_path.rglob("*.pdf")):
        year, term = parse_pdf_context(pdf_path, folder_path)
        key = (str(year), str(term))
        term_pdfs.setdefault(key, []).append(pdf_path)

    all_dfs = []
    processed_any = False

    for (year, term), pdf_paths in sorted(term_pdfs.items()):
        out_dir = outputs_root / str(year) / str(term)
        if not _should_process_term(out_dir, pdf_paths, only_new=only_new):
            continue

        processed_any = True
        # Aynı term altındaki PDF’lerin kelime sayılarını tek bir Counter’da biriktir.
        counter = Counter()
        for pdf_path in pdf_paths:
            text = extract_text_from_pdf(str(pdf_path))
            words = process_text(text, english_vocab)
            counter.update(words)

        # total_tokens: bu term için (tüm PDF’ler birleşik) toplam token sayısı.
        total_tokens = int(sum(counter.values()))
        df = pd.DataFrame(counter.items(), columns=["word", "frequency"])
        df["year"] = year
        df["term"] = term
        df["source_pdf"] = ", ".join(sorted(set(p.name for p in pdf_paths)))
        df["total_tokens"] = total_tokens
        # relative_freq: frekansı token sayısına bölerek normalize edilmiş oran.
        df["relative_freq"] = df["frequency"] / total_tokens if total_tokens else 0.0
        df = df.sort_values(["frequency", "word"], ascending=[False, True], ignore_index=True)
        all_dfs.append(df)

        save_outputs(df, out_dir, base_name="words")

    return (pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()), processed_any


def save_global_dataset(df, outputs_root="outputs"):
    # Tüm year/term sonuçlarını tek bir global dataset’e çevirir.
    # - long: satır bazında (word, year, term, frequency, total_tokens, relative_freq, ...)
    # - pivot: word x (year, term) matrisi (relative_freq ile)
    outputs_root = Path(outputs_root)
    out_dir = outputs_root / "_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / "all_terms_long.csv"
    long_xlsx = out_dir / "all_terms_long.xlsx"
    df.to_csv(long_csv, index=False, encoding="utf-8-sig")

    pivot = df.pivot_table(
        index="word",
        columns=["year", "term"],
        values="relative_freq",
        aggfunc="sum",
        fill_value=0.0,
    )

    with pd.ExcelWriter(long_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="long")
        pivot.to_excel(writer, sheet_name="pivot")


def load_all_term_outputs(outputs_root="outputs"):
    # outputs/<year>/<term>/words.csv dosyalarının tamamını okuyup tek DataFrame’de birleştirir.
    # _all klasörü bu taramada dahil edilmez (zaten birleşik çıktı).
    outputs_root = Path(outputs_root)
    parts = []
    for csv_path in sorted(outputs_root.glob("*/*/words.csv")):
        if "_all" in csv_path.parts:
            continue
        try:
            parts.append(pd.read_csv(csv_path))
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------
# ÇALIŞTIR
# ----------------------------
if __name__ == "__main__":
    # Komut satırından çalıştırma senaryosu:
    # Örnek:
    #   python script.py --pdf-dir pdfler --outputs-dir outputs --only-new
    #
    # --only-new verilirse: çıktıları güncel olan klasörler atlanır (hız için).
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="pdfler")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--only-new", action="store_true")
    args = parser.parse_args()

    # PDF’leri işle ve term bazında yeni üretim yapıldı mı bilgisini al.
    df_new, processed_any = process_all_pdfs(
        args.pdf_dir,
        outputs_root=args.outputs_dir,
        only_new=args.only_new,
    )

    if processed_any:
        # Yeni üretim yapıldıysa tüm çıktıları birleştirip global long/pivot dataset oluştur.
        df_all = load_all_term_outputs(args.outputs_dir)
        if not df_all.empty:
            save_global_dataset(df_all, outputs_root=args.outputs_dir)
            # Konsola hızlı bir özet: en sık 20 kelime.
            print(
                df_all.groupby("word")["frequency"]
                .sum()
                .sort_values(ascending=False)
                .head(20)
            )
    else:
        print("Yeni PDF bulunamadı veya tüm çıktılar güncel.")
