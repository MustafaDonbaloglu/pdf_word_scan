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
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

def _ensure_nltk_resource(resource_path, package_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name)


def ensure_nltk_data():
    _ensure_nltk_resource("corpora/stopwords", "stopwords")
    _ensure_nltk_resource("corpora/wordnet", "wordnet")
    _ensure_nltk_resource("corpora/words", "words")

# ----------------------------
# PDF'ten metin çekme
# ----------------------------
def extract_text_from_pdf(pdf_path):
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

        if (
            len(text.strip()) < 100
            and convert_from_path is not None
            and pytesseract is not None
        ):
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img) or ""

    except Exception as e:
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
    return set(w.lower() for w in nltk_words.words())


def _is_english_word(word, english_vocab):
    if re.fullmatch(r"[ivxlcdm]+", word) and len(word) <= 6:
        return False

    if len(word) <= 2:
        return word in {"go", "us", "we"}

    if word in english_vocab:
        return True
    return bool(wordnet.synsets(word))


def process_text(text, english_vocab):
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{base_name}.csv"
    xlsx_path = out_dir / f"{base_name}.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="words")


def _latest_mtime(paths):
    latest = 0.0
    for p in paths:
        try:
            latest = max(latest, p.stat().st_mtime)
        except Exception:
            continue
    return latest


def _outputs_mtime(out_dir):
    out_dir = Path(out_dir)
    csv_path = out_dir / "words.csv"
    xlsx_path = out_dir / "words.xlsx"
    if not csv_path.exists() or not xlsx_path.exists():
        return None
    return min(csv_path.stat().st_mtime, xlsx_path.stat().st_mtime)


def _read_existing_source_pdfs(out_dir):
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
    folder_path = Path(folder_path)
    outputs_root = Path(outputs_root)

    ensure_nltk_data()
    english_vocab = _build_english_vocab()

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
        counter = Counter()
        for pdf_path in pdf_paths:
            text = extract_text_from_pdf(str(pdf_path))
            words = process_text(text, english_vocab)
            counter.update(words)

        total_tokens = int(sum(counter.values()))
        df = pd.DataFrame(counter.items(), columns=["word", "frequency"])
        df["year"] = year
        df["term"] = term
        df["source_pdf"] = ", ".join(sorted(set(p.name for p in pdf_paths)))
        df["total_tokens"] = total_tokens
        df["relative_freq"] = df["frequency"] / total_tokens if total_tokens else 0.0
        df = df.sort_values(["frequency", "word"], ascending=[False, True], ignore_index=True)
        all_dfs.append(df)

        save_outputs(df, out_dir, base_name="words")

    return (pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()), processed_any


def save_global_dataset(df, outputs_root="outputs"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="pdfler")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--only-new", action="store_true")
    args = parser.parse_args()

    df_new, processed_any = process_all_pdfs(
        args.pdf_dir,
        outputs_root=args.outputs_dir,
        only_new=args.only_new,
    )

    if processed_any:
        df_all = load_all_term_outputs(args.outputs_dir)
        if not df_all.empty:
            save_global_dataset(df_all, outputs_root=args.outputs_dir)
            print(
                df_all.groupby("word")["frequency"]
                .sum()
                .sort_values(ascending=False)
                .head(20)
            )
    else:
        print("Yeni PDF bulunamadı veya tüm çıktılar güncel.")
