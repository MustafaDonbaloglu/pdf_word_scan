"""
Streamlit arayüzü (app.py)

Bu dosya, `script.py` ile üretilen çıktı dosyalarını (özellikle `outputs/_all/all_terms_long.csv`)
okuyup görselleştiren bir Streamlit uygulamasıdır.

Sunum için akış özeti:
- Veri Hazırlama: Uzun (long) format CSV okunur, yıl bilgisi normalize edilir.
- Özellik Üretimi: Yıl-kelime bazında frekanslar ve göreli frekanslar hesaplanır.
- Basit Tahmin: Her kelimenin yıllara göre göreli frekans trendi lineer olarak uydurulur.
- Değerlendirme: Tahmin edilen Top-N ile gerçek Top-N karşılaştırılır (Precision@N, Jaccard).
- Güven Testi: İsteğe bağlı olarak gerçek değerler çıktı dosyalarından veya bir PDF’ten çıkarılarak
  Wilson güven aralığı ile precision@N için belirsizlik gösterilir.
- İndirmeler: Sonuçlar Excel/PDF olarak indirilebilir.
"""

import io
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception as e:
    raise SystemExit(
        "Streamlit kurulu değil. Kurulum: pip install streamlit\n" + str(e)
    )


@dataclass(frozen=True)
class Score:
    # Tahmin performansını tek bir yerde toplamak için küçük bir veri taşıyıcı (immutable).
    # hits: Top-N listelerdeki kesişim eleman sayısı
    # n: karşılaştırılan Top-N uzunluğu
    # precision_at_n: hits / n
    # jaccard: |A∩B| / |A∪B|
    hits: int
    n: int
    precision_at_n: float
    jaccard: float


def _to_int_year(value: object) -> int | None:
    # CSV’den gelen yıl alanı kimi zaman "2020", kimi zaman "2020.0" ya da boş/bozuk olabilir.
    # Bu yardımcı fonksiyon:
    # - Girdiyi string’e çevirip kırpar
    # - Tam sayıya çevirmeyi dener
    # - Başarısız olursa None döndürür (sonrasında satır elenir)
    try:
        return int(str(value).strip())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_long_dataset(csv_path: str) -> pd.DataFrame:
    # `script.py` tarafından üretilen "long format" dataset’i okur ve normalize eder.
    # Beklenen kolonlar:
    # - word: kelime (normalize edilmiş)
    # - frequency: sayım
    # - year: klasör yapısından gelen yıl (string olabilir)
    # - term: alt klasör / dönem / kategori bilgisi
    # - source_pdf: kaynak PDF adı(ları)
    # Opsiyonel:
    # - total_tokens: ilgili yıl/term/PDF grubundaki toplam token sayısı
    df = pd.read_csv(csv_path)
    required = {"word", "frequency", "year", "term", "source_pdf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Beklenen kolonlar eksik: {sorted(missing)}")

    # Eski/eksik çıktı setleriyle uyumluluk: total_tokens yoksa boş kolon eklenir.
    if "total_tokens" not in df.columns:
        df["total_tokens"] = np.nan

    # Yılı int’e çevirip bozuk satırları temizle.
    df["year_int"] = df["year"].map(_to_int_year)
    df = df.dropna(subset=["year_int"]).copy()
    df["year_int"] = df["year_int"].astype(int)

    # frequency alanını sayısala çevir: bozuk değerleri 0’a düşür.
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def aggregate_year_word(df_long: pd.DataFrame) -> pd.DataFrame:
    # Amaç: Yıl-kelime seviyesinde tek bir tablo üretmek.
    # - frequency: ilgili yıldaki toplam kelime sayımı (term/PDF bazında toplanmış)
    # - total_tokens: ilgili yıldaki toplam token sayısı (term/PDF bazında toplanmış)
    # - relative_freq: frequency / total_tokens (oran; yıl içi normalize edilmiş ölçü)
    base = df_long.copy()

    # total_tokens yıl/term/source_pdf düzeyinde benzersiz olmalı; tekrarları düşürüp yıl bazında topla.
    tokens = base.drop_duplicates(["year_int", "term", "source_pdf"])[
        ["year_int", "term", "source_pdf", "total_tokens"]
    ].copy()
    tokens["total_tokens"] = pd.to_numeric(tokens["total_tokens"], errors="coerce").fillna(0)
    year_tokens = tokens.groupby("year_int", as_index=False)["total_tokens"].sum()

    # Kelime frekanslarını yıl-kelime düzeyinde topla.
    word_freq = (
        base.groupby(["year_int", "word"], as_index=False)["frequency"].sum().copy()
    )
    out = word_freq.merge(year_tokens, on="year_int", how="left")

    # 0 token durumunda bölme hatasını engelle: relative_freq’i 0 kabul et.
    out["relative_freq"] = np.where(
        out["total_tokens"] > 0,
        out["frequency"] / out["total_tokens"],
        0.0,
    )
    return out


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    # Wilson güven aralığı (binom oransal metrikler için daha dengeli).
    # Burada successes=hits, n=Top-N uzunluğu.
    # z=1.96 yaklaşık %95 güven düzeyine karşılık gelir.
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (
        z
        * ((p * (1 - p) / n) + (z**2) / (4 * (n**2))) ** 0.5
        / denom
    )
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def _top_words(df_year_word: pd.DataFrame, year: int, metric: str, top_n: int) -> pd.DataFrame:
    # Seçilen yıl için "gerçek" Top-N listesini üretir.
    # metric:
    # - "frequency": ham sayım
    # - "relative_freq": yıl içi normalize oran
    part = df_year_word[df_year_word["year_int"] == year].copy()
    if part.empty:
        return pd.DataFrame(columns=["word", metric])
    part = part.sort_values(metric, ascending=False, ignore_index=True)
    # Görünümde kullanışlı alanları döndür: word + metrik + yardımcı metrikler.
    cols = ["word", metric, "frequency", "relative_freq"]
    seen: set[str] = set()
    deduped_cols: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            deduped_cols.append(c)
    return part[deduped_cols].head(top_n)


def _predict_top_words(
    df_year_word: pd.DataFrame,
    target_year: int,
    train_max_year: int,
    top_n: int,
    max_vocab: int = 5000,
) -> pd.DataFrame:
    # Basit bir trend-tabanlı tahmin yaklaşımı:
    # 1) Eğitim verisi olarak target_year'dan önceki yıllar alınır.
    # 2) Çok büyük sözlükleri sınırlamak için en sık geçen max_vocab kelime seçilir.
    # 3) Her kelime için (yıl -> relative_freq) doğrusal regresyon (np.polyfit) yapılır.
    # 4) target_year için göreli frekans tahmin edilir ve Top-N olarak sıralanır.
    #
    # Not: Bu bir "zaman serisi" modeli değildir; mevsimsellik/ani kırılmalar yakalanmaz.
    train = df_year_word[df_year_word["year_int"] <= train_max_year].copy()
    train = train[train["year_int"] < target_year]
    if train.empty:
        return pd.DataFrame(columns=["word", "predicted_relative_freq", "slope"]) 

    # Sözlük kısıtı: hesaplamayı hızlandırmak ve gürültüyü azaltmak için en sık kelimeler.
    vocab = (
        train.groupby("word")["frequency"].sum().sort_values(ascending=False).head(max_vocab)
    )
    train = train[train["word"].isin(vocab.index)]

    # Her kelimeye ayrı doğrusal fit: y = slope * year + intercept
    records: list[tuple[str, float, float]] = []
    for word, g in train.groupby("word"):
        x = g["year_int"].to_numpy(dtype=float)
        y = g["relative_freq"].to_numpy(dtype=float)
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            pred = float(slope * target_year + intercept)
        else:
            # Tek gözlem varsa trend çıkarılamaz: son değeri sabit kabul et.
            slope = 0.0
            pred = float(y[-1])
        if pred < 0:
            # Negatif oran anlamlı değil: 0’a kırp.
            pred = 0.0
        records.append((word, pred, float(slope)))

    out = pd.DataFrame(records, columns=["word", "predicted_relative_freq", "slope"])
    out = out.sort_values("predicted_relative_freq", ascending=False, ignore_index=True)
    return out.head(top_n)


def _score_predictions(pred_words: list[str], actual_words: list[str]) -> Score:
    # Top-N listesini set’e çevirip basit kesişim/union metrikleri hesaplar.
    # Precision@N: tahmin listesindeki elemanların kaç tanesi gerçekten de Top-N’de?
    # Jaccard: iki listenin benzerliği (kesişim / birleşim).
    pred_set = set(pred_words)
    actual_set = set(actual_words)
    hits = len(pred_set & actual_set)
    n = max(1, len(pred_words))
    precision = hits / n
    denom = len(pred_set | actual_set) or 1
    jaccard = hits / denom
    return Score(hits=hits, n=len(pred_words), precision_at_n=precision, jaccard=jaccard)


@st.cache_resource(show_spinner=False)
def _load_cefr_analyzer():
    try:
        from cefrpy import CEFRAnalyzer
    except Exception:
        return None
    try:
        return CEFRAnalyzer()
    except Exception:
        return None


_CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
_CEFR_SCORE = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def _coerce_cefr_label(value: object) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name in _CEFR_LEVELS:
        return name
    s = str(value)
    if s in _CEFR_LEVELS:
        return s
    if "." in s:
        tail = s.rsplit(".", 1)[-1]
        if tail in _CEFR_LEVELS:
            return tail
    return None


def _get_cefr_cache() -> dict[str, str | None]:
    cache = st.session_state.get("cefr_cache")
    if isinstance(cache, dict):
        return cache
    st.session_state["cefr_cache"] = {}
    return st.session_state["cefr_cache"]


def _prime_cefr_cache(words: list[str]) -> None:
    analyzer = _load_cefr_analyzer()
    if analyzer is None:
        return
    cache = _get_cefr_cache()
    for w in words:
        if w in cache:
            continue
        try:
            cache[w] = _coerce_cefr_label(analyzer.get_average_word_level_CEFR(w))
        except Exception:
            cache[w] = None


def _annotate_df_with_cefr(df: pd.DataFrame, word_col: str = "word") -> pd.DataFrame:
    out = df.copy()
    words = out[word_col].astype(str).tolist() if word_col in out.columns else []
    _prime_cefr_cache(words)
    cache = _get_cefr_cache()
    out["cefr"] = [cache.get(w) for w in words]
    return out


def _cefr_distribution(
    df: pd.DataFrame,
    word_col: str = "word",
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]] | None:
    analyzer = _load_cefr_analyzer()
    if analyzer is None:
        return None

    base = df[[word_col]].copy()
    if weight_col is not None and weight_col in df.columns:
        base[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    else:
        base["_w"] = 1.0
        weight_col = "_w"

    words = base[word_col].astype(str).tolist()
    _prime_cefr_cache(words)
    cache = _get_cefr_cache()
    base["cefr"] = [cache.get(w) for w in words]

    known = base[base["cefr"].notna()].copy()
    unknown = base[base["cefr"].isna()].copy()

    total_w = float(base[weight_col].sum()) or 1.0
    known_w = float(known[weight_col].sum())
    b2p_w = float(known[known["cefr"].isin(["B2", "C1", "C2"])][weight_col].sum())

    dist = (
        known.groupby("cefr", as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "weight"})
    )
    all_rows = pd.DataFrame({"cefr": _CEFR_LEVELS})
    dist = all_rows.merge(dist, on="cefr", how="left")
    dist["weight"] = dist["weight"].fillna(0.0)
    dist["pct_total"] = dist["weight"] / total_w
    dist["pct_known"] = dist["weight"] / (known_w or 1.0)

    mean_level = 0.0
    if known_w > 0:
        mean_level = float(
            sum(
                _CEFR_SCORE.get(r["cefr"], 0) * float(r["weight"])
                for r in dist.to_dict("records")
            )
            / known_w
        )

    stats = {
        "total_weight": total_w,
        "known_weight": known_w,
        "unknown_weight": float(unknown[weight_col].sum()),
        "known_coverage": known_w / total_w,
        "b2plus_share_total": b2p_w / total_w,
        "b2plus_share_known": b2p_w / (known_w or 1.0),
        "mean_level": mean_level,
        "unique_words": float(pd.Series(words).nunique()),
    }
    return dist, stats


def _df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    # Streamlit download_button için DataFrame’i bellek içi (in-memory) Excel dosyasına çevirir.
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


def _df_to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes | None:
    # Basit tablo-PDF çıktısı üretir.
    # - `fpdf2` kurulu değilse None döndürür (arayüzde bilgilendirme yapılır).
    # - Büyük tabloları taşmamak için ilk 500 satırla sınırlar.
    try:
        from fpdf import FPDF, XPos, YPos
    except Exception:
        return None

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 8, text=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=9)

        # DataFrame’i düz metin satırlarına çevir: "col1 | col2 | ..."
        cols = [c for c in df.columns]
        lines = [" | ".join(cols)]
        for row in df.itertuples(index=False):
            values = [str(v) for v in row]
            lines.append(" | ".join(values))

        # Sayfa genişliğine göre satırları wrap ederek PDF’e bas.
        page_width = pdf.w - pdf.l_margin - pdf.r_margin
        for line in lines[:500]:
            safe = str(line).replace("\t", " ").replace("\r", " ")
            safe = "\n".join(
                textwrap.wrap(safe, width=140, break_long_words=True, break_on_hyphens=False)
            )
            pdf.multi_cell(page_width, 5, text=safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # fpdf çıktısı bazen str döndürebildiği için bytes’e normalize et.
        data = pdf.output(dest="S")
        if isinstance(data, str):
            return data.encode("latin-1", errors="replace")
        return bytes(data)
    except Exception:
        return None


def _find_year_pdf_paths(pdf_root: Path, year: int) -> list[Path]:
    # Doğrulama için kullanılabilecek PDF’leri, `pdfler/<year>/.../*.pdf` altında arar.
    year_dir = pdf_root / str(year)
    if not year_dir.exists():
        return []
    return sorted(year_dir.rglob("*.pdf"))


@st.cache_resource(show_spinner=False)
def _load_nlp_resources():
    # `script.py` içindeki NLP bağımlılıklarını (NLTK veri setleri + sözlük) bir kez yükler.
    # cache_resource: Streamlit yeniden çalıştığında ağır kaynaklar yeniden yüklenmesin.
    import script

    script.ensure_nltk_data()
    english_vocab = script._build_english_vocab()
    return script, english_vocab


def _extract_actual_from_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    # Kullanıcı PDF yüklediğinde (veya varsayılan bir PDF seçildiğinde) gerçek kelime dağılımını çıkarır.
    # Akış:
    # - PDF’i geçici dosyaya yaz (kütüphaneler dosya yolu bekliyor)
    # - script.extract_text_from_pdf ile metni al
    # - script.process_text ile tokenize + stopword + lemmatize + İngilizce filtresi
    # - frekans ve relative_freq hesapla
    import tempfile

    script, english_vocab = _load_nlp_resources()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as f:
        f.write(pdf_bytes)
        f.flush()
        text = script.extract_text_from_pdf(f.name)
    words = script.process_text(text, english_vocab)
    counts = pd.Series(words).value_counts().reset_index()
    counts.columns = ["word", "frequency"]
    counts["frequency"] = counts["frequency"].astype(int)
    counts["relative_freq"] = counts["frequency"] / max(1, int(counts["frequency"].sum()))
    return counts


def main():
    # Streamlit uygulamasının giriş noktası.
    # Burada:
    # - Çıktı dosyası kontrol edilir
    # - Kullanıcıdan yıl/Top-N/metrik/validasyon kaynağı alınır
    # - Tahmin/gerçek karşılaştırması ve grafikler çizilir
    # - Güven testi ve indirmeler sunulur
    st.set_page_config(page_title="PDF Kelime Analiz Arayüzü", layout="wide")
    st.title("PDF Kelime Analiz Arayüzü")

    # Proje kökü: app.py'nin bulunduğu dizin
    root = Path(__file__).resolve().parent
    long_csv = root / "outputs" / "_all" / "all_terms_long.csv"
    if not long_csv.exists():
        st.error("`outputs/_all/all_terms_long.csv` bulunamadı. Önce `python script.py` çalıştırın.")
        st.stop()

    # Dataset’i oku ve yıl-kelime agregasyonunu oluştur.
    df_long = load_long_dataset(str(long_csv))
    df_year_word = aggregate_year_word(df_long)
    years = sorted(df_year_word["year_int"].unique().tolist())
    min_year, max_year = min(years), max(years)

    # Sol panel: kullanıcı ayarları.
    with st.sidebar:
        st.subheader("Ayarlar")
        year = st.selectbox(
            "Analiz yılı",
            years,
            index=years.index(2020) if 2020 in years else len(years) - 1,
        )
        top_n = st.slider("Top N kelime", min_value=10, max_value=200, value=50, step=10)
        metric = st.radio(
            "Sıralama metriği",
            ["relative_freq", "frequency"],
            index=0,
            horizontal=True,
        )
        validate_source = st.radio(
            "Güven testi doğrulama kaynağı",
            ["Hazır çıktılar (outputs)", "PDF ile teyit"],
            index=0,
        )
        st.divider()
        st.subheader("CEFR")
        cefr_enabled = st.checkbox("CEFR analizi göster", value=True)
        cefr_weighting = st.radio(
            "CEFR ağırlık",
            ["Frekans ağırlıklı", "Benzersiz kelime"],
            index=0,
            horizontal=True,
        )
        cefr_exam_max_words = st.slider(
            "Sınav analizi kelime limiti",
            min_value=200,
            max_value=5000,
            value=2000,
            step=200,
        )

    # Ekran üst bilgileri: seçilen yıl ve eğitim aralığı.
    st.write(f"Elimizdeki veri aralığı: {min_year}–{max_year}")
    st.write(f"Seçilen yıl: {year} (eğitim: {min_year}–{year-1})")

    # Eğitim setinin özetini de göstermek için (seçilen yıldan önceki tüm yıllar).
    train_agg = df_year_word[df_year_word["year_int"] < year].copy()
    if not train_agg.empty:
        # Eğitim dönemi boyunca toplam token: relative_freq’i yıl üstü normalize etmek için kullanılır.
        train_total_tokens = (
            df_long[df_long["year_int"] < year]
            .drop_duplicates(["year_int", "term", "source_pdf"])["total_tokens"]
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum()
        )
        train_word = train_agg.groupby("word", as_index=False)["frequency"].sum()
        train_word["relative_freq"] = np.where(
            train_total_tokens > 0,
            train_word["frequency"] / train_total_tokens,
            0.0,
        )
        train_top = train_word.sort_values(metric, ascending=False, ignore_index=True).head(top_n)
    else:
        train_top = pd.DataFrame(columns=["word", "frequency", "relative_freq"])

    # İki kolonlu yerleşim: sol (metrikler + tablolar), sağ (grafikler).
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Doğruluk testi")
        # Tahmin: geçmiş yıllardaki trendlerden seçilen yılın Top-N kelimelerini kestir.
        predicted = _predict_top_words(
            df_year_word,
            target_year=year,
            train_max_year=max_year,
            top_n=top_n,
        )
        # Gerçek: seçilen yılda gerçekten en sık çıkan Top-N kelimeler.
        actual = _top_words(df_year_word, year=year, metric=metric, top_n=top_n)

        if predicted.empty or actual.empty:
            st.warning("Tahmin veya gerçek veri üretilemedi.")
        else:
            # Hızlı performans özeti.
            score = _score_predictions(
                pred_words=predicted["word"].tolist(),
                actual_words=actual["word"].tolist(),
            )
            st.metric("Precision@N", f"{score.precision_at_n:.3f}")
            st.metric("Jaccard", f"{score.jaccard:.3f}")
            st.write(f"Eşleşen kelime: {score.hits}/{score.n}")

        # Tahmin tablosunu göster.
        st.dataframe(predicted, width="stretch")

        if not train_top.empty:
            st.subheader("Eğitim verisi (seçilen yıldan önce)")
            st.dataframe(train_top, width="stretch")

    with col_right:
        st.subheader("Seçilen yılın en sık kelimeleri")
        if actual.empty:
            st.info("Bu yıl için çıktı bulunamadı.")
        else:
            # Bar chart: Top-N kelimelerin dağılımı.
            view = actual[["word", metric]].set_index("word")
            st.bar_chart(view)

            try:
                import matplotlib.pyplot as plt

                # Pasta grafiği: Top 10'u daha görsel bir şekilde özetlemek için.
                pie_df = actual.head(10).copy()
                fig, ax = plt.subplots()
                ax.pie(pie_df[metric], labels=pie_df["word"], autopct="%1.1f%%")
                ax.set_title(f"Top 10 (yıl={year})")
                st.pyplot(fig)
            except Exception:
                pass

        if not train_top.empty:
            st.subheader("Eğitim verisi grafikleri")
            st.bar_chart(train_top[["word", metric]].set_index("word"))
            try:
                import matplotlib.pyplot as plt

                # Eğitim döneminin Top 10 dağılımını ayrı pasta grafiğiyle göster.
                pie_df = train_top.head(10).copy()
                fig, ax = plt.subplots()
                ax.pie(pie_df[metric], labels=pie_df["word"], autopct="%1.1f%%")
                ax.set_title(f"Top 10 (eğitim: {min_year}–{year-1})")
                st.pyplot(fig)
            except Exception:
                pass

    st.divider()
    st.subheader("Güven testi")

    # Güven testi için "gerçek" referansı iki yoldan üretiyoruz:
    # 1) Hazır çıktılar: all_terms_long üzerinden relative_freq’e göre sıralama
    # 2) PDF ile teyit: doğrudan PDF’ten çıkarılan kelime dağılımı
    validation_actual_df: pd.DataFrame | None = None
    if validate_source == "PDF ile teyit":
        pdf_root = root / "pdfler"
        candidates = _find_year_pdf_paths(pdf_root, year)
        default_pdf = str(candidates[0]) if candidates else None
        uploaded = st.file_uploader("Doğrulama PDF yükle (opsiyonel)", type=["pdf"])

        if uploaded is not None:
            validation_actual_df = _extract_actual_from_pdf(uploaded.read())
            st.caption("Gerçek değerler yüklenen PDF’ten çıkarıldı.")
        elif default_pdf is not None:
            try:
                validation_actual_df = _extract_actual_from_pdf(Path(default_pdf).read_bytes())
                st.caption(f"Gerçek değerler dosyadan çıkarıldı: {Path(default_pdf).name}")
            except Exception as e:
                st.warning(f"PDF ile teyit başarısız: {e}")
    else:
        # Hazır çıktılarla teyit: seçilen yıl için geniş Top listesi al (sonra Top-N alınacak).
        validation_actual_df = _top_words(df_year_word, year=year, metric="relative_freq", top_n=5000)

    if st.button("Güven testi çalıştır", type="primary"):
        if predicted.empty or validation_actual_df is None or validation_actual_df.empty:
            st.error("Güven testi için tahmin ve gerçek veri gerekli.")
        else:
            # Referans kelimeleri relative_freq'e göre sırala ve Top-N'i al.
            actual_words = validation_actual_df.sort_values(
                "relative_freq", ascending=False, ignore_index=True
            )["word"].head(top_n).tolist()
            pred_words = predicted["word"].tolist()
            score = _score_predictions(pred_words=pred_words, actual_words=actual_words)
            lo, hi = _wilson_interval(score.hits, score.n, z=1.96)
            st.write(
                f"Güven aralığı (Wilson, %95) precision@N için: {lo:.3f} – {hi:.3f} (N={score.n})"
            )
            st.write(f"Eşleşen kelime: {score.hits}/{score.n}")

    st.divider()
    st.subheader("2026 tahmini")
    # Uygulamada ayrıca "gelecek yıl" örneği olarak 2026 tahmini gösterilir.
    pred_2026 = _predict_top_words(
        df_year_word,
        target_year=2026,
        train_max_year=max_year,
        top_n=top_n,
    )
    st.dataframe(pred_2026, width="stretch")

    st.divider()
    st.subheader("CEFR analizi")
    if not cefr_enabled:
        st.info("CEFR analizi kapalı.")
    else:
        analyzer = _load_cefr_analyzer()
        if analyzer is None:
            st.warning("CEFR analizi için `cefrpy` gerekli. Kurulum: pip install cefrpy")
        else:
            use_weights = cefr_weighting == "Frekans ağırlıklı"

            t1, t2, t3, t4 = st.tabs(
                [
                    f"{year} gerçek",
                    f"{year} tahmin",
                    "2026 tahmin",
                    "Sınav zorluk",
                ]
            )

            with t1:
                df_words = actual[["word", "frequency"]].copy() if not actual.empty else pd.DataFrame()
                if df_words.empty:
                    st.info("CEFR analizi için gerçek veri yok.")
                else:
                    annotated = _annotate_df_with_cefr(df_words, word_col="word")
                    dist_stats = _cefr_distribution(
                        annotated,
                        word_col="word",
                        weight_col="frequency" if use_weights else None,
                    )
                    if dist_stats is None:
                        st.info("CEFR analizi çalıştırılamadı.")
                    else:
                        dist, stats = dist_stats
                        st.metric("Bilinen kapsam", f"{stats['known_coverage']:.1%}")
                        st.metric("B2+ oranı", f"{stats['b2plus_share_total']:.1%}")
                        st.metric("Ortalama seviye (1-6)", f"{stats['mean_level']:.2f}")
                        st.dataframe(dist, width="stretch")
                        b2plus = annotated[annotated["cefr"].isin(["B2", "C1", "C2"])].copy()
                        if not b2plus.empty:
                            st.subheader("B2 ve üstü kelimeler")
                            show_cols = ["word", "cefr"] + (["frequency"] if "frequency" in b2plus.columns else [])
                            st.dataframe(b2plus[show_cols], width="stretch")

            with t2:
                df_words = (
                    predicted[["word", "predicted_relative_freq"]].copy()
                    if not predicted.empty
                    else pd.DataFrame()
                )
                if df_words.empty:
                    st.info("CEFR analizi için tahmin verisi yok.")
                else:
                    annotated = _annotate_df_with_cefr(df_words, word_col="word")
                    dist_stats = _cefr_distribution(
                        annotated,
                        word_col="word",
                        weight_col="predicted_relative_freq" if use_weights else None,
                    )
                    if dist_stats is None:
                        st.info("CEFR analizi çalıştırılamadı.")
                    else:
                        dist, stats = dist_stats
                        st.metric("Bilinen kapsam", f"{stats['known_coverage']:.1%}")
                        st.metric("B2+ oranı", f"{stats['b2plus_share_total']:.1%}")
                        st.metric("Ortalama seviye (1-6)", f"{stats['mean_level']:.2f}")
                        st.dataframe(dist, width="stretch")
                        b2plus = annotated[annotated["cefr"].isin(["B2", "C1", "C2"])].copy()
                        if not b2plus.empty:
                            st.subheader("B2 ve üstü kelimeler")
                            st.dataframe(b2plus[["word", "cefr"]], width="stretch")

            with t3:
                df_words = (
                    pred_2026[["word", "predicted_relative_freq"]].copy()
                    if not pred_2026.empty
                    else pd.DataFrame()
                )
                if df_words.empty:
                    st.info("CEFR analizi için 2026 tahmin verisi yok.")
                else:
                    annotated = _annotate_df_with_cefr(df_words, word_col="word")
                    dist_stats = _cefr_distribution(
                        annotated,
                        word_col="word",
                        weight_col="predicted_relative_freq" if use_weights else None,
                    )
                    if dist_stats is None:
                        st.info("CEFR analizi çalıştırılamadı.")
                    else:
                        dist, stats = dist_stats
                        st.metric("Bilinen kapsam", f"{stats['known_coverage']:.1%}")
                        st.metric("B2+ oranı", f"{stats['b2plus_share_total']:.1%}")
                        st.metric("Ortalama seviye (1-6)", f"{stats['mean_level']:.2f}")
                        st.dataframe(dist, width="stretch")
                        b2plus = annotated[annotated["cefr"].isin(["B2", "C1", "C2"])].copy()
                        if not b2plus.empty:
                            st.subheader("B2 ve üstü kelimeler")
                            st.dataframe(b2plus[["word", "cefr"]], width="stretch")

            with t4:
                df_exam = (
                    df_long.groupby(["year_int", "term", "word"], as_index=False)["frequency"]
                    .sum()
                    .copy()
                )
                rows: list[dict[str, object]] = []
                for (y, term), g in df_exam.groupby(["year_int", "term"]):
                    g = g.sort_values("frequency", ascending=False, ignore_index=True).head(cefr_exam_max_words)
                    g = _annotate_df_with_cefr(g, word_col="word")
                    dist_stats = _cefr_distribution(
                        g,
                        word_col="word",
                        weight_col="frequency" if use_weights else None,
                    )
                    if dist_stats is None:
                        continue
                    _, stats = dist_stats
                    mean = float(stats["mean_level"])
                    lvl = None
                    if mean > 0:
                        idx = int(round(mean)) - 1
                        idx = max(0, min(idx, len(_CEFR_LEVELS) - 1))
                        lvl = _CEFR_LEVELS[idx]
                    rows.append(
                        {
                            "year": int(y),
                            "term": str(term),
                            "difficulty_score": mean,
                            "difficulty_cefr": lvl,
                            "b2plus_pct": stats["b2plus_share_total"],
                            "known_pct": stats["known_coverage"],
                            "sample_words": int(stats["unique_words"]),
                        }
                    )

                out = pd.DataFrame(rows)
                if out.empty:
                    st.info("Sınav zorluk tablosu üretilemedi.")
                else:
                    out = out.sort_values("difficulty_score", ascending=False, ignore_index=True)
                    st.dataframe(out, width="stretch")

    # İndirme butonları: 2026 tahminini Excel/PDF olarak sun.
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            "2026 tahmini Excel indir",
            data=_df_to_excel_bytes(pred_2026, sheet_name="pred_2026"),
            file_name=f"pred_2026_top{top_n}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dl_col2:
        pdf_bytes = _df_to_pdf_bytes(pred_2026, title=f"2026 Tahmini Top {top_n}")
        if pdf_bytes is None:
            st.caption("PDF için `pip install fpdf2` gerekli.")
        else:
            st.download_button(
                "2026 tahmini PDF indir",
                data=pdf_bytes,
                file_name=f"pred_2026_top{top_n}.pdf",
                mime="application/pdf",
            )
    with dl_col3:
        if not actual.empty:
            st.download_button(
                f"{year} gerçek Excel indir",
                data=_df_to_excel_bytes(actual, sheet_name=f"actual_{year}"),
                file_name=f"actual_{year}_top{top_n}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.subheader("İndirmeler")
    # Ek indirmeler: seçilen yıl tahmini, PDF/Excel ve eğitim seti özet çıktısı.
    d1, d2, d3 = st.columns(3)
    with d1:
        if not predicted.empty:
            st.download_button(
                f"{year} tahmin Excel indir",
                data=_df_to_excel_bytes(predicted, sheet_name=f"pred_{year}"),
                file_name=f"pred_{year}_top{top_n}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with d2:
        if not predicted.empty:
            pdf_bytes = _df_to_pdf_bytes(predicted, title=f"{year} Tahmini Top {top_n}")
            if pdf_bytes is None:
                st.caption("PDF için `pip install fpdf2` gerekli.")
            else:
                st.download_button(
                    f"{year} tahmin PDF indir",
                    data=pdf_bytes,
                    file_name=f"pred_{year}_top{top_n}.pdf",
                    mime="application/pdf",
                )
    with d3:
        if not train_top.empty:
            st.download_button(
                f"Eğitim top Excel indir",
                data=_df_to_excel_bytes(train_top, sheet_name="train_top"),
                file_name=f"train_before_{year}_top{top_n}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    # Modül doğrudan çalıştırıldığında Streamlit uygulamasını başlatacak ana fonksiyon çağrısı.
    main()

