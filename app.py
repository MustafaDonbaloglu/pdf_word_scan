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
    hits: int
    n: int
    precision_at_n: float
    jaccard: float


def _to_int_year(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_long_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"word", "frequency", "year", "term", "source_pdf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Beklenen kolonlar eksik: {sorted(missing)}")

    if "total_tokens" not in df.columns:
        df["total_tokens"] = np.nan

    df["year_int"] = df["year"].map(_to_int_year)
    df = df.dropna(subset=["year_int"]).copy()
    df["year_int"] = df["year_int"].astype(int)
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def aggregate_year_word(df_long: pd.DataFrame) -> pd.DataFrame:
    base = df_long.copy()
    tokens = base.drop_duplicates(["year_int", "term", "source_pdf"])[
        ["year_int", "term", "source_pdf", "total_tokens"]
    ].copy()
    tokens["total_tokens"] = pd.to_numeric(tokens["total_tokens"], errors="coerce").fillna(0)
    year_tokens = tokens.groupby("year_int", as_index=False)["total_tokens"].sum()

    word_freq = (
        base.groupby(["year_int", "word"], as_index=False)["frequency"].sum().copy()
    )
    out = word_freq.merge(year_tokens, on="year_int", how="left")
    out["relative_freq"] = np.where(
        out["total_tokens"] > 0,
        out["frequency"] / out["total_tokens"],
        0.0,
    )
    return out


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
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
    part = df_year_word[df_year_word["year_int"] == year].copy()
    if part.empty:
        return pd.DataFrame(columns=["word", metric])
    part = part.sort_values(metric, ascending=False, ignore_index=True)
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
    train = df_year_word[df_year_word["year_int"] <= train_max_year].copy()
    train = train[train["year_int"] < target_year]
    if train.empty:
        return pd.DataFrame(columns=["word", "predicted_relative_freq", "slope"]) 

    vocab = (
        train.groupby("word")["frequency"].sum().sort_values(ascending=False).head(max_vocab)
    )
    train = train[train["word"].isin(vocab.index)]

    records: list[tuple[str, float, float]] = []
    for word, g in train.groupby("word"):
        x = g["year_int"].to_numpy(dtype=float)
        y = g["relative_freq"].to_numpy(dtype=float)
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            pred = float(slope * target_year + intercept)
        else:
            slope = 0.0
            pred = float(y[-1])
        if pred < 0:
            pred = 0.0
        records.append((word, pred, float(slope)))

    out = pd.DataFrame(records, columns=["word", "predicted_relative_freq", "slope"])
    out = out.sort_values("predicted_relative_freq", ascending=False, ignore_index=True)
    return out.head(top_n)


def _score_predictions(pred_words: list[str], actual_words: list[str]) -> Score:
    pred_set = set(pred_words)
    actual_set = set(actual_words)
    hits = len(pred_set & actual_set)
    n = max(1, len(pred_words))
    precision = hits / n
    denom = len(pred_set | actual_set) or 1
    jaccard = hits / denom
    return Score(hits=hits, n=len(pred_words), precision_at_n=precision, jaccard=jaccard)


def _df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


def _df_to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes | None:
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

        cols = [c for c in df.columns]
        lines = [" | ".join(cols)]
        for row in df.itertuples(index=False):
            values = [str(v) for v in row]
            lines.append(" | ".join(values))

        page_width = pdf.w - pdf.l_margin - pdf.r_margin
        for line in lines[:500]:
            safe = str(line).replace("\t", " ").replace("\r", " ")
            safe = "\n".join(
                textwrap.wrap(safe, width=140, break_long_words=True, break_on_hyphens=False)
            )
            pdf.multi_cell(page_width, 5, text=safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        data = pdf.output(dest="S")
        if isinstance(data, str):
            return data.encode("latin-1", errors="replace")
        return bytes(data)
    except Exception:
        return None


def _find_year_pdf_paths(pdf_root: Path, year: int) -> list[Path]:
    year_dir = pdf_root / str(year)
    if not year_dir.exists():
        return []
    return sorted(year_dir.rglob("*.pdf"))


@st.cache_resource(show_spinner=False)
def _load_nlp_resources():
    import script

    script.ensure_nltk_data()
    english_vocab = script._build_english_vocab()
    return script, english_vocab


def _extract_actual_from_pdf(pdf_bytes: bytes) -> pd.DataFrame:
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
    st.set_page_config(page_title="PDF Kelime Analiz Arayüzü", layout="wide")
    st.title("PDF Kelime Analiz Arayüzü")

    root = Path(__file__).resolve().parent
    long_csv = root / "outputs" / "_all" / "all_terms_long.csv"
    if not long_csv.exists():
        st.error("`outputs/_all/all_terms_long.csv` bulunamadı. Önce `python script.py` çalıştırın.")
        st.stop()

    df_long = load_long_dataset(str(long_csv))
    df_year_word = aggregate_year_word(df_long)
    years = sorted(df_year_word["year_int"].unique().tolist())
    min_year, max_year = min(years), max(years)

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

    st.write(f"Elimizdeki veri aralığı: {min_year}–{max_year}")
    st.write(f"Seçilen yıl: {year} (eğitim: {min_year}–{year-1})")

    train_agg = df_year_word[df_year_word["year_int"] < year].copy()
    if not train_agg.empty:
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

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Doğruluk testi")
        predicted = _predict_top_words(
            df_year_word,
            target_year=year,
            train_max_year=max_year,
            top_n=top_n,
        )
        actual = _top_words(df_year_word, year=year, metric=metric, top_n=top_n)

        if predicted.empty or actual.empty:
            st.warning("Tahmin veya gerçek veri üretilemedi.")
        else:
            score = _score_predictions(
                pred_words=predicted["word"].tolist(),
                actual_words=actual["word"].tolist(),
            )
            st.metric("Precision@N", f"{score.precision_at_n:.3f}")
            st.metric("Jaccard", f"{score.jaccard:.3f}")
            st.write(f"Eşleşen kelime: {score.hits}/{score.n}")

        st.dataframe(predicted, width="stretch")

        if not train_top.empty:
            st.subheader("Eğitim verisi (seçilen yıldan önce)")
            st.dataframe(train_top, width="stretch")

    with col_right:
        st.subheader("Seçilen yılın en sık kelimeleri")
        if actual.empty:
            st.info("Bu yıl için çıktı bulunamadı.")
        else:
            view = actual[["word", metric]].set_index("word")
            st.bar_chart(view)

            try:
                import matplotlib.pyplot as plt

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

                pie_df = train_top.head(10).copy()
                fig, ax = plt.subplots()
                ax.pie(pie_df[metric], labels=pie_df["word"], autopct="%1.1f%%")
                ax.set_title(f"Top 10 (eğitim: {min_year}–{year-1})")
                st.pyplot(fig)
            except Exception:
                pass

    st.divider()
    st.subheader("Güven testi")

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
        validation_actual_df = _top_words(df_year_word, year=year, metric="relative_freq", top_n=5000)

    if st.button("Güven testi çalıştır", type="primary"):
        if predicted.empty or validation_actual_df is None or validation_actual_df.empty:
            st.error("Güven testi için tahmin ve gerçek veri gerekli.")
        else:
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
    pred_2026 = _predict_top_words(
        df_year_word,
        target_year=2026,
        train_max_year=max_year,
        top_n=top_n,
    )
    st.dataframe(pred_2026, width="stretch")

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
    main()

