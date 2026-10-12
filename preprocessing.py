import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

try:
    from nltk.corpus import stopwords

    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

DATA_DIR = Path(__file__).parent / "data"

FILES = {
    # "dsa_5tahun.csv": "Data Science and Analytics",
    "sib_5tahun.csv": "Sistem Informasi Bisnis",
    # "infor_5tahun.csv": "Informatika",
}

MAP_STATUS = {
    "Selesai": 0,
    "Dalam Pengerjaan": 1,
    "Baru" : 2,
}

try:
  stop_words_en = set(stopwords.words("english"))
except Exception:
  stop_words_en = set()

keep_words = {
    "tidak",
    "bukan",
    "belum",
    "tanpa",
    "no",
    "not",
    "nor",
    "only",
    "akhir",
    "baru",
    "diri",
    "lanjut",
    "keluar",
    "awal",
    "lalu",
    "masalah",
    "hari",
    "dua",
    "tiga",
    "satu",
    "pertama",
    "kedua",
    "luar",
    "tengah",
    "besar",
    "kecil",
    "tinggi",
    "lama",
    "up",
    "out",
    "off",
    "over",
    "under",
    "same",
    "all",
}

stop_id = set(StopWordRemoverFactory().get_stop_words()) | {
    "berdasarkan",
    "beberapa",
    "berbagai",
}
stop_en = set(stopwords.words("english")) | {"using"}

stop_domain = {
    # kata pembuka/penghubung
    "analisis",
    "implementasi",
    "penerapan",
    "menggunakan",
    "metode",
    "berbasis",
    "studi",
    "kasus",
    "perancangan",
    "pembuatan",
    "pengembangan",
    "rancang",
    "bangun",
    "aplikasi",
    "sistem",
    "informasi",
    "meningkatkan",
    # nama perusahaan/badan usaha
    "pt",
    "cv",
    "ud",
    "x",
    "xyz",
    "tbk",
}

STOPWORDS = stop_id | stop_en | stop_domain

NORMALISASI = {
    "pemesanan": "pesan",  # Sastrawi -> "mesan"
    "pemetaan": "peta",  # Sastrawi -> "meta"
    "pengecekan": "cek",  # Sastrawi -> "kece"
    "application": "aplikasi",
    "app": "aplikasi",
    "apps": "aplikasi",
    "mobile application": "aplikasi mobile",
    "mobile app": "aplikasi mobile",
    "web application": "aplikasi web",
    "web app": "aplikasi web",
    "web": "website",
    "websites": "website",
    "web based": "berbasis web",
    "berbasis website": "berbasis web",
    "system": "sistem",
    "systems": "sistem",
    "information system": "sistem informasi",
    "monitoring": "pemantauan",
    "tracking": "pelacakan",
    "management": "manajemen",
    "inventory": "inventaris",
    "expert system": "sistem pakar",
    "recommender system": "sistem rekomendasi",
    "recommendation system": "sistem rekomendasi",
    "artificial intelligence": "ai",
    "kecerdasan buatan": "ai",
    "pencarian": "cari",
    "pembelajaran": "ajar",
    "pembuatan": "buat",
    "perancangan": "rancang",
    "penerapan": "terap",
    "implementasi": "terap",
    "analisis": "analisis",
    "analisa": "analisis",
    "analysis": "analisis",
    "klasifikasi": "klasifikasi",
    "prediksi": "prediksi",
    "forecasting": "prediksi",
    "deteksi": "deteksi",
    "pengenalan": "pengenalan",
    "optimalisasi": "optimalisasi",
    "rekomendasi": "rekomendasi",
    "sentimen": "sentimen",
}

JANGAN_STEM = {
    "petra",
    "bali",
    "kediri",
    "keunikan",
    "pengangguran",
    "babai",
    "tolis",
    "pengadaan",
    "ai",
    "cnn",
    "rnn",
    "lstm",
    "svm",
    "knn",
    "nlp",
    "llm",
    "yolo",
    "pso",
    "saw",
    "arima",
}

stemmer = StemmerFactory().create_stemmer()


def preprocess(judul):
    # biar ga ada nilai kosong
    t = "" if judul is None else str(judul)
    t = judul.lower()

    t = re.sub(r"(?<=\w)[*'’](?=\w)", "", t)
    t = re.sub(r"\b([a-z0-9])-(?=\w)", r"\1", t)
    t = re.sub(r"[\-\u2013\u2014/]", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)

    tokens = t.split()
    tokens = [w for w in tokens if not w.isdigit()]
    tokens = [NORMALISASI.get(w, w) for w in tokens]
    tokens = [w for w in tokens if w not in STOPWORDS and (len(w) > 1 or w == "r")]
    tokens = [w if w in JANGAN_STEM else stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


# CLEANING
def baca_csv(path):
    for enc in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Encoding file {path} tidak dikenali")


def proses_file(file, jurusan):
    df = baca_csv(DATA_DIR / file)
    df.columns = df.columns.str.strip().str.lower()
    print(f"\n{file}")
    print("Ukuran data awal :", df.shape)

    print("\nMissing value per kolom:\n", df.isna().sum())
    print("\nBaris duplikat    :", df.duplicated().sum())
    print("NRP duplikat      :", df["nrp"].duplicated().sum())
    print("Judul duplikat    :", df["judul"].str.lower().str.strip().duplicated().sum())
    print("Panjang NRP       :", df["nrp"].str.len().unique())
    print("Sel berisi newline / spasi ganda:")
    print(df.apply(lambda s: s.str.contains(r"\n|\s{2,}", na=False).sum()))
    print("\nNilai unik status:\n", df["status"].value_counts())

    for c in df.columns:
        df[c] = (
            df[c]
            .str.replace(r"_x0092_|\x92", "'", regex=True)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    df = df.rename(columns={"periode_sia": "periode"})
    # df = df.drop(columns="periode", errors="ignore")

    if "unit" in df.columns:
        df = df.rename(columns={"unit": "jurusan"})
    else:
        df["jurusan"] = jurusan

    df["jurusan"] = df["jurusan"].replace("Sistem Informasi", "Sistem Informasi Bisnis")
    
    def get_period_weight(val):
        v = str(val).lower()
        if "genap" in v:
            return 2
        elif "gasal" in v:
            return 1
        return 0
    
    # Hapus duplicate entry utk NRP sama (ambil yg terbaru)
    if "tahun" in df.columns:
        df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").fillna(0)
        periode_col = "periode" if "periode" in df.columns else None
        df["_periode_weight"] = (
            df[periode_col].apply(get_period_weight) if periode_col else 0
        )
        df = df.sort_values(by=["tahun", "_periode_weight"], kind="stable")
        
    dup_mask = df.duplicated(subset=["nrp"], keep="last")
    df = df[~dup_mask].drop(
        columns=["_periode_weight", "periode"], errors="ignore"
    ).reset_index(drop=True)
    
    print(
        f"Jumlah data setelah duplikat dibersihkan: {len(df)}"
    )
    
    # Pembimbing -> pembimbing_1 & pembimbing_2 ("|" atau ";")
    # Jika hanya 1 pembimbing -> pembimbing_2 kosong
    pemb = df["pembimbing"].str.split(r"\s*[|;]\s*", expand=True)
    df["pembimbing_1"] = pemb[0]
    df["pembimbing_2"] = pemb[1] if pemb.shape[1] > 1 else None
    df = df.drop(columns="pembimbing")

    # Encode status
    df["status"] = df["status"].map(MAP_STATUS)
    df = df[df["status"].notna()].reset_index(drop=True)
    df["status"] = df["status"].astype(int)

    # Preprocessing judul
    df["judul_clean"] = df["judul"].apply(preprocess)

    freq = Counter(" ".join(df["judul_clean"]).split())
    print("\n20 kata terbanyak setelah preprocessing:\n", freq.most_common(20))
    print(
        "Judul kosong setelah preprocessing:",
        (df["judul_clean"].str.strip() == "").sum(),
    )

    kolom = [
        "nrp",
        "nama",
        "jurusan",
        "pembimbing_1",
        "pembimbing_2",
        "judul",
        "judul_clean",
        "semester",
        "tahun",
        "status",
    ]
    output = DATA_DIR / file.replace(".csv", "_clean.csv")
    df[kolom].to_csv(output, index=False)
    print("\nUkuran data akhir:", df[kolom].shape)
    print("Tersimpan ke", output)


if __name__ == "__main__":
    for file, jurusan in FILES.items():
        proses_file(file, jurusan)