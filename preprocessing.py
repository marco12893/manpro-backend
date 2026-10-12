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
}

stemmer = StemmerFactory().create_stemmer()


def preprocess(judul):
    t = judul.lower()

    t = re.sub(r"(?<=\w)[*'’](?=\w)", "", t)
    t = re.sub(r"\b([a-z0-9])-(?=\w)", r"\1", t)

    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)

    tokens = t.split()
    tokens = [w for w in tokens if not w.isdigit()]
    tokens = [NORMALISASI.get(w, w) for w in tokens]
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
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
    df = df.drop(columns="periode", errors="ignore")

    if "unit" in df.columns:
        df = df.rename(columns={"unit": "jurusan"})
    else:
        df["jurusan"] = jurusan

    df["jurusan"] = df["jurusan"].replace("Sistem Informasi", "Sistem Informasi Bisnis")

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