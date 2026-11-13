# -*- coding: utf-8 -*-
"""Fungsi inti segmentasi pemakaian air PDAM: pembersihan, fitur, K-Medoids (PAM), Random Forest, prediksi."""
import io, re
import numpy as np
import pandas as pd
from scipy import stats
import kmedoids
from sklearn.metrics import (pairwise_distances, davies_bouldin_score, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BULAN = [f"M{i:02d}" for i in range(1, 13)]
NAMA_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
FITUR = ["rata_rata", "std", "cv", "maks", "bulan_nol", "tren"]
LOG_KOLOM = ["rata_rata", "std", "maks"]
SEED = 42

PETA_BULAN = {"januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6, "juli": 7,
              "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
              "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "agu": 8, "agt": 8,
              "sep": 9, "okt": 10, "nov": 11, "des": 12}


# ------------------------------------------------------------------ data
def buat_data_dummy(n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    n_r, n_s = int(n * 0.06), int(n * 0.54)
    n_t = n - n_r - n_s
    pola = lambda m, sd, k: np.clip(rng.normal(m, sd, size=(k, 12)), 0, None)
    rendah = pola(2, 5, n_r); rendah[rng.random(rendah.shape) < 0.75] = 0
    sedang = pola(11, 5, n_s)
    tinggi = pola(27, 7.5, n_t) + np.linspace(0, 2, 12)
    X = np.vstack([rendah, sedang, tinggi]).round(0)
    ids = [f"1303010{str(i).zfill(5)}" for i in range(n)]
    gol = rng.choice([1, 12, 1, 1, 1, 2, 4], size=n)
    rayon = rng.choice(["01", "02", "03", "12"], size=n)
    alamat = [f"Jorong {rng.integers(1, 9)} Nagari {rng.integers(1, 12)}" for _ in range(n)]
    X[rng.integers(0, n, 20), rng.integers(0, 12, 20)] = -rng.integers(1, 30, 20)
    X[rng.integers(0, n, 8), rng.integers(0, 12, 8)] = rng.integers(300, 900, 8)
    out = {}
    for j, b in enumerate(BULAN):
        d = pd.DataFrame({"kode_rayon": rayon, "no_langganan": ids, "alamat": alamat,
                          "golongan": gol, "pemakaian": X[:, j]})
        d.loc[rng.integers(0, n, 5), "pemakaian"] = np.nan
        out[b] = d
    out["M05"] = pd.concat([out["M05"], out["M05"].iloc[:15]])
    return out


def bulan_dari_nama(nama):
    n = nama.lower()
    for k, v in sorted(PETA_BULAN.items(), key=lambda x: -len(x[0])):
        if k in n:
            return v
    m = re.search(r"(?<!\d)(0?[1-9]|1[0-2])(?!\d)", n)
    return int(m.group(1)) if m else None


def standarkan_kolom(df):
    peta = {}
    for c in df.columns:
        k = str(c).lower().replace("_", " ").strip()
        if "rayon" in k: peta[c] = "kode_rayon"
        elif "langganan" in k or "pelanggan" in k: peta[c] = "no_langganan"
        elif "alamat" in k: peta[c] = "alamat"
        elif "gol" in k: peta[c] = "golongan"
        elif "pakai" in k or "pemakaian" in k or "m3" in k or "volume" in k: peta[c] = "pemakaian"
    df = df.rename(columns=peta)
    butuh = ["kode_rayon", "no_langganan", "alamat", "golongan", "pemakaian"]
    hilang = [c for c in butuh if c not in df.columns]
    if hilang:
        raise ValueError(f"Kolom tidak ditemukan: {hilang}. Kolom terbaca: {list(df.columns)}")
    return df[butuh]


def baca_berkas(daftar):
    """daftar: list (nama, bytes). Mengembalikan dict M01..M12 -> DataFrame."""
    out = {}
    for nama, isi in daftar:
        b = bulan_dari_nama(nama)
        if b is None:
            raise ValueError(f"Bulan tidak terbaca dari nama berkas: {nama}")
        df = pd.read_csv(io.BytesIO(isi)) if nama.lower().endswith(".csv") else pd.read_excel(io.BytesIO(isi))
        out[f"M{b:02d}"] = standarkan_kolom(df)
    return out


def baca_rekap(isi_bytes):
    """CSV rekap lebar (pemisah ; atau ,) -> dict M01..M12 seperti berkas bulanan."""
    teks = isi_bytes.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(teks), sep=None, engine="python", dtype=str)
    peta = {}
    for c in df.columns:
        k = str(c).lower().strip()
        if "rayon" in k: peta["kode_rayon"] = c
        elif "langganan" in k: peta["no_langganan"] = c
        elif "alamat" in k: peta["alamat"] = c
        elif k in ("gol", "golongan"): peta["golongan"] = c
    nomor = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "jun": 6, "jul": 7, "agu": 8, "agt": 8,
             "sep": 9, "okt": 10, "nov": 11, "des": 12}
    bulan_kol = {}
    for c in df.columns:
        k = str(c).lower().strip()[:3]
        if k in nomor and any(ch.isdigit() for ch in str(c)):
            bulan_kol[nomor[k]] = c
    if len(peta) < 4 or len(bulan_kol) < 12:
        raise ValueError(f"Kolom tidak lengkap. Terbaca identitas {list(peta)} dan bulan {sorted(bulan_kol)}")
    out = {}
    for no in range(1, 13):
        d = pd.DataFrame({k: df[v] for k, v in peta.items()})
        d["pemakaian"] = pd.to_numeric(df[bulan_kol[no]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        out[f"M{no:02d}"] = d
    return out


def bersihkan(bulanan, golongan, faktor_lonjakan=3.0, batas_min=200):
    alur = []
    bagian = []
    for b, d in sorted(bulanan.items()):
        d = d.copy()
        d["no_langganan"] = d["no_langganan"].astype(str).str.strip()
        d = d.drop_duplicates("no_langganan", keep="first")
        bagian.append(d.set_index("no_langganan")[["pemakaian"]].rename(columns={"pemakaian": b}))
    wide = pd.concat(bagian, axis=1)
    for b in BULAN:
        if b not in wide.columns:
            wide[b] = np.nan
    ident = (pd.concat([d[["no_langganan", "kode_rayon", "alamat", "golongan"]] for d in bulanan.values()])
             .assign(no_langganan=lambda x: x["no_langganan"].astype(str).str.strip(),
                     golongan=lambda x: pd.to_numeric(x["golongan"], errors="coerce"))
             .drop_duplicates("no_langganan", keep="last").set_index("no_langganan"))
    data = ident.join(wide[BULAN], how="inner").reset_index()
    alur.append(("Gabungan berkas bulanan (pelanggan unik)", len(data), ""))
    data = data[data["golongan"].isin(golongan)].copy()
    alur.append((f"Seleksi golongan {golongan}", len(data), ""))
    data = data[~data[BULAN].isna().all(axis=1)].copy()
    alur.append(("Hapus pelanggan tanpa catatan 12 bulan", len(data), ""))

    neg = data[BULAN] < 0
    n_neg = int(neg.sum().sum()); n_neg_pel = int(neg.any(axis=1).sum())
    X1 = data[BULAN].mask(neg)
    alur.append(("Nilai negatif menjadi kosong", n_neg, f"sel, pada {n_neg_pel} pelanggan"))
    q1, q3 = X1.quantile(.25, axis=1), X1.quantile(.75, axis=1)
    batas = np.maximum(q3 + faktor_lonjakan * (q3 - q1), batas_min)
    lonjak = X1.gt(batas, axis=0)
    n_lon = int(lonjak.sum().sum())
    X2 = X1.mask(lonjak)
    alur.append(("Lonjakan menjadi kosong", n_lon, f"sel, di atas maks(Q3 + {faktor_lonjakan:g} x IQR; {batas_min:g} m3)"))
    nonaktif = X2.fillna(0).sum(axis=1) == 0
    alur.append(("Pelanggan nonaktif dikeluarkan", int(nonaktif.sum()), "tanpa pemakaian selama 12 bulan"))
    keep = ~nonaktif
    n_imp = int(X2.loc[keep].isna().sum().sum())
    med = X2.median(axis=1)
    data[BULAN] = X2.apply(lambda kol: kol.fillna(med))
    data = data[keep].reset_index(drop=True)
    alur.append(("Sel diisi median pelanggan", n_imp, "sel"))
    alur.append(("Data bersih", len(data), ""))
    tabel = pd.DataFrame(alur, columns=["Langkah", "Jumlah", "Keterangan"])
    return data, tabel


# ------------------------------------------------------------------ fitur
def bentuk_fitur(df, kolom):
    X = df[kolom].to_numpy(dtype=float)
    t = np.arange(1, X.shape[1] + 1)
    rata = X.mean(axis=1)
    std = X.std(axis=1, ddof=1)
    cv = np.divide(std, rata, out=np.zeros_like(std), where=rata > 0)
    return pd.DataFrame({"rata_rata": rata, "std": std, "cv": cv, "maks": X.max(axis=1),
                         "bulan_nol": (X == 0).sum(axis=1), "tren": np.polyfit(t, X.T, 1)[0]}, index=df.index)


def normalisasi(fitur):
    sebelum = fitur[FITUR].apply(stats.skew)
    f = fitur[FITUR].copy(); f[LOG_KOLOM] = np.log1p(f[LOG_KOLOM])
    sesudah = f.apply(stats.skew)
    sc = StandardScaler()
    Z = pd.DataFrame(sc.fit_transform(f), columns=FITUR, index=fitur.index)
    return Z, sc, pd.DataFrame({"sebelum": sebelum, "sesudah": sesudah})


# ------------------------------------------------------------------ K-Medoids
def pam(D, k, seed=SEED):
    return kmedoids.pam(D, k, init="build", max_iter=100, random_state=seed)


def uji_k(Z, rentang, sampel=None, seed=SEED, progres=None):
    Zv = Z.to_numpy(dtype=np.float32)
    if sampel and sampel < len(Zv):
        Zv = Zv[np.random.default_rng(seed).choice(len(Zv), sampel, replace=False)]
    D = pairwise_distances(Zv).astype(np.float32)
    rows = []
    for i, k in enumerate(rentang):
        r = pam(D, k)
        rows.append({"K": k, "Total cost": r.loss, "Iterasi": r.n_iter,
                     "Silhouette": kmedoids.silhouette(D, r.labels)[0],
                     "DBI": davies_bouldin_score(Zv, r.labels)})
        if progres: progres((i + 1) / len(rentang))
    return pd.DataFrame(rows).set_index("K")


def klasterisasi(Z, fitur, k):
    Zv = Z.to_numpy(dtype=np.float32)
    D = pairwise_distances(Zv).astype(np.float32)
    r = pam(D, k)
    lab = np.array(r.labels); med = np.array(r.medoids)
    sil, sil_s = kmedoids.silhouette(D, lab, samples=True)
    sil_k = pd.Series(sil_s).groupby(lab).mean()
    biaya = [float(D[lab == i][:, med[i]].sum()) for i in range(k)]
    dbi = davies_bouldin_score(Zv, lab)
    profil = fitur[FITUR].assign(klaster=lab).groupby("klaster").agg(
        jumlah=("rata_rata", "size"), rata_rata=("rata_rata", "mean"), std=("std", "mean"),
        cv=("cv", "mean"), maks=("maks", "mean"), bulan_nol=("bulan_nol", "mean"), tren=("tren", "mean"))
    profil["persentase"] = profil["jumlah"] / profil["jumlah"].sum() * 100
    urut = profil["rata_rata"].sort_values().index.tolist()
    nama = ["Rendah", "Sedang", "Tinggi"] if k == 3 else [f"Tingkat {i + 1}" for i in range(k)]
    label_map = {int(c): nama[i] for i, c in enumerate(urut)}
    profil["segmen"] = [label_map[int(i)] for i in profil.index]
    profil["cost"] = biaya
    profil["silhouette"] = sil_k.reindex(profil.index).values
    del D
    return {"labels": lab, "medoids": med, "loss": r.loss, "iter": r.n_iter, "silhouette": sil,
            "dbi": dbi, "profil": profil, "label_map": label_map, "nama": nama}


# ------------------------------------------------------------------ Random Forest
def siapkan_X(fitur, data, kolom_rayon=None):
    X = fitur[FITUR].copy(); X[LOG_KOLOM] = np.log1p(X[LOG_KOLOM])
    X["golongan"] = data["golongan"].astype(int).values
    ray = pd.get_dummies(data["kode_rayon"].astype(str), prefix="rayon").astype(int)
    ray.index = X.index
    X = pd.concat([X, ray], axis=1)
    if kolom_rayon is not None:
        X = X.reindex(columns=kolom_rayon, fill_value=0)
    return X


def latih_rf(X, y, urutan, rasio_uji=0.2, pohon=200, seed=SEED, cv=True):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=rasio_uji, stratify=y, random_state=seed)
    rf = RandomForestClassifier(n_estimators=pohon, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    p, r, f, s = precision_recall_fscore_support(yte, pred, labels=urutan, zero_division=0)
    tab = pd.DataFrame({"Presisi": p, "Recall": r, "F1-score": f, "Dukungan": s}, index=urutan)
    tab.loc["Rata-rata makro"] = [p.mean(), r.mean(), f.mean(), s.sum()]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    hasil = {"model": rf, "kolom": list(X.columns), "akurasi": accuracy_score(yte, pred), "tabel": tab,
             "cm": pd.DataFrame(confusion_matrix(yte, pred, labels=urutan), index=urutan, columns=urutan),
             "importance": imp, "n_latih": len(Xtr), "n_uji": len(Xte),
             "split": pd.DataFrame({"Latih": ytr.value_counts(), "Uji": yte.value_counts()}).reindex(urutan)}
    if cv:
        sk = cross_val_score(RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                    random_state=seed, n_jobs=-1),
                             X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="f1_macro")
        hasil["cv_f1"] = (sk.mean(), sk.std())
    return hasil


# ------------------------------------------------------------------ prediksi
REKOMENDASI = {
    "Rendah": "Verifikasi meter dan status sambungan (kemungkinan tidak aktif atau meter bermasalah); tinjau kebutuhan layanan.",
    "Sedang": "Pelayanan reguler; pantau perubahan pola secara berkala.",
    "Tinggi": "Prioritaskan perencanaan debit dan tekanan; pantau kebocoran dan pelanggan dengan tren naik.",
}


def prediksi_satu(pemakaian, golongan, rayon, km, rf_dini):
    p = np.array(pemakaian, dtype=float)
    if len(p) < 6:
        raise ValueError("Minimal 6 bulan pemakaian.")
    if (p < 0).any():
        raise ValueError("Pemakaian tidak boleh negatif.")
    df = pd.DataFrame([p], columns=[f"M{i + 1:02d}" for i in range(len(p))])
    out = {}
    if len(p) == 12:
        f = bentuk_fitur(df, list(df.columns))
        fl = f[FITUR].copy(); fl[LOG_KOLOM] = np.log1p(fl[LOG_KOLOM])
        z = km["scaler"].transform(fl)
        d = np.linalg.norm(km["medoid_z"] - z, axis=1)
        out["segmen_medoid"] = km["label_map"][int(d.argmin())]
        out["jarak"] = {f"M{i + 1}": round(float(v), 4) for i, v in enumerate(d)}
    f6 = bentuk_fitur(df.iloc[:, :6], list(df.columns[:6]))
    x = f6[FITUR].copy(); x[LOG_KOLOM] = np.log1p(x[LOG_KOLOM]); x["golongan"] = int(golongan)
    for kol in rf_dini["kolom"]:
        if kol.startswith("rayon_"):
            x[kol] = int(kol == f"rayon_{rayon}")
    x = x[rf_dini["kolom"]]
    pr = rf_dini["model"].predict_proba(x)[0]
    out["segmen_dini"] = rf_dini["model"].classes_[pr.argmax()]
    out["peluang"] = {c: float(v) for c, v in zip(rf_dini["model"].classes_, pr)}
    return out
