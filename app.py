# -*- coding: utf-8 -*-
"""Aplikasi Streamlit: Segmentasi Penggunaan Air Pelanggan PDAM Kabupaten Sijunjung
(K-Medoids Clustering dan Random Forest)."""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pdam_core as core
from pdam_core import BULAN, NAMA_BULAN, FITUR

st.set_page_config(page_title="Segmentasi Air PDAM Sijunjung", page_icon="💧", layout="wide")
sns.set_theme(style="whitegrid")
S = st.session_state
S.setdefault("hasil", {})
H = S["hasil"]

LANGKAH = ["Beranda", "1. Data", "2. Fitur dan K", "3. K-Medoids", "4. Random Forest", "5. Prediksi", "6. Unduh hasil"]

with st.sidebar:
    st.title("💧 PDAM Sijunjung")
    st.caption("Segmentasi pelanggan rumah tangga dengan K-Medoids dan Random Forest")
    halaman = st.radio("Menu", LANGKAH, label_visibility="collapsed")
    st.divider()
    st.markdown("**Status proses**")
    for kunci, label in [("data", "Data bersih"), ("Z", "Fitur"), ("km", "K-Medoids"),
                         ("rf1", "RF skenario 1"), ("rf2", "RF skenario 2")]:
        st.write(("✅ " if kunci in H else "⬜ ") + label)
    if st.button("Reset semua hasil"):
        S["hasil"] = {}
        st.rerun()


def butuh(*kunci):
    hilang = [k for k in kunci if k not in H]
    if hilang:
        st.warning("Selesaikan langkah sebelumnya terlebih dahulu.")
        st.stop()


# =========================================================== Beranda
if halaman == "Beranda":
    st.title("Segmentasi Penggunaan Air Pelanggan PDAM Kabupaten Sijunjung")
    st.markdown(
        """
Aplikasi ini menjalankan tahapan penelitian secara berurutan:

1. **Data**: unggah 12 berkas pemakaian bulanan, gabungkan, seleksi golongan, dan bersihkan.
2. **Fitur dan K**: bentuk enam fitur (rata-rata, simpang baku, CV, maksimum, bulan nol, tren), transformasi `log1p`, standardisasi z-score, lalu uji jumlah klaster dengan Silhouette dan DBI.
3. **K-Medoids**: kelompokkan pelanggan dengan PAM, beri nama segmen setelah profil klaster dibaca.
4. **Random Forest**: klasifikasi segmen dengan dua skenario (fitur 12 bulan dan fitur semester pertama).
5. **Prediksi**: tentukan segmen satu pelanggan atau banyak pelanggan sekaligus.
6. **Unduh hasil**: tabel untuk Bab IV dalam satu berkas Excel.

Format berkas bulanan: kolom kode rayon, no langganan, alamat, golongan, dan pemakaian. Nama berkas harus memuat nama atau nomor bulan, misalnya `01_Januari.xlsx`.
        """
    )
    st.info("Data dummy tersedia di menu Data untuk mencoba alur. Angka dari data dummy tidak boleh dipakai pada tesis.")

# =========================================================== 1. Data
elif halaman == "1. Data":
    st.header("1. Data: gabung, seleksi, dan bersihkan")
    sumber = st.radio("Sumber data", ["Unggah 1 berkas rekap (CSV)", "Unggah 12 berkas bulanan", "Data dummy (uji coba)"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    gol = c1.multiselect("Golongan dipakai", [1, 2, 4, 5, 10, 12, 15, 35], default=[1, 12],
                         help="Golongan 01 rumah tangga A dan golongan 12 rumah tangga B.")
    faktor = c2.number_input("Pengali IQR (batas Q3 + k × IQR)", 1.0, 10.0, 3.0, 0.5)
    batas = c3.number_input("Batas lonjakan minimum (m³)", 10, 1000, 200, 10,
                            help="Sesuaikan dengan batas teknis rumah tangga dari PDAM.")
    bulanan = None
    if sumber.startswith("Unggah 1 berkas"):
        rekap = st.file_uploader("Berkas rekap (csv): kolom Kode Rayon, No Langganan, Alamat, Gol, Jan 2025 ... Des 2025", type=["csv"])
        if rekap is not None:
            try:
                bulanan = core.baca_rekap(rekap.getvalue())
                st.success(f"Rekap terbaca: {len(bulanan['M01']):,} baris, 12 bulan.".replace(",", "."))
            except Exception as e:
                st.error(f"Gagal membaca berkas: {e}")
    elif sumber.startswith("Unggah"):
        berkas = st.file_uploader("Berkas bulanan (xlsx atau csv)", type=["xlsx", "xls", "csv"],
                                  accept_multiple_files=True)
        if berkas:
            try:
                bulanan = core.baca_berkas([(f.name, f.getvalue()) for f in berkas])
                st.success(f"Terbaca {len(bulanan)} bulan: " + ", ".join(NAMA_BULAN[int(b[1:]) - 1] for b in sorted(bulanan)))
                if len(bulanan) < 12:
                    st.warning("Berkas kurang dari 12 bulan; bulan yang hilang dianggap kosong dan akan diimputasi.")
            except Exception as e:
                st.error(f"Gagal membaca berkas: {e}")
    else:
        n = st.slider("Jumlah pelanggan dummy", 500, 15000, 3000, 500)
        bulanan = core.buat_data_dummy(n)
    if bulanan is not None and st.button("Gabungkan dan bersihkan", type="primary"):
        with st.spinner("Memproses..."):
            data, alur = core.bersihkan(bulanan, gol, faktor, batas)
        S["hasil"] = {"data": data, "alur": alur, "dummy": not sumber.startswith("Unggah")}
        st.rerun()
    if "data" in H:
        data = H["data"]
        if H.get("dummy"):
            st.warning("Memakai data dummy.")
        st.subheader("Alur jumlah data")
        st.dataframe(H["alur"], hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.metric("Pelanggan bersih", f"{len(data):,}".replace(",", "."))
        b.write(data["golongan"].value_counts().rename("pelanggan").to_frame())
        st.subheader("Analisis eksploratif")
        desk = data[BULAN].describe().T; desk.index = NAMA_BULAN
        st.dataframe(desk.style.format("{:,.2f}"), width="stretch")
        fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
        ax[0].plot(NAMA_BULAN, data[BULAN].mean().values, marker="o"); ax[0].set_title("Rata-rata pemakaian per bulan (m³)")
        ax[1].hist(data[BULAN].mean(axis=1), bins=60, color="#3b78b4"); ax[1].set_title("Distribusi rata-rata tahunan (m³)")
        sns.boxplot(data=data[BULAN].set_axis(NAMA_BULAN, axis=1), ax=ax[2], fliersize=1); ax[2].set_title("Sebaran per bulan")
        plt.tight_layout(); st.pyplot(fig)
        st.dataframe(data.head(20), width="stretch")

# =========================================================== 2. Fitur dan K
elif halaman == "2. Fitur dan K":
    st.header("2. Pembentukan fitur dan penentuan K")
    butuh("data")
    data = H["data"]
    if st.button("Bentuk fitur, transformasi, dan standardisasi", type="primary") or "Z" in H:
        if "Z" not in H:
            fitur = core.bentuk_fitur(data, BULAN)
            Z, sc, skew = core.normalisasi(fitur)
            H.update({"fitur": fitur, "Z": Z, "scaler": sc, "skew": skew})
        st.subheader("Statistik fitur (nilai asli)")
        st.dataframe(H["fitur"].describe().T.style.format("{:,.4f}"), width="stretch")
        a, b = st.columns(2)
        with a:
            st.subheader("Kemencengan (skewness)")
            st.caption("Rata_rata, Std, dan Maks ditransformasi ln(1 + x).")
            st.dataframe(H["skew"].style.format("{:,.4f}"), width="stretch")
        with b:
            st.subheader("Korelasi antarfitur (z-score)")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(H["Z"].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
        st.divider()
        st.subheader("Uji jumlah klaster")
        c1, c2, c3 = st.columns(3)
        k_min = c1.number_input("K minimum", 2, 10, 2)
        k_maks = c2.number_input("K maksimum", 3, 12, 6)
        sampel = c3.number_input("Sampel untuk uji (0 = semua)", 0, 20000, 6000 if len(data) > 8000 else 0, 500,
                                 help="Matriks jarak penuh memakai sekitar 0,9 GB untuk 15 ribu pelanggan.")
        if st.button("Jalankan uji K"):
            bar = st.progress(0.0)
            H["uji_k"] = core.uji_k(H["Z"], range(int(k_min), int(k_maks) + 1), int(sampel) or None, progres=bar.progress)
        if "uji_k" in H:
            u = H["uji_k"]
            st.dataframe(u.style.format({"Total cost": "{:,.2f}", "Silhouette": "{:.4f}", "DBI": "{:.4f}"}), width="stretch")
            fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
            for a_, kol in zip(ax, ["Total cost", "Silhouette", "DBI"]):
                a_.plot(u.index, u[kol], marker="o"); a_.set_title(kol); a_.set_xlabel("K")
            plt.tight_layout(); st.pyplot(fig)
            st.caption(f"Silhouette tertinggi pada K = {u['Silhouette'].idxmax()}; DBI terendah pada K = {u['DBI'].idxmin()}.")

# =========================================================== 3. K-Medoids
elif halaman == "3. K-Medoids":
    st.header("3. Klasterisasi K-Medoids (PAM)")
    butuh("Z")
    k = st.number_input("Jumlah klaster (K)", 2, 10, 3)
    if st.button("Jalankan K-Medoids", type="primary"):
        with st.spinner("Menghitung matriks jarak dan PAM..."):
            H["km"] = core.klasterisasi(H["Z"], H["fitur"], int(k))
            H["data"] = H["data"].assign(klaster=H["km"]["labels"],
                                         segmen=pd.Series(H["km"]["labels"]).map(H["km"]["label_map"]).values)
            H["fitur"] = H["fitur"].assign(segmen=H["data"]["segmen"].values)
            for kunci in ("rf1", "rf2"):
                H.pop(kunci, None)
    if "km" in H:
        km = H["km"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total cost", f"{km['loss']:,.2f}")
        c2.metric("Iterasi", km["iter"])
        c3.metric("Silhouette", f"{km['silhouette']:.4f}")
        c4.metric("DBI", f"{km['dbi']:.4f}")
        s = km["silhouette"]
        st.caption("Kriteria Kaufman dan Rousseeuw: " + ("kuat" if s > .70 else "cukup" if s > .50 else
                   "lemah" if s > .25 else "tidak berstruktur") + ".")
        st.subheader("Profil klaster dan label segmen")
        pf = km["profil"].copy(); pf.index = [f"C{i + 1}" for i in pf.index]
        st.dataframe(pf.style.format({"rata_rata": "{:.2f}", "std": "{:.2f}", "cv": "{:.2f}", "maks": "{:.2f}",
                                      "bulan_nol": "{:.2f}", "tren": "{:.2f}", "persentase": "{:.2f}",
                                      "cost": "{:,.2f}", "silhouette": "{:.4f}"}), width="stretch")
        st.caption("Nama segmen ditetapkan setelah klaster terbentuk, berdasarkan urutan rata-rata pemakaian.")
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(6, 4))
            for nm in km["nama"]:
                ax.plot(NAMA_BULAN, H["data"].loc[H["data"]["segmen"] == nm, BULAN].mean().values, marker="o", label=nm)
            ax.legend(); ax.set_title("Rata-rata pemakaian bulanan per segmen (m³)")
            st.pyplot(fig)
        with b:
            from sklearn.decomposition import PCA
            p = PCA(2, random_state=42).fit_transform(H["Z"].to_numpy())
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=p[:, 0], y=p[:, 1], hue=H["data"]["segmen"], s=8, palette="Set2", ax=ax)
            ax.set_title("Sebaran segmen (proyeksi PCA, ilustrasi)")
            st.pyplot(fig)
        st.subheader("Medoid akhir (z-score)")
        med = H["Z"].iloc[km["medoids"]].copy()
        med.insert(0, "no_langganan", H["data"].loc[km["medoids"], "no_langganan"].values)
        med.insert(0, "data_ke", km["medoids"] + 1)
        med.index = [f"M{i + 1}" for i in range(len(med))]
        st.dataframe(med.style.format({c: "{:.4f}" for c in FITUR}), width="stretch")
        st.subheader("Contoh perhitungan jarak satu pelanggan")
        idx = st.number_input("Data ke-", 1, len(H["data"]), 1) - 1
        x = H["Z"].iloc[idx].to_numpy()
        for m, mi in enumerate(km["medoids"]):
            kud = (x - H["Z"].iloc[mi].to_numpy()) ** 2
            st.write(f"Jarak ke M{m + 1}: √{kud.sum():.4f} = **{np.sqrt(kud.sum()):.4f}**")
        st.write(f"Pelanggan {H['data'].loc[idx, 'no_langganan']} → klaster C{km['labels'][idx] + 1} "
                 f"(segmen {H['data'].loc[idx, 'segmen']}).")

# =========================================================== 4. Random Forest
elif halaman == "4. Random Forest":
    st.header("4. Klasifikasi Random Forest")
    butuh("km")
    c1, c2 = st.columns(2)
    pohon = c1.number_input("Jumlah pohon", 50, 1000, 200, 50)
    uji = c2.slider("Proporsi data uji", 0.1, 0.4, 0.2, 0.05)
    cv = st.checkbox("Hitung validasi silang 5-lipat (lebih lama)", value=True)
    if st.button("Latih model", type="primary"):
        data, fitur = H["data"], H["fitur"]
        y = data["segmen"]
        nama = H["km"]["nama"]
        with st.spinner("Melatih skenario 1 dan 2..."):
            H["rf1"] = core.latih_rf(core.siapkan_X(fitur, data), y, nama, uji, int(pohon), cv=cv)
            f6 = core.bentuk_fitur(data, BULAN[:6])
            H["rf2"] = core.latih_rf(core.siapkan_X(f6, data), y, nama, uji, int(pohon), cv=cv)
    if "rf1" in H:
        nama = H["km"]["nama"]
        ringkas = pd.DataFrame({"Akurasi": [H["rf1"]["akurasi"], H["rf2"]["akurasi"]],
                                "F1-makro": [H["rf1"]["tabel"].loc["Rata-rata makro", "F1-score"],
                                             H["rf2"]["tabel"].loc["Rata-rata makro", "F1-score"]]},
                               index=["Skenario 1: fitur 12 bulan", "Skenario 2: fitur Januari–Juni"])
        st.dataframe(ringkas.style.format("{:.4f}"), width="stretch")
        st.info("Skenario 1 memakai fitur yang sama dengan pembentuk label, sehingga akurasi tinggi tidak mencerminkan "
                "kemampuan prediksi pelanggan baru. Skenario 2 lebih jujur sebagai ukuran prediksi dini.")
        tab1, tab2 = st.tabs(["Skenario 1 (12 bulan)", "Skenario 2 (6 bulan)"])
        for tab, kunci in ((tab1, "rf1"), (tab2, "rf2")):
            with tab:
                r = H[kunci]
                a, b = st.columns(2)
                with a:
                    st.write(f"Latih: {r['n_latih']} | Uji: {r['n_uji']}")
                    st.dataframe(r["tabel"].style.format({"Presisi": "{:.4f}", "Recall": "{:.4f}",
                                                          "F1-score": "{:.4f}", "Dukungan": "{:.0f}"}), width="stretch")
                    if "cv_f1" in r:
                        st.write(f"F1-makro validasi silang 5-lipat: {r['cv_f1'][0]:.4f} ± {r['cv_f1'][1]:.4f}")
                with b:
                    fig, ax = plt.subplots(figsize=(4.5, 3.6))
                    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
                    st.pyplot(fig)
                fig, ax = plt.subplots(figsize=(7, 3.5))
                r["importance"].head(10).iloc[::-1].plot.barh(color="#3b78b4", ax=ax)
                ax.set_title("Feature importance (10 teratas)")
                st.pyplot(fig)

# =========================================================== 5. Prediksi
elif halaman == "5. Prediksi":
    st.header("5. Prediksi segmen pelanggan")
    butuh("km", "rf2")
    km = {"scaler": H["scaler"], "medoid_z": H["Z"].iloc[H["km"]["medoids"]].to_numpy(),
          "label_map": H["km"]["label_map"]}
    rf_dini = {"model": H["rf2"]["model"], "kolom": H["rf2"]["kolom"]}
    rayons = sorted(H["data"]["kode_rayon"].astype(str).unique())
    golongan = sorted(H["data"]["golongan"].astype(int).unique())
    mode = st.radio("Mode", ["Satu pelanggan", "Banyak pelanggan (berkas)"], horizontal=True)
    if mode == "Satu pelanggan":
        a, b, c = st.columns(3)
        g = a.selectbox("Golongan", golongan)
        r = b.selectbox("Kode rayon", rayons)
        n = c.slider("Jumlah bulan tersedia", 6, 12, 12)
        cols = st.columns(6)
        nilai = [cols[i % 6].number_input(NAMA_BULAN[i], 0.0, 2000.0, 0.0, 1.0, key=f"b{i}") for i in range(n)]
        if st.button("Prediksi", type="primary"):
            try:
                out = core.prediksi_satu(nilai, g, r, km, rf_dini)
                seg = out.get("segmen_medoid", out["segmen_dini"])
                st.success(f"Segmen: **{seg}**")
                if "segmen_medoid" in out:
                    st.write("Berdasarkan medoid terdekat (12 bulan). Jarak:", out["jarak"])
                st.write(f"Prediksi dini Random Forest (6 bulan pertama): **{out['segmen_dini']}**")
                st.bar_chart(pd.Series(out["peluang"], name="Peluang"))
                st.write("**Rekomendasi:**", core.REKOMENDASI.get(seg, "-"))
            except Exception as e:
                st.error(str(e))
    else:
        st.caption("Berkas xlsx atau csv berisi kolom no_langganan, kode_rayon, golongan, dan M01–M12 "
                   "(atau minimal M01–M06).")
        f = st.file_uploader("Berkas pelanggan", type=["xlsx", "csv"])
        if f is not None:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            butuhkol = {"no_langganan", "kode_rayon", "golongan"} | set(BULAN[:6])
            if not butuhkol <= set(df.columns):
                st.error(f"Kolom wajib: {sorted(butuhkol)}")
            else:
                hasil = []
                for _, row in df.iterrows():
                    bln = [row[b] for b in BULAN if b in df.columns and pd.notna(row[b])]
                    try:
                        o = core.prediksi_satu(bln, row["golongan"], str(row["kode_rayon"]), km, rf_dini)
                        hasil.append(o.get("segmen_medoid", o["segmen_dini"]))
                    except Exception as e:
                        hasil.append(f"galat: {e}")
                df["segmen"] = hasil
                st.dataframe(df, width="stretch")
                st.download_button("Unduh hasil (CSV)", df.to_csv(index=False).encode("utf-8"), "prediksi_segmen.csv")

# =========================================================== 6. Unduh
elif halaman == "6. Unduh hasil":
    st.header("6. Unduh hasil")
    butuh("km")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        H["alur"].to_excel(xl, sheet_name="alur_data", index=False)
        H["data"][BULAN].describe().T.to_excel(xl, sheet_name="statistik_bulanan")
        H["skew"].to_excel(xl, sheet_name="skewness")
        if "uji_k" in H: H["uji_k"].to_excel(xl, sheet_name="uji_K")
        pf = H["km"]["profil"].copy(); pf.index = [f"C{i + 1}" for i in pf.index]
        pf.to_excel(xl, sheet_name="profil_klaster")
        for kunci, nm in (("rf1", "rf_skenario1"), ("rf2", "rf_skenario2")):
            if kunci in H:
                H[kunci]["tabel"].to_excel(xl, sheet_name=nm)
                H[kunci]["cm"].to_excel(xl, sheet_name=nm + "_cm")
                H[kunci]["importance"].to_frame("importance").to_excel(xl, sheet_name=nm + "_imp")
        H["data"][["no_langganan", "kode_rayon", "golongan", "klaster", "segmen"] + BULAN].to_excel(
            xl, sheet_name="data_bersegmen", index=False)
    st.download_button("Unduh tabel Bab IV (Excel)", buf.getvalue(), "tabel_bab4.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    if H.get("dummy"):
        st.warning("Hasil ini berasal dari data dummy.")
