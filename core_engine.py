import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from math import pi

# Membaca data dari excel
file_path = "data_masjid.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="Sheet1")
df = df_raw.iloc[1:].copy()
df.columns = [
    "kode_provinsi", "nama_provinsi", "nasional",
    "besar", "raya", "agung", "jami", "bersejarah",
    "publik", "jumlah", "tahun"
]
df = df[[
    "nama_provinsi", "besar", "raya", "agung",
    "jami", "bersejarah", "publik", "jumlah", "tahun"
]]

# Konversi Ke Numerik
for col in ["besar", "raya", "agung", "jami", "bersejarah", "publik", "jumlah"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_excel("output_data_tipologi_masjid_bersih.xlsx", index=False)
print("✅ Data setelah dibersihkan:")
print(df.head(), "\n")

# === 3. Normalisasi (Min-Max Scaling per tahun) ===
df_normalized_list = []

for tahun, group in df.groupby("tahun"):
    group_norm = group.copy()
    for col in ["besar", "raya", "agung", "jami", "bersejarah", "publik", "jumlah"]:
        min_val = group[col].min()
        max_val = group[col].max()
        group_norm[col] = (group[col] - min_val) / (max_val - min_val)
    group_norm["tahun"] = tahun
    df_normalized_list.append(group_norm)

df_normalized = pd.concat(df_normalized_list, ignore_index=True)

# === 4. Interpretasi nilai normalisasi ===
def interpret(value):
    if value < 0.33:
        return "Rendah"
    elif value < 0.66:
        return "Menengah"
    else:
        return "Tinggi"

hasil_list = []
for i, row in df.iterrows():
    tahun = row["tahun"]
    group = df[df["tahun"] == tahun]
    for col in ["besar", "raya", "agung", "jami", "bersejarah", "publik", "jumlah"]:
        min_val = group[col].min()
        max_val = group[col].max()
        nilai_asli = row[col]
        hasil_norm = round((nilai_asli - min_val) / (max_val - min_val), 2)
        hasil_list.append({
            "Tahun": tahun,
            "Provinsi": row["nama_provinsi"],
            "Kolom": col,
            "Nilai Asli": int(nilai_asli),
            "Min": int(min_val),
            "Max": int(max_val),
            "Hasil Normalisasi": hasil_norm,
            "Makna": interpret(hasil_norm)
        })

df_hasil = pd.DataFrame(hasil_list)
df_hasil.sort_values(["Tahun", "Provinsi", "Kolom"], inplace=True)
df_hasil.to_excel("output_hasil_normalisasi_tipologi_masjid.xlsx", index=False, float_format="%.2f")

print("✅ Normalisasi per tahun selesai!")
print("💾 Hasil tersimpan sebagai 'output_hasil_normalisasi_tipologi_masjid.xlsx'")
print("\n📊 Contoh hasil normalisasi per tahun:")
print(df_hasil.head(10))

df_hasil["Provinsi"] = df_hasil["Provinsi"].str.upper().str.strip()
df_wide = df_hasil.pivot_table(
    index=["Tahun","Provinsi"],
    columns="Kolom",
    values="Hasil Normalisasi",
    aggfunc="mean"
).reset_index()

df_wide_rounded = df_wide.copy()
df_wide_rounded.iloc[:, 1:] = df_wide_rounded.iloc[:, 1:].round(2)

print("\n✅ Data siap untuk clustering (rata-rata per provinsi):")
print(df_wide_rounded.head())

output_file = "output_hasil_clustering.xlsx"
df_wide_rounded.to_excel(output_file, index=False)

print(f"✅ Data berhasil disimpan ke file: {output_file}")

# KODE LANJUTAN
X = df_wide_rounded[["besar","raya","agung","jami","bersejarah","publik","jumlah"]]

inertia = []  # Menyimpan nilai total within-cluster sum of squares (SSE)
K_range = range(2, 10)  # Uji dari 2 sampai 9 cluster

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

print("\n📊 Hasil Perhitungan Nilai Inertia (SSE) untuk Setiap Jumlah Cluster:")
print("-----------------------------------------------------------")
print(f"{'Jumlah Cluster (k)':<20} {'Nilai Inertia (SSE)':<20}")
print("-----------------------------------------------------------")

for k, sse in zip(K_range, inertia):
    print(f"{k:<20} {sse:<20.2f}")

print("-----------------------------------------------------------")

# Visualisasi hasil Elbow Method
plt.figure(figsize=(7, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='b')
plt.title("Elbow Method")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia (Total Within-Cluster SSE)")
plt.grid(True)
plt.show()

# === 6. Jalankan K-Means Akhir ===
best_k = 3  # jumlah cluster yang kamu pilih
print(f"\n🔥 Jumlah cluster optimal berdasarkan Silhouette Score: k = {best_k}")

# Jalankan model K-Means
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_wide_rounded["Cluster"] = kmeans_final.fit_predict(X)

# Hitung Silhouette Score
final_silhouette = silhouette_score(X, kmeans_final.labels_)
print(f"\n💡 Silhouette Score akhir untuk k = {best_k}: {final_silhouette:.3f}")

# === 7. Buat Tabel Hasil K-Means ===
df_cluster_result = df_wide_rounded[["Tahun", "Provinsi", "Cluster"] + list(X.columns)]
df_cluster_result.sort_values(["Cluster", "Provinsi"], inplace=True)

print("\n📋 Contoh Tabel Hasil K-Means Clustering:")
print(df_cluster_result.head(10))

# Simpan hasil clustering
output_kmeans_file = "output_hasil_kmeans_tipologi_masjid.xlsx"
df_cluster_result.to_excel(output_kmeans_file, index=False)
print(f"\n💾 Hasil akhir clustering disimpan ke '{output_kmeans_file}'")

# === 8. Analisis dan Penataan Ulang Cluster ===
# Hitung rata-rata tiap variabel per cluster
cluster_summary = (
    df_cluster_result.groupby("Cluster")[["besar", "raya", "agung", "jami",
                                          "bersejarah", "publik", "jumlah"]]
    .mean()
    .round(2)
)

# Urutkan berdasarkan kolom 'jumlah' dari rendah ke tinggi
ranking = cluster_summary.reset_index().sort_values("jumlah", ascending=True).reset_index(drop=True)

# Buat label urut (0=Rendah, 1=Menengah, 2=Tinggi)
labels = ["Rendah", "Menengah", "Tinggi"]
ranking["Label"] = labels[:len(ranking)]
ranking["Cluster_Baru"] = range(len(ranking))

# Mapping cluster lama ke urutan baru
old_to_new = dict(zip(ranking["Cluster"], ranking["Cluster_Baru"]))
label_map = dict(zip(ranking["Cluster_Baru"], ranking["Label"]))

# Ganti cluster lama di dataset hasil clustering
df_cluster_result["Cluster_Asli"] = df_cluster_result["Cluster"]
df_cluster_result["Cluster"] = df_cluster_result["Cluster"].map(old_to_new)

# Update dictionary global agar semua bagian selaras
cluster_label_map = label_map
cluster_summary_sorted = ranking.copy()

print("\n✅ Klaster sudah diurutkan:")
print(cluster_summary_sorted[["Cluster_Baru", "Label", "jumlah"]])

# === 9. Pivot Cluster Lintas Tahun ===
df_cluster_result["Provinsi"] = df_cluster_result["Provinsi"].str.upper().str.strip()
pivot_cluster = (
    df_cluster_result.pivot_table(
        index="Provinsi",
        columns="Tahun",
        values="Cluster",
        aggfunc="first"
    )
    .reset_index()
)

pivot_cluster.columns = pivot_cluster.columns.map(str)
tahun_cols = [c for c in pivot_cluster.columns if c.isdigit()]

# === 10. Tentukan Cluster Dominan Berdasarkan Frekuensi (modus) ===
def get_dominant_cluster(row):
    values = [v for v in row[tahun_cols] if pd.notna(v)]
    if len(values) == 0:
        return None
    return pd.Series(values).mode().iloc[0]

pivot_cluster["Cluster Dominan"] = pivot_cluster.apply(get_dominant_cluster, axis=1)
pivot_cluster["Keterangan"] = pivot_cluster["Cluster Dominan"].map(cluster_label_map)
pivot_cluster["Keterangan"] = pivot_cluster["Keterangan"].fillna("Tidak Ada Data")

# === 11. Hitung Konsistensi Tiap Provinsi ===
def calc_consistency(row):
    values = [v for v in row[tahun_cols] if pd.notna(v)]
    if len(values) == 0:
        return 0
    dominant = row["Cluster Dominan"]
    return round((values.count(dominant) / len(values)) * 100, 1)

pivot_cluster["Konsistensi (%)"] = pivot_cluster.apply(calc_consistency, axis=1)

# === 12. Simpan Hasil ke Excel ===
print("\n📈 Interpretasi Cluster per Provinsi (berdasarkan frekuensi dominan):")
print(pivot_cluster.head(10))

output_interpretasi_file = "output_interpretasi_cluster_tipologi_lintas_tahun.xlsx"
with pd.ExcelWriter(output_interpretasi_file) as writer:
    pivot_cluster.to_excel(writer, sheet_name="Interpretasi Cluster", index=False)
    cluster_summary_sorted.to_excel(writer, sheet_name="Rata2_per_Cluster")

print(f"\n💾 File interpretasi lintas tahun disimpan ke '{output_interpretasi_file}'")
print("📊 Sheet 1: Interpretasi Cluster")
print("📊 Sheet 2: Rata-rata tiap variabel per cluster (sudah diurutkan)")

# === 13. Analisis Centroid Tiap Cluster ===
from scipy.spatial.distance import cdist

centroids = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=["besar", "raya", "agung", "jami", "bersejarah", "publik", "jumlah"]
)
centroids["Cluster_Asli"] = range(best_k)
centroids["Cluster"] = centroids["Cluster_Asli"].map(old_to_new)
centroids["Label"] = centroids["Cluster"].map(cluster_label_map)
centroids = centroids.sort_values("Cluster").reset_index(drop=True)
centroids = centroids.round(3)

dist_matrix = pd.DataFrame(
    cdist(
        centroids[["besar","raya","agung","jami","bersejarah","publik","jumlah"]],
        centroids[["besar","raya","agung","jami","bersejarah","publik","jumlah"]]
    ),
    index=[f"Cluster {i}" for i in centroids["Cluster"]],
    columns=[f"Cluster {i}" for i in centroids["Cluster"]]
).round(3)

print("\n🎯 Analisis Centroid Tiap Cluster:")
print(centroids)

print("\n📏 Jarak Antar Centroid (Euclidean Distance):")
print(dist_matrix)

with pd.ExcelWriter(output_interpretasi_file, mode="a", engine="openpyxl") as writer:
    centroids.to_excel(writer, sheet_name="Centroid_per_Cluster", index=False)
    dist_matrix.to_excel(writer, sheet_name="Jarak_Centroid")

print("\n💾 Hasil analisis centroid disimpan ke file Excel:")
print(f"📂 {output_interpretasi_file}")
print("📊 Sheet 3: Centroid_per_Cluster")
print("📊 Sheet 4: Jarak_Centroid")

# === 14. Visualisasi Centroid Tiap Cluster ===
fig, axes = plt.subplots(1, best_k, figsize=(5 * best_k, 5), sharey=True)
if best_k == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    data = centroids.loc[centroids["Cluster"] == i, ["besar","raya","agung","jami",
                                                     "bersejarah","publik","jumlah"]].iloc[0]
    ax.bar(data.index, data.values, color='skyblue')
    ax.set_title(f"Cluster {i} ({cluster_label_map[i]})", fontsize=12, weight='bold')
    ax.set_ylabel("Nilai Normalisasi (0-1)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("Profil Rata-Rata (Centroid) Tiap Cluster (Urut: Rendah → Tinggi)",
             fontsize=14, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 1️⃣ Baca file peta
peta = gpd.read_file("indonesia-edit.geojson")

# 2️⃣ Samakan format nama provinsi
peta["state"] = peta["state"].str.upper().str.strip()
pivot_cluster["Provinsi"] = pivot_cluster["Provinsi"].str.upper().str.strip()

# 3️⃣ Ambil data hasil akhir interpretasi (Cluster Dominan)
df_peta = pivot_cluster[["Provinsi", "Cluster Dominan", "Keterangan", "Konsistensi (%)"]].copy()
df_peta.rename(columns={"Cluster Dominan": "Cluster"}, inplace=True)

# 4️⃣ Gabungkan dengan peta
merged = peta.merge(df_peta, left_on="state", right_on="Provinsi", how="left")

# 5️⃣ Pastikan warna sesuai urutan (0=Rendah, 1=Menengah, 2=Tinggi)
cmap = mcolors.ListedColormap(["#4B9CD3", "#F4D35E", "#EE6352"])

# 6️⃣ Plot peta tanpa legend bawaan
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
merged.plot(
    column="Cluster",
    cmap=cmap,
    legend=False,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

# 7️⃣ Legend manual (sesuai urutan)
legend_patches = [
    mpatches.Patch(color="#4B9CD3", label="Rendah (Cluster 0)"),
    mpatches.Patch(color="#F4D35E", label="Menengah (Cluster 1)"),
    mpatches.Patch(color="#EE6352", label="Tinggi (Cluster 2)")
]
plt.legend(
    handles=legend_patches,
    title="Keterangan Cluster Dominan",
    loc="lower left",
    fontsize=10,
    title_fontsize=11,
    frameon=True,
    facecolor="white",
    edgecolor="black"
)

# 8️⃣ Judul dan tampilkan
plt.title("Peta Sebaran Tipologi Masjid di Indonesia (Cluster Dominan Lintas Tahun)\n"
          "(0=Rendah, 1=Menengah, 2=Tinggi)",
          fontsize=14, weight='bold', pad=15)
ax.set_axis_off()
plt.tight_layout()

# 9️⃣ Simpan hasil peta
plt.savefig("peta_cluster_dominan_tipologi_masjid.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Peta spasial berhasil dibuat berdasarkan hasil interpretasi dominan lintas tahun!")
print("💾 Disimpan sebagai 'peta_cluster_dominan_tipologi_masjid.png'")

# 🔟 Simpan hasil GeoDataFrame ke file GeoJSON
output_geojson = "hasil_peta_cluster_dominan_tipologi_masjid.geojson"
merged.to_file(output_geojson, driver="GeoJSON")
print(f"💾 File GeoJSON hasil disimpan ke '{output_geojson}'")