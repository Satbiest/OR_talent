import streamlit as st
import pandas as pd
import numpy as np
import io
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Konfigurasi Halaman Utama ---
st.set_page_config(page_title="Kuat Bersama Telkom", layout="wide")

# --- DEFINISI FUNGSI PENDUKUNG ---
def convert_to_months(duration):
    years, months = 0, 0
    parts = str(duration).split()
    for i in range(len(parts)):
        if 'tahun' in parts[i]:
            try:
                years = int(parts[i - 1])
            except:
                years = 0
        elif 'bulan' in parts[i]:
            try:
                months = int(parts[i - 1])
            except:
                months = 0
    return years * 12 + months

def preprocess_dataframe(df, nama):
    if set(['Role', 'Responsibilities', 'Skill 1', 'Skill 2']).issubset(df.columns):
        df = df.drop_duplicates(subset=['Role', 'Responsibilities', 'Skill 1', 'Skill 2']).reset_index(drop=True)
    if 'LAMA KERJA BERJALAN' in df.columns:
        df['Durasi Bulan'] = df['LAMA KERJA BERJALAN'].apply(convert_to_months)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.replace('Artificial Intelligence Engineer', 'AI Engineer')
    st.success(f"✅ Dataset {nama} berhasil dipraproses.")
    return df

def minmax_scaling(series):
    if series.max() == series.min():
        return 0.5 # Mengembalikan nilai tengah jika semua sama untuk menghindari pembagian nol
    return (series - series.min()) / (series.max() - series.min())

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi Aplikasi")

# Pilihan Halaman Utama
if st.sidebar.button("🏠 Home"):
    st.session_state['current_page'] = 'Home'

st.sidebar.markdown("---")
st.sidebar.header("📂 Data Based")

# Pilihan Halaman/Tab di bawah "Data Based"
if st.sidebar.button("📁 Upload Data"):
    st.session_state['current_page'] = 'Upload Data'
if st.sidebar.button("📊 Hasil Pencarian Talent Sesuai Use Case"):
    st.session_state['current_page'] = 'Hasil Pencarian Talent Sesuai Use Case'
if st.sidebar.button("🤝 Co-worker Detection"):
    st.session_state['current_page'] = 'Co-worker Detection'
if st.sidebar.button("🏆 Hasil Akhir"):
    st.session_state['current_page'] = 'Hasil Akhir'

st.sidebar.markdown("---")
st.sidebar.header("👤 User Based")
# Pilihan Halaman di bawah "User Based"
if st.sidebar.button("🔍 User Input Talent Search"):
    st.session_state['current_page'] = 'User Input Talent Search'


# Inisialisasi halaman saat aplikasi pertama kali dijalankan
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

# --- KONTEN HALAMAN ---

# --- Halaman Home ---
if st.session_state['current_page'] == 'Home':
    st.title("💪 Kuat Bersama Telkom: Talent Matching & Network Analyzer")
    st.markdown("""
    Selamat datang di aplikasi **Kuat Bersama Telkom**! Aplikasi ini dirancang untuk membantu Anda menemukan talenta terbaik yang sesuai dengan kebutuhan *use case* atau proyek, serta menganalisis jaringan kerja antar-talenta.

    ---
    ### 🛠️ Tools dan Library yang Digunakan:

    Aplikasi prototipe ini dibangun menggunakan teknologi dan *library* Python terkini untuk pemrosesan data, pemodelan, dan antarmuka pengguna interaktif:

    * **Bahasa Pemrograman:** Python 🐍
    * **Framework Aplikasi Web:** `streamlit`
    * **Manipulasi dan Analisis Data:** `pandas`
    * **Komputasi Numerik:** `numpy`
    * **Pemrosesan Teks & Embeddings:** `sentence-transformers` (`SentenceTransformer`)
        * Model yang digunakan: `all-MiniLM-L6-v2`
    * **Machine Learning (Klasterisasi & Reduksi Dimensi):** `scikit-learn` (`sklearn`)
        * `sklearn.metrics.pairwise.cosine_similarity`
        * `sklearn.decomposition.PCA`
        * `sklearn.cluster.KMeans`
        * `sklearn.metrics.silhouette_score`
    * **Penanganan File Input/Output:** `io`, `BytesIO`
    * **Penulisan File Excel:** `openpyxl` (digunakan secara internal oleh Pandas untuk `.to_excel`)

    ---
    ### 🚀 Fitur Utama:

    * **Upload Data Fleksibel:** Dukungan untuk mengunggah berbagai dataset penting (`Skill Inventory`, `Use Case Requirement`, `Data Pegawai`, `Evaluation Score`, `Histori Pekerjaan`, `Data Penugasan`).
    * **Pencocokan Talent:** Menganalisis kemiripan antara *skillset* dan *use case* menggunakan *sentence embeddings* dan sistem skoring terintegrasi.
    * **Klasterisasi Skill:** Mengelompokkan talenta berdasarkan *skill* mereka untuk identifikasi *talent pool* yang relevan.
    * **Deteksi Rekan Kerja:** Mengidentifikasi hubungan rekan kerja berdasarkan penugasan dan histori proyek.
    * **Rekomendasi Talent:** Memberikan rekomendasi talenta terbaik yang disesuaikan dengan *use case* tertentu, mempertimbangkan faktor *skill*, evaluasi, dan kedekatan rekan kerja.

    ---
    Untuk memulai, silakan gunakan menu navigasi di *sidebar* kiri untuk memilih halaman yang ingin Anda akses.
    """)

# --- Halaman Upload Data ---
elif st.session_state['current_page'] == 'Upload Data':
    st.title("📁 Upload Data")
    st.markdown("---")
    df_skillinv_file = st.file_uploader("Upload file 📘 **Skill Inventory (df_skillinv)**", type=["xlsx"], key="skillinv")
    df_ureq_file = st.file_uploader("Upload file 📗 **Use Case Requirement (df_ureq)**", type=["xlsx"], key="ureq")
    df_talent_file = st.file_uploader("Upload file 📙 **Data Pegawai (df_talent)**", type=["xlsx"], key="talent")
    df_eval_file = st.file_uploader("Upload file 📒 **Evaluation Score (df_eval)**", type=["xlsx"], key="eval")
    uploaded_hist = st.file_uploader("Upload file 📗 **Histori Pekerjaan (df_hist)**", type=["xlsx"])
    uploaded_assign = st.file_uploader("Upload file 🟣 **Data Penugasan (df_assign)**", type=["xlsx"], key="assign")
    
    if df_skillinv_file:
        st.session_state['df_skillinv'] = pd.read_excel(df_skillinv_file)
        st.success("✅ File Skill Inventory berhasil diunggah.")
    if df_ureq_file:
        st.session_state['df_ureq'] = pd.read_excel(df_ureq_file)
        st.success("✅ File Use Case Requirement berhasil diunggah.")
    if df_talent_file:
        st.session_state['df_talent'] = pd.read_excel(df_talent_file)
        st.success("✅ File Data Pegawai berhasil diunggah.")
    if df_eval_file:
        st.session_state['df_eval'] = pd.read_excel(df_eval_file)
        st.success("✅ File Evaluation Score berhasil diunggah.")
    if uploaded_hist:
        st.session_state['df_hist'] = pd.read_excel(uploaded_hist)
        st.success("✅ File Histori Pekerjaan berhasil diunggah.")
    if uploaded_assign:
        st.session_state['df_assign'] = pd.read_excel(uploaded_assign)
        st.success("✅ File Data Penugasan berhasil diunggah.")

# --- Halaman Hasil Pencarian Talent Sesuai Use Case ---
elif st.session_state['current_page'] == 'Hasil Pencarian Talent Sesuai Use Case':
    st.title("📊 Hasil Pencarian Talent Sesuai Use Case")
    st.markdown("---")

    # Pastikan semua dataframe yang dibutuhkan tersedia
    if 'df_skillinv' in st.session_state and 'df_ureq' in st.session_state and \
       'df_talent' in st.session_state and 'df_eval' in st.session_state and \
       'df_hist' in st.session_state:

        df_skillinv = preprocess_dataframe(st.session_state['df_skillinv'].copy(), "Skill Inventory")
        df_ureq = preprocess_dataframe(st.session_state['df_ureq'].copy(), "Use Case Requirement")
        df_talent = st.session_state['df_talent'].copy()
        df_eval = st.session_state['df_eval'].copy()
        df_hist = st.session_state['df_hist'].copy()

        # ------------------------------
        # AGG_SENTENCES
        # ------------------------------
        skillsets = df_skillinv.columns[2:-1].tolist()
        df_ureq['agg_sentences'] = df_ureq['Responsibilities'] + " " + df_ureq['Skill 1'].fillna('') + " " + df_ureq['Skill 2'].fillna('')

        st.subheader("Menghitung Kemiripan Antar Skillset dan Use Case")
        with st.spinner('Memuat model dan menghitung kemiripan...'):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            results = []

            for idx, row in df_ureq.iterrows():
                agg_sentence = row['agg_sentences']
                corpus = [agg_sentence] + skillsets
                sentence_embeddings = model.encode(corpus)
                cosine_sim = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
                for i in range(len(skillsets)):
                    results.append([
                        row['Responsibilities'], row['Skill 1'], row['Skill 2'],
                        row['Role'], row['agg_sentences'], skillsets[i], cosine_sim[0][i]
                    ])

            df_results = pd.DataFrame(results, columns=[
                'Responsibilities', 'Skill 1', 'Skill 2', 'Role', 'agg_sentences', 'Skillset', 'Similarity score'
            ])
            df_results = df_results.sort_values(by=['agg_sentences', 'Similarity score'], ascending=[True, False])

        st.subheader("📊 Hasil Kemiripan Awal (df_results)")
        st.dataframe(df_results, use_container_width=True)

        threshold_input = st.text_input("🎯 Masukkan threshold Similarity Score (contoh: 0.3 atau 0,3)", value="0.2999")

        try:
            threshold = float(threshold_input.replace(",", "."))
            df_results_filtered = df_results[df_results['Similarity score'] >= threshold]

            unique_ids_roles = df_skillinv[['UNIQUE ID', 'Role']].drop_duplicates()
            unique_ids_roles.rename(columns={'Role': 'Role Person'}, inplace=True)
            merged_df = df_results_filtered.merge(unique_ids_roles, how='cross')

            def get_skill_score(row):
                unique_id = row['UNIQUE ID']
                skill_needed = row['Skillset']
                if skill_needed in df_skillinv.columns:
                    score = df_skillinv.loc[df_skillinv['UNIQUE ID'] == unique_id, skill_needed]
                    return score.values[0] if not score.empty else None
                return None

            merged_df['Skill Score'] = merged_df.apply(get_skill_score, axis=1)
            merged_df['Skill 1'].fillna('', inplace=True)
            merged_df['Skill 2'].fillna('', inplace=True)

            df_search = merged_df.groupby(
                ['Responsibilities', 'Skill 1', 'Skill 2', 'Role', 'agg_sentences', 'UNIQUE ID', 'Role Person'],
                as_index=False
            ).agg(
                Skillset=('Skillset', lambda x: list(x)),
                Avg_SkillScore=('Skill Score', 'mean'),
            )

            st.subheader("🔍 Hasil Search: Kecocokan Talenta dengan Requirement")
            st.dataframe(df_search, use_container_width=True)

            buffer = io.BytesIO()
            df_search.to_excel(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📅 Download df_search.xlsx",
                data=buffer,
                file_name="df_search.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            df_ureq['Combined'] = df_ureq['Responsibilities'] + '|' + df_ureq['Skill 1'].fillna('') + '|' + df_ureq['Skill 2'].fillna('')
            df_search['Combined'] = df_search['Responsibilities'] + '|' + df_search['Skill 1'].fillna('') + '|' + df_search['Skill 2'].fillna('')

            df_missingtask = df_ureq[~df_ureq['Combined'].isin(df_search['Combined'])]

            if df_missingtask.empty:
                st.success("✅ Semua job telah terplot!")
            else:
                st.warning("⚠️ Berikut job yang belum ter-plot skill-nya:")
                df_missingtask = df_missingtask.drop(columns=['Combined', 'Relevant Use Cases', 'agg_sentences'], errors='ignore')
                df_missingtask = df_missingtask.rename(columns={'Role': 'Role Task'})
                st.dataframe(df_missingtask, use_container_width=True)

            df_hasilSearch = df_search
            st.session_state['df_hasilSearch'] = df_hasilSearch # Simpan untuk akses di tab lain

            # ------------------------------
            # CLUSTERING df_skillinv
            # ------------------------------
            st.subheader("🌐 Klasterisasi Skill Inventory")
            null_columns = df_skillinv.columns[df_skillinv.isnull().any()].tolist()
            numerical_columns = df_skillinv.select_dtypes(include=[np.number]).columns.tolist()
            df_numerical = df_skillinv[numerical_columns]
            df_numerical_cleaned = df_numerical.dropna(axis=1)

            if not df_numerical_cleaned.empty and len(df_numerical_cleaned) >= 2: # Minimal 2 sampel untuk klasterisasi
                pca = PCA(n_components=min(2, len(df_numerical_cleaned.columns))) # Min 2 components or less if few columns
                df_pca = pca.fit_transform(df_numerical_cleaned)

                best_k = 2
                best_score = -1
                silhouette_scores = []

                # Max k should be less than number of samples and greater than 1 for silhouette score
                max_k_cluster = min(len(df_pca) - 1, 8)
                if max_k_cluster >= 2:
                    for k in range(2, max_k_cluster + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                        cluster_labels = kmeans.fit_predict(df_pca)
                        score = silhouette_score(df_pca, cluster_labels)
                        silhouette_scores.append(score)
                        if score > best_score:
                            best_score = score
                            best_k = k
                    
                    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
                    df_skillinv['Cluster'] = kmeans_best.fit_predict(df_pca)
                    df_skillinv_clustered = df_skillinv.copy()
                    st.session_state['df_skillinv_clustered'] = df_skillinv_clustered # Simpan untuk akses di tab lain

                    cluster_counts = df_skillinv_clustered['Cluster'].value_counts()
                    st.write(f"Klaster terbaik: {best_k} dengan skor: {best_score:.4f}")
                    st.dataframe(cluster_counts.reset_index().rename(columns={'index': 'Cluster', 'Cluster': 'Jumlah'}), use_container_width=True)
                    st.dataframe(df_skillinv_clustered, use_container_width=True)
                else:
                    st.warning("⚠️ Tidak cukup data untuk melakukan klasterisasi dengan lebih dari 1 klaster.")
            else:
                st.warning("⚠️ Tidak cukup data numerik atau sampel untuk melakukan klasterisasi pada Skill Inventory.")


        except Exception as e:
            st.error(f"❗ Input tidak valid atau terjadi error: {e}")
            
        # ------------------------------
        # MERGING df_hasilSearch dengan Cluster
        # ------------------------------
        st.subheader("🧬 Penggabungan Data dengan Hasil Klasterisasi")

        try:
            if 'df_skillinv_clustered' in st.session_state:
                df_hasilSearch = df_hasilSearch.merge(
                    st.session_state['df_skillinv_clustered'][['UNIQUE ID', 'Cluster']], 
                    on='UNIQUE ID', 
                    how='left'
                )
                st.success("✅ Penggabungan df_hasilSearch dengan hasil klasterisasi berhasil dilakukan.")
                st.dataframe(df_hasilSearch, use_container_width=True)

                buffer_clustered = io.BytesIO()
                df_hasilSearch.to_excel(buffer_clustered, index=False)
                buffer_clustered.seek(0)

                st.download_button(
                    label="📁 Download df_hasilSearch_Clustered.xlsx",
                    data=buffer_clustered,
                    file_name="df_hasilSearch_Clustered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Data klasterisasi Skill Inventory belum tersedia.")

        except Exception as e:
            st.error(f"❗ Terjadi error saat penggabungan: {e}")

        # ------------------------------
        # INPUT TAMBAHAN: df_talent dan df_eval
        # ------------------------------
        st.subheader("🧾 Validasi Kesesuaian ID Pegawai dan Evaluasi")
        # Praproses df_talent
        if 'LAMA KERJA BERJALAN' in df_talent.columns:
            df_talent['Durasi Bulan'] = df_talent['LAMA KERJA BERJALAN'].apply(convert_to_months)
            st.success("✅ Kolom 'Durasi Bulan' berhasil ditambahkan ke df_talent.")

        # Validasi kelengkapan UNIQUE ID
        unique_ids_eval = set(df_eval["UNIQUE ID"])
        unique_ids_talent = set(df_talent["UNIQUE ID"])

        missing_in_eval = unique_ids_talent - unique_ids_eval
        missing_in_talent = unique_ids_eval - unique_ids_talent

        if not missing_in_eval and not missing_in_talent:
            st.success("✅ Data pegawai dengan Data Evaluasi pegawai relevan")
        else:
            if missing_in_eval:
                st.warning("⚠️ UNIQUE ID yang tidak ada di df_eval:")
                st.write(list(missing_in_eval))
            if missing_in_talent:
                st.warning("⚠️ UNIQUE ID yang tidak ada di df_talent:")
                st.write(list(missing_in_talent))

        # Merge menjadi DF_agg_talent
        DF_agg_talent = pd.merge(df_talent, df_eval, on='UNIQUE ID', how='inner')
        st.session_state['DF_agg_talent'] = DF_agg_talent # Simpan untuk akses di tab lain
        st.subheader("📌 Agregasi Pegawai dan Evaluasi Pegawai (DF_agg_talent)")
        
        st.subheader("⚙️ Pengaturan Bobot Penilaian")

        # Input bobot dari user
        bobot_capability_score = st.number_input(
            "Masukkan bobot Capability Score", min_value=0.0, max_value=1.0, value=0.8, step=0.05, key='bobot_cap_score_tab2'
        )
        bobot_durasi = st.number_input(
            "Masukkan bobot Durasi (Bulan)", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key='bobot_durasi_tab2'
        )

        # Validasi agar jumlah bobot tidak lebih dari 1 (opsional)
        if bobot_capability_score + bobot_durasi > 1.0:
            st.warning("Total bobot sebaiknya tidak lebih dari 1. Silakan sesuaikan.")

        # Hitung kolom scoring_eval
        if 'Durasi Bulan' in DF_agg_talent.columns and 'Capability Score' in DF_agg_talent.columns:
            DF_agg_talent['scoring_eval'] = (
                DF_agg_talent['Durasi Bulan'] * bobot_durasi
                + DF_agg_talent['Capability Score'] * bobot_capability_score
            )
            st.success("Kolom 'scoring_eval' berhasil ditambahkan.")
        else:
            st.error("Kolom 'Durasi Bulan' atau 'Capability Score' tidak ditemukan dalam DataFrame.")

        st.dataframe(DF_agg_talent, use_container_width=True)          

        # Validasi dengan df_hasilSearch
        st.subheader("🧮 Validasi Agregasi dengan Hasil Search")

        unique_ids_agg_talent = set(DF_agg_talent['UNIQUE ID'])
        unique_ids_hasilSearch = set(df_hasilSearch['UNIQUE ID'])

        missing_in_agg = unique_ids_hasilSearch - unique_ids_agg_talent
        missing_in_search = unique_ids_agg_talent - unique_ids_hasilSearch

        if not missing_in_agg and not missing_in_search:
            st.success("✅ Semua UNIQUE ID saling cocok antara DF_agg_talent dan df_hasilSearch.")
        else:
            if missing_in_agg:
                st.warning("⚠️ UNIQUE ID dari hasil pencocokan usecase yang tidak ada di data agregat talent:")
                st.write(", ".join(str(uid) for uid in missing_in_agg))
            if missing_in_search:
                st.warning("⚠️ UNIQUE ID dari data agregat talent yang tidak ada di hasil pencocokan use case:")
                st.write(", ".join(str(uid) for uid in missing_in_search))

        st.subheader("🔗 Menggabungkan Data Pencarian dengan Talent Agg")

        # Merge berdasarkan UNIQUE ID
        if 'UNIQUE ID' in df_hasilSearch.columns and 'UNIQUE ID' in DF_agg_talent.columns:
            DF_search_talent_merged = pd.merge(df_hasilSearch, DF_agg_talent, on='UNIQUE ID', how='inner')

            # Hapus kolom 'NO' jika ada
            if 'NO' in DF_search_talent_merged.columns:
                DF_search_talent_merged = DF_search_talent_merged.drop(columns=['NO'])

            st.session_state['DF_search_talent_merged'] = DF_search_talent_merged # Simpan untuk akses di tab lain
            st.success(f"Berhasil menggabungkan data. Jumlah baris hasil: {len(DF_search_talent_merged)}")
            st.dataframe(DF_search_talent_merged)

            # Simpan ke XLSX menggunakan openpyxl
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                DF_search_talent_merged.to_excel(writer, index=False, sheet_name='Hasil Merge')
            xlsx_data = output.getvalue()

            st.download_button(
                label="📥 Download Hasil Merge sebagai XLSX",
                data=xlsx_data,
                file_name="hasil_merge_talent.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Kolom 'UNIQUE ID' tidak ditemukan di salah satu DataFrame.")
            
        st.header("Proses Skoring dan Penyatuan Data")
        
        if df_hist is not None and 'DF_search_talent_merged' in st.session_state:
            DF_search_talent_merged = st.session_state['DF_search_talent_merged'].copy()
            # Hitung job_count
            DF_hist_count = df_hist.groupby("UNIQUE ID")["PRODUCT / USECASE"].nunique().reset_index()
            DF_hist_count.columns = ["UNIQUE ID", "job_count"]

            # Gabungkan dengan DF_search_talent_merged
            DF_search_clust_talent_merged_count = pd.merge(
                DF_search_talent_merged,
                DF_hist_count[['UNIQUE ID', 'job_count']],
                on='UNIQUE ID',
                how='left'
            )
            DF_search_clust_talent_merged_count['job_count'] = DF_search_clust_talent_merged_count['job_count'].fillna(0)
            st.session_state['DF_search_clust_talent_merged_count'] = DF_search_clust_talent_merged_count # Simpan untuk akses di tab lain

            # Input user untuk koefisien
            st.subheader("Penyesuaian Koefisien Skoring")
            a = st.number_input("Koefisien a (Avg_SkillScore)", value=0.42, key='coeff_a_tab2')
            b = st.number_input("Koefisien b (scoring_eval)", value=0.48, key='coeff_b_tab2')
            c = st.number_input("Koefisien c (job_count)", value=0.1, key='coeff_c_tab2')
            r = st.number_input("Koefisien r (pengali jika Role cocok)", value=1.2, key='coeff_r_tab2')

            # Hitung skor d
            DF_search_clust_talent_merged_count['d'] = (
                DF_search_clust_talent_merged_count['Avg_SkillScore'] * a +
                DF_search_clust_talent_merged_count['scoring_eval'] * b +
                DF_search_clust_talent_merged_count['job_count'] * c
            )

            # Hitung finalscore
            DF_search_clust_talent_merged_count['finalscore'] = DF_search_clust_talent_merged_count.apply(
                lambda row: row['d'] * r if row['Role Person'] == row['Role'] else row['d'], axis=1
            )

            # Min-Max Scaling finalscore berdasarkan Responsibilities
            DF_search_clust_talent_merged_count['finalscore_scaled'] = DF_search_clust_talent_merged_count.groupby(
                'Responsibilities')['finalscore'].transform(minmax_scaling)

            # Tampilkan hasil akhir
            st.subheader("📊 Hasil Rekomendasi Pencarian Akhir")
            st.dataframe(DF_search_clust_talent_merged_count)

            # Export ke Excel
            output_excel = BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                DF_search_clust_talent_merged_count.to_excel(writer, index=False, sheet_name="Hasil Rekomendasi")
            output_excel.seek(0)

            st.download_button(
                label="📥 Download Hasil Last Harbour Pencocokan dalam Excel",
                data=output_excel,
                file_name="hasil_rekomendasi_talent.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("⚠️ Mohon unggah data histori dan pastikan proses penggabungan sebelumnya sudah selesai.")

    else:
        st.warning("⚠️ Mohon unggah semua file yang diperlukan di halaman 'Upload Data' terlebih dahulu.")


# --- Halaman Co-worker Detection ---
elif st.session_state['current_page'] == 'Co-worker Detection':
    st.title("🤝 Co-worker Detection")
    st.markdown("---")

    df_assign = st.session_state.get('df_assign')
    df_hist = st.session_state.get('df_hist')

    if df_assign is not None and df_hist is not None:
        st.success("✅ Data `df_assign` dan `df_hist` berhasil dimuat untuk deteksi rekan kerja.")
        
        # --- Operasi pada df_assign untuk DF_assign_cowork ---
        st.markdown("#### Deteksi Rekan Kerja Berdasarkan Penugasan (`df_assign`)")
        DF_assign_cowork = pd.DataFrame(columns=['UNIQUE ID', 'co_work', 'weight'])

        progress_text_assign = "Mencari rekan kerja di df_assign..."
        my_bar_assign = st.progress(0, text=progress_text_assign)

        total_rows_assign = len(df_assign)
        for i, row in df_assign.iterrows():
            my_bar_assign.progress((i + 1) / total_rows_assign, text=f"{progress_text_assign} {int((i + 1) / total_rows_assign * 100)}%")

            unique_id = row['UNIQUE ID']
            tribe = row['TRIBE/BIDANG']
            product = row['PRODUCT/USECASE']
            squad = row['SQUAD']
            
            # Mencari rekan kerja berdasarkan TRIBE/BIDANG
            coworkers_tribe = df_assign[df_assign['TRIBE/BIDANG'] == tribe]['UNIQUE ID'].unique()
            coworkers_tribe = [coworker for coworker in coworkers_tribe if coworker != unique_id]
            
            for cowork in coworkers_tribe:
                exists_ab = ((DF_assign_cowork['UNIQUE ID'] == unique_id) & (DF_assign_cowork['co_work'] == cowork)).any()
                exists_ba = ((DF_assign_cowork['UNIQUE ID'] == cowork) & (DF_assign_cowork['co_work'] == unique_id)).any()

                if exists_ab:
                    DF_assign_cowork.loc[(DF_assign_cowork['UNIQUE ID'] == unique_id) & (DF_assign_cowork['co_work'] == cowork), 'weight'] += 1
                elif exists_ba:
                    DF_assign_cowork.loc[(DF_assign_cowork['UNIQUE ID'] == cowork) & (DF_assign_cowork['co_work'] == unique_id), 'weight'] += 1
                else:
                    DF_assign_cowork = pd.concat([DF_assign_cowork, pd.DataFrame([{'UNIQUE ID': unique_id, 'co_work': cowork, 'weight': 0}])], ignore_index=True)


            # Mencari rekan kerja berdasarkan TRIBE/BIDANG dan PRODUCT/USECASE
            coworkers_product = df_assign[(df_assign['TRIBE/BIDANG'] == tribe) & (df_assign['PRODUCT/USECASE'] == product)]['UNIQUE ID'].unique()
            coworkers_product = [coworker for coworker in coworkers_product if coworker != unique_id]
            
            for cowork in coworkers_product:
                exists_ab = ((DF_assign_cowork['UNIQUE ID'] == unique_id) & (DF_assign_cowork['co_work'] == cowork)).any()
                exists_ba = ((DF_assign_cowork['UNIQUE ID'] == cowork) & (DF_assign_cowork['co_work'] == unique_id)).any()

                if exists_ab:
                    DF_assign_cowork.loc[(DF_assign_cowork['UNIQUE ID'] == unique_id) & (DF_assign_cowork['co_work'] == cowork), 'weight'] += 1
                elif exists_ba:
                    DF_assign_cowork.loc[(DF_assign_cowork['UNIQUE ID'] == cowork) & (DF_assign_cowork['co_work'] == unique_id), 'weight'] += 1
                else:
                    DF_assign_cowork = pd.concat([DF_assign_cowork, pd.DataFrame([{'UNIQUE ID': unique_id, 'co_work': cowork, 'weight': 1}])], ignore_index=True)

            # Mencari rekan kerja berdasarkan TRIBE/BIDANG, PRODUCT/USECASE, dan SQUAD
            coworkers_squad = df_assign [(df_assign['TRIBE/BIDANG'] == tribe) & 
                                        (df_assign['PRODUCT/USECASE'] == product) & 
                                        (df_assign['SQUAD'] == squad)]['UNIQUE ID'].unique()
            coworkers_squad = [coworker for coworker in coworkers_squad if coworker != unique_id]
            
            for cowork in coworkers_squad:
                exists_ab = ((DF_assign_cowork['UNIQUE ID'] == unique_id) & (DF_assign_cowork['co_work'] == cowork)).any()
                exists_ba = ((DF_assign_cowork['UNIQUE ID'] == cowork) & (DF_assign_cowork['co_work'] == unique_id)).any()

                if exists_ab:
                    DF_assign_cowork.loc[(DF_assign_cowork['UNIQUE ID'] == unique_id) & (DF_assign_cowork['co_work'] == cowork), 'weight'] += 1
                elif exists_ba:
                    DF_assign_cowork.loc[(DF_assign_cowork['UNIQUE ID'] == cowork) & (DF_assign_cowork['co_work'] == unique_id), 'weight'] += 1
                else:
                    DF_assign_cowork = pd.concat([DF_assign_cowork, pd.DataFrame([{'UNIQUE ID': unique_id, 'co_work': cowork, 'weight': 1}])], ignore_index=True)
        
        DF_assign_cowork = DF_assign_cowork.drop_duplicates().reset_index(drop=True)
        my_bar_assign.empty()
        st.success("✅ Deteksi rekan kerja dari `df_assign` selesai.")
        # st.dataframe(DF_assign_cowork, use_container_width=True) # Tidak ditampilkan langsung

        # --- Operasi pada df_hist untuk DF_hist_cowork ---
        st.markdown("#### Deteksi Rekan Kerja Berdasarkan Histori Pekerjaan (`df_hist`)")
        DF_hist_cowork = pd.DataFrame(columns=["UNIQUE ID", "co_work", "weight"])

        progress_text_hist = "Mencari rekan kerja di df_hist..."
        my_bar_hist = st.progress(0, text=progress_text_hist)

        total_rows_hist = len(df_hist)
        for i, row in df_hist.iterrows():
            my_bar_hist.progress((i + 1) / total_rows_hist, text=f"{progress_text_hist} {int((i + 1) / total_rows_hist * 100)}%")

            unique_id = row["UNIQUE ID"]
            product = row["PRODUCT / USECASE"]
            
            coworkers = df_hist[
                (df_hist["PRODUCT / USECASE"] == product) & 
                (df_hist["UNIQUE ID"] != unique_id)
            ]["UNIQUE ID"].unique()
            
            for cowork in coworkers:
                exists = (
                    ((DF_hist_cowork["UNIQUE ID"] == unique_id) & 
                    (DF_hist_cowork["co_work"] == cowork))
                ).any() or (
                    ((DF_hist_cowork["UNIQUE ID"] == cowork) & 
                    (DF_hist_cowork["co_work"] == unique_id))
                ).any()
                
                if not exists:
                    DF_hist_cowork = pd.concat([DF_hist_cowork, pd.DataFrame([{"UNIQUE ID": unique_id, "co_work": cowork, "weight": 1}])], ignore_index=True)
        
        my_bar_hist.empty()
        st.success("✅ Deteksi rekan kerja dari `df_hist` selesai.")
        # st.dataframe(DF_hist_cowork, use_container_width=True) # Tidak ditampilkan langsung

        # --- Penggabungan kedua DataFrame ---
        st.markdown("#### Penggabungan Hasil Deteksi Rekan Kerja")
        combined = pd.concat([DF_hist_cowork, DF_assign_cowork])

        combined["pair_key"] = combined.apply(
            lambda x: "-".join(sorted([str(x["UNIQUE ID"]), str(x["co_work"])] if pd.notna(x["UNIQUE ID"]) and pd.notna(x["co_work"]) else ["", ""])),
            axis=1
        )

        aggregated = combined.groupby("pair_key").agg({
            "weight": "sum"
        }).reset_index()

        aggregated[['UNIQUE ID_temp', 'co_work_temp']] = aggregated['pair_key'].str.split('-', expand=True)
        
        DF_Neigh = aggregated[['UNIQUE ID_temp', 'co_work_temp', 'weight']].copy()
        DF_Neigh.rename(columns={'UNIQUE ID_temp': 'UNIQUE ID', 'co_work_temp': 'co_work'}, inplace=True)
        st.session_state['DF_Neigh'] = DF_Neigh # Simpan DF_Neigh ke session state
        
        st.success("✅ Penggabungan dan agregasi rekan kerja selesai.")
        # st.dataframe(DF_Neigh, use_container_width=True) # Tidak ditampilkan

        # --- Membuat DF_Neigh_redundant ---
        st.markdown("#### DataFrame Rekan Kerja Redundant (`DF_Neigh_redundant`)")
        DF_Neigh_redundant = DF_Neigh.copy()

        rows_to_add = []
        for _, row in DF_Neigh.iterrows():
            rows_to_add.append({
                'UNIQUE ID': row['co_work'],
                'co_work': row['UNIQUE ID'],
                'weight': row['weight']
            })
        
        DF_Neigh_redundant = pd.concat([DF_Neigh_redundant, pd.DataFrame(rows_to_add)], ignore_index=True)
        st.session_state['DF_Neigh_redundant'] = DF_Neigh_redundant # Simpan DF_Neigh_redundant ke session state
        
        st.dataframe(DF_Neigh_redundant, use_container_width=True)

        buffer_redundant = io.BytesIO()
        DF_Neigh_redundant.to_excel(buffer_redundant, index=False)
        buffer_redundant.seek(0)

        st.download_button(
            label="📥 Download DF_Neigh_redundant.xlsx",
            data=buffer_redundant,
            file_name="DF_Neigh_redundant.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.warning("⚠️ Mohon unggah file 'Data Penugasan (df_assign)' dan 'Histori Pekerjaan (df_hist)' di halaman 'Upload Data' terlebih dahulu.")

# --- Halaman Hasil Akhir ---
elif st.session_state['current_page'] == 'Hasil Akhir':
    st.title("🏆 Hasil Akhir Rekomendasi Talent")
    st.markdown("---")

    # Ambil dataframe dari session state
    DF_search_clust_talent_merged_count = st.session_state.get('DF_search_clust_talent_merged_count')
    DF_Neigh_redundant = st.session_state.get('DF_Neigh_redundant')

    if DF_search_clust_talent_merged_count is not None and DF_Neigh_redundant is not None:
        st.success("✅ Data yang diperlukan untuk hasil akhir telah tersedia.")
        
        responsibilities_list = DF_search_clust_talent_merged_count['Responsibilities'].unique().tolist()
        
        st.markdown("#### Daftar Responsibilities yang Tersedia:")
        for i, responsibility in enumerate(responsibilities_list, start=1):
            st.write(f"**{i}.** {responsibility}")

        st.markdown("---")

        st.markdown("#### Masukkan Pilihan Anda:")
        n_input = st.number_input("Berapa banyak talent terbaik yang ingin ditampilkan (n)?", min_value=1, value=5, key='n_talent_final')
        max_x_value = len(responsibilities_list) if len(responsibilities_list) > 0 else 1
        x_input = st.number_input("Masukkan nomor (indeks) dari Responsibility yang ingin Anda pilih (x)?", min_value=1, max_value=max_x_value, value=1, key='x_responsibility_final')

        def searching_streamlit(n, x, df_talent_scores, df_cowork_redundant, resp_list):
            if not (1 <= x <= len(resp_list)):
                st.warning(f"⚠️ Nomor Responsibility (x={x}) tidak valid. Masukkan nomor antara 1 dan {len(resp_list)}.")
                return

            selected_responsibility = resp_list[x - 1]
            
            st.markdown(f"### Analisis untuk Tugas: **{selected_responsibility}**")
            
            filtered_df = df_talent_scores[
                df_talent_scores['Responsibilities'] == selected_responsibility
            ].copy()
            
            filtered_df = filtered_df.sort_values(by='finalscore_scaled', ascending=False)

            top_n_unique_ids = []
            final_top_n_df = pd.DataFrame()
            seen_ids = set()

            for idx, row in filtered_df.iterrows():
                if row['UNIQUE ID'] not in seen_ids:
                    top_n_unique_ids.append(row['UNIQUE ID'])
                    final_top_n_df = pd.concat([final_top_n_df, pd.DataFrame([row])], ignore_index=True)
                    seen_ids.add(row['UNIQUE ID'])
                if len(top_n_unique_ids) >= n:
                    break
            
            st.markdown(f"#### {n} Talent Terbaik untuk Tugas ini:")
            if not final_top_n_df.empty:
                display_top_n = final_top_n_df[['UNIQUE ID', 'Role Person', 'finalscore_scaled']].reset_index(drop=True)
                display_top_n.index = display_top_n.index + 1
                st.dataframe(display_top_n, use_container_width=True)
            else:
                st.info("Tidak ada talent yang ditemukan untuk Responsibility ini.")
                return

            L = final_top_n_df['UNIQUE ID'].tolist()
            
            st.markdown("#### Pertimbangkan Rekan Kerja Terdekat dari Talent Terbaik:")
            co_worker_results = []
            for unique_id in L:
                co_work_df = df_cowork_redundant[df_cowork_redundant['UNIQUE ID'] == unique_id]
                sorted_co_work_df = co_work_df.sort_values(by='weight', ascending=False).head(3)
                
                if not sorted_co_work_df.empty:
                    for _, co_row in sorted_co_work_df.iterrows():
                        co_worker_results.append({
                            'Talent Utama': unique_id,
                            'Rekan Kerja': co_row['co_work'],
                            'Bobot Kedekatan': co_row['weight']
                        })
                else:
                    co_worker_results.append({
                        'Talent Utama': unique_id,
                        'Rekan Kerja': 'Tidak ditemukan',
                        'Bobot Kedekatan': 'N/A'
                    })
            
            if co_worker_results:
                st.dataframe(pd.DataFrame(co_worker_results), use_container_width=True)
            else:
                st.info("Tidak ada data rekan kerja yang ditemukan.")

            if not final_top_n_df.empty and 'Cluster' in final_top_n_df.columns:
                clusters_available = final_top_n_df['Cluster'].dropna()
                if not clusters_available.empty:
                    majority_cluster = clusters_available.mode()[0]
                    st.markdown(f"#### Talent dari Klaster Mayoritas (Klaster {int(majority_cluster)}) yang Relevan:")
                    
                    cluster_df = df_talent_scores[
                        (df_talent_scores['Cluster'] == majority_cluster) &
                        (df_talent_scores['Responsibilities'] == selected_responsibility)
                    ].copy()

                    unique_ids_in_cluster = set()
                    top_n_unique_in_cluster = []
                    for _, row in cluster_df.nlargest(len(cluster_df), 'finalscore').iterrows():
                        if row['UNIQUE ID'] not in unique_ids_in_cluster:
                            unique_ids_in_cluster.add(row['UNIQUE ID'])
                            top_n_unique_in_cluster.append(row)
                        if len(top_n_unique_in_cluster) >= n:
                            break

                    if top_n_unique_in_cluster:
                        display_cluster_talent = pd.DataFrame(top_n_unique_in_cluster)[['UNIQUE ID', 'Role Person', 'finalscore']].reset_index(drop=True)
                        display_cluster_talent.index = display_cluster_talent.index + 1
                        st.dataframe(display_cluster_talent, use_container_width=True)
                    else:
                        st.info(f"Tidak ada talent lain yang relevan di klaster {int(majority_cluster)} untuk Responsibility ini.")
                else:
                    st.info("Tidak dapat menentukan klaster mayoritas karena data klaster tidak tersedia.")
            else:
                st.info("Informasi klaster tidak tersedia untuk talent terpilih.")

        if st.button("Jalankan Pencarian", key='run_final_search'):
            if DF_search_clust_talent_merged_count is not None and DF_Neigh_redundant is not None:
                searching_streamlit(n_input, x_input, DF_search_clust_talent_merged_count, DF_Neigh_redundant, responsibilities_list)
            else:
                st.error("❗ Data yang diperlukan belum lengkap. Pastikan Anda telah mengunggah semua file dan proses di tab sebelumnya sudah selesai.")
    else:
        st.warning("⚠️ Mohon pastikan semua file telah diunggah dan proses di halaman 'Hasil Pencarian Talent Sesuai Use Case' serta 'Co-worker Detection' telah selesai untuk melihat hasil akhir.")

# --- Halaman User Input Talent Search ---
elif st.session_state['current_page'] == 'User Input Talent Search':
    st.title("🔍 User Input Talent Search")
    st.markdown("---")

    st.info("Masukkan detail pekerjaan yang Anda butuhkan di bawah ini untuk mencari talenta yang sesuai.")

    # Ambil dataframe dari session state
    df_skillinv = st.session_state.get('df_skillinv')
    df_talent = st.session_state.get('df_talent')
    df_eval = st.session_state.get('df_eval')
    df_hist = st.session_state.get('df_hist')
    df_assign = st.session_state.get('df_assign')
    DF_search_clust_talent_merged_count = st.session_state.get('DF_search_clust_talent_merged_count') # Diperlukan untuk daftar role

    # Cek ketersediaan data yang diperlukan
    if df_skillinv is None or df_talent is None or df_eval is None or df_hist is None or df_assign is None or DF_search_clust_talent_merged_count is None:
        st.warning("⚠️ Mohon unggah semua file yang diperlukan di halaman 'Upload Data' dan jalankan proses di 'Hasil Pencarian Talent Sesuai Use Case' terlebih dahulu untuk menjalankan fitur ini.")
    else:
        # Dapatkan daftar Role unik dari DF_search_clust_talent_merged_count
        role_options = ['Pilih Role Pekerjaan'] + sorted(DF_search_clust_talent_merged_count['Role'].unique().tolist())

        # Input manual dari user
        user_responsibility = st.text_area("Deskripsi Pekerjaan (Responsibilities)", "Contoh: Menganalisis data besar untuk menemukan pola tren dan membuat model prediktif.")
        user_skill1 = st.text_input("Skill Utama (Skill 1)", "Data Science")
        user_skill2 = st.text_input("Skill Pendukung (Skill 2)", "Python")
        user_role = st.selectbox("Pilihan Role Pekerjaan", role_options, index=0) # Tambahkan dropdown untuk Role

        # Panggil fungsi preprocessing jika belum
        df_skillinv_processed = preprocess_dataframe(df_skillinv.copy(), "Skill Inventory")

        # Inisialisasi model SentenceTransformer (jika belum ada)
        @st.cache_resource
        def load_sentence_transformer_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        model = load_sentence_transformer_model()

        if st.button("Cari Talent", key='user_search_button'):
            if user_role == 'Pilih Role Pekerjaan':
                st.warning("⚠️ Mohon pilih Role Pekerjaan terlebih dahulu.")
                st.stop()

            with st.spinner('Mencari talent sesuai input Anda...'):
                # 1. Buat "ureq" dummy dari input user
                user_ureq_data = {
                    'Responsibilities': [user_responsibility],
                    'Skill 1': [user_skill1],
                    'Skill 2': [user_skill2],
                    'Role': [user_role] # Gunakan input role dari dropdown
                }
                df_user_ureq = pd.DataFrame(user_ureq_data)
                df_user_ureq['agg_sentences'] = df_user_ureq['Responsibilities'] + " " + df_user_ureq['Skill 1'].fillna('') + " " + df_user_ureq['Skill 2'].fillna('')

                # 2. Hitung Kemiripan Antara Input User dengan Skillset yang Ada
                skillsets = df_skillinv_processed.columns[2:-1].tolist()
                user_agg_sentence = df_user_ureq['agg_sentences'].iloc[0]
                
                corpus_user = [user_agg_sentence] + skillsets
                sentence_embeddings_user = model.encode(corpus_user)
                cosine_sim_user = cosine_similarity([sentence_embeddings_user[0]], sentence_embeddings_user[1:])

                results_user = []
                for i in range(len(skillsets)):
                    results_user.append([
                        user_responsibility, user_skill1, user_skill2,
                        user_role, user_agg_sentence, skillsets[i], cosine_sim_user[0][i]
                    ])
                
                df_results_user = pd.DataFrame(results_user, columns=[
                    'Responsibilities', 'Skill 1', 'Skill 2', 'Role', 'agg_sentences', 'Skillset', 'Similarity score'
                ])
                df_results_user = df_results_user.sort_values(by='Similarity score', ascending=False)
                
                st.subheader("💡 Kemiripan Input Anda dengan Skillset Telkom:")
                st.dataframe(df_results_user, use_container_width=True)

                # Ambil threshold dari tab Hasil Pencarian Talent (atau default)
                threshold_val = 0.2999
                if 'threshold_input' in st.session_state:
                     try:
                         threshold_val = float(st.session_state.threshold_input.replace(",", "."))
                     except AttributeError:
                         threshold_val = float(st.session_state.threshold_input)

                df_results_user_filtered = df_results_user[df_results_user['Similarity score'] >= threshold_val]
                
                # 3. Lanjutkan dengan Logika Pencarian Mirip df_search
                unique_ids_roles = df_skillinv_processed[['UNIQUE ID', 'Role']].drop_duplicates()
                unique_ids_roles.rename(columns={'Role': 'Role Person'}, inplace=True)
                
                merged_df_user = df_results_user_filtered.merge(unique_ids_roles, how='cross')
                
                def get_skill_score_user(row, df_skillinv_data):
                    unique_id = row['UNIQUE ID']
                    skill_needed = row['Skillset']
                    if skill_needed in df_skillinv_data.columns:
                        score = df_skillinv_data.loc[df_skillinv_data['UNIQUE ID'] == unique_id, skill_needed]
                        return score.values[0] if not score.empty else None
                    return None
                
                merged_df_user['Skill Score'] = merged_df_user.apply(lambda r: get_skill_score_user(r, df_skillinv_processed), axis=1)
                merged_df_user['Skill 1'].fillna('', inplace=True)
                merged_df_user['Skill 2'].fillna('', inplace=True)

                df_search_user = merged_df_user.groupby(
                    ['Responsibilities', 'Skill 1', 'Skill 2', 'Role', 'agg_sentences', 'UNIQUE ID', 'Role Person'],
                    as_index=False
                ).agg(
                    Skillset=('Skillset', lambda x: list(x)),
                    Avg_SkillScore=('Skill Score', 'mean'),
                )
                
                st.subheader("🔍 Kecocokan Talenta dengan Kebutuhan Anda:")
                if not df_search_user.empty:
                    st.dataframe(df_search_user, use_container_width=True)
                else:
                    st.info("Tidak ada talenta yang cocok dengan kriteria dan threshold yang diberikan.")
                    st.stop()

                # 4. Klasterisasi df_skillinv_processed
                df_skillinv_clustered = st.session_state.get('df_skillinv_clustered')
                if df_skillinv_clustered is None:
                    st.warning("Klasterisasi Skill Inventory sedang diproses ulang (sebaiknya lakukan di halaman 'Hasil Pencarian Talent' terlebih dahulu).")
                    numerical_columns = df_skillinv_processed.select_dtypes(include=[np.number]).columns.tolist()
                    df_numerical_cleaned_local = df_skillinv_processed[numerical_columns].dropna(axis=1)

                    if not df_numerical_cleaned_local.empty and len(df_numerical_cleaned_local) >= 2:
                        pca_local = PCA(n_components=min(2, len(df_numerical_cleaned_local.columns)))
                        df_pca_local = pca_local.fit_transform(df_numerical_cleaned_local)
                        
                        max_k_cluster_local = min(len(df_pca_local) - 1, 8)
                        if max_k_cluster_local >=2:
                            best_k_local = 2
                            best_score_local = -1
                            for k in range(2, max_k_cluster_local + 1):
                                kmeans_local = KMeans(n_clusters=k, random_state=42, n_init='auto')
                                cluster_labels_local = kmeans_local.fit_predict(df_pca_local)
                                score_local = silhouette_score(df_pca_local, cluster_labels_local)
                                if score_local > best_score_local:
                                    best_score_local = score_local
                                    best_k_local = k
                            kmeans_best_local = KMeans(n_clusters=best_k_local, random_state=42, n_init='auto')
                            df_skillinv_processed['Cluster'] = kmeans_best_local.fit_predict(df_pca_local)
                            df_skillinv_clustered = df_skillinv_processed.copy()
                            st.session_state['df_skillinv_clustered'] = df_skillinv_clustered
                        else:
                            st.warning("Tidak cukup data untuk klasterisasi ulang.")
                            df_skillinv_clustered = df_skillinv_processed.assign(Cluster=0) # Default cluster 0
                    else:
                        st.warning("Tidak cukup data numerik untuk klasterisasi ulang.")
                        df_skillinv_clustered = df_skillinv_processed.assign(Cluster=0) # Default cluster 0
                
                df_search_user = df_search_user.merge(
                    df_skillinv_clustered[['UNIQUE ID', 'Cluster']],
                    on='UNIQUE ID',
                    how='left'
                )

                # 5. Agregasi Pegawai dan Evaluasi Pegawai (DF_agg_talent)
                DF_agg_talent_user = pd.merge(df_talent, df_eval, on='UNIQUE ID', how='inner')
                if 'LAMA KERJA BERJALAN' in DF_agg_talent_user.columns:
                    DF_agg_talent_user['Durasi Bulan'] = DF_agg_talent_user['LAMA KERJA BERJALAN'].apply(convert_to_months)
                
                bobot_cap_score_user = st.session_state.get('bobot_cap_score_tab2', 0.8)
                bobot_durasi_user = st.session_state.get('bobot_durasi_tab2', 0.2)

                if 'Durasi Bulan' in DF_agg_talent_user.columns and 'Capability Score' in DF_agg_talent_user.columns:
                    DF_agg_talent_user['scoring_eval'] = (
                        DF_agg_talent_user['Durasi Bulan'] * bobot_durasi_user
                        + DF_agg_talent_user['Capability Score'] * bobot_cap_score_user
                    )
                else:
                    st.error("Kolom 'Durasi Bulan' atau 'Capability Score' tidak ditemukan untuk perhitungan scoring_eval.")
                    st.stop()

                DF_search_talent_merged_user = pd.merge(df_search_user, DF_agg_talent_user, on='UNIQUE ID', how='inner')

                # 6. Hitung job_count dari df_hist
                DF_hist_count = df_hist.groupby("UNIQUE ID")["PRODUCT / USECASE"].nunique().reset_index()
                DF_hist_count.columns = ["UNIQUE ID", "job_count"]

                DF_search_clust_talent_merged_count_user = pd.merge(
                    DF_search_talent_merged_user,
                    DF_hist_count[['UNIQUE ID', 'job_count']],
                    on='UNIQUE ID',
                    how='left'
                )
                DF_search_clust_talent_merged_count_user['job_count'] = DF_search_clust_talent_merged_count_user['job_count'].fillna(0)

                # 7. Hitung finalscore
                a_user = st.session_state.get('coeff_a_tab2', 0.42)
                b_user = st.session_state.get('coeff_b_tab2', 0.48)
                c_user = st.session_state.get('coeff_c_tab2', 0.1)
                r_user = st.session_state.get('coeff_r_tab2', 1.2)

                DF_search_clust_talent_merged_count_user['d'] = (
                    DF_search_clust_talent_merged_count_user['Avg_SkillScore'] * a_user +
                    DF_search_clust_talent_merged_count_user['scoring_eval'] * b_user +
                    DF_search_clust_talent_merged_count_user['job_count'] * c_user
                )

                DF_search_clust_talent_merged_count_user['finalscore'] = DF_search_clust_talent_merged_count_user.apply(
                    lambda row: row['d'] * r_user if row['Role Person'] == row['Role'] else row['d'], axis=1
                )
                DF_search_clust_talent_merged_count_user['finalscore_scaled'] = DF_search_clust_talent_merged_count_user.groupby(
                    'Responsibilities')['finalscore'].transform(minmax_scaling)

                # 8. Tampilkan hasil akhir
                st.markdown("### 🏆 Rekomendasi Talent Terbaik untuk Kebutuhan Anda:")
                n_display_user = st.number_input("Berapa banyak talent rekomendasi yang ingin ditampilkan?", min_value=1, value=5, key='n_display_user')
                
                final_recommendations = DF_search_clust_talent_merged_count_user.sort_values(by='finalscore_scaled', ascending=False)
                
                unique_final_recommendations = pd.DataFrame()
                seen_final_ids = set()
                for idx, row in final_recommendations.iterrows():
                    if row['UNIQUE ID'] not in seen_final_ids:
                        unique_final_recommendations = pd.concat([unique_final_recommendations, pd.DataFrame([row])], ignore_index=True)
                        seen_final_ids.add(row['UNIQUE ID'])
                    if len(unique_final_recommendations) >= n_display_user:
                        break

                if not unique_final_recommendations.empty:
                    display_final = unique_final_recommendations[['UNIQUE ID', 'Role Person', 'finalscore_scaled', 'Cluster', 'Avg_SkillScore']].reset_index(drop=True)
                    display_final.index = display_final.index + 1
                    st.dataframe(display_final, use_container_width=True)

                    st.markdown("#### Pertimbangkan Rekan Kerja Terdekat dari Talent Rekomendasi:")
                    DF_Neigh_redundant_user = st.session_state.get('DF_Neigh_redundant')
                    if DF_Neigh_redundant_user is not None:
                        co_worker_results_user = []
                        for unique_id in unique_final_recommendations['UNIQUE ID'].tolist():
                            co_work_df_user = DF_Neigh_redundant_user[DF_Neigh_redundant_user['UNIQUE ID'] == unique_id]
                            sorted_co_work_df_user = co_work_df_user.sort_values(by='weight', ascending=False).head(3)
                            
                            if not sorted_co_work_df_user.empty:
                                for _, co_row in sorted_co_work_df_user.iterrows():
                                    co_worker_results_user.append({
                                        'Talent Utama': unique_id,
                                        'Rekan Kerja': co_row['co_work'],
                                        'Bobot Kedekatan': co_row['weight']
                                    })
                            else:
                                co_worker_results_user.append({
                                    'Talent Utama': unique_id,
                                    'Rekan Kerja': 'Tidak ditemukan',
                                    'Bobot Kedekatan': 'N/A'
                                })
                        
                        if co_worker_results_user:
                            st.dataframe(pd.DataFrame(co_worker_results_user), use_container_width=True)
                        else:
                            st.info("Tidak ada data rekan kerja yang ditemukan untuk talent rekomendasi ini.")
                    else:
                        st.warning("⚠️ Data rekan kerja belum tersedia. Mohon proses 'Co-worker Detection' terlebih dahulu.")

                    if not unique_final_recommendations.empty and 'Cluster' in unique_final_recommendations.columns:
                        clusters_available_user = unique_final_recommendations['Cluster'].dropna()
                        if not clusters_available_user.empty:
                            majority_cluster_user = clusters_available_user.mode()[0]
                            st.markdown(f"#### Talent Lain dari Klaster Mayoritas (Klaster {int(majority_cluster_user)}) yang Relevan:")
                            
                            cluster_df_user = DF_search_clust_talent_merged_count_user[
                                (DF_search_clust_talent_merged_count_user['Cluster'] == majority_cluster_user) &
                                (DF_search_clust_talent_merged_count_user['Responsibilities'] == user_responsibility)
                            ].copy()

                            unique_ids_in_cluster_user = set()
                            top_n_unique_in_cluster_user = []
                            existing_top_n_ids = set(unique_final_recommendations['UNIQUE ID'].tolist())

                            for _, row in cluster_df_user.nlargest(len(cluster_df_user), 'finalscore').iterrows():
                                if row['UNIQUE ID'] not in unique_ids_in_cluster_user and row['UNIQUE ID'] not in existing_top_n_ids:
                                    unique_ids_in_cluster_user.add(row['UNIQUE ID'])
                                    top_n_unique_in_cluster_user.append(row)
                                if len(top_n_unique_in_cluster_user) >= n_display_user:
                                    break

                            if top_n_unique_in_cluster_user:
                                display_cluster_talent_user = pd.DataFrame(top_n_unique_in_cluster_user)[['UNIQUE ID', 'Role Person', 'finalscore']].reset_index(drop=True)
                                display_cluster_talent_user.index = display_cluster_talent_user.index + 1
                                st.dataframe(display_cluster_talent_user, use_container_width=True)
                            else:
                                st.info(f"Tidak ada talent tambahan yang relevan di klaster {int(majority_cluster_user)} untuk Responsibility ini.")
                        else:
                            st.info("Tidak dapat menentukan klaster mayoritas karena data klaster tidak tersedia untuk rekomendasi.")
                    else:
                        st.info("Informasi klaster tidak tersedia untuk rekomendasi talent.")

                else:
                    st.info("Tidak ada rekomendasi yang dapat ditampilkan berdasarkan input Anda.")
