# Penerapan Singular Value Decomposition (SVD) pada Latent Semantic Indexing untuk Sistem Temu Balik Informasi Basis Data Kecelakaan Pesawat

## 1. Deskripsi Program

Program ini adalah mesin pencari (*Information Retrieval*) untuk korpus narasi laporan investigasi kecelakaan pesawat dari Aviation Safety Network (`asn.flightsafety.org`). Sistem membandingkan dua pendekatan pencarian:

| Mode | Cara Kerja | Karakteristik |
| :--- | :--- | :--- |
| **VSM (*baseline*)** | Menghitung *cosine similarity* langsung pada matriks TF-IDF berdimensi penuh ($m \times n$). | Hanya mengenali kecocokan kata persis (*exact keyword matching*). |
| **SVD-LSI (utama)** | Menghitung *cosine similarity* pada ruang laten berdimensi $k$ hasil *Truncated SVD*. | Menangkap relasi semantik dan sinonim (misalnya `powerplant failure` terhubung ke `engine failure`). |

### Alur Pemrosesan

1. **Ingest:** Mengekstrak teks dari file PDF pada `--raw-dir` (default `data/raw/`), membersihkan *boilerplate*, dan menyimpannya ke `--out` (default `data/processed/dataset_asn.csv`).
2. **Preprocessing:** Melakukan *case folding*, penghapusan tanda baca dan angka, tokenisasi, penghapusan *stopwords*, serta *stemming* (opsional, bisa dimatikan via `--no-stem`).
3. **Pembobotan TF-IDF:** Membentuk matriks *term-document* $A \in \mathbb{R}^{m \times n}$ sesuai skema `--tf` dan penyaringan `--min-df`.
4. **Dekomposisi SVD:** Menguraikan $A = U \Sigma V^T$ menggunakan `numpy.linalg.svd`, lalu memangkasnya menjadi hampiran peringkat-$k$:
   $$A_k = U_k \Sigma_k V_k^T$$
5. **Proyeksi *Query*:** Memetakan vektor *query* $q \in \mathbb{R}^{m \times 1}$ ke ruang laten berdimensi-$k$:
   $$\hat{q} = q^T U_k \Sigma_k^{-1}$$
   sementara setiap dokumen ke-$j$ direpresentasikan oleh baris ke-$j$ dari matriks $V_k$.
6. **Pemeringkatan & Evaluasi:** Mengurutkan dokumen dengan *cosine similarity*, lalu menghitung metrik evaluasi ($\text{P@}N$, $\text{MAP}$, $\text{MRR}$), uji sensitivitas nilai $k$, grafik, dan ekspor tabel.

Seluruh perhitungan inti (TF-IDF, SVD, pemangkasan peringkat-$k$, proyeksi *query*, dan *cosine similarity*) ditulis secara eksplisit menggunakan `numpy` tanpa `scikit-learn` agar transparan dan selaras dengan formulasi matematis di makalah.

---

## 2. Glosarium Variabel, Simbol Matematis, dan Metrik

Bagian ini menjelaskan seluruh variabel dan simbol yang muncul pada rumus, kode program, terminal, serta keluaran tabel/grafik.

### 2.1 Variabel Matriks & Aljabar Linier

| Simbol / Variabel | Dimensi / Tipe | Penjelasan |
| :--- | :--- | :--- |
| $m$ | Bilangan bulat | Jumlah *term* (kata unik) dalam *vocabulary* setelah *preprocessing* dan penyaringan `--min-df`. |
| $n$ | Bilangan bulat | Jumlah total dokumen laporan kecelakaan pada korpus. |
| $r$ (`rank_r`) | $r \le \min(m, n)$ | Peringkat (*rank*) numerik matriks $A$, yaitu jumlah nilai singular positif ($\sigma_i > 0$). Jika $r < n$, terdapat dokumen yang bergantung linier (misalnya duplikat atau teks kosong). |
| $A$ | $m \times n$ | Matriks *term-document* berbobot TF-IDF. Baris mewakili *term*, kolom mewakili dokumen. |
| $U$ | $m \times r$ | Matriks vektor singular kiri (*term-concept*) yang kolom-kolomnya saling ortonormal ($U^T U = I_r$). |
| $\Sigma$ (`Sigma`) | $r \times r$ | Matriks diagonal berisi nilai singular terurut menurun $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$. |
| $V^T$ (`Vt`) | $r \times n$ | Transpos matriks vektor singular kanan (*concept-document*) yang baris-barisnya saling ortonormal ($V^T V = I_r$). |
| $k$ (`--k`) | $1 \le k \le r$ | Dimensi ruang laten (*rank truncation*), yaitu jumlah konsep abstrak / nilai singular terbesar yang dipertahankan. |
| $U_k, \Sigma_k, V_k^T$ | $(m \times k), (k \times k), (k \times n)$ | Submatriks hasil pemangkasan SVD ke $k$ komponen utama pertama. |
| $A_k$ | $m \times n$ | Matriks hampiran peringkat-$k$ ($A_k = U_k \Sigma_k V_k^T$), merupakan hampiran terbaik terhadap $A$ menurut Teorema Eckart-Young-Mirsky. |
| $q$ | $m \times 1$ | Vektor *query* pengguna dalam ruang TF-IDF asli menggunakan bobot IDF korpus. |
| $\hat{q}$ (`q_hat`) | $1 \times k$ | Vektor *query* setelah diproyeksikan ke ruang laten berdimensi-$k$ ($\hat{q} = q^T U_k \Sigma_k^{-1}$). |

### 2.2 Variabel Pembobotan & Validasi Numerik (`build`)

| Variabel | Rumus / Definisi | Penjelasan |
| :--- | :--- | :--- |
| $\text{TF}(t, d)$ | `raw`: $f_{t,d}$ <br>`log`: $1 + \ln(f_{t,d})$ | *Term Frequency* — seberapa sering *term* $t$ muncul di dokumen $d$. Skema `log` meredam dominasi kata yang muncul berulang kali pada laporan panjang. |
| $\text{df}(t)$ | $1 \le \text{df}(t) \le n$ | *Document Frequency* — jumlah dokumen yang mengandung *term* $t$ minimal satu kali. |
| $\text{IDF}(t)$ | $\ln(n / \text{df}(t)) + 1$ | *Inverse Document Frequency* — memberi bobot lebih besar pada kata spesifik dan bobot kecil pada kata yang muncul di hampir semua laporan. |
| $\text{sim}(u, v)$ | $\frac{u \cdot v}{\|u\|_2 \|v\|_2}$ | *Cosine Similarity* — ukuran kemiripan arah dua vektor. Pada VSM bernilai $[0, 1]$; pada LSI bernilai $[-1, 1]$. |
| `energi_persen` | $\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \times 100\%$ | Persentase total varians/energi matriks $A$ yang berhasil dipertahankan pada dimensi $k$. |
| `galat_frobenius` | $\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$ | Besar kesalahan rekonstruksi antara matriks asli $A$ dan matriks hampiran $A_k$ dalam norma Frobenius. |
| `ortogonalitas` | $\|U^T U - I\|_F$, $\|V^T V - I\|_F$ | Uji numerik keortonormalan faktor SVD. Nilai mendekati $0$ (orde $10^{-12}$ s.d. $10^{-13}$) membuktikan dekomposisi stabil. |

### 2.3 Variabel Metrik Evaluasi (`eval`)

| Metrik | Rentang | Penjelasan |
| :--- | :---: | :--- |
| **$\text{P@}N$ (`P@5`)** | $0 \text{ s.d. } 1$ | *Precision at $N$* — proporsi dokumen relevan di antara $N$ hasil peringkat teratas. Contoh: `P@5 = 0.80` berarti 4 dari 5 hasil teratas tepat sasaran. |
| **$\text{AP}$** | $0 \text{ s.d. } 1$ | *Average Precision* — rata-rata presisi pada setiap posisi ditemukannya dokumen relevan untuk **satu *query***. Menghargai sistem yang meletakkan dokumen relevan lebih dekat ke puncak daftar. |
| **$\text{MAP}$** | $0 \text{ s.d. } 1$ | *Mean Average Precision* — rata-rata nilai $\text{AP}$ di seluruh *query* uji. Menjadi indikator utama kualitas pemeringkatan menyeluruh. |
| **$\text{RR}$** | $0 \text{ s.d. } 1$ | *Reciprocal Rank* — nilai $1 / \text{rank}_1$, di mana $\text{rank}_1$ adalah posisi dokumen relevan pertama yang muncul untuk satu *query* (peringkat 1 $= 1{,}0$; peringkat 2 $= 0{,}5$; peringkat 4 $= 0{,}25$). |
| **$\text{MRR}$** | $0 \text{ s.d. } 1$ | *Mean Reciprocal Rank* — rata-rata $\text{RR}$ dari seluruh *query* uji. Mengukur seberapa cepat pengguna menemukan jawaban relevan pertama. |
| **`--k-sweep`** | Daftar integer (default `K_SWEEP` di `lsi/config.py`) | Daftar nilai $k$ yang diuji secara beruntun saat `eval` untuk menghasilkan tabel dan grafik sensitivitas dimensi $k$. |

---

## 3. Struktur Direktori

```text
aviation-accident-lsi/
├── README.md                      # Dokumentasi program
├── requirements.txt               # Kebutuhan library untuk program
├── main.py                        # Entry point CLI utama: ingest, build, search, eval
├── batch_search.py                # Runner pencarian mode batch (file) & terminal (CLI/interaktif)
├── data/
│   ├── raw/                       # PDF asli ASN dikelompokkan per tahun (2000–2026)
│   ├── processed/
│   │   ├── dataset_asn.csv        # Dataset utama hasil ekstraksi PDF (1.713 dokumen)
│   │   └── index_cache.npz        # Cache matriks TF-IDF dan faktor U, Sigma, Vt (otomatis)
│   └── eval/
│       └── queries.json           # Contoh 100 query batch
├── lsi/
│   ├── config.py                  # Path default dan konstanta awal (dapat dioverride lewat CLI)
│   ├── schemas.py                 # Dataclass struktur data antar-modul
│   ├── io/
│   │   ├── pdf_reader.py          # Ekstraksi teks PDF menjadi objek Document
│   │   └── loader.py              # Baca/tulis CSV, JSON, dan cache .npz
│   ├── nlp/
│   │   ├── cleaner.py             # Pembersihan boilerplate PDF, metadata, & label heuristik
│   │   └── tokenizer.py           # Tokenisasi, stopword removal, dan stemming
│   ├── engine/
│   │   ├── vectorizer.py          # Vocabulary dan pembobotan matriks TF-IDF
│   │   ├── svd_model.py           # SVD, truncation k, proyeksi query, energi, & galat
│   │   ├── similarity.py          # Norma Euclidean dan cosine similarity
│   │   └── retriever.py           # Orkestrasi pencarian VSM vs LSI & manajemen cache
│   └── analysis/
│       ├── evaluation.py          # Perhitungan P@N, AP/MAP, MRR, dan sensitivitas k
│       ├── visualization.py       # Grafik scree plot, energi kumulatif, & perbandingan skor
│       └── exporter.py            # Ekspor tabel ke CSV, Markdown, LaTeX, dan log
├── tests/
│   ├── test_svd_math.py           # Pengujian ortogonalitas & galat rekonstruksi SVD
│   └── test_retriever.py          # Pengujian alur pencarian, cache, evaluasi, & NLP
└── output/
    ├── figures/                   # Grafik .png untuk makalah
    ├── tables/                    # Tabel .csv, .md, .tex (termasuk subfolder batch/)
    └── logs/                      # Ringkasan log evaluasi
```

---

## 4. Cara Menjalankan

### 4.1 Prasyarat dan Instalasi

Membutuhkan Python 3.10 atau lebih baru. Jalankan seluruh perintah dari direktori root `aviation-accident-lsi/`.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Alur Kerja Utama (Parameter Dapat Disesuaikan Pengguna)

Semua jalur berkas, dimensi $k$, jumlah hasil teratas, skema pembobotan, dan direktori keluaran memiliki nilai bawaan tetapi **dapat diubah langsung melalui argumen CLI** sesuai kebutuhan Anda:

```bash
# 1. Ekstrak PDF dari direktori masukan pilihan Anda ke berkas CSV tujuan
python main.py ingest --raw-dir data/raw --out data/processed/dataset_asn.csv --min-words 8

# 2. Bangun indeks TF-IDF + SVD dengan parameter (k, tf, min-df, cache, output-dir) sesuai kebutuhan
python main.py build --data data/processed/dataset_asn.csv --cache data/processed/index_cache.npz \
  --k 150 --tf log --min-df 1 --output-dir output

# 3. Jalankan pencarian dengan query, mode, jumlah hasil (--top), dan nama berkas ekspor Anda sendiri
python main.py search "windshear on final approach" --k 150 --top 10 --mode both \
  --export md csv tex --out-name hasil_windshear --output-dir output

# 4. Jalankan evaluasi dengan berkas query dan daftar nilai k (--k-sweep) yang dapat ditentukan sendiri
python main.py eval --queries data/eval/queries.json --k 150 --top 5 \
  --k-sweep 10 25 50 100 150 200 300 --output-dir output --log-name eval_100_queries
```

> **Catatan tentang `category`:** Saat `ingest`, kolom `category` diisi secara heuristik (`UNLABELED` bila tidak ada kata kunci cocok). Pastikan label `category` pada dataset masukan sudah diverifikasi sebelum menjalankan `eval`.

### 4.3 Pencarian Mode Batch & Terminal (`batch_search.py`)

Menjalankan `main.py search ... --export` berulang kali dengan `--out-name` yang sama akan menimpa berkas sebelumnya. Gunakan **`batch_search.py`** yang memuat indeks **satu kali di memori** lalu mengarsipkan setiap hasil ekspor secara unik ke `--out-dir` sekaligus menggabungkannya ke `--summary-name`.

```bash
# A. Mode Batch penuh dari berkas JSON/TXT (ubah --file, --k, --top, --out-dir, --summary-name sesuai kebutuhan)
python batch_search.py --file data/eval/queries.json --k 150 --top 5 \
  --export md csv tex --out-dir output/tables/batch --summary-name rekap_semua_query.md

# A2. Mode Batch dengan filter kategori atau batas jumlah uji cepat
python batch_search.py --file data/eval/queries.json --category WEATHER --k 150 --top 5
python batch_search.py --file data/eval/queries.json --category CFIT --limit 5 --k 100 --mode lsi

# B. Mode Terminal dengan daftar query representatif dari beragam insiden
python batch_search.py \
  --queries "uncontained engine failure" "overran the wet runway" "microburst on final approach" "controlled flight into terrain" \
  --k 150 --top 5 --no-export

# C. Mode Terminal Interaktif (ketik query satu per satu; indeks hanya dimuat sekali di awal)
python batch_search.py --interactive --k 150 --top 5 --mode both --no-export
```

---

## 5. Referensi Lengkap Opsi & Variabel Perintah

### 5.1 Opsi Indeks & Output Bersama (`build`, `search`, `eval`, dan `batch_search.py`)

| Opsi CLI | Default | Fungsi & Dampak |
| :--- | :--- | :--- |
| `--data PATH` | `data/processed/dataset_asn.csv` | Menentukan lokasi berkas korpus masukan (`.csv` atau `.json`). |
| `--cache PATH` | `data/processed/index_cache.npz` | Menentukan lokasi penyimpanan/pembacaan cache indeks `.npz`. |
| `--k N` | `150` | Menetapkan jumlah dimensi ruang laten SVD yang dipertahankan ($1 \le k \le r$). |
| `--top N` | `5` | Jumlah dokumen peringkat teratas yang ditampilkan dan dipakai untuk $\text{P@}N$. |
| `--tf {raw,log}` | `log` | Memilih formula *Term Frequency*: `raw` (hitungan mentah) atau `log` ($1 + \ln(\text{count})$). |
| `--min-df N` | `1` | Membuang *term* yang muncul di kurang dari $N$ dokumen. Menaikkannya (misal `3`) memangkas *noise* kata langka dan memperkecil $m$. |
| `--no-stem` | nonaktif | Menonaktifkan proses *stemming* (mempertahankan bentuk kata asli). |
| `--no-cache` | nonaktif | Mengabaikan file cache `.npz` dan memaksa penghitungan ulang TF-IDF serta SVD dari awal. |

### 5.2 Subperintah `main.py`

| Perintah | Opsi Khusus (Dapat Diubah Pengguna) | Keluaran |
| :--- | :--- | :--- |
| `python main.py ingest` | `--raw-dir PATH`, `--out PATH`, `--min-words N` | Berkas dataset CSV pada `--out` |
| `python main.py build` | Opsi indeks, `--output-dir PATH` | Cache `.npz` pada `--cache`, `scree_plot.png`, `energy_plot.png` di `<output-dir>/figures/` |
| `python main.py search "<query>"` | Opsi indeks, `--mode {both,vsm,lsi}`, `--export csv md tex`, `--output-dir PATH`, `--out-name NAMA` | Tabel peringkat di terminal; `<output-dir>/tables/<out-name>.*` bila `--export` aktif |
| `python main.py eval` | Opsi indeks, `--queries PATH`, `--k-sweep K1 K2 ...`, `--output-dir PATH`, `--log-name NAMA` | Tabel, grafik, dan log evaluasi di `<output-dir>/` |

**Opsi khusus `batch_search.py`:**
* `--file PATH`: Berkas daftar *query* (`.json` terstruktur atau `.txt` satu *query* per baris; default `data/eval/queries.json`).
* `--queries Q1 Q2 ...`: Daftar *query* langsung dari terminal.
* `--interactive`: Mode interaktif di terminal.
* `--mode {both,vsm,lsi}`, `--export {md,csv,tex}`, `--no-export`, `--category NAMA`, `--limit N`, `--out-dir PATH`, `--summary-name FILE`.

**Rincian keluaran `eval` (di dalam `--output-dir`):**
* `tables/`: `metrik_per_query.csv`, `metrik_ringkasan.{csv,md,tex}`, `sensitivitas_k.{csv,md,tex}`, `ranking_per_query.md`.
* `figures/`: `scree_plot.png`, `energy_plot.png`, `map_vs_k.png`, `skor_vs_k.png`, `skor_q<n>.png`.
* `logs/`: `<log-name>.log` (default `eval_summary.log`).

### 5.3 Pengujian

```bash
# Jalankan seluruh unit test matematis SVD, retriever, cache, evaluasi, & NLP
python -m pytest -q
```

---

## 6. Format Data

### 6.1 Dataset Korpus (`.csv` atau `.json`)
Kolom wajib: `doc_id` dan `narrative`.

| Kolom | Tipe | Keterangan |
| :--- | :--- | :--- |
| `doc_id` | String | Pengenal unik dokumen (misalnya `ASN-2004-...`). |
| `date` | String | Tanggal insiden (`YYYY-MM-DD`). |
| `aircraft_type` | String | Tipe pesawat terbang. |
| `phase` | String | Fase penerbangan saat kejadian (`Takeoff`, `Approach`, `Landing`, dll.). |
| `category` | String | Label kategori insiden (`ENGINE`, `RUNWAY`, `WEATHER`, `CFIT`, atau kategori lain buatan Anda). Hanya digunakan sebagai *ground truth* saat `eval`, tidak dipakai saat proses pencarian. |
| `narrative` | String | Teks narasi laporan kecelakaan yang diindeks oleh sistem. |
| `source_url` | String | Tautan atau rujukan sumber laporan asli. |

### 6.2 Berkas Query (`queries.json`, `queries.json`, atau `.txt`)

Untuk format `.json`:

```json
[
  {
    "id": 1,
    "query": "powerplant failure",
    "target_category": "ENGINE"
  },
  {
    "id": 2,
    "query": "overran the wet runway",
    "target_category": "RUNWAY"
  },
  {
    "id": 3,
    "query": "windshear on final approach",
    "target_category": "WEATHER"
  },
  {
    "id": 4,
    "query": "controlled flight into terrain",
    "target_category": "CFIT"
  }
]
```

Untuk `batch_search.py --file`, Anda juga dapat memakai berkas `.txt` biasa berisi satu *query* per baris (baris diawali `#` dianggap komentar). Sebuah dokumen dianggap relevan terhadap suatu *query* saat `eval` apabila nilai `category` dokumen tersebut sama dengan `target_category`.

---

## 7. Catatan Teknis

* **SVD Ekonomi:** Menggunakan `full_matrices=False` sehingga $U \in \mathbb{R}^{m \times r}$ dengan $r = \min(m, n)$. Untuk seluruh $k < r$, hasil pemangkasan identik secara matematis dengan SVD penuh namun jauh lebih hemat memori.
* **Penanganan *Out-of-Vocabulary*:** Kata pada *query* yang tidak ada di *vocabulary* korpus diabaikan. Jika seluruh kata pada *query* tidak dikenal, sistem mengembalikan skor `0` pada kedua mode.
* **Validasi Cache Otomatis:** Cache `.npz` otomatis diabaikan bila isi dataset atau parameter indeks berubah. Perintah `build` selalu membangun ulang dan menimpa file pada `--cache`.
* **Batasan Ekstraksi PDF:** Parsing metadata (`Date`, `Type`, `Phase`, `Narrative`) memakai pola *regex* pada `lsi/nlp/cleaner.py`. File PDF hasil pindai gambar (*scanned image* tanpa lapisan teks) tidak dapat diekstrak langsung tanpa OCR.
