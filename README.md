# Dokumentasi Proyek Analisis Sentimen Ulasan Google Play Store

## 1. Deskripsi Proyek
Proyek ini bertujuan untuk membangun dan mengevaluasi model *Deep Learning* dan *Machine Learning* untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi yang diperoleh dari Google Play Store.

Analisis sentimen dilakukan dengan mengklasifikasikan ulasan ke dalam tiga kategori sentimen, yaitu:
- **Positif:** ulasan yang menunjukkan kepuasan atau pengalaman positif pengguna.
- **Netral:** ulasan yang menunjukkan pendapat netral atau tidak memiliki kecenderungan sentimen yang kuat.
- **Negatif:** ulasan yang menunjukkan ketidakpuasan atau pengalaman negatif pengguna.

Proyek ini mencakup seluruh alur pengembangan model, mulai dari pengumpulan data (*web scraping*), eksplorasi data, pembersihan teks, preprocessing, pelatihan model, hingga evaluasi performa model.
Tiga model yang dikembangkan adalah Support Vector Machine (SVM), Random Forest (RF), dan Bidirectional Gated Recurrent Unit (BiGRU).

## 2. Teknologi dan Library
Proyek dikembangkan menggunakan Python dengan memanfaatkan beberapa library untuk pengolahan data, pemrosesan teks, pemodelan, visualisasi, dan evaluasi.

| Library | Kegunaan |
|---|---|
| pandas | Manipulasi dan analisis dataset |
| NumPy | Operasi numerik |
| google-play-scraper | Mengambil ulasan dari Google Play Store |
| BeautifulSoup4 | Membantu pemrosesan HTML jika diperlukan |
| re | Pembersihan teks menggunakan regular expression |
| Sastrawi | Stemming bahasa Indonesia |
| NLTK | Pemrosesan bahasa alami |
| scikit-learn | TF-IDF, SVM, Random Forest, pembagian data, dan evaluasi |
| TensorFlow / Keras | Pembuatan dan pelatihan model BiGRU |
| Matplotlib | Visualisasi hasil analisis dan evaluasi |
| Seaborn | Visualisasi statistik dan confusion matrix |
| WordCloud | Visualisasi kata yang sering muncul |
| Joblib | Penyimpanan dan pemuatan model machine learning |
| Jupyter Notebook | Lingkungan untuk menjalankan eksperimen dan analisis |

Daftar library di atas merupakan gambaran teknologi yang digunakan. Library yang benar-benar diperlukan harus disesuaikan dengan seluruh import pada file scraping dan notebook proyek.

## 3. Pengumpulan Data (Web Scraping)
Pengumpulan data dilakukan dengan mengambil ulasan pengguna dari Google Play Store menggunakan library `google-play-scraper`.
Data ulasan dapat dikumpulkan dari beberapa aplikasi agar dataset mencakup variasi pengalaman dan pendapat pengguna.

Informasi yang dikumpulkan dapat mencakup:
- ID ulasan (reviewId).
- Rating atau skor ulasan (score).
- Isi ulasan (content).
- Tanggal ulasan (at).

Data hasil scraping kemudian disimpan dalam format CSV atau JSON agar dapat digunakan pada proses analisis dan pelatihan model.

### Hasil scraping
Dataset hasil scraping menjadi sumber utama untuk seluruh eksperimen analisis sentimen. Sebelum digunakan, data diperiksa untuk mengetahui jumlah ulasan, kelengkapan kolom, nilai kosong, duplikasi, dan distribusi rating.

**Catatan:** jumlah data akhir mengikuti hasil scraping yang berhasil dijalankan. Jumlah data setelah proses pembersihan dapat berbeda dari jumlah data mentah.

## 4. Exploratory Data Analysis (EDA)
Exploratory Data Analysis dilakukan untuk memahami karakteristik dataset sebelum proses pemodelan.

Tahapan EDA meliputi:
1. Memeriksa dimensi dataset menggunakan jumlah baris dan kolom.
2. Memeriksa tipe data dan struktur dataset.
3. Mengidentifikasi nilai kosong dan data duplikat.
4. Menganalisis distribusi rating pengguna.
5. Mengamati panjang ulasan dan distribusi teks.
6. Mengidentifikasi kata yang paling sering muncul.
7. Memvisualisasikan distribusi data menggunakan grafik dan WordCloud.

Hasil EDA digunakan untuk memahami karakteristik data dan menentukan langkah preprocessing yang sesuai.

## 5. Data Preprocessing dan Text Cleaning
Tahapan preprocessing dilakukan untuk mempersiapkan ulasan agar dapat digunakan oleh model machine learning dan deep learning.

### 5.1 Pembersihan data
Data diperiksa untuk menangani nilai kosong, ulasan duplikat, dan teks yang tidak dapat digunakan dalam proses pelatihan.

### 5.2 Text cleaning
Pembersihan teks dilakukan untuk mengurangi elemen yang tidak relevan, antara lain:

- URL dan tautan.
- Tag HTML.
- Emoji dan simbol tertentu.
- Angka atau karakter yang tidak diperlukan sesuai strategi preprocessing.
- Spasi berlebih.
- Perbedaan huruf kapital dan huruf kecil melalui proses *lowercasing*.

Pembersihan tetap perlu mempertahankan informasi yang berpengaruh terhadap sentimen, seperti kata negasi yang dapat mengubah makna kalimat.

### 5.3 Normalisasi dan preprocessing bahasa
Jika diterapkan, tahapan lanjutan meliputi:

- Normalisasi kata tidak baku menjadi kata yang lebih standar.
- Penghapusan stopwords yang tidak relevan.
- Tokenisasi teks.
- Stemming menggunakan Sastrawi untuk mengembalikan kata ke bentuk dasarnya.

Langkah yang digunakan dapat berbeda antara model SVM, Random Forest, dan BiGRU karena setiap model memiliki kebutuhan representasi teks yang berbeda.

### 5.4 Pelabelan sentimen
Ulasan dikelompokkan ke dalam tiga kelas sentimen: negatif, netral, dan positif.
Pelabelan dapat menggunakan rating sebagai dasar awal, aturan berbasis leksikon, atau metode lain yang telah ditentukan dalam notebook. Jika label diturunkan dari rating, hasil klasifikasi perlu dipahami sebagai prediksi label berdasarkan aturan tersebut, bukan selalu sebagai penilaian sentimen yang telah diverifikasi secara manual.

## 6. Pelatihan Model
Tiga model dikembangkan menggunakan pendekatan representasi teks dan pembagian dataset yang berbeda.

### 6.1 Support Vector Machine (SVM) dengan TF-IDF
**Pembagian dataset: 80% data latih dan 20% data uji.**
SVM digunakan sebagai model machine learning untuk mengklasifikasikan ulasan berdasarkan representasi numerik dari teks.

Tahapan pelatihan:
1. Mengambil teks yang telah melalui preprocessing.
2. Mengubah teks menjadi representasi numerik menggunakan TF-IDF (*Term Frequency–Inverse Document Frequency*).
3. Membagi dataset menjadi data latih dan data uji dengan rasio 80:20.
4. Melatih model SVM menggunakan data latih.
5. Melakukan prediksi terhadap data uji.
6. Mengevaluasi hasil prediksi.

TF-IDF memberikan bobot pada kata berdasarkan frekuensi kemunculannya dalam dokumen dan keseluruhan kumpulan dokumen. Representasi ini digunakan oleh SVM untuk mempelajari pola yang membedakan kelas sentimen.

### 6.2 Random Forest (RF) dengan TF-IDF
**Pembagian dataset: 70% data latih dan 30% data uji.**
Random Forest merupakan algoritma ensemble yang menggabungkan prediksi dari sejumlah decision tree untuk menghasilkan klasifikasi.

Tahapan pelatihan:
1. Mengambil teks yang telah melalui preprocessing.
2. Mengubah teks menjadi fitur numerik menggunakan TF-IDF.
3. Membagi dataset menjadi data latih dan data uji dengan rasio 70:30.
4. Melatih model Random Forest.
5. Menghasilkan prediksi sentimen pada data uji.
6. Mengevaluasi performa model.

Penggunaan TF-IDF memungkinkan teks diubah menjadi fitur numerik yang dapat digunakan oleh Random Forest. Namun, representasi TF-IDF dapat menghasilkan matriks berdimensi tinggi, sehingga penggunaan memori dan waktu pelatihan perlu diperhatikan.

### 6.3 Bidirectional Gated Recurrent Unit (BiGRU)
**Pembagian dataset: 70% data latih, 15% data validasi, dan 15% data uji.**
BiGRU merupakan arsitektur deep learning berbasis recurrent neural network yang memproses urutan token dari dua arah, yaitu maju dan mundur. Pendekatan ini memungkinkan model mempelajari konteks kata dari kedua arah dalam sebuah ulasan.

Tahapan pelatihan:
1. Mengambil teks yang telah melalui preprocessing.
2. Melakukan tokenisasi untuk mengubah kata menjadi urutan indeks numerik.
3. Melakukan padding agar urutan teks memiliki panjang yang seragam.
4. Membagi dataset menjadi data latih, validasi, dan uji dengan rasio 70:15:15.
5. Membangun arsitektur BiGRU menggunakan TensorFlow/Keras.
6. Melatih model menggunakan data latih.
7. Menggunakan data validasi untuk memantau proses pelatihan dan membantu pemilihan konfigurasi model.
8. Mengevaluasi model menggunakan data uji.

Tokenisasi dan padding diperlukan karena BiGRU memproses data teks sebagai urutan token, bukan sebagai fitur TF-IDF seperti pada dua model sebelumnya.

## 7. Evaluasi Model
Evaluasi dilakukan untuk mengukur kemampuan masing-masing model dalam mengklasifikasikan sentimen pada data yang tidak digunakan untuk pelatihan.

Metrik evaluasi yang dapat digunakan meliputi:
- **Accuracy:** proporsi prediksi yang benar dari seluruh data uji.
- **Precision:** ketepatan prediksi untuk suatu kelas sentimen.
- **Recall:** proporsi data suatu kelas yang berhasil dikenali model.
- **F1-score:** rata-rata harmonis antara precision dan recall.
- **Confusion matrix:** visualisasi perbandingan label aktual dengan label hasil prediksi.

Untuk klasifikasi tiga kelas, evaluasi sebaiknya tidak hanya berfokus pada accuracy. Precision, recall, dan F1-score untuk masing-masing kelas juga perlu diperhatikan, terutama jika distribusi data tidak seimbang.
Hasil evaluasi setiap model dicatat berdasarkan eksperimen yang benar-benar dijalankan. Model terbaik dipilih dengan mempertimbangkan performa pada data uji, keseimbangan antar-kelas, waktu pelatihan, dan kebutuhan implementasi.

## 8. Perbandingan Model

| Model | Representasi teks | Pembagian dataset |
|---|---|---|
| SVM | TF-IDF | 80% training, 20% testing |
| Random Forest | TF-IDF | 70% training, 30% testing |
| BiGRU | Tokenisasi dan padding | 70% training, 15% validation, 15% testing |

Perbandingan ini menggunakan konfigurasi eksperimen yang berbeda. Oleh karena itu, perbedaan skor tidak hanya dipengaruhi oleh algoritma, tetapi juga dapat dipengaruhi oleh pembagian dataset, preprocessing, representasi teks, dan konfigurasi pelatihan.
Untuk perbandingan yang lebih adil, evaluasi tambahan dapat dilakukan menggunakan data uji yang sama dan prosedur preprocessing yang konsisten, jika sesuai dengan rancangan eksperimen.

## 9. Instalasi dan Konfigurasi Environment
Proyek menggunakan Python dan sejumlah library eksternal. Untuk memastikan lingkungan eksekusi memiliki dependensi yang diperlukan, tersedia file `requirements.txt`.
File tersebut mencatat nama paket Python beserta versi yang digunakan atau batasan versinya. Instalasi dependensi perlu dilakukan sebelum menjalankan scraping dan notebook pelatihan.

### Persyaratan awal
- Python dengan versi yang kompatibel dengan seluruh library proyek.
- pip untuk instalasi paket Python.
- Jupyter Notebook atau JupyterLab, maupun VS Code dengan ekstensi Python dan Jupyter.
- Koneksi internet untuk instalasi dependensi dan proses scraping.

### Langkah instalasi

**1. Buka terminal pada direktori utama proyek.**
Pastikan direktori aktif berisi file `requirements.txt`, kode scraping, dan notebook pelatihan.

**2. Buat virtual environment.**
```bash
python -m venv .venv
```

**3. Aktifkan virtual environment.**
Windows Command Prompt:
```bat
.venv\Scripts\activate
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Linux atau macOS:
```bash
source .venv/bin/activate
```

**4. Perbarui pip.**
```bash
python -m pip install --upgrade pip
```

**5. Instal seluruh dependensi.**
```bash
pip install -r requirements.txt
```

Perintah tersebut menginstal paket yang tercantum di dalam `requirements.txt`. Jika instalasi berhasil, lingkungan siap digunakan sesuai dependensi dan konfigurasi proyek.

**6. Jalankan scraping dan notebook.**
Jalankan kode scraping untuk mengumpulkan ulasan jika dataset belum tersedia. Setelah dataset berhasil disimpan, buka notebook pelatihan, pastikan kernel menggunakan virtual environment yang sama, lalu jalankan sel notebook secara berurutan.

### Catatan penggunaan Google Colab
Jika proyek dijalankan di Google Colab, dependensi perlu diinstal pada runtime Colab. File `requirements.txt` dapat diunggah terlebih dahulu, kemudian dijalankan menggunakan:
```python
%pip install -r requirements.txt
```

Jika file berada di Google Drive, pastikan Drive telah di-mount dan path file ditentukan dengan benar.

## 10. Struktur Direktori Proyek
Struktur direktori yang disarankan adalah sebagai berikut:

```text
analisis-sentimen/
├── scraping.py
├── analisis_sentimen.ipynb
├── requirements.txt
├── ulasan.csv
└── README.md
```

Keterangan:
- `scraping.py`: kode untuk mengambil ulasan dari Google Play Store.
- `analisis_sentimen.ipynb`: notebook yang berisi EDA, preprocessing, pelatihan, dan evaluasi model.
- `requirements.txt`: daftar dependensi Python yang diperlukan.
- `ulasan.csv`: dataset hasil scraping.
- `README.md`: dokumentasi proyek.

Jika menggunakan beberapa notebook atau file dataset tambahan, struktur tersebut dapat disesuaikan dengan berkas yang dikumpulkan.

## 11. Reproduksibilitas Eksperimen
Untuk memudahkan orang lain menjalankan proyek, beberapa hal perlu diperhatikan:
1. Gunakan versi Python yang kompatibel dengan dependensi proyek.
2. Catat dependensi dan versi paket dalam `requirements.txt`.
3. Pastikan path dataset tidak hanya mengarah ke direktori lokal milik pengembang.
4. Simpan parameter penting, seperti random seed, rasio pembagian data, konfigurasi model, dan jumlah epoch.
5. Jalankan notebook dari awal hingga akhir untuk memastikan seluruh tahapan dapat direproduksi.
6. Simpan hasil evaluasi berdasarkan eksperimen yang benar-benar dilakukan.

File `requirements.txt` membantu menyamakan dependensi, tetapi tidak selalu menjamin lingkungan identik pada semua sistem operasi atau perangkat keras.

## 12. Kesimpulan
Proyek ini menerapkan tiga pendekatan klasifikasi sentimen terhadap ulasan Google Play Store, yaitu SVM dengan TF-IDF, Random Forest dengan TF-IDF, dan BiGRU dengan tokenisasi serta padding.
Proses pengembangan mencakup pengumpulan data, EDA, data preprocessing, text cleaning, pelabelan sentimen, pelatihan model, dan evaluasi performa.
Dengan menyediakan kode scraping, dataset, notebook pelatihan, serta `requirements.txt`, proyek dapat digunakan sebagai dokumentasi proses pengembangan dan sebagai dasar untuk eksperimen lanjutan maupun implementasi model analisis sentimen pada aplikasi lain.
