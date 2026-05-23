# Submission 2: Wine Quality MLOps System

Nama: Aqil Aziz

Username Dicoding: aqilaziz

## Ringkasan Proyek

| | Deskripsi |
| --- | --- |
| Dataset | Wine Quality Data Set - Red Wine dari UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/186/wine+quality |
| Masalah | Menentukan apakah red wine memiliki kualitas baik berdasarkan atribut kimia seperti acidity, sugar, sulfur dioxide, pH, sulphates, dan alcohol. |
| Solusi machine learning | Model klasifikasi biner untuk memprediksi `good_quality`. Label bernilai 1 jika skor kualitas asli minimal 6, dan 0 jika skor di bawah 6. |
| Target | Menghasilkan pipeline TFX end-to-end, model serving, dan monitoring sederhana agar sistem dapat dioperasikan sebagai layanan cloud. |
| Metode pengolahan data | Dataset CSV dinormalisasi menjadi fitur numerik snake_case. Komponen TFX Transform menstandarkan setiap fitur numerik dengan z-score. |
| Arsitektur model | Keras feed-forward neural network: input numerik terstandarisasi, Dense 32 ReLU, Dropout 0.2, Dense 16 ReLU, Dense 1 sigmoid. |
| Metrik evaluasi | Binary Accuracy dan AUC. Evaluator memberi blessing jika binary accuracy minimal 0.65 dan AUC minimal 0.70. |
| Performa model | Evaluator menghasilkan blessing dengan binary accuracy 0.778, AUC 0.853, dan loss 0.475 pada split evaluasi. |

## Machine Learning Pipeline

Pipeline dibuat dengan TensorFlow Extended (TFX) dan dijalankan memakai Apache Beam melalui `InteractiveContext`. Seluruh artifact tersimpan pada direktori `aqilaziz-pipeline`.

Komponen pipeline:

1. CsvExampleGen
2. StatisticsGen
3. SchemaGen
4. ExampleValidator
5. Transform
6. Trainer
7. Resolver
8. Evaluator
9. Pusher

Model yang telah lulus validasi tersedia pada `aqilaziz-pipeline/serving_model`.

## Deployment

Direktori ini menyertakan `Dockerfile` untuk menjalankan model dengan TensorFlow Serving. Konfigurasi ini dapat digunakan pada Railway, Heroku container stack, Cloud Run, atau platform cloud lain yang mendukung Docker.

Demo cloud untuk mengakses sistem tersedia di:

https://aqilaziz.github.io/dicoding-mlops-aqilaziz-final/

Docker serving command:

```bash
docker build -t aqilaziz-wine-serving .
docker run --rm -p 8501:8501 aqilaziz-wine-serving
```

Endpoint TensorFlow Serving:

```text
POST /v1/models/wine-quality:predict
GET  /monitoring/prometheus/metrics
```

## Monitoring

Prometheus dikonfigurasi pada direktori `monitoring`. Konfigurasi memuat target TensorFlow Serving dan endpoint metrik cloud demo.

```bash
docker build -t aqilaziz-wine-monitoring monitoring
docker run --rm -p 9090:9090 aqilaziz-wine-monitoring
```

Endpoint metrik demo:

https://aqilaziz.github.io/dicoding-mlops-aqilaziz-final/metrics.txt

## Struktur Proyek

```text
.
|-- aqilaziz-pipeline/          # Output artifact TFX dan serving_model
|-- data/
|   |-- winequality-red.csv
|-- data_source/
|   |-- winequality-red-raw.csv
|-- docs/                       # Demo cloud GitHub Pages
|-- modules/
|   |-- wine_transform.py
|   |-- wine_trainer.py
|-- monitoring/
|   |-- Dockerfile
|   |-- prometheus.yml
|   |-- README.md
|-- scripts/
|   |-- create_notebook.py
|   |-- prepare_data.py
|-- serving/
|   |-- prometheus.config
|-- aqilaziz-pipeline.ipynb
|-- aqilaziz-testing.ipynb
|-- Dockerfile
|-- README.md
|-- requirements.txt
|-- sample_request.json
|-- railway.toml
```

## Cara Menjalankan Pipeline

Gunakan Python 3.10 di Linux/WSL atau Google Colab.

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
jupyter nbconvert --to notebook --execute aqilaziz-pipeline.ipynb --output aqilaziz-pipeline.ipynb
```

## Catatan

Railway pada akun yang tersedia menunjukkan status trial expired, sehingga deployment Docker disiapkan sebagai konfigurasi cloud-ready. Demo cloud GitHub Pages digunakan sebagai web app publik untuk mengakses model demo dan endpoint monitoring statis.
