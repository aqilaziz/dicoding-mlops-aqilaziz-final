# Monitoring

Direktori ini memuat konfigurasi Prometheus untuk memantau sistem machine learning.

Target utama:

- `wine-quality-tf-serving`: endpoint metrik TensorFlow Serving pada `/monitoring/prometheus/metrics`.

File penting:

- `prometheus.yml`: konfigurasi Prometheus untuk scrape endpoint TF Serving di Railway.
- `prometheus.config`: konfigurasi monitoring yang digunakan TensorFlow Serving.
- `aqilaziz-monitoring.png`: bukti Prometheus UI dengan target TF Serving berstatus UP.

Jalankan Prometheus:

```bash
docker build -t aqilaziz-wine-monitoring monitoring
docker run --rm -p 9090:9090 aqilaziz-wine-monitoring
```
