# Monitoring

Direktori ini memuat konfigurasi Prometheus untuk memantau sistem machine learning.

Target utama:

- `wine-quality-tf-serving`: endpoint metrik TensorFlow Serving pada `/monitoring/prometheus/metrics`.
- `wine-quality-demo-cloud`: metrik demo cloud statis pada GitHub Pages.

Jalankan Prometheus:

```bash
docker build -t aqilaziz-wine-monitoring monitoring
docker run --rm -p 9090:9090 aqilaziz-wine-monitoring
```
