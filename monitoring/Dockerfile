FROM prom/prometheus:v2.52.0

COPY prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 9090

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --web.listen-address=0.0.0.0:${PORT:-9090} --storage.tsdb.path=/prometheus"]
