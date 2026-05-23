FROM tensorflow/serving:2.15.1

ENV MODEL_NAME=wine-quality
ENV REST_PORT=8501

COPY aqilaziz-pipeline/serving_model /models/wine-quality
COPY serving/prometheus.config /models/monitoring/prometheus.config

EXPOSE 8501

CMD tensorflow_model_server \
    --port=8500 \
    --rest_api_port=${PORT:-8501} \
    --model_name=${MODEL_NAME} \
    --model_base_path=/models/${MODEL_NAME} \
    --monitoring_config_file=/models/monitoring/prometheus.config
