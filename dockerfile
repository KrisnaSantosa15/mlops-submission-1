FROM tensorflow/serving:latest
COPY ./serving_model_dir /models/obesity_model
ENV MODEL_NAME=obesity_model