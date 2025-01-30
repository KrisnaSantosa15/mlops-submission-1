FROM tensorflow/serving:latest

COPY ./serving_model /models/obesity_model
ENV MODEL_NAME=obesity_model
EXPOSE 8501