FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    fastapi==0.162.0 \
    uvicorn==0.54.0 \
    pydantic==2.14.3 \
    pandas==3.1.1 \
    numpy==2.5.4 \
    scipy==1.18.1 \
    scikit-learn==1.9.1 \
    skops==0.16.0

COPY docker_wheels/xgboost-3.4.1-py3-none-manylinux_2_28_x86_64.whl /tmp/

RUN pip install --no-cache-dir --no-deps /tmp/xgboost-3.4.1-py3-none-manylinux_2_28_x86_64.whl \
    && rm /tmp/xgboost-3.4.1-py3-none-manylinux_2_28_x86_64.whl
COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]