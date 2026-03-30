#!/bin/bash
# Inicia o visualizador interativo dos ciclos de Otto.
# Abre automaticamente no navegador em  http://localhost:8501

cd "$(dirname "$0")"
exec .venv/bin/streamlit run app_ciclos.py \
    --server.headless false \
    --server.port 8501 \
    --theme.base dark
