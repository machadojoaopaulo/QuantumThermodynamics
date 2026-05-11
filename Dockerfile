FROM python:3.11-slim

WORKDIR /app

# Dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código da aplicação
COPY thermodynamics_functions.py .
COPY app.py .

# Hugging Face Spaces usa porta 7860 por padrão
EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:server"]
