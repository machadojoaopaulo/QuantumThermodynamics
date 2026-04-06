#!/usr/bin/env bash
# Inicia o QuantumThermo Dashboard
# Duplo-clique ou: ./dashboard.sh

cd "$(dirname "$0")"

# Abre o navegador automaticamente após 2s
(sleep 2 && xdg-open http://127.0.0.1:8050) &

.venv/bin/python quantum_dashboard.py
