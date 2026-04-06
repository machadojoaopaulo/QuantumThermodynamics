#!/usr/bin/env bash
# Pré-computa os mapas de regiões em alta resolução
# Executar uma vez antes do dashboard: ./precomputar.sh

cd "$(dirname "$0")"
.venv/bin/python precompute_regions.py
