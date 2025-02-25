#!/bin/bash
streamlit run run.py --server.address 0.0.0.0 &
python -m uvicorn api:app --host 0.0.0.0 --port 8000
