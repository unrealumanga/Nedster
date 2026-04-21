#!/bin/bash
echo "Starting Nedster Web Dashboard on http://127.0.0.1:8008"
cd dashboard
uvicorn main:app --port 8008
