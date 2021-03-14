from fastapi import FastAPI
from starlette.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer Style Prediction Service is running!'

