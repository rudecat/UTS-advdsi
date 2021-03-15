import joblib
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI
from io import StringIO
from pydantic import BaseModel
from pytorch.py import PytorchMultiClass, PytorchDataset
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
from starlette.responses import JSONResponse

class Review(BaseModel):
    brewery_name: str
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float

app = FastAPI()

# Load Trained PyTorch model
model = PytorchMultiClass(108)
model.load_state_dict(torch.load('/models/torch25.model.state_dict',map_location='cpu'))
model.eval()

# Load all preprocessing items
df_cat = joblib.load('/models/df_cat')
sc = joblib.load('/models/standardScaler')
ore = joblib.load('/models/OrdinalEncoderTarget')
ohe = joblib.load('/models/OneHotEncoderBeerType')

@app.get("/")
def read_root():
    return {
        "Project Objective": "This project is to predict the type of beer based on users review on appearance, aroma, taste and brewery.",
        "Available Services":[
            {   "service-name": "Project Description",
                "link": "https://agile-earth-24058.herokuapp.com/",
                "Action": "GET"
            },
            {   "service-name": "Swagger Doc",
                "link": "https://agile-earth-24058.herokuapp.com/docs",
                "Action": "GET"
            },
            {   "service-name": "Health Check",
                "link": "https://agile-earth-24058.herokuapp.com/health",
                "Action": "GET"
            },
            {   "service-name": "Model Architecture",
                "link": "https://agile-earth-24058.herokuapp.com/model/architecture",
                "Action": "GET"
            },
            {   "service-name": "Beer Type Prediction",
                "link": "https://agile-earth-24058.herokuapp.com/beer/type/",
                "Action": "POST"
            },
            {   "service-name": "(List of) Beers Type Prediction",
                "link": "https://agile-earth-24058.herokuapp.com/beers/type/",
                "Action": "POST"
            }
        ]
    }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer Style Prediction Service is running!'

@app.get('/model/architecture/', status_code=200)
def getArchitecture():
    #Store Stdout into a variable for response payload
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    print(model)
    sys.stdout = old_stdout
    result_string = result.getvalue()
    return result_string

@app.post("/beer/type")
async def pred_beer(review: Review):
    temp_dataset = preprocessing([review])
    outputs = predict(temp_dataset, model=model, batch_size=10, device=torch.device('cpu'))
    return JSONResponse( ore.inverse_transform(ohe.inverse_transform(outputs)).tolist() )

@app.post("/beers/type")
async def pred_beers(reviews: List[Review]):
    temp_dataset = preprocessing(reviews)
    outputs = predict(temp_dataset, model=model, batch_size=10, device=torch.device('cpu'))
    return JSONResponse( ore.inverse_transform(ohe.inverse_transform(outputs)).tolist() )

def preprocessing(reviews: List[Review]):
    # Construct the data into correct format for prediction
    df_temp = pd.DataFrame(columns=["brewery_name","review_aroma","review_appearance","review_palate","review_taste"])
    for review in reviews:
        df_temp = df_temp.append({
            "brewery_name": review.brewery_name, 
            "review_aroma": review.review_aroma,
            "review_appearance": review.review_appearance,
            "review_palate": review.review_palate,
            "review_taste": review.review_taste
        },ignore_index=True)
    # Preprocessing
    df_temp = df_temp.join(df_cat, on='brewery_name')
    df_temp = df_temp.drop(columns=['brewery_name'])

    y = pd.DataFrame(np.zeros((len(reviews),1)))
    return PytorchDataset(X=df_temp, y=y)

def predict(test_data, model, batch_size, device, generate_batch=None):
    
    # Set model to evaluation mode
    model.eval()
    outputs = []
    
    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        # Set no update to gradients
        with torch.no_grad():
            # Make predictions
            output = model(feature)
            outputs = output.detach().cpu().numpy()

    return outputs
