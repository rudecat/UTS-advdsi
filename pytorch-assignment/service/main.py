from fastapi import FastAPI
from typing import List, Optional
from starlette.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pydantic import BaseModel
from io import StringIO
import sys

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 128)
        # self.layer_2 = nn.Linear(128, 32)
        # self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(128, 104)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        # x = F.dropout(F.relu(self.layer_2(x)), training=self.training)
        # x = F.dropout(F.relu(self.layer_3(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

class Review(BaseModel):
    brewery_name: str
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float

app = FastAPI()

model = PytorchMultiClass(108)
model.load_state_dict(torch.load('/models/torch25.model.state_dict',map_location='cpu'))
model.eval()
df_cat = joblib.load('/models/df_cat')
sc = joblib.load('/models/standardScaler')
ore = joblib.load('/models/OrdinalEncoderTarget')
ore_brewer = joblib.load('/models/OrdinalEncoderBrewer')
ohe = joblib.load('/models/OneHotEncoderBeerType')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer Style Prediction Service is running!'

@app.get('/model/architecture/', status_code=200)
def getArchitecture():
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    print(model)
    sys.stdout = old_stdout
    result_string = result.getvalue()
    result_string = result_string.replace('\n', "\n")
    return result_string

@app.post("/beer/type")
async def pred_beer(review: Review):
    temp_dataset = preprocessing([review])
    outputs = predict(temp_dataset, model=model, batch_size=10, device=torch.device('cpu'))
    print(ore.inverse_transform(ohe.inverse_transform(outputs)))
    return JSONResponse( ore.inverse_transform(ohe.inverse_transform(outputs)).tolist() )

@app.post("/beers/type")
async def pred_beers(reviews: List[Review]):
    temp_dataset = preprocessing(reviews)
    outputs = predict(temp_dataset, model=model, batch_size=10, device=torch.device('cpu'))
    print(outputs)
    print(ore.inverse_transform(ohe.inverse_transform(outputs)))
    return JSONResponse( ore.inverse_transform(ohe.inverse_transform(outputs)).tolist() )

def preprocessing(reviews: List[Review]):
    df_temp = pd.DataFrame(columns=["brewery_name","review_aroma","review_appearance","review_palate","review_taste"])
    for review in reviews:
        df_temp = df_temp.append({
            "brewery_name": review.brewery_name, 
            "review_aroma": review.review_aroma,
            "review_appearance": review.review_appearance,
            "review_palate": review.review_palate,
            "review_taste": review.review_taste
        },ignore_index=True)
    # df_cat = joblib.load('/models/df_cat')
    df_temp = df_temp.join(df_cat, on='brewery_name')
    df_temp = df_temp.drop(columns=['brewery_name'])

    print(df_temp)

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

    print(outputs)
    return outputs

class PytorchDataset(Dataset):
        
    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)
    
    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]
        
    def __len__ (self):
        return len(self.X_tensor)
    
    def to_tensor(self, data):
        return torch.Tensor(np.array(data))
