import pandas as pd

from typing import Optional
from fastapi import FastAPI
from DataModel import DataModel
from joblib import load
from typing import List

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def make_predictions(dataModel: List[DataModel]):
    lista = []
    for i in dataModel:
        lista.append(i.dict())
    df = pd.DataFrame(lista)
    df.columns = dataModel[0].columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    dic = {"resultado": result.tolist()}
    return dic

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
