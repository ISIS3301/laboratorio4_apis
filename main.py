from typing import Optional
from fastapi import FastAPI
from DataModel import DataModel
from joblib import load, dump
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

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


@app.post("/retrain")
def make_retrain(dataModel: List[DataModel]):
    lista = []
    print(lista)
    for i in dataModel:
        lista.append(i.dict())
    df = pd.DataFrame(lista)
    df.columns = dataModel[0].columns()
    ret = retrain(df)
    return ret



def retrain(df):
    df.dropna()

    X_train, X_test, Y_train, Y_test = train_test_split(df, df["Admission Points"], test_size=0.2, random_state=1)
    ct = ColumnTransformer(
        [("gre_preprocess", MinMaxScaler(), ["GRE Score", "University Rating", "SOP", "CGPA", "Research"]),
        ("drop_columns", "drop", ['Serial No.', "TOEFL Score", "LOR"])])
    pipeline = Pipeline(
        [
            ('feature_selection', ct),
            ('model', LinearRegression())
        ]
    )
    pipeline = pipeline.fit(X_train, Y_train)
    preds_train = pipeline.predict(X_train)
    preds_test = pipeline.predict(X_test)

    RMSE_train = mean_squared_error(Y_train, preds_train)
    MAE_train = mean_absolute_error(Y_train, preds_train)
    r2_train = r2_score(Y_train, preds_train)

    RMSE_test = mean_squared_error(Y_test, preds_test)
    MAE_test = mean_absolute_error(Y_test, preds_test)
    r2_test = r2_score(Y_test, preds_test)
    filename = './assets/modelo.joblib'

    

    dump(pipeline, filename)
    ret = {"RMSE-train": RMSE_train, "MAE-train": MAE_train, "R^2-train": r2_train,
            "RMSE-test": RMSE_test, "MAE-test": MAE_test, "R^2-test": r2_test}
    json_object = json.dumps(ret, indent = 4) 
    return json_object