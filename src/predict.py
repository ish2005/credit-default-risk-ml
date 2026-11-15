import pandas as pd
import skops.io as sio

from src.features import prepare_features

model = sio.load(
    "models/model.skops",
    trusted=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]
)

def predict(df):
    X=prepare_features(df)
    probability=model.predict_proba(X)[:,1]
    prediction=(probability>=0.60).astype(int)
    return probability,prediction
if __name__=='__main__':
    df = pd.read_csv(r"data/default of credit card clients.csv",header=1)
    X_raw=df.drop(columns=['default payment next month'])
    X=prepare_features(X_raw)
    probability,prediction=predict(X)
    print('Probabilities:')
    print(probability[:5])
    print('\nPredictions:')
    print(prediction[:5])