from fastapi import FastAPI
from api.schemas import CustomerData
from src.predict import predict
app=FastAPI(
    title='Credit Default Prediction API',
    description='Predicts credit card default using XGBoost.',
    version='1.0.0'
)
@app.get('/')
def home():
    return {'message':'Credit default API is running.'}
@app.post('/predict')
def predict_default(customer:CustomerData):
    data=customer.model_dump()
    probability,prediction=predict(
        __import__('pandas').DataFrame([data])
    )
    return {
        'default_probability':float(probability[0]),
        'prediction':int(prediction[0])
    }