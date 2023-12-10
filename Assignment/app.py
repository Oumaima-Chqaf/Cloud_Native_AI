from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pickle
import numpy as np  # Import NumPy for array operations

app = FastAPI()

# Load your model
with open("model.pkl", "rb") as f_model, open("transformer.pkl", "rb") as f_transformer:
    model = pickle.load(f_model)
    transformer = pickle.load(f_transformer)


@app.get("/")
def home():
    return {"message": "Welcome to FastAPI for sentiment analysis!"}


class ReviewInput(BaseModel):
    review: str


@app.post("/predict/")
async def predict(review_input: ReviewInput):
    review = review_input.review
    # Perform prediction using the loaded model
    processed_review = preprocess_review(review)
    prediction_result = model.predict(processed_review.reshape(1, -1))[0]

    return JSONResponse(content={"sentiment_prediction": prediction_result})


def preprocess_review(review):
    # Apply any necessary preprocessing steps using the loaded transformer (vectorizer)
    processed_review = transformer.transform([review])
    return processed_review.toarray()
