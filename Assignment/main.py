from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your model and transformer
with open("model.pkl", "rb") as f_model, open("transformer.pkl", "rb") as f_transformer:
    model = pickle.load(f_model)
    transformer = pickle.load(f_transformer)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(request: Request, review: str = Form(...)):
    processed_review = preprocess_review(review)
    prediction_result = model.predict(processed_review.reshape(1, -1))[0]
    return templates.TemplateResponse("result.html", {"request": request, "prediction_result": prediction_result})


def preprocess_review(review):
    processed_review = transformer.transform([review])
    return processed_review.toarray()
