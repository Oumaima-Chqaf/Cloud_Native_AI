from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI!"}

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, data: dict):
    # Get input data from the request body
    category = data.get("category")
    sellable_online = data.get("sellable_online")
    other_colors = data.get("other_colors")
    depth = data.get("depth")
    height = data.get("height")
    width = data.get("width")

    # Perform prediction using the model
    # Replace this with your actual prediction logic based on the input features
    prediction_result = model.predict([[category,sellable_online, other_colors, depth, height, width]])[0]

    # Render the HTML template with the prediction result
    return templates.TemplateResponse("result_template.html", {"request": request, "prediction_result": prediction_result})
