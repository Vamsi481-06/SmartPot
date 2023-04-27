from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from API_Loader import load_keras_model # Import the function directly
import numpy as np
from PIL import Image

app = FastAPI()
# Load your trained model
model = load_keras_model("models/testmodel.h5")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Preprocess the input data and make predictions using the model
    predictions = await make_prediction(model, image)

    # Return the predictions as JSON
    return JSONResponse(content=predictions)

async def make_prediction(model, image: UploadFile):
    # Load the image file
    image_data = Image.open(image.file)

    # Preprocess the image (e.g., resize, convert to grayscale, normalize, etc.)
    # This example assumes a grayscale image with a fixed size
    processed_image = image_data.convert("RGB").resize((256, 256))

    # Convert the PIL image to a NumPy array and add an extra dimension for batch size
    input_array = np.array(processed_image)[np.newaxis, :, :]

    # Make predictions using the model
    predictions = model.predict(input_array)

    # Post-process the predictions (e.g., convert to a Python list or dictionary)
    result = predictions.tolist()

    return result

