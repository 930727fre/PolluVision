from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from typing import Optional
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import rag
from sky_test_v1 import validate_image
from model_v3 import air_pridiction
import os

app = FastAPI()

@app.post("/api/upload")
async def upload_data(
    query: str = Form(...),  # Required field for the query text
    image_path: Optional[str] = Form(None)  # Optional field for the image path
):
    response_text = ""
    
    # Process query
    query = query.replace("'", "").replace('"', "")
    print(f"Received query: {query}")

    if image_path:
        # Validate the provided image path
        if not os.path.exists(image_path):
            return JSONResponse(content={"error": "Image file not found."}, status_code=400)
        
        # Read and process the image
        try:
            with open(image_path, "rb") as img_file:
                image_content = img_file.read()
            pil_image = Image.open(io.BytesIO(image_content))
            pil_image = pil_image.convert("RGB")
            pil_image = pil_image.resize((224, 224))  # Resize image for the model
            img_array = img_to_array(pil_image) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            print(f"Image processed successfully, shape: {img_array.shape}")

            # Validate the image using sky_test_v1
            if not validate_image(img_array):
                return {"response": "Invalid image input (not the sky)."}
            
            # Predict using model_v3
            cnn_result = air_pridiction(img_array)
            print(f"CNN Prediction Result: {cnn_result}")

            # Generate response using RAG
            response_text += rag.initialize(query, cnn_result)
        except Exception as e:
            return JSONResponse(content={"error": f"Error processing image: {str(e)}"}, status_code=500)
    else:
        # Generate response without an image
        response_text = rag.initialize(query, " ")

    return {"response": response_text}
