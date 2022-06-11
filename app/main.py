from typing import Optional
from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter

import os
from random import randint

from fastapi.responses import FileResponse
import uuid

from app.inference import get_category, plot_category, show_heatmap
from datetime import datetime


# -----------------------------------------------------------------------------------------------------------------
# FAST API APPLICATION

app = FastAPI()

# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Instantiate Jinja2Templates to be able to render HTML files
templates = Jinja2Templates(directory="app/templates")


# -----------------------------------------------------------------------------------------------------------------
 # GET AND POST METHODS

# GET method to render index.html
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    #print('GET METHOD IS WORKING JOSE DANIEL')
    return templates.TemplateResponse("index.html", {"request": request})

# POST method to retreive image from the form located in index.html and render it to result.html.
@app.post("/result")
async def create_file(request: Request, file: UploadFile = File(...), ):
    
    #file.filename = f"{uuid.uuid4()}"
    contents = await file.read()  # <-- Important!

    IMAGEDIR = "app/static/test_images/"
    input_image_path = f"{IMAGEDIR}{file.filename}"  # Path for input image file
     
     # example of how you can save the file (input image file)
    with open(input_image_path, "wb") as f:
        f.write(contents)
    
    # -----------------------------------------------------------------------------------
    # IGNORE. 

    # This is code to select random images from the image directory
    #files = os.listdir(IMAGEDIR)
    #random_index = randint(0, len(files) - 1)
    #path = f"{IMAGEDIR}{files[random_index]}"
    #response = FileResponse(path)              # FileResponse expects a path and It will render the image.
    # -----------------------------------------------------------------------------------
     
     # Make PREDICTIONS and get the HEATMAP

    # Get predicted label for input image using tflite model
    category = get_category(img=input_image_path) 
    # Get predicted label for input image using gram_cam model
    heatmap_img = show_heatmap(img=input_image_path)

    # Save the heatmap image in the heatmap image path
    heatmap_img_path = f"{IMAGEDIR}heatmap_{file.filename}"
    heatmap_img = heatmap_img.save(heatmap_img_path)
    
    # These parameters are rendered in the result.html file
    original_image = file.filename
    heatmap_img_name = f"heatmap_{file.filename}" 
 
    return templates.TemplateResponse("result.html", {"request": request, 
                                                      "category":category, 
                                                      "original_image":original_image,
                                                      "heatmap_image":heatmap_img_name,})


# -----------------------------------------------------------------------------------------------------------------
# SIMPLE INTRO FASTAPI

# @app.get("/items")
# async def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}
# -----------------------------------------------------------------------------------------------------------------
