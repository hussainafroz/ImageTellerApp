from fastapi import FastAPI, UploadFile
import io  # io is used to handle binary data in memorey as if it were a file
from PIL import Image  # image preprocessing


from Huggingface import model_pipeline

# create an instance of the FastApi application
app = FastAPI()

# @ is a Python decorator , -> is used to apply a decorator to a function or method
# app.get("/") , -> it indicates that the function its decorating should handle HTTP GET requests to the root URL("/")


@app.get("/")
def read_root():
    return {"Hello": "world"}


@app.post("/ask")
def ask(text: str, image: UploadFile):
    content = image.file.read()

    # open is function from pillow , used to open image files
    # io.BytesIO(content) -> this creates an in-memory binary stream from the 'content'(i.e the binary data of the image)

    # by passing io.BytesIO(content) to Image.open() you create a binary data as if it were reading from a file.
    image = Image.open(io.BytesIO(content))

    # model_pipeline from huggingface.py
    result = model_pipeline(text, image)
    return result
