import os
import io
import time
import PIL.Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from google import genai
from google.genai.errors import ClientError

app = FastAPI(title="PDF/Image Text Extraction API")

# Global exception handler to always return JSON responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Retrieve the API key from an environment variable.
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

# Initialize the GenAI client.
client = genai.Client(api_key=API_KEY)

def extract_text_from_image(img):
    """
    Extracts text from a PIL image using the Google GenAI API.
    Includes error handling for RESOURCE_EXHAUSTED errors.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    "Extract the text from the image. Do not write anything except the extracted content",
                    img,
                ]
            )
            return response.text
        except ClientError as e:
            # Extract error code from the exception arguments
            error_code = e.args[0] if e.args and isinstance(e.args[0], int) else None
            if error_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff before retrying
                    continue
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="API resource exhausted. Please try again later."
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing image: {str(e)}"
                )

@app.post("/upload", summary="Upload a PDF or image file", response_description="Returns extracted text as JSON")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Read file content.
    file_contents = await file.read()
    output_text = ""
    
    if file.filename.lower().endswith(".pdf"):
        try:
            # Convert PDF bytes to images.
            images = convert_from_bytes(file_contents, dpi=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting PDF: {str(e)}")
        
        # Process each page.
        for idx, img in enumerate(images, start=1):
            page_text = extract_text_from_image(img)
            output_text += f"### Page {idx}\n\n{page_text}\n\n"
    else:
        try:
            # Process the file as an image.
            img = PIL.Image.open(io.BytesIO(file_contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
        
        output_text += extract_text_from_image(img) + "\n\n"
    
    # Return the extracted text in a JSON response.
    return JSONResponse(content={"extracted_text": output_text})

@app.get("/", summary="Health Check")
async def root():
    return JSONResponse(content={"message": "API is up and running."})
