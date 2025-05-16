# Use an official Python runtime as a parent image.
FROM python:3.10-slim

# Install system dependencies (poppler-utils is required by pdf2image).
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy requirements and install Python dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code.
COPY main.py .

# Expose the port FastAPI will run on.
EXPOSE 7860

# Command to run the FastAPI app using uvicorn via the module interface.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
