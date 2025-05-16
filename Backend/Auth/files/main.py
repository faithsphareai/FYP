# main.py

from fastapi import FastAPI
from auth import router as auth_router

app = FastAPI(
    title="User Authentication API",
    description="Handles user signup, login, profile update, and avatar management.",
    version="1.0.0"
)

# Include the auth router under the /auth prefix
app.include_router(auth_router)

# Optional root route to verify the app is running
@app.get("/")
async def root():
    return {"message": "Welcome to the User Auth API"}
