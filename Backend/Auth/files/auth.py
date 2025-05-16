# auth.py
import os
import uuid
import logging
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient
import gridfs

from models import User, UserUpdate, Token, LoginResponse
from config import CONNECTION_STRING, SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS

load_dotenv()

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# Updated MongoDB initialization: now using CONNECTION_STRING from config.py
client = MongoClient(CONNECTION_STRING)
db = client.users_database
users_collection = db.users
# GridFS instance for storing avatars
fs = gridfs.GridFS(db, collection="avatars")

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
router = APIRouter(prefix="/auth", tags=["auth"])

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(email: str) -> Optional[dict]:
    return users_collection.find_one({"email": email})

def authenticate_user(email: str, password: str) -> Optional[dict]:
    user = get_user(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    algorithm = "HS256"
    return jwt.encode(to_encode, SECRET_KEY, algorithm=algorithm)

def create_access_token(email: str) -> str:
    return create_token({"sub": email}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

def create_refresh_token(email: str) -> str:
    return create_token({"sub": email}, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = get_user(email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def save_avatar_file_to_gridfs(file: UploadFile) -> str:
    allowed_types = ["image/jpeg", "image/png", "image/gif"]
    if file.content_type not in allowed_types:
        logger.error(f"Unsupported file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only JPEG, PNG, and GIF are accepted."
        )
    try:
        contents = await file.read()
        file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)
        logger.info(f"Avatar stored in GridFS with file_id: {file_id}")
        return str(file_id)
    except Exception as e:
        logger.exception("Failed to store avatar in GridFS")
        raise HTTPException(status_code=500, detail="Could not store avatar file in MongoDB.")

@router.post("/signup", response_model=Token)
async def signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    avatar: Optional[UploadFile] = File(None)
):
    try:
        _ = User(name=name, email=email, password=password)
    except Exception as e:
        logger.error(f"Validation error during signup: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    if get_user(email):
        logger.warning(f"Attempt to register already existing email: {email}")
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(password)
    user_data = {
        "name": name,
        "email": email,
        "hashed_password": hashed_password,
        "chat_histories": []
    }
    if avatar:
        file_id = await save_avatar_file_to_gridfs(avatar)
        user_data["avatar"] = file_id
    users_collection.insert_one(user_data)
    logger.info(f"New user registered: {email}")
    return {
        "access_token": create_access_token(email),
        "refresh_token": create_refresh_token(email),
        "token_type": "bearer"
    }

@router.post("/login", response_model=LoginResponse)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for: {form_data.username}")
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    logger.info(f"User logged in: {user['email']}")
    avatar_url = None
    if "avatar" in user and user["avatar"]:
        avatar_url = f"/auth/avatar/{user['avatar']}"
    return {
        "access_token": create_access_token(user["email"]),
        "refresh_token": create_refresh_token(user["email"]),
        "token_type": "bearer",
        "name": user["name"],
        "avatar": avatar_url
    }

@router.get("/user/data")
async def get_user_data(request: Request, current_user: dict = Depends(get_current_user)):
    avatar_url = None
    if "avatar" in current_user and current_user["avatar"]:
        avatar_url = f"/auth/avatar/{current_user['avatar']}"
    return {
        "name": current_user["name"],
        "email": current_user["email"],
        "avatar": avatar_url,
        "chat_histories": current_user.get("chat_histories", [])
    }

@router.put("/user/update")
async def update_user(
    request: Request,
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if email is not None:
        update_data["email"] = email
    if password is not None:
        try:
            _ = User(name=current_user["name"], email=current_user["email"], password=password)
        except Exception as e:
            logger.error(f"Password validation error during update: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        update_data["hashed_password"] = get_password_hash(password)
    if avatar:
        file_id = await save_avatar_file_to_gridfs(avatar)
        update_data["avatar"] = file_id
    if not update_data:
        logger.info("No update parameters provided")
        raise HTTPException(status_code=400, detail="No update parameters provided")
    users_collection.update_one({"email": current_user["email"]}, {"$set": update_data})
    logger.info(f"User updated: {current_user['email']}")
    return {"message": "User updated successfully"}

@router.post("/logout")
async def logout(request: Request, current_user: dict = Depends(get_current_user)):
    logger.info(f"User logged out: {current_user['email']}")
    return {"message": "User logged out successfully"}

from bson import ObjectId

@router.get("/avatar/{file_id}")
async def get_avatar(file_id: str):
    try:
        # Convert the file_id string to an ObjectId before fetching
        file = fs.get(ObjectId(file_id))
        return StreamingResponse(file, media_type=file.content_type)
    except Exception as e:
        logger.error(f"Avatar not found for file_id {file_id}: {e}")
        raise HTTPException(status_code=404, detail="Avatar not found")
