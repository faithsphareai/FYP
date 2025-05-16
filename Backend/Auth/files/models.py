from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional

# === Authentication Models ===
class User(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str

    @validator("password")
    def validate_password(cls, value):
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not any(char.isdigit() for char in value):
            raise ValueError("Password must include at least one number.")
        if not any(char.isupper() for char in value):
            raise ValueError("Password must include at least one uppercase letter.")
        if not any(char.islower() for char in value):
            raise ValueError("Password must include at least one lowercase letter.")
        if not any(char in "!@#$%^&*()-_+=<>?/" for char in value):
            raise ValueError("Password must include at least one special character.")
        return value

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr]
    password: Optional[str]

    @validator("password")
    def validate_password(cls, value):
        if value is not None:
            if len(value) < 8:
                raise ValueError("Password must be at least 8 characters long.")
            if not any(char.isdigit() for char in value):
                raise ValueError("Password must include at least one number.")
            if not any(char.isupper() for char in value):
                raise ValueError("Password must include at least one uppercase letter.")
            if not any(char.islower() for char in value):
                raise ValueError("Password must include at least one lowercase letter.")
            if not any(char in "!@#$%^&*()-_+=<>?/" for char in value):
                raise ValueError("Password must include at least one special character.")
        return value

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class LoginResponse(Token):
    name: str
    avatar: Optional[str] = None

class TokenData(BaseModel):
    email: Optional[str] = None
