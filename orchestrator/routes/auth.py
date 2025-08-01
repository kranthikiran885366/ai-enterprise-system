"""Authentication routes."""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from shared_libs.auth import create_access_token, get_current_user
from services.auth_service import AuthService


router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    is_admin: bool = False


class UserResponse(BaseModel):
    username: str
    email: str
    is_active: bool
    is_admin: bool
    roles: list


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint."""
    # Get auth service from app state
    from main import app
    auth_service = app.state.auth_service
    
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=dict)
async def register(user_data: UserCreate):
    """Register a new user."""
    from main import app
    auth_service = app.state.auth_service
    
    success = await auth_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        is_admin=user_data.is_admin
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create user"
        )
    
    return {"message": "User created successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    from main import app
    auth_service = app.state.auth_service
    
    user = await auth_service.get_user_by_username(current_user["sub"])
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(**user)


auth_router = router
