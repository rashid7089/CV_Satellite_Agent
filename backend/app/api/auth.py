from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import create_token, current_user, hash_password, verify_password
from app.database import get_db
from app.models import User
from app.schemas import AuthCredentials, TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["authentication"])


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if "@" not in value or value.startswith("@") or value.endswith("@"):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return value


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(credentials: AuthCredentials, db: Session = Depends(get_db)):
    email = normalize_email(credentials.email)
    user = User(email=email, password_hash=hash_password(credentials.password))
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    return TokenOut(access_token=create_token(user), user=user)


@router.post("/login", response_model=TokenOut)
def login(credentials: AuthCredentials, db: Session = Depends(get_db)):
    email = normalize_email(credentials.email)
    user = db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return TokenOut(access_token=create_token(user), user=user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user
