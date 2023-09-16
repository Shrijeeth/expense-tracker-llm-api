from fastapi import APIRouter
from src.endpoints import index


router = APIRouter()
router.include_router(index.router)