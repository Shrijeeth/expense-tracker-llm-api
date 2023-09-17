from fastapi import APIRouter
from src.endpoints import index, predict


router = APIRouter()
router.include_router(index.router)
router.include_router(predict.router)