from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix='/index',
    tags=["Index"],
    responses={404: {"description": "Not Found"}}
)


@router.get("/")
async def index():
    return JSONResponse(content={"message": "Expense Tracker LLM"})
