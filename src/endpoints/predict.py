from fastapi import APIRouter
from fastapi.responses import JSONResponse

import os

from src.dto.predict import predict_dto
from src.services.predict import get_expense

router = APIRouter(
    prefix='/predict',
    tags=["Predict"],
    responses={404: {"description": "Not Found"}}
)

@router.post("/")
async def predict(request: predict_dto):
    try:
        auth_token = os.getenv("AUTH_TOKEN")
        if (auth_token != request.token):
            return JSONResponse({
                "success": False,
                "error": "Unauthorized"
            })
        expense_json = await get_expense(request.user_prompt)
        return JSONResponse({
            "success": True,
            "data": expense_json
        })
    except Exception as error:
        return JSONResponse({
            "success": False,
            "error": str(error)
        })