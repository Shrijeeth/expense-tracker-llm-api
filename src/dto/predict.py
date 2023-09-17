from pydantic import BaseModel

class predict_dto(BaseModel):
    user_prompt: str