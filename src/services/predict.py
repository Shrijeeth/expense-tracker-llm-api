import pandas as pd
import json

from src.utils import ModelSingleton

async def get_expense(user_prompt: str):
    model_singleton = ModelSingleton()
    model = await model_singleton.load_model()
    data = pd.DataFrame([{
        'instruction': user_prompt
    }])
    prediction = model.predict(dataset=data)[0]
    response = json.loads(prediction['json_answer_response'][0][0])
    return response