import os
import torch
import json
import google.generativeai as palm

from src.utils import load_model, generate_text_palm    

async def get_prompt_for_llama2(user_prompt):
    template = f"Below is a input that describes an expense. Write a response in json format that appropriately completes the request.\nResponse is a json string with fields - account_type (CREDIT or DEBIT), category, sub_category, reason (detailed reason if provided with context), third_party - person who gave to got the money (Amount in Indian Rupees).\nGenerate appropriate response json string for the input expense. Response must be in only json string format strictly.\n\n### Input:\n{user_prompt}\n\n### Response:\n"
    return template
    
async def get_prompt_for_palm_api(user_prompt):
    template = f"""Below is a input that describes an expense. Write a response in json format that appropriately completes the request.\nResponse is a json string with fields - account_type (CREDIT or DEBIT), category, sub_category, reason (detailed reason if provided with context), third_party - person who gave to got the money (Amount in Indian Rupees).\nGenerate appropriate response json string for the input expense. Response (json_answer) must be in only json string format strictly.
input: {user_prompt}
output:"""
    return template

async def get_expense_from_palm_api(user_prompt: str):
    defaults = {
        'model': 'tunedModels/expensetrackermodelv2-510p26eit79f',
        'temperature': 0.4,
        'candidate_count': 1,
        'top_k': 50,
        'top_p': 1.0,
        'max_output_tokens': 256,
        'stop_sequences': [],
        'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
    }
    prompt = await get_prompt_for_palm_api(user_prompt=user_prompt)
    response = await generate_text_palm(
        **defaults,
        prompt=prompt
    )
    result = json.loads(str(response.result))
    output = {}
    for k,v in result.items():
        if k.lower() == "category" or k.lower() == "sub_category":
            output[k.lower()] = v.upper()
        else:
            output[k.lower()] = v
    return output

async def get_expense_from_model(user_prompt: str):
    prompt = await get_prompt_for_llama2(user_prompt=user_prompt)
    model, tokenizer = await load_model("Fduv/Expense-Tracker-Llama-V2-Instruction_Fine_Tuned")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], 
            max_new_tokens=256, temperature=0.1, top_k=50, top_p=1.0, typical_p=1.0,
            encoder_repetition_penalty=1.0, num_beams=1, repetition_penalty=1.0,
        )
        result = json.loads(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].split("Response:\n")[1])
    return result
    
async def get_expense(user_prompt: str):
    try:
        response = await get_expense_from_palm_api(user_prompt)
        if response == "" or not response:
            response = await get_expense_from_palm_api(user_prompt)
        return response
    except Exception as error:
        print(error)
        if os.getenv("IS_LOCAL_MODEL_REQUIRED") == 1:
            response = await get_expense_from_model(user_prompt)
            return response
        raise error