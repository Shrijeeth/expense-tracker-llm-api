import torch
import json

from src.utils import load_model

async def get_prompt(user_prompt):
    template = f"Below is a input that describes an expense. Write a response in json format that appropriately completes the request.\nResponse is a json string with fields - account_type (CREDIT or DEBIT), category, sub_category, reason (detailed reason if provided with context), third_party - person who gave to got the money (Amount in Indian Rupees).\nGenerate appropriate response json string for the input expense. Response must be in only json string format strictly.\n\n### Input:\n{user_prompt}\n\n### Response:\n"
    return template

async def get_expense(user_prompt: str):
    prompt = await get_prompt(user_prompt)
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
    
