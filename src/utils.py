from ludwig.api import LudwigModel

model = None

async def load_model():
    global model
    if model is None:
        model = LudwigModel.load('./src/models/expense_tracker_llm')
    return model

def download_llama_2():
    from huggingface_hub import snapshot_download

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    snapshot_download(model_name)