from ludwig.api import LudwigModel

class ModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    async def load_model(self):
        if self.model is None:
            self.model = await self._load_model()
        return self.model

    async def _load_model(self):
        model = LudwigModel.load('./src/models/expense_tracker_llm')
        return model

def download_llama_2():
    from huggingface_hub import snapshot_download

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    snapshot_download(model_name)