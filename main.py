from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.api import router as api_router
from dotenv import load_dotenv
import modal

from src.utils import download_llama_2, load_model

load_dotenv()
app = FastAPI()

image = modal.Image.micromamba().micromamba_install(
    "cudatoolkit=11.7",
    "cudnn=8.1.0",
    "cuda-nvcc",
    "scipy",
    channels=["conda-forge", "nvidia"],
    ).pip_install_from_requirements(
        requirements_txt=r"./requirements.txt"
    ).run_function(download_llama_2, secret=modal.Secret.from_name("expense-tracker-token"), mounts=[modal.Mount.from_local_dir("./src/models", remote_path="/root/src/models")])
stub = modal.Stub("expense-tracker", image=image)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event(event_type="startup")
async def startup_event():
    await load_model("Fduv/Expense-Tracker-Llama-V2-Instruction_Fine_Tuned")

app.include_router(api_router)

@stub.function(
        gpu="any",
        secret=modal.Secret.from_name("expense-tracker-token"), 
        mounts=[modal.Mount.from_local_dir("./src/models", remote_path="/root/src/models")]
)
@modal.asgi_app(label='expense-tracker-llm-app')
def fastapi_app():
    return app
