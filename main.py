from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.api import router as api_router
import modal

app = FastAPI()

stub = modal.Stub("expense-tracker")

image = modal.Image.debian_slim().pip_install_from_requirements(requirements_txt=r"./requirements.txt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)

@stub.function(image=image, gpu="any", mounts=[modal.Mount.from_local_dir("./src/models", remote_path="/root/src/models")])
@modal.asgi_app()
def fastapi_app():
    return app