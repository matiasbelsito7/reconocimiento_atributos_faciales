"""App principal de FastAPI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from facial_attributes.api.dependencies import init_pipeline
from facial_attributes.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Gestionar lifecycle de la aplicación.

    Carga el pipeline de inferencia al iniciar y lo libera al apagar.
    """
    logger.info("Iniciando servicio de inferencia...")
    init_pipeline()
    logger.info("Servicio listo.")
    yield
    logger.info("Apagando servicio...")


app = FastAPI(
    title="Facial Attribute Recognition API",
    description="API para predicción de atributos faciales visualmente observables.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
