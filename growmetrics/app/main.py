"""
FastAPI application entrypoint for the Growmetrics AI Chemistry Video Request Service.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Growmetrics AI Chemistry Video Request Service",
    description=(
        "Submit chemistry concept queries and receive AI-generated "
        "educational video explanations. Built with FastAPI, Manim, "
        "Edge-TTS, and a 4-stage quality gate reliability framework."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "growmetrics-chemistry-video-service"}
