import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base, SessionLocal
from app.routers import datasets, models, deployments, monitoring, logs
from app.services.deployment_manager import deployment_manager

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Ship It ML — AutoMLOps Engine",
    description="Automated MLOps lifecycle REST API backend with models training, registry, drift detection, and auto-retraining.",
    version="1.0.0"
)

# CORS Middleware
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(deployments.router)
app.include_router(monitoring.router)
app.include_router(logs.router)

@app.on_event("startup")
def startup_event():
    # 1. Create database tables if they do not exist
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    
    # 2. Reload the active deployed model on startup
    db = SessionLocal()
    try:
        deployment_manager.load_active_deployment_on_startup(db)
    finally:
        db.close()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Ship It ML backend service is running."
    }
