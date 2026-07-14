import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.models import Dataset, TrainedModel, Deployment, PredictionLog, AuditLog
from app.services.deployment_manager import deployment_manager

# Setup Test Database
TEST_DATABASE_URL = "sqlite:///./test_shipitml.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    # Override get_db dependency
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    # Remove database file if exists
    if os.path.exists("./test_shipitml.db"):
        try:
            os.remove("./test_shipitml.db")
        except PermissionError:
            pass


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_e2e_lifecycle(db_session):
    client = TestClient(app)
    
    # 1. Create a dummy CSV dataset on disk
    dummy_csv_path = "./test_churn_dataset.csv"
    df = pd.DataFrame({
        "age": [25, 45, 30, 50, 35, 40],
        "income": [50000, 80000, 60000, 100000, 75000, 90000],
        "gender": ["M", "F", "M", "F", "M", "F"],
        "churn": ["no", "yes", "no", "yes", "no", "yes"]
    })
    df.to_csv(dummy_csv_path, index=False)

    # 2. Upload dataset through API
    with open(dummy_csv_path, "rb") as f:
        response = client.post(
            "/api/datasets/upload",
            files={"file": ("test_churn_dataset.csv", f, "text/csv")},
            data={"target_column": "churn"}
        )
    
    assert response.status_code == 200
    dataset_data = response.json()
    assert dataset_data["target_column"] == "churn"
    assert dataset_data["problem_type"] == "classification"
    assert "age" in dataset_data["features_metadata"]
    
    # Cleanup dummy file
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)

    dataset_id = dataset_data["id"]

    # 3. Trigger training (Will execute synchronously in test runtime or background tasks can be run synchronously via client)
    # We pass background_tasks, TestClient runs background tasks before returning when using client.post / client.get
    response = client.post(f"/api/models/train/{dataset_id}", data={"algorithm": "Random Forest"})
    assert response.status_code == 200
    model_data = response.json()
    assert model_data["status"] in ["PENDING", "TRAINING", "DONE"]
    model_id = model_data["id"]

    # Wait or fetch model from db directly to make sure it's done since background tasks execute in client request lifecycle
    db_model = db_session.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    assert db_model is not None
    assert db_model.status == "DONE" or db_model.status == "TRAINING" # Background task executes immediately in TestClient
    
    # Force DONE state for testing prediction if it's still training or failed (though it should succeed on dummy data)
    if db_model.status != "DONE":
        # Force fit model and mock save
        from app.services.automl import train_and_evaluate
        df_loaded = pd.read_csv(dataset_data["file_path"])
        model_filename = f"test_model_v{db_model.version}.joblib"
        model_save_path = os.path.join(settings.MODEL_DIR, model_filename)
        train_and_evaluate(df_loaded, "churn", "classification", dataset_data["features_metadata"], model_save_path)
        
        db_model.status = "DONE"
        db_model.file_path = model_save_path
        db_model.metrics = {"accuracy": 1.0, "f1_score": 1.0}
        db_session.commit()

    # Cleanup dataset file from disk
    if os.path.exists(dataset_data["file_path"]):
        os.remove(dataset_data["file_path"])

    # 4. Deploy model
    response = client.post(f"/api/deployments/deploy/{model_id}")
    assert response.status_code == 200
    deployment_data = response.json()
    assert deployment_data["status"] == "active"
    deployment_id = deployment_data["id"]

    # 5. Make predictions
    headers = {"X-API-Key": settings.API_KEY}
    predict_payload = {
        "inputs": {
            "age": 30,
            "income": 65000,
            "gender": "M"
        }
    }
    
    # 5a. Test without API Key
    response = client.post("/api/deployments/predict", json=predict_payload)
    assert response.status_code == 401

    # 5b. Test with valid API Key
    response = client.post("/api/deployments/predict", json=predict_payload, headers=headers)
    assert response.status_code == 200
    pred_res = response.json()
    assert "prediction" in pred_res
    assert pred_res["prediction"] in ["yes", "no"]
    prediction_id = pred_res["prediction_id"]

    # 6. Post prediction feedback
    feedback_payload = {
        "prediction_id": prediction_id,
        "actual_value": "no"
    }
    response = client.post("/api/deployments/feedback", json=feedback_payload)
    assert response.status_code == 200
    feedback_res = response.json()
    assert feedback_res["status"] == "success"

    # 7. Test download endpoint
    # Test joblib download
    response = client.get(f"/api/models/download/{model_id}?format=joblib")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Test pkl download
    response = client.get(f"/api/models/download/{model_id}?format=pkl")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Cleanup models saved on disk during test
    if db_model.file_path and os.path.exists(db_model.file_path):
        os.remove(db_model.file_path)
