# 🚀 Ship It ML — AutoMLOps Engine

> **An end-to-end automated MLOps platform** that handles the full machine learning lifecycle: dataset upload → AI-powered profiling → AutoML training → model registry → deployment → drift detection → auto-retraining.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?style=flat&logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red?style=flat)](https://sqlalchemy.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-blue?style=flat&logo=mlflow)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [AutoML Pipeline](#-automl-pipeline)
- [Drift Detection & Auto-Retraining](#-drift-detection--auto-retraining)
- [AI-Powered Analysis](#-ai-powered-analysis-gemini)
- [Running Tests](#-running-tests)
- [Screenshots](#-screenshots)

---

## 🌟 Overview

**Ship It ML** is a production-ready AutoMLOps platform built to automate the repetitive and complex parts of the ML lifecycle. Upload any CSV dataset, let the engine profile it, train and compare multiple ML models automatically, deploy the best one, and continuously monitor it for data drift — all through a clean REST API and a modern dashboard UI.

Whether you're a data scientist who wants fast experimentation or an MLOps engineer who needs a reliable retraining pipeline, Ship It ML handles it end-to-end.

---

## ✨ Features

### 🤖 AutoML Engine
- **Automatic dataset profiling** — detects target column, problem type (classification/regression), feature types, and data quality issues
- **Multi-model training** — simultaneously trains Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Hyperparameter tuning** with `RandomizedSearchCV` for optimal performance
- **Smart preprocessing** — automatic numeric coercion (handles `₹599`, `6,531` formats), median imputation, standard scaling, one-hot encoding
- **ID/URL column detection** — automatically excludes useless high-cardinality columns
- **Best model selection** — F1-score for classification, R² for regression

### 📊 Model Registry
- Register, version, and tag models (champion/challenger)
- MLflow integration for experiment tracking and artifact storage
- Upload custom pre-trained `.joblib` model files with feature schemas

### 🚀 Deployment Management
- Deploy any registered model as the active inference endpoint
- `/predict` endpoint for real-time inference with automatic feature preprocessing
- Logs all predictions to a production data store for drift monitoring

### 📡 Drift Detection & Auto-Retraining
- **Evidently AI** powered statistical drift detection (KS-test, Chi-squared, etc.)
- Compares training reference data vs. live production prediction logs
- Configurable drift threshold (default: 30% of features drifted)
- **Automatic retraining** triggered when drift is detected — trains fresh models on updated data and promotes the best one

### 🧠 AI-Powered Analysis (Gemini)
- **Dataset profiler** — uses Google Gemini (`gemini-2.0-flash`) to generate domain descriptions, suggest the best target column, and classify column roles
- **Data quality advisor** — detects formatted numerics, high-cardinality columns, high null %, and useless text columns
- Graceful fallback to local heuristic analyzers when Gemini API is unavailable

### 📝 Audit Logging
- All key system events are logged (training jobs, deployments, drift checks, retraining) with timestamps and metadata

### 🖥️ Dashboard UI
- Overview stats and system health
- Dataset upload with AI profiling results
- Model leaderboard with metrics comparison
- Deployment control panel
- Monitoring and drift visualization
- Audit logs viewer
- Custom model upload interface

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Frontend (Next.js)                     │
│  Dashboard │ Upload │ Leaderboard │ Deploy │ Monitor │ Logs  │
└─────────────────────────────┬────────────────────────────────┘
                              │ REST API (CORS)
┌─────────────────────────────▼────────────────────────────────┐
│                    Backend (FastAPI)                         │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐  ┌─────────┐  │
│  │ Datasets │  │  Models  │  │ Deployments │  │Monitoring│  │
│  │  Router  │  │  Router  │  │   Router    │  │  Router  │  │
│  └────┬─────┘  └────┬─────┘  └──────┬──────┘  └────┬────┘  │
│       │              │               │               │       │
│  ┌────▼──────────────▼───────────────▼───────────────▼────┐  │
│  │                   Service Layer                        │  │
│  │  AutoML │ AI Analyzer │ Drift │ Registry │ Retrainer  │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌──────────┐  ┌──────────▼──────┐  ┌──────────────────┐   │
│  │  MLflow  │  │  SQLAlchemy ORM │  │  Joblib Models   │   │
│  │ Registry │  │  (SQLite / PG)  │  │  (.joblib files) │   │
│  └──────────┘  └─────────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Gemini 2.0 Flash  │
                    │  (AI Profiling &    │
                    │   Quality Advisor)  │
                    └────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend Framework** | FastAPI 0.111 |
| **ORM / Database** | SQLAlchemy 2.0 + SQLite (default) / PostgreSQL |
| **ML Training** | scikit-learn 1.5, XGBoost 2.0, LightGBM 4.3 |
| **Experiment Tracking** | MLflow 2.13 |
| **Drift Detection** | Evidently AI 0.4 |
| **AI Analysis** | Google Gemini 2.0 Flash (`google-generativeai`) |
| **Model Serialization** | Joblib |
| **Frontend Framework** | Next.js 16 (App Router) |
| **Frontend Language** | TypeScript |
| **UI Components** | React 19 + Lucide Icons |
| **Charts** | Recharts |
| **Styling** | Tailwind CSS v4 |
| **API Validation** | Pydantic v2 + pydantic-settings |
| **Testing** | Pytest |

---

## 📁 Project Structure

```
Ship-It-ML/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Settings (env vars, defaults)
│   │   ├── database.py          # SQLAlchemy engine & session
│   │   ├── models.py            # ORM models (Dataset, Model, Deployment, etc.)
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   ├── middleware.py        # API key auth middleware
│   │   ├── routers/
│   │   │   ├── datasets.py      # Dataset upload & profiling endpoints
│   │   │   ├── models.py        # Model training & registry endpoints
│   │   │   ├── deployments.py   # Deploy, predict, production logs endpoints
│   │   │   ├── monitoring.py    # Drift detection & retraining endpoints
│   │   │   └── logs.py          # Audit log endpoints
│   │   └── services/
│   │       ├── automl.py        # Core AutoML training engine
│   │       ├── ai_analyzer.py   # Gemini AI dataset profiler & quality advisor
│   │       ├── drift.py         # Evidently drift calculation
│   │       ├── retrainer.py     # Auto-retraining pipeline
│   │       ├── registry.py      # MLflow model registry helpers
│   │       ├── deployment_manager.py  # In-memory model loader
│   │       └── audit_logger.py  # Structured event logging
│   ├── tests/
│   │   ├── test_e2e.py          # End-to-end API tests
│   │   ├── test_drift.py        # Drift detection unit tests
│   │   └── test_preprocessing.py # Preprocessing unit tests
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root layout with font & metadata
│   │   │   ├── page.tsx         # Main dashboard page
│   │   │   └── globals.css      # Global styles
│   │   ├── components/
│   │   │   ├── Sidebar.tsx           # Navigation sidebar
│   │   │   ├── OverviewDashboard.tsx # System stats & health
│   │   │   ├── DatasetUpload.tsx     # CSV upload + AI profiling UI
│   │   │   ├── ModelLeaderboard.tsx  # Model comparison table
│   │   │   ├── MonitoringDrift.tsx   # Drift scores & charts
│   │   │   ├── UploadCustomModel.tsx # Custom model upload
│   │   │   └── AuditLogs.tsx         # Audit event timeline
│   │   └── lib/
│   │       └── api.ts           # Typed API client (all endpoints)
│   ├── package.json
│   └── next.config.ts
│
├── models/                      # Saved .joblib model files
├── test/                        # Sample CSV datasets
│   ├── amazon.csv
│   └── mymoviedb.csv
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (Python 3.13 recommended)
- **Node.js 18+** and npm
- (Optional) **PostgreSQL** for production database
- (Optional) **MLflow server** running locally on port 5000

### 1. Clone the Repository

```bash
git clone https://github.com/msraghavendra/AutoMLOPS-Engineer.git
cd AutoMLOPS-Engineer
```

### 2. Backend Setup

```bash
cd backend

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies (use --prefer-binary to avoid building from source)
pip install --prefer-binary -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```env
DATABASE_URL=sqlite:///./shipitml.db
MLFLOW_TRACKING_URI=http://localhost:5000
API_KEY=your_secret_api_key_here
CORS_ORIGINS=http://localhost:3000
DEFAULT_DRIFT_THRESHOLD=0.3
UPLOAD_DIR=uploads
MODEL_DIR=models
GEMINI_API_KEY=your_gemini_api_key_here   # Optional: enables AI profiling
```

Start the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The dashboard will be live at `http://localhost:3000`

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./shipitml.db` | Database connection string |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URL |
| `API_KEY` | `ship_it_ml_secret_api_key_2026` | API key for authentication |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `DEFAULT_DRIFT_THRESHOLD` | `0.3` | Fraction of drifted features to trigger alert |
| `UPLOAD_DIR` | `uploads` | Directory for uploaded CSV files |
| `MODEL_DIR` | `models` | Directory for saved `.joblib` model files |
| `GEMINI_API_KEY` | _(empty)_ | Google Gemini API key for AI analysis |

> **Tip:** Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/). Without it, the system falls back to rule-based heuristic analysis.

---

## 📡 API Reference

All endpoints are prefixed and documented at `/docs` (Swagger UI) and `/redoc`.

### Datasets
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/datasets/upload` | Upload a CSV file |
| `POST` | `/datasets/{id}/analyze` | AI-powered dataset profiling |
| `POST` | `/datasets/{id}/quality` | Data quality analysis |
| `GET`  | `/datasets/` | List all datasets |
| `GET`  | `/datasets/{id}` | Get dataset details |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/models/train` | Trigger AutoML training on a dataset |
| `GET`  | `/models/` | List all trained models |
| `GET`  | `/models/{id}` | Get model details and metrics |
| `POST` | `/models/upload` | Upload a custom pre-trained model |
| `DELETE` | `/models/{id}` | Delete a model |

### Deployments
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/deployments/` | Deploy a model to production |
| `GET`  | `/deployments/` | List all deployments |
| `GET`  | `/deployments/active` | Get currently active deployment |
| `POST` | `/deployments/{id}/predict` | Run inference with deployed model |
| `GET`  | `/deployments/{id}/production-logs` | Get logged production predictions |

### Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/monitoring/drift` | Calculate data drift for active deployment |
| `POST` | `/monitoring/retrain` | Trigger manual retraining |
| `GET`  | `/monitoring/history` | Get drift history |

### Audit Logs
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/logs/` | List all audit log events |

---

## 🤖 AutoML Pipeline

The training pipeline (`automl.py`) follows these steps:

```
CSV Upload
    ↓
Smart Preprocessing
  ├── Coerce formatted numerics (₹, $, commas, %)
  ├── Detect & exclude ID/URL columns automatically
  ├── Impute missing values (median for numeric, mode for categorical)
  ├── StandardScaler for numerical features
  └── OneHotEncoder (max 50 categories) for categorical features
    ↓
Problem Type Detection
  ├── Classification: target is categorical OR < 15 unique values
  └── Regression: target is numeric with ≥ 15 unique values
    ↓
Train 4 Models in Parallel
  ├── Logistic Regression / Linear Regression
  ├── Random Forest
  ├── XGBoost
  └── LightGBM
    ↓
RandomizedSearchCV Hyperparameter Tuning
  └── (falls back to direct fit for high-cardinality classification)
    ↓
Evaluate on held-out 20% test set
  ├── Classification: Accuracy, F1, Precision, Recall, ROC-AUC
  └── Regression: R², MAE, MSE, RMSE
    ↓
Select Best Model (F1 or R²)
    ↓
Refit on Full Dataset → Save .joblib artifact
    ↓
Register in Model Registry
```

---

## 📡 Drift Detection & Auto-Retraining

When a model is deployed, every prediction is logged to the production data store. The drift detection pipeline:

1. **Loads reference data** — the original training CSV for the deployed model
2. **Loads current data** — all production prediction logs since deployment
3. **Runs Evidently AI** `DataDriftPreset` — statistical tests per feature (KS-test for numeric, Chi-squared for categorical)
4. **Calculates drift score** — ratio of drifted features
5. **Triggers alert** if drift score ≥ `DEFAULT_DRIFT_THRESHOLD` (default: 30%)
6. **Auto-retraining** — if triggered, runs a full AutoML pipeline on the original dataset, promotes the new champion, and redeploys it

```
Production Predictions (logged)
    ↓
compare with Training Reference Data
    ↓
Evidently AI Drift Report
    ↓
Drift Score > Threshold?
  ├── NO  → Log "no drift detected"
  └── YES → Trigger Retraining → Deploy New Champion
```

---

## 🧠 AI-Powered Analysis (Gemini)

When `GEMINI_API_KEY` is configured, the AI analyzer uses **Gemini 2.0 Flash** to:

**Dataset Profiling:**
- Identify the domain (e.g., "E-commerce sales data", "Movie recommendation dataset")
- Suggest the optimal target column
- Classify each column as `feature`, `target`, or `ignore`
- Detect formatted numerics, ID columns, URL columns, and free-text blobs

**Data Quality Advisor:**
- Score overall dataset quality (0-100)
- Flag issues: formatted numerics, high cardinality, high null %, useless text
- Recommend fixes for each issue

> Without a Gemini API key, all analysis falls back gracefully to a local rule-based heuristic analyzer with no loss of core functionality.

---

## 🧪 Running Tests

```bash
cd backend

# Activate virtual environment first
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_e2e.py -v          # End-to-end API tests
pytest tests/test_drift.py -v        # Drift detection tests
pytest tests/test_preprocessing.py -v # Preprocessing tests
```

---

## 📸 Screenshots

> The dashboard provides a full-featured MLOps control panel:

| Section | Description |
|---------|-------------|
| **Overview** | System health, active deployment status, model count |
| **Dataset Upload** | Drag-and-drop CSV upload with instant AI profiling |
| **Model Leaderboard** | Side-by-side metric comparison across all trained models |
| **Monitoring** | Drift score timeline and per-feature drift breakdown |
| **Audit Logs** | Full event history of all system actions |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Raghavendra M S**  
GitHub: [@msraghavendra](https://github.com/msraghavendra)

---

> Built with ❤️ for automating the boring parts of MLOps so you can focus on what matters — building great models.
