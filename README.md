# ff-ml
Machine learning predictions for fantasy football

# Fantasy Football Player Performance Predictor (MLOps Pipeline)

This project demonstrates a real-world MLOps pipeline using a fantasy football player performance predictor as the example. The goal is to simulate a production-grade ML system, complete with data ingestion, model training, evaluation, versioning, serving, and monitoring.

## 1. Problem Setup

Before you begin, choose a problem that has:

- Regularly updated or streamed data (e.g., sports stats, tweets, sensor readings)
- A clear prediction or classification goal (e.g., weekly fantasy points)
- Publicly accessible APIs or datasets

## 2. Data Ingestion

Goal: Regularly pull in new data

Tools:
- APIs or Web Scraping: `requests`, `BeautifulSoup`, or `Selenium`
- Scheduled Jobs: `cron`, Apache Airflow, or Prefect
- Storage: PostgreSQL, SQLite, Amazon S3, or local files (CSV, Parquet, Feather)

## 3. Data Processing & Feature Engineering

Goal: Clean and transform data into model-ready format

Tools:
- Processing: `pandas`, `dask`, `polars`
- Validation: Great Expectations
- Development: JupyterLab, VS Code
- SQL pipelines: SQLAlchemy or raw SQL

## 4. Model Training & Evaluation

Goal: Train your model on the latest data and validate its performance

Tools:
- ML Models: `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
- Experiment Tracking: MLflow, Weights & Biases
- Pipelines: `sklearn.pipeline.Pipeline`
- Tuning: `GridSearchCV`, Optuna, Ray Tune

## 5. Model Versioning & Registry

Goal: Save and manage trained models with versioning

Tools:
- Registry: MLflow Model Registry
- File Tracking: DVC, Git LFS
- Portability: ONNX for model export

## 6. CI/CD & Automation

Goal: Automatically retrain and test the model when new data or code changes arrive

Tools:
- CI/CD: GitHub Actions, GitLab CI, Jenkins
- Common Steps:
  - Unit tests (pytest)
  - Data validation
  - Model training & evaluation
  - Model promotion to registry

## 7. Deployment

Goal: Serve the trained model online for real-time or batch prediction

Tools:
- REST API: FastAPI, Flask
- Containerization: Docker
- Production Serving: `uvicorn`, `gunicorn`, `nginx`
- Scaling: Kubernetes, docker-compose, AWS ECS/Fargate

## 8. Prediction Pipeline

Goal: Run predictions on new data

Tools:
- Scheduled Jobs: cron, Airflow
- Real-time Serving: FastAPI + streaming input
- Event Streams (Advanced): Kafka, RabbitMQ, Redis Streams
- Feature Store (Enterprise): Feast

## 9. Monitoring & Logging

Goal: Track model performance, drift, and system health

Tools:
- Metrics: Prometheus + Grafana
- Logs: Sentry, OpenTelemetry, ELK Stack
- Drift Detection: Evidently AI, Fiddler

## 10. Optional: AutoML & Retraining Loop

Goal: Automatically retrain when model performance drops

Tools:
- Feedback loop based on evaluation thresholds
- Trigger retraining pipeline via Airflow or CI/CD

## Postgres SQL DB Info
User: 'ffuser'
Password: 'ffpass' 
Database: 'ffdb'