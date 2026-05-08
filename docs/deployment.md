# 🚀 Deployment Guide

This document describes how to deploy the production-grade MLOps platform on GCP.

The platform includes:

* FastAPI prediction API
* MLflow tracking & model registry
* Prefect orchestration
* Prometheus + Grafana monitoring
* automated CI/CD pipelines
* Terraform-based infrastructure provisioning

---

# 🏗️ Deployment Architecture

```text
GitHub Actions
↓
Container Build + Trivy Scan
↓
Artifact Registry
↓
Cloud Run Deployment
↓
Prediction API + MLflow
↓
Monitoring + Retraining Flows
```

Infrastructure components:

* Cloud Run
* Artifact Registry
* GCS
* GitHub Actions
* Terraform
* Prometheus
* Grafana

Authentication is handled via Workload Identity Federation (OIDC).

No long-lived cloud credentials are stored in GitHub.

---

# ☁️ Required GCP Services

Enable the following APIs in GCP:

* Cloud Run API
* Artifact Registry API
* IAM Credentials API
* Cloud Build API
* Secret Manager API
* Cloud Storage API

---

# 📦 Requirements

Before deployment, install:

* Docker
* Terraform
* Python 3.12
* uv
* gcloud CLI

You also need:

* a GCP project
* billing enabled
* a configured service account
* GitHub Actions OIDC / WIF setup

---

# ⚙️ Infrastructure Provisioning

Terraform is used to provision the required cloud infrastructure on GCP.

Provisioned resources include:

* Artifact Registry repository
* GCS bucket for model artifacts and monitoring history
* Cloud Run services
* IAM service account bindings

## Navigate to the infrastructure directory

```bash
cd infrastructure
```

## Create Terraform variables

```bash
cp terraform.tfvars.example terraform.tfvars
```

Example Terraform variables:

```hcl
project_id          = "your-gcp-project"
region              = "europe-west1"
bucket_name         = "mlops-churn-bucket"
artifact_repository = "mlops-repo"
```

## Initialize Terraform

```bash
terraform init
```

## Preview infrastructure changes

```bash
terraform plan
```

## Apply infrastructure

```bash
terraform apply
```

Recommended production practices:

* use remote Terraform state
* enable state locking
* avoid committing tfstate files
* restrict IAM permissions to least privilege

---

# 🔐 GitHub Actions Authentication

CI/CD authentication is handled via:

* Workload Identity Federation (OIDC)
* short-lived credentials
* dedicated deployment service account

Example deployment service account roles:

* Cloud Run Admin
* Artifact Registry Writer
* Storage Admin
* Service Account User

---

# 🔐 GitHub Actions Configuration

Configure repository secrets in:

```text
Settings
→ Secrets and Variables
→ Actions
```

## Required Secrets

```text
GCP_WIF_PROVIDER
GCP_SA_EMAIL
API_KEY
```

## Required Variables

```text
GCP_REGION
GCP_ARTIFACT_REPO
MLFLOW_URL
GCP_PROJECT_ID
GCP_BUCKET_NAME
```

---

# 🔄 CI/CD Pipeline

The CI/CD pipeline is implemented with GitHub Actions.

Key pipeline stages:

1. Ruff linting
2. unit + integration tests
3. API smoke tests
4. Docker image builds
5. Trivy vulnerability scanning
6. Artifact Registry push
7. Cloud Run deployment

Security highlights:

* authentication via Workload Identity Federation (OIDC)
* no long-lived GCP credentials stored in GitHub
* deployments blocked on failed tests or critical vulnerabilities

Pipeline definition:

```text
.github/workflows/main.yml
```

Deployment is triggered automatically on pushes to:

```text
main
```

---

# 🚀 Deployment Flow

Deployment process:

1. push changes to `main`
2. GitHub Actions runs tests and smoke tests
3. Docker images are built
4. Trivy vulnerability scan is executed
5. images are pushed to Artifact Registry
6. Cloud Run services are updated
7. new revisions become active automatically

Separate Cloud Run services are deployed for:

* prediction API
* MLflow tracking server

This allows independent scaling and deployments.

---

# 🐳 Local Development Deployment

Start all local services:

```bash
docker-compose up -d
```

This launches:

* FastAPI API
* MLflow
* Prefect
* PostgreSQL
* Prometheus
* Grafana

---

# 🧠 Initial Model Bootstrap

After deployment, the system requires an initial trained model.

Run:

```bash
make train-force
```

This executes:

* ingestion
* validation
* feature engineering
* training
* MLflow registration
* champion model setup

Without this step, the API cannot serve predictions.

---

# 📊 Monitoring & Observability

The platform provides:

* feature drift detection
* prediction logging
* performance monitoring
* retraining triggers
* operational metrics
* Prometheus metrics
* Grafana dashboards

Monitoring configuration:

```text
configs/monitoring.yaml
```

---

# 🔁 Automated Retraining

The platform supports automated retraining workflows.

Retraining flow:

```text
monitoring
↓
Prefect retraining flow
↓
model training
↓
MLflow registration
↓
champion evaluation
```

Retraining can be triggered by:

* drift thresholds
* performance degradation
* scheduled orchestration flows

Prefect deployment configuration:

```text
prefect.yaml
```

---

# 📈 Service Endpoints

## Prediction API

```text
POST /predict
```

## Liveness Check

```text
GET /livez
```

## Metrics Endpoint

```text
GET /metrics
```

## Swagger UI

```text
/docs
```

---

# 🌐 Live Services

Add deployed service URLs here.

Example:

```text
Prediction API:
https://your-api-url

Swagger:
https://your-api-url/docs

MLflow:
https://your-mlflow-url

Grafana:
https://your-grafana-url
```

---

# ⚙️ Environment Configuration

Environment-specific settings are managed via:

```text
configs/dev.yaml
configs/staging.yaml
configs/prod.yaml
configs/gcp.yaml
```

Environment variables are loaded from:

```bash
.env
```

Example:

```bash
cp .env.example .env
```

---

# 🔒 Security

Security controls included in this platform:

* non-root Docker containers
* Workload Identity Federation (OIDC)
* vulnerability scanning with Trivy
* smoke tests before deployment
* isolated Cloud Run services
* environment-based configuration

---

# 🧪 Deployment Verification

Verify the deployment after rollout.

## API health

```bash
curl https://YOUR_API_URL/livez
```

## Metrics endpoint

```bash
curl https://YOUR_API_URL/metrics
```

## Swagger UI

```text
https://YOUR_API_URL/docs
```

---

# 📁 Important Directories

```text
infrastructure/        Terraform infrastructure
.github/workflows/     CI/CD pipelines
configs/               environment configuration
flows/                 Prefect orchestration flows
monitoring/            Prometheus configuration
docs/                  deployment documentation
```

---

# 🚀 Production Notes

This project is designed as a production-oriented MLOps showcase focused on:

* operational ML systems
* cloud-native deployment
* CI/CD automation
* observability
* retraining workflows
* infrastructure reproducibility

The emphasis is on reliable ML operations and maintainable production systems.
