# Template migration to v0.3.0

This document tracks the adoption of the reusable MLOps project template by
the existing churn-prediction project.

## Migration principles

- Existing project behavior must remain covered by tests.
- Project-specific model and business logic remains owned by this repository.
- Reusable infrastructure and lifecycle code should converge towards the
  project template.
- Each migration phase must leave the project in a runnable state.
- Template files must not replace project-specific files without an explicit
  comparison.

## Ownership

### Project-specific

The following areas remain project-owned:

- churn data ingestion and validation
- churn feature engineering
- dataset splitting and versioning
- classification model construction and training
- evaluation and business decision policies
- SHAP-based model explanations
- churn-specific API routes and schemas
- customer-facing documentation and case-study material
- project-specific MLflow and Cloud SQL requirements

### Template-owned

The following areas should eventually follow the template:

- Copier metadata
- generic configuration loading
- filesystem and GCS abstractions
- training-pipeline contracts and lifecycle records
- Prefect orchestration adapters
- MLflow candidate registration and promotion
- serving-release manifests and active pointers
- generic FastAPI middleware and health endpoints
- Prometheus metrics and Grafana provisioning
- dependency and container security scans
- Terraform bootstrap and Cloud Run deployment foundations
- template update documentation

### Merge required

These areas contain relevant code on both sides and require manual
reconciliation:

- Python package layout
- pyproject.toml and uv.lock
- Dockerfile and Compose configuration
- Makefile
- environment configuration
- API bootstrap and application construction
- inference model management
- serving-release publication
- monitoring and drift detection
- GitHub Actions
- Terraform infrastructure
- operational documentation

## Migration phases

1. Register the project with Copier v0.3.0.
2. Normalize repository metadata and development tooling.
3. Move the Python package to `src/mlops_churn_prediction`.
4. Adopt shared configuration, utilities and storage abstractions.
5. Connect churn-specific training components to template contracts.
6. Adopt pipeline lifecycle and Prefect orchestration.
7. Reconcile MLflow registration, promotion and serving releases.
8. Reconcile the FastAPI serving layer.
9. Integrate monitoring, feedback and drift functionality.
10. Reconcile Docker, CI/CD and Google Cloud infrastructure.
11. Run end-to-end and template-update verification.
12. Remove superseded legacy files.

## Current template baseline

- Template version: v0.3.0
- Task type: classification
- Project slug: mlops-churn-prediction
- Migration branch: chore/adopt-template-v0.3.0