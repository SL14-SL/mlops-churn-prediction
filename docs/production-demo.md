# Google Cloud Production Demo

## Purpose and Scope

This guide describes the cost-conscious Google Cloud deployment used to prove
that the customer-churn training, release and serving lifecycle works outside
the local Docker Compose environment.

It is a production-oriented portfolio deployment, not a continuously operated
enterprise platform.

## Cloud Resources

Terraform provisions:

- an Artifact Registry repository;
- a GCS artifact bucket;
- a Cloud SQL PostgreSQL instance and MLflow database;
- a Secret Manager secret for the MLflow database password;
- an MLflow Cloud Run service connected to Cloud SQL;
- the `churn-prediction-api` Cloud Run service;
- service accounts and IAM bindings;
- Workload Identity Federation for GitHub Actions.

<p align="center">
  <img
    src="images/cloud_run_mlflow_cloud_sql.png"
    width="100%"
    alt="MLflow Cloud Run service connected to Cloud SQL and GCS"
  >
</p>

<p align="center">
  <em>
    Persistent production MLflow deployment using Cloud SQL for metadata,
    Secret Manager for credentials and GCS for model artifacts.
  </em>
</p>

Prefect Cloud records production training flows. Prometheus, Grafana and
Alertmanager remain part of the local operational demonstration unless a
separate cloud monitoring stack is deployed.

Avoid manually creating a second API service with another name. Terraform and
GitHub Actions must target the same `churn-prediction-api` service.

## Required Local Configuration

The Makefile includes and exports `.env`. Configure at least:

```text
API_KEY=<production-api-key>
PREFECT_API_URL=https://api.prefect.cloud/api/accounts/.../workspaces/...
PREFECT_API_KEY=<prefect-cloud-key>
GCP_PROJECT_ID=<project-id>
GCP_REGION=europe-west1
GCP_BUCKET_NAME=<artifact-bucket>
GCP_ARTIFACT_REPO=<artifact-registry-path>
MLFLOW_UI_URL=https://<mlflow-service>.run.app
MLFLOW_TRACKING_URI=https://<mlflow-service>.run.app
PRODUCTION_API_URL=https://<api-service>.run.app
PREDICTION_API_URL=https://<api-service>.run.app/predict
MLFLOW_DATABASE_INSTANCE=mlflow-postgres-dev
```

Do not commit `.env`, API keys, Prefect keys, service-account credentials or
Slack webhooks.

Validate the values without printing secrets:

```bash
make debug-prod-env
make check-prod-env
```

The validation must reject local MLflow, API and Prefect URLs for production
targets.

Authenticate locally when necessary:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
```

## Provision Infrastructure

```bash
terraform -chdir=infrastructure init
terraform -chdir=infrastructure fmt -check
terraform -chdir=infrastructure validate
terraform -chdir=infrastructure plan -out=tfplan
terraform -chdir=infrastructure apply tfplan
```
Read the generated service URLs:

```bash
terraform -chdir=infrastructure output -raw mlflow_url
terraform -chdir=infrastructure output -raw prediction_api_url
terraform -chdir=infrastructure output -raw artifacts_bucket_name
```
Store the raw URL values in `.env`. Do not include placeholder brackets,
variable assignments from a shell command or additional quotation marks.

Review replacements and deletions before applying. Confirm Cloud Run names,
IAM targets, bucket operations and Workload Identity conditions.

Terraform plan files, `.terraform/`, state and variable files containing local
values must not be committed.

### Existing Cloud Run resources

If a resource already exists but Terraform does not own it, import it instead
of creating a second service. For the API, the resource address and import ID
must match the current Terraform block.

Run a fresh plan after every import and require `0 to destroy` unless a deletion
is explicitly intended.

## GitHub Actions Configuration

Repository configuration includes:

### Secrets

- `GCP_WIF_PROVIDER`;
- `GCP_SA_EMAIL`;
- `API_KEY`;
- production integration secrets required by the workflow.

### Variables

- `DEPLOY_GCP`;
- `GCP_PROJECT_ID`;
- `GCP_REGION`;
- `GCP_ARTIFACT_REPO`;
- `GCP_BUCKET_NAME`;
- `MLFLOW_URL`.

Set `DEPLOY_GCP` to `true` only while the Terraform-managed infrastructure
exists:

```bash
gh variable set DEPLOY_GCP --body true
```

Set it back to `false` before destroying the infrastructure: 

```bash
gh variable set DEPLOY_GCP --body false
```

The Workload Identity provider condition must reference the exact GitHub
repository and branch, for example:

```text
attribute.repository == 'SL14-SL/mlops-churn-prediction'
assertion.ref == 'refs/heads/main'
```

A repository-name mismatch causes `unauthorized_client` during federated-token
generation.

## CI/CD Deployment

GitHub Actions performs:

1. Ruff validation;
2. unit and integration tests;
3. API smoke testing;
4. Terraform validation or planning;
5. API and MLflow image builds;
6. Trivy vulnerability scans;
7. Artifact Registry publication;
8. Cloud Run deployment.

Application images use immutable Git SHA tags. The production API should not
remain on the Terraform placeholder image after the deployment workflow.

Inspect deployed images:

```bash
gcloud run services describe churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='value(spec.template.spec.containers[0].image)'

gcloud run services describe mlflow-server \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='value(spec.template.spec.containers[0].image)'
```

## Upload Raw Demo Data

```bash
make upload-raw-prod
```

Confirm the object:

```bash
gcloud storage ls \
  "gs://${GCP_BUCKET_NAME}/data/raw/"
```

## Bootstrap Production

A fresh MLflow registry has no Champion and no complete serving release:

```bash
make train-bootstrap-prod
```

The target:

- validates the production environment;
- ensures that Cloud SQL and the MLflow health endpoint are available;
- uploads raw churn data;
- runs training through Prefect Cloud;
- registers the first Champion;
- publishes and activates an immutable release;
- reloads and verifies the API.

Do not use bootstrap once a Champion exists in the current MLflow backend.
For subsequent forced candidate runs:

```bash
make train-force-prod
```

Forced training still respects Champion/challenger promotion gates.
When recent labeled production evidence is available, promotion uses that data
as its primary evaluation dataset and applies reference-validation gates as a
separate safety layer.

## Verify Production

Resolve the API base URL from Terraform:

```bash
API_URL="$(
  terraform -chdir=infrastructure \
    output -raw prediction_api_url
)"
```

Check the services:

```bash
curl -fsS "$MLFLOW_UI_URL/health"
curl -fsS "$API_URL/livez" | jq .
curl -fsS "$API_URL/readyz" | jq .
curl -fsS "$API_URL/health" | jq .
```

Readiness must identify the active release, numeric model version, MLflow run
ID, model URI, decision threshold and loaded feature schema.

Execute authenticated semantic inference:

```bash
make predict-test-prod
```

The response must contain:

- a non-null `customer_id`;
- a finite churn probability between zero and one;
- a customer value;
- a configured retention action;
- finite expected value;
- release and model lineage metadata.

## Inspect Cloud Run and Logs

```bash
gcloud run services describe churn-prediction-api \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(metadata.name,status.url,status.traffic,status.conditions)'

gcloud run services describe mlflow-server \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION" \
  --format='yaml(metadata.name,status.url,status.traffic,status.conditions)'
```

Inspect recent errors:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND severity>=ERROR' \
  --project "$GCP_PROJECT_ID" \
  --freshness=30m \
  --limit=100
```

## Cost-Conscious Persistent MLflow Configuration

The demonstration uses:

- Cloud SQL for PostgreSQL as the persistent tracking and registry backend;
- GCS for MLflow artifacts and immutable serving releases;
- Secret Manager for the database password;
- an MLflow Cloud Run service with scale-to-zero enabled;
- a maximum of one MLflow Cloud Run instance;
- Terraform-managed infrastructure that is destroyed after verification.

Cloud Run revision replacement does not remove MLflow experiments, registered
models or aliases because this metadata is stored in Cloud SQL. Model artifacts
remain independently persisted in GCS.

Cloud SQL does not scale to zero and is therefore the largest ongoing cost of
the demonstration. For this portfolio deployment, it is provisioned only for
bootstrap, verification and evidence collection.

## Verify MLflow Persistence

Persistence was verified by creating a new MLflow Cloud Run revision after the
production Champion had been registered.

The verification confirmed:

- the MLflow health endpoint remained available;
- the registered churn model remained present;
- the `champion` alias still referenced the same model version;
- the model run ID remained unchanged;
- the run remained in `FINISHED` state;
- the model artifact URI still referenced GCS;
- the production API continued to pass semantic verification.

<p align="center">
  <img
    src="images/mlflow_persistence_verification.png"
    width="100%"
    alt="Successful MLflow persistence verification"
  >
</p>

<p align="center">
  <em>
    Champion version, run lineage and GCS artifact location verified after
    MLflow Cloud Run revision replacement.
  </em>
</p>

## Infrastructure Drift

Terraform should remain the source of truth for service memory, scaling,
environment variables and names. Reconcile emergency `gcloud run services
update` changes back into Terraform.

Run after any manual change:

```bash
terraform -chdir=infrastructure plan
```

## Teardown

Disable GitHub cloud deployments before destroying the infrastructure:

```bash
gh variable set DEPLOY_GCP --body false
```

Review and destroy the Terraform-managed resources:

```bash
terraform -chdir=infrastructure plan -destroy
terraform -chdir=infrastructure destroy
terraform -chdir=infrastructure state list
```

An empty state listing confirms that no Terraform-managed resources remain.
The remote Terraform state bucket is retained separately because Terraform
does not manage its own backend bucket.

Verify removal of the primary billable resources:

```bash
gcloud sql instances describe mlflow-postgres-dev \
  --project "$GCP_PROJECT_ID"

gcloud artifacts repositories list \
  --project "$GCP_PROJECT_ID" \
  --location "$GCP_REGION"
```

A `404` response for the Cloud SQL instance is expected after successful
destruction. Review retained state buckets separately and do not delete the
active Terraform backend bucket.

## Related Documentation

- [Architecture](architecture.md)
- [Serving releases](serving-releases.md)
- [Monitoring and SLOs](monitoring-and-slos.md)
- [Operations runbook](operations-runbook.md)

