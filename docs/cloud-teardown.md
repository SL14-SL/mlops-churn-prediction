# Google Cloud teardown

This document describes the controlled removal of a Google Cloud
environment created for **Customer Churn Prediction**.

The teardown is intentionally separated into two stages:

1. remove the application infrastructure;
2. remove the bootstrap infrastructure and Terraform state.

Always destroy application environments before destroying the bootstrap
resources they depend on.

The recorded demonstration completed teardown on 6 October 2026: 29
application resources and 18 bootstrap resources were destroyed, both Terraform
states were empty, and SQL, Cloud Run, storage, Artifact Registry and Secret
Manager listings returned no demo resources. The GCP project itself remains.

## Safety rules

Before continuing:

- confirm the Google Cloud project ID;
- confirm the deployment environment;
- inspect every Terraform destroy plan;
- preserve any model releases, monitoring data or logs that are still
  required;
- ensure no other environment uses the bootstrap state bucket or GitHub
  Workload Identity configuration.

Do not delete the bootstrap resources while an application environment
still has resources in its remote Terraform state.

## Set the teardown context

Set explicit shell variables:

```bash
GCP_PROJECT_ID="your-gcp-project-id"
DEPLOYMENT_ENVIRONMENT="dev"
GCP_REGION="europe-west1"

gcloud config set project "$GCP_PROJECT_ID"

printf 'Project:     %s\n' "$GCP_PROJECT_ID"
printf 'Environment: %s\n' "$DEPLOYMENT_ENVIRONMENT"
printf 'Region:      %s\n' "$GCP_REGION"
```

Confirm the active Google Cloud account and project:

```bash
gcloud auth list

gcloud config get-value project
```

## Read the bootstrap outputs

The bootstrap Terraform state supplies the remote state bucket:

```bash
TF_STATE_BUCKET="$(
  terraform \
    -chdir=infrastructure/terraform-bootstrap \
    output \
    -raw \
    terraform_state_bucket
)"

printf 'Terraform state bucket: %s\n' \
  "$TF_STATE_BUCKET"
```

If the bootstrap output is unavailable, identify the bucket before
continuing:

```bash
gcloud storage buckets list \
  --project "$GCP_PROJECT_ID"
```

## Back up the application state

Initialize the application stack against its existing remote state:

```bash
terraform \
  -chdir=infrastructure/terraform \
  init \
  -input=false \
  -reconfigure \
  -backend-config="bucket=${TF_STATE_BUCKET}" \
  -backend-config="prefix=mlops-churn-prediction/${DEPLOYMENT_ENVIRONMENT}"
```

Inspect the resources currently tracked:

```bash
terraform \
  -chdir=infrastructure/terraform \
  state list
```

Create a local state backup:

```bash
terraform \
  -chdir=infrastructure/terraform \
  state pull \
  > "/tmp/mlops-churn-prediction-${DEPLOYMENT_ENVIRONMENT}-before-destroy.tfstate"
```

The backup can contain sensitive infrastructure data. Keep it outside
the repository and delete it when it is no longer required.

## Unlock Application Deletion Protection

The MLflow database has Terraform deletion protection. Temporarily change
`deletion_protection = true` to `false` in
`infrastructure/terraform/mlflow_database.tf`.

The artifact bucket also requires `force_destroy_artifact_bucket=true` if its
objects are intentionally being removed. Apply those protection changes before
creating the destroy plan; a destroy plan alone does not persist the unlock.
Use the same project, region, environment and repository values as deployment.

```bash
terraform -chdir=infrastructure/terraform plan \
  -out=unlock-destroy.tfplan \
  -var="gcp_project_id=${GCP_PROJECT_ID}" \
  -var="region=${GCP_REGION}" \
  -var="environment=${DEPLOYMENT_ENVIRONMENT}" \
  -var="artifact_registry_repository=mlops-churn-prediction-images" \
  -var="container_image=unused" \
  -var="force_destroy_artifact_bucket=true" \
  -target=google_sql_database_instance.mlflow \
  -target=google_storage_bucket.artifacts
terraform -chdir=infrastructure/terraform show -no-color unlock-destroy.tfplan
terraform -chdir=infrastructure/terraform apply unlock-destroy.tfplan
```

The targeted plan is limited to the explicit protection changes and their
Terraform dependencies. Stop if it proposes unrelated changes or resource
removal. Do not commit the temporary database protection change.

## Review the application destroy plan

Create a saved destroy plan:

```bash
terraform \
  -chdir=infrastructure/terraform \
  plan \
  -destroy \
  -input=false \
  -out=destroy.tfplan \
  -var="gcp_project_id=${GCP_PROJECT_ID}" \
  -var="environment=${DEPLOYMENT_ENVIRONMENT}" \
  -var="container_image=unused" \
  -var="region=${GCP_REGION}" \
  -var="artifact_registry_repository=mlops-churn-prediction-images" \
  -var="force_destroy_artifact_bucket=true"
```

Render the plan for review:

```bash
terraform \
  -chdir=infrastructure/terraform \
  show \
  -no-color \
  destroy.tfplan \
  > /tmp/mlops-churn-prediction-destroy-plan.txt
```

Review the affected resources:

```bash
grep -E \
  '^  # .* will be destroyed|^Plan:' \
  /tmp/mlops-churn-prediction-destroy-plan.txt
```

Stop if the plan includes resources outside the intended project and
environment.

## Preserve or remove application artifacts

The environment artifact bucket can contain:

- serving manifests, feature schemas and prediction probes;
- model artifacts;
- monitoring data;
- prediction logs;
- processed datasets.

List the bucket before destruction:

```bash
ARTIFACT_BUCKET="${GCP_PROJECT_ID}-mlops-churn-prediction-${DEPLOYMENT_ENVIRONMENT}-artifacts"

gcloud storage du \
  --summarize \
  --readable-sizes \
  "gs://${ARTIFACT_BUCKET}" \
  2>/dev/null \
  || true
```

Copy required objects to an approved backup location before continuing.

The bucket is versioned and may intentionally block deletion while it
contains objects. Do not enable force deletion until the retained data
has been reviewed.

## Destroy the application infrastructure

Apply the exact reviewed plan:

```bash
terraform \
  -chdir=infrastructure/terraform \
  apply \
  -input=false \
  destroy.tfplan
```

Restore the database protection setting after destruction:

```bash
git restore -- infrastructure/terraform/mlflow_database.tf
```

Use `git restore` only when that file has no unrelated uncommitted changes;
otherwise restore just the temporary protection edit.

Confirm that the application state is empty:

```bash
terraform \
  -chdir=infrastructure/terraform \
  state list
```

The command should produce no resource addresses.

Verify the main managed services:

```bash
gcloud run services list \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"

gcloud artifacts repositories list \
  --project "$GCP_PROJECT_ID" \
  --location "$GCP_REGION"

gcloud storage buckets list \
  --project "$GCP_PROJECT_ID"
```

## Decide whether to retain the bootstrap

Keep the bootstrap infrastructure when:

- another environment still uses the state bucket;
- another branch or repository deployment still uses the Workload
  Identity provider;
- the project will be deployed again soon.

The bootstrap resources have low operational cost and are protected
against accidental deletion.

Continue only when every dependent application environment has already
been destroyed.

## Temporarily unlock bootstrap destruction

The bootstrap state bucket uses both:

```hcl
force_destroy = false

lifecycle {
  prevent_destroy = true
}
```

These settings intentionally make complete teardown a manual operation.

In the generated project, temporarily edit:

```text
infrastructure/terraform-bootstrap/main.tf
```

Change:

```hcl
force_destroy = false
```

to:

```hcl
force_destroy = true
```

and change:

```hcl
prevent_destroy = true
```

to:

```hcl
prevent_destroy = false
```

Do not commit these temporary protection changes.

Persist the bucket force-deletion setting before its destroy plan:

```bash
terraform -chdir=infrastructure/terraform-bootstrap plan \
  -target=google_storage_bucket.terraform_state -out=bootstrap-unlock.tfplan
terraform -chdir=infrastructure/terraform-bootstrap show -no-color bootstrap-unlock.tfplan
terraform -chdir=infrastructure/terraform-bootstrap apply bootstrap-unlock.tfplan
```

Keep the original bootstrap variable file and local state available. Review the
plan to confirm that only the intended bucket protection is changing.

## Review the bootstrap destroy plan

Inspect the resources tracked by the bootstrap state:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  state list
```

Create and review a saved plan:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  plan \
  -destroy \
  -input=false \
  -out=bootstrap-destroy.tfplan
```

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  show \
  -no-color \
  bootstrap-destroy.tfplan \
  > /tmp/mlops-churn-prediction-bootstrap-destroy-plan.txt
```

```bash
grep -E \
  '^  # .* will be destroyed|^Plan:' \
  /tmp/mlops-churn-prediction-bootstrap-destroy-plan.txt
```

The plan should be limited to the bootstrap state bucket, Workload
Identity resources, deployment service account, IAM bindings and
supporting Google APIs managed by the bootstrap module.

## Destroy the bootstrap infrastructure

Apply the reviewed bootstrap plan:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  apply \
  -input=false \
  bootstrap-destroy.tfplan
```

Confirm that its state is empty:

```bash
terraform \
  -chdir=infrastructure/terraform-bootstrap \
  state list
```

Restore the protected configuration immediately:

```bash
git restore -- \
  infrastructure/terraform-bootstrap/main.tf
```

Confirm that no temporary source changes remain:

```bash
git status --short
```

## Remove GitHub deployment configuration

After the Google Cloud resources have been destroyed, remove the
environment secret:

```bash
gh secret delete \
  API_KEY \
  --env "$DEPLOYMENT_ENVIRONMENT"
```

Also remove the database secret if the environment is being retired:

```bash
gh secret delete MLFLOW_DATABASE_PASSWORD --env "$DEPLOYMENT_ENVIRONMENT"
```

Remove an environment-specific service-name override if configured:

```bash
gh variable delete \
  CLOUD_RUN_SERVICE_NAME \
  --env "$DEPLOYMENT_ENVIRONMENT" \
  2>/dev/null \
  || true
```

Repository-level deployment variables can be removed after all
environments have been retired:

```bash
for variable_name in \
  GCP_PROJECT_ID \
  GCP_REGION \
  ARTIFACT_REGISTRY_REPOSITORY \
  GCP_WORKLOAD_IDENTITY_PROVIDER \
  GCP_DEPLOY_SERVICE_ACCOUNT \
  TF_STATE_BUCKET \
  GCS_STORAGE_LOCATION \
  ALLOW_UNAUTHENTICATED \
  MLFLOW_TRACKING_URI \
  PREFECT_API_URL \
  TRAINING_OPERATOR_EMAIL
do
  gh variable delete \
    "$variable_name" \
    2>/dev/null \
    || true
done
```

Do not remove repository-level variables while another GitHub
Environment still uses them.

## Final verification

Check Cloud SQL and secrets as well as the other managed resources:

```bash
gcloud sql instances list --project "$GCP_PROJECT_ID"
gcloud secrets list --project "$GCP_PROJECT_ID"
```

Check for remaining relevant resources:

```bash
gcloud run services list \
  --project "$GCP_PROJECT_ID" \
  --region "$GCP_REGION"

gcloud artifacts repositories list \
  --project "$GCP_PROJECT_ID" \
  --location "$GCP_REGION"

gcloud storage buckets list \
  --project "$GCP_PROJECT_ID"

gcloud iam service-accounts list \
  --project "$GCP_PROJECT_ID"
```

Also inspect the Google Cloud Billing console if the project contains
resources that are not managed by this repository.

Terraform only destroys resources recorded in its state. Unrelated or
manually created resources require separate review.

## Local cleanup

Remove the local destroy plans and temporary state backup after the
teardown has been verified:

```bash
rm \
  -f \
  infrastructure/terraform/destroy.tfplan \
  infrastructure/terraform/unlock-destroy.tfplan \
  infrastructure/terraform-bootstrap/bootstrap-unlock.tfplan \
  infrastructure/terraform-bootstrap/bootstrap-destroy.tfplan \
  "/tmp/mlops-churn-prediction-${DEPLOYMENT_ENVIRONMENT}-before-destroy.tfstate" \
  /tmp/mlops-churn-prediction-destroy-plan.txt \
  /tmp/mlops-churn-prediction-bootstrap-destroy-plan.txt
```

Unset the shell variables:

```bash
unset \
  GCP_PROJECT_ID \
  DEPLOYMENT_ENVIRONMENT \
  GCP_REGION \
  TF_STATE_BUCKET \
  ARTIFACT_BUCKET
```
