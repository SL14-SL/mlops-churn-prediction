output "artifact_registry_repository" {
  description = "Artifact Registry repository resource name."
  value       = google_artifact_registry_repository.containers.name
}

output "artifact_registry_image_prefix" {
  description = "Prefix used when publishing container images."
  value = join(
    "/",
    [
      "${var.region}-docker.pkg.dev",
      var.gcp_project_id,
      google_artifact_registry_repository.containers.repository_id,
    ],
  )
}

output "artifact_bucket_name" {
  description = "GCS bucket used for MLOps artifacts."
  value       = google_storage_bucket.artifacts.name
}

output "api_service_account_email" {
  description = "Service account used by the serving API."
  value       = google_service_account.api.email
}

output "api_key_secret_name" {
  description = "Secret Manager secret containing the API key."
  value       = google_secret_manager_secret.api_key.secret_id
}

output "cloud_run_service_name" {
  description = "Name of the deployed Cloud Run service."
  value = try(
    google_cloud_run_v2_service.api[0].name,
    null,
  )
}

output "cloud_run_service_uri" {
  description = "URI assigned to the Cloud Run service."
  value = try(
    google_cloud_run_v2_service.api[0].uri,
    null,
  )
}

output "mlflow_database_instance" {
  description = "Cloud SQL instance hosting the MLflow database."
  value       = google_sql_database_instance.mlflow.name
}

output "mlflow_database_connection_name" {
  description = "Cloud SQL connection name used by MLflow."
  value       = google_sql_database_instance.mlflow.connection_name
}

output "mlflow_service_account_email" {
  description = "Service account used by the MLflow server."
  value       = google_service_account.mlflow.email
}

output "mlflow_database_password_secret_name" {
  description = "Secret containing the MLflow database password."
  value       = google_secret_manager_secret.mlflow_database_password.secret_id
}