resource "google_service_account" "mlflow" {
  project      = var.gcp_project_id
  account_id   = "churn-${var.environment}-mlflow"
  display_name = "${local.name_prefix} MLflow"

  depends_on = [
    google_project_service.required["iam.googleapis.com"],
  ]
}

resource "google_project_iam_member" "mlflow_sql_client" {
  project = var.gcp_project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlflow.email}"

  depends_on = [
    google_project_service.sqladmin,
  ]
}

resource "google_storage_bucket_iam_member" "mlflow_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

resource "google_secret_manager_secret" "mlflow_database_password" {
  project   = var.gcp_project_id
  secret_id = "${local.name_prefix}-mlflow-db-password"

  replication {
    auto {}
  }

  depends_on = [
    google_project_service.required["secretmanager.googleapis.com"],
  ]
}

resource "google_secret_manager_secret_iam_member" "mlflow_database_password" {
  project   = var.gcp_project_id
  secret_id = google_secret_manager_secret.mlflow_database_password.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlflow.email}"
}