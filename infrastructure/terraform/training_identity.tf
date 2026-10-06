resource "google_service_account" "training" {
  project      = var.gcp_project_id
  account_id   = "churn-${var.environment}-training"
  display_name = "${local.name_prefix} training"

  depends_on = [
    google_project_service.required["iam.googleapis.com"],
  ]
}

resource "google_storage_bucket_iam_member" "training_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.training.email}"
}

resource "google_cloud_run_v2_service_iam_member" "training_mlflow_invoker" {
  count = var.deploy_mlflow ? 1 : 0

  project  = var.gcp_project_id
  location = google_cloud_run_v2_service.mlflow[0].location
  name     = google_cloud_run_v2_service.mlflow[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.training.email}"
}

resource "google_service_account_iam_member" "training_operator" {
  count = trimspace(var.training_operator_email) != "" ? 1 : 0

  service_account_id = google_service_account.training.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "user:${var.training_operator_email}"
}