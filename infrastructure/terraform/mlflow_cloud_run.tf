resource "google_cloud_run_v2_service" "mlflow" {
  count = var.deploy_mlflow ? 1 : 0

  project             = var.gcp_project_id
  name                = "${local.name_prefix}-mlflow"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.mlflow.email
    timeout         = "300s"

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    volumes {
      name = "cloudsql"

      cloud_sql_instance {
        instances = [
          google_sql_database_instance.mlflow.connection_name,
        ]
      }
    }

    containers {
      image = var.mlflow_container_image

      ports {
        container_port = 5000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }

        cpu_idle = true
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      env {
        name  = "CLOUD_SQL_CONNECTION_NAME"
        value = google_sql_database_instance.mlflow.connection_name
      }

      env {
        name  = "MLFLOW_DB_USER"
        value = "mlflow"
      }

      env {
        name  = "MLFLOW_DB_NAME"
        value = google_sql_database.mlflow.name
      }

      env {
        name = "MLFLOW_DB_PASSWORD"

        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mlflow_database_password.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "MLFLOW_ARTIFACT_ROOT"
        value = "gs://${google_storage_bucket.artifacts.name}/mlflow-artifacts"
      }

      env {
        name  = "MLFLOW_SERVER_ALLOWED_HOSTS"
        value = "*"
      }

      startup_probe {
        timeout_seconds   = 3
        period_seconds    = 10
        failure_threshold = 30

        http_get {
          path = "/health"
          port = 5000
        }
      }

      liveness_probe {
        timeout_seconds = 3
        period_seconds  = 30

        http_get {
          path = "/health"
          port = 5000
        }
      }
    }
  }

  lifecycle {
    precondition {
      condition     = length(trimspace(var.mlflow_container_image)) > 0
      error_message = "mlflow_container_image is required when deploy_mlflow is true."
    }
  }

  depends_on = [
    google_project_service.required["run.googleapis.com"],
    google_sql_database.mlflow,
    google_project_iam_member.mlflow_sql_client,
    google_storage_bucket_iam_member.mlflow_artifacts,
    google_secret_manager_secret_iam_member.mlflow_database_password,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "api_mlflow_invoker" {
  count = var.deploy_mlflow ? 1 : 0

  project  = var.gcp_project_id
  location = google_cloud_run_v2_service.mlflow[0].location
  name     = google_cloud_run_v2_service.mlflow[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.api.email}"
}