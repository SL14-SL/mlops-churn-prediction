resource "google_project_service" "sqladmin" {
  project            = var.gcp_project_id
  service            = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

# Public IP provides connectivity for the Cloud Run Cloud SQL Auth Proxy.
# Connector enforcement rejects direct connections; no authorized networks.
# trivy:ignore:GCP-0017
resource "google_sql_database_instance" "mlflow" {
  project          = var.gcp_project_id
  name             = "${local.name_prefix}-mlflow-db"
  region           = var.region
  database_version = "POSTGRES_15"

  deletion_protection = true

  settings {
    edition           = "ENTERPRISE"
    tier              = "db-f1-micro"
    availability_type = "ZONAL"
    activation_policy = "ALWAYS"

    disk_type       = "PD_SSD"
    disk_size       = 10
    disk_autoresize = true

    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }

    connector_enforcement = "REQUIRED"

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
    }
  }

  depends_on = [
    google_project_service.sqladmin,
  ]
}

resource "google_sql_database" "mlflow" {
  project  = var.gcp_project_id
  name     = "mlflowdb"
  instance = google_sql_database_instance.mlflow.name
}