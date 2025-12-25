variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "credentials_path" {
  type        = string
  description = "Path to the service-account JSON key"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "gcs_location" {
  type    = string
  default = "US"
}

variable "bucket_name" {
  type        = string
  description = "Globally unique GCS bucket name for raw news data"
  default     = "news-recommender-raw-data-tf-unique"
}
