terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.34"
    }
  }
}

provider "google" {
  project     = var.project_id
  credentials = file(var.credentials_path)
  region      = var.region
}

# 1. Raw-data bucket
resource "google_storage_bucket" "raw_news_data" {
  name                        = var.bucket_name
  location                    = var.gcs_location
  force_destroy               = true
  uniform_bucket_level_access = true
}

# 2. BigQuery dataset
resource "google_bigquery_dataset" "news_warehouse" {
  dataset_id                 = "news_recommender_dw"
  friendly_name              = "News Recommender Data Warehouse"
  description                = "Contains all tables for the news recommender project."
  location                   = var.gcs_location
  delete_contents_on_destroy = true
}
