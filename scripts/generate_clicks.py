# /scripts/generate_clicks.py - FINAL VERSION

import os
import sys
from pyspark.sql import SparkSession
import random
from datetime import datetime

# --- Configuration ---
GCP_PROJECT_ID = "news-recommendation-473806"
GCS_BUCKET_NAME = "news-recommender-raw-data-tf-unique"
BIGQUERY_DATASET = "news_recommender_dw"
BIGQUERY_ARTICLE_TABLE = "dim_articles"

# Service Account Key Path
SERVICE_ACCOUNT_KEY_PATH = os.getenv("SERVICE_ACCOUNT_KEY_PATH")

# Get absolute paths to JARs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
JARS_DIR = os.path.join(PROJECT_ROOT, "jars")

BIGQUERY_JAR = os.path.join(
    JARS_DIR, "spark-bigquery-with-dependencies_2.12-0.34.0.jar"
)
GCS_JAR = os.path.join(JARS_DIR, "gcs-connector-hadoop3-2.2.18.jar")

print(" Initializing Spark Session...")
print(f"Service Account Key exists: {os.path.exists(SERVICE_ACCOUNT_KEY_PATH)}")

# --- Spark Session Initialization ---
spark = (
    SparkSession.builder.appName("ClickstreamGenerator")
    .config("spark.jars", f"{BIGQUERY_JAR},{GCS_JAR}")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.executor.memory", "1g")
    .config("spark.driver.memory", "1g")
    # GCS configurations
    .config(
        "spark.hadoop.fs.gs.impl",
        "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
    )
    .config(
        "spark.hadoop.fs.AbstractFileSystem.gs.impl",
        "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
    )
    .config("spark.hadoop.fs.gs.project.id", GCP_PROJECT_ID)
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
    .config(
        "spark.hadoop.google.cloud.auth.service.account.json.keyfile",
        SERVICE_ACCOUNT_KEY_PATH,
    )
    .config("temporaryGcsBucket", GCS_BUCKET_NAME)
    .getOrCreate()
)

print(" Spark session created successfully!")

# Set Hadoop configuration
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
hadoop_conf.set(
    "fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
)
hadoop_conf.set("fs.gs.project.id", GCP_PROJECT_ID)
hadoop_conf.set("google.cloud.auth.service.account.enable", "true")
hadoop_conf.set(
    "google.cloud.auth.service.account.json.keyfile", SERVICE_ACCOUNT_KEY_PATH
)

print(" Using service account credentials...")

# --- Read from BigQuery ---
print(f" Reading articles from BigQuery...")
articles_df = (
    spark.read.format("bigquery")
    .option("table", f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_ARTICLE_TABLE}")
    .option("parentProject", GCP_PROJECT_ID)
    .load()
)

article_ids = [
    row.article_id for row in articles_df.select("article_id").distinct().collect()
]
print(f" Found {len(article_ids)} unique articles.")

# --- Generate Click Data ---
print(" Generating fake user clickstream data...")
num_users, num_clicks = 500, 1000
user_ids = list(range(1, num_users + 1))

click_data = [
    {
        "user_id": random.choice(user_ids),
        "article_id": random.choice(article_ids),
        "click_timestamp": datetime.now(),
    }
    for _ in range(num_clicks)
]

clicks_df = spark.createDataFrame(click_data)
print(f" Generated {clicks_df.count()} click events.")
clicks_df.show(5)

# --- Write to GCS ---
output_path = f"gs://{GCS_BUCKET_NAME}/raw_clicks/{datetime.now().strftime('%Y-%m-%d')}"
print(f" Writing to GCS: {output_path}...")

import time

start_time = time.time()

clicks_df.write.mode("overwrite").parquet(output_path)

end_time = time.time()
print(
    f" Success! Written {clicks_df.count()} records to GCS in {end_time - start_time:.2f} seconds"
)

spark.stop()
print(" Spark session stopped.")
