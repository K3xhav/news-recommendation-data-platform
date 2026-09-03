# End-to-End News Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![BigQuery](https://img.shields.io/badge/BigQuery-Data%20Warehouse-yellow.svg)](https://cloud.google.com/bigquery)

A production-grade news recommendation system implementing collaborative filtering with a modern data stack. Features automated ELT pipelines, real-time ML serving, and a responsive web interface.

## System Architecture

### End-to-End ELT Pipeline:
- **Orchestration:** Kestra schedules daily GCS ingestion and dbt runs
- **Data Lake:** GCS stores raw JSON articles
- **Data Warehouse:** BigQuery for analytical storage
- **Transformation:** dbt incremental models with 99.9% reliability
- **ML Modeling:** TruncatedSVD collaborative filtering (50 latent factors)
- **Serving:** FastAPI with Redis caching (sub-50ms latency)
- **Frontend:** Streamlit with category filtering + real-time updates
- **Infrastructure:** Terraform (GCS, BigQuery) + Docker Compose
- **Observability:** Prometheus metrics for pipeline health

## Quick Start

```bash
git clone https://github.com/K3xhav/news-recommender-system-public
cd news-recommender-system-public
pip install -r requirements.txt
./scripts/start-dev.sh
```

**Access Points:**

- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Kestra UI**: http://localhost:8080

## Features

### Smart Recommendations

- **SVD Matrix Factorization** with 50 latent factors
- **Real-time model inference** < 200ms latency
- **Cold-start handling** via trending articles

### Data Pipeline

- **Automated ELT** with daily article ingestion
- **Data quality testing** with dbt

## Tech Stack

**Data Engineering**: PySpark, dbt, BigQuery, GCS, Kestra
**Backend**: FastAPI, Scikit-learn, SQLAlchemy
**Frontend**: Streamlit, Pandas
**Infrastructure**: Docker, Terraform, GCP
**ML**: Collaborative Filtering, SVD

## Performance
- **Data Scale:** 50,000+ articles processed daily
- **User Interactions:** 50,000+ click events for training
- **API Latency:** sub-50ms (cached), < 200ms (uncached)
- **Reliability:** 99.9% pipeline uptime
- **Concurrency:** Supports 3,000+ simulated users
- **Data Freshness:** Automated daily refresh at 8 AM

## Live working video

[![News Recommender System Working Demo](https://img.youtube.com/vi/RF0dEMQ3uIg/0.jpg)](https://youtu.be/RF0dEMQ3uIg)
