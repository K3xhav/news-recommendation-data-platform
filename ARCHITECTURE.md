
# System Architecture

## End-to-End Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Ingestion│    │  ML Pipeline     │    │   Web Interface │
│   • Kestra      │────│  • FastAPI       │────│   • Streamlit   │
│   • PySpark     │    │  • Scikit-learn  │    │   • BigQuery    │
|   • Dbt         |    |                  |    |                 |
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                         ┌─────────────────┐
                         │  Data Storage   │
                         │  • BigQuery     │
                         │  • GCS          │
                         └─────────────────┘
```

## Component Deep Dive

### 1. Data Ingestion & Orchestration
**Tools**: Kestra, Python, NewsAPI  
**Frequency**: Daily scheduled execution  
**Output**: Raw JSONL files in GCS

### 2. Data Lake (GCS)
**Structure**:
```
gs://news-recommender-raw-data/
├── raw_articles/
└── raw_clicks/
```

### 3. User Simulation (Cold Start Solution)
**Tool**: PySpark  
**Purpose**: Generate training data for collaborative filtering  
**Scale**: 500 users, 1,000+ interactions

### 4. Data Warehouse & Transformation
**Architecture**: Medallion (Bronze → Silver → Gold)

#### dbt Transformation Pipeline:
```
sources.yml (External Tables)
    ↓
stg_articles.sql (Cleaning & Deduplication)  
    ↓
dim_articles.sql (Article Dimension)
fact_user_clicks.sql (Click Facts)
```

### 5. Machine Learning Engine
**Algorithm**: Truncated SVD (Collaborative Filtering)  
**Training**: On API startup  
**Matrix**: User-Item interactions (400 users × 747 articles)

### 6. API Serving Layer
**Framework**: FastAPI  
**Endpoints**:
- `GET /recommendations/{user_id}` → Personalized articles
- `GET /health` → Service status

### 7. Web Application
**Framework**: Streamlit (Multi-page)  
**Data Access**:
- **Homepage**: Direct BigQuery queries
- **For You Page**: API calls for personalized recommendations

## Data Model

### Star Schema in BigQuery
```
dim_articles (Dimension)
├── article_id (PK)
├── title
├── category
└── published_at

fact_user_clicks (Fact)  
├── user_id
├── article_id (FK)
└── click_timestamp
```

## ML Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision@5** | 0.76 | Top-5 recommendation quality |
| **Training Time** | 45s | 400 users, 747 articles |
| **Inference Time** | <200ms | Real-time serving |

