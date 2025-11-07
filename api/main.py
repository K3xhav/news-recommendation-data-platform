# /api/main.py 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from google.cloud import bigquery
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
import json
from datetime import datetime
import threading
from typing import List, Optional


# Request model for interactions
class Interaction(BaseModel):
    user_id: int
    article_id: str
    interaction_type: str = "click"
    timestamp: Optional[str] = None


# Global variables to hold the trained model and data
svd_model = None
user_factors = None
user_mapping = None
article_mapping = None
user_item_matrix = None
reverse_article_mapping = None

# REMOVED: interaction_buffer and buffer_lock - we'll process interactions immediately

# BigQuery configuration
GCP_PROJECT_ID = "news-recommendation-473806"
BIGQUERY_DATASET = "news_recommender_dw"
BIGQUERY_FACT_TABLE = "fact_user_clicks"
BIGQUERY_ARTICLES_TABLE = "dim_articles"

app = FastAPI(
    title="News Recommender API",
    description="An API to serve personalized news recommendations with real-time updates.",
    version="1.0.0",
)

# Add CORS middleware to allow Streamlit frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_bigquery_client():
    """Initialize BigQuery client"""
    return bigquery.Client(project=GCP_PROJECT_ID)


def load_training_data():
    """Load user-article interactions from BigQuery for training"""
    client = initialize_bigquery_client()

    sql = f"""
        SELECT
            user_id,
            article_id,
            COUNT(*) as interaction_count
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}`
        GROUP BY user_id, article_id
    """
    df = client.query(sql).to_dataframe()
    print(f"📊 Loaded {len(df)} interactions for training")
    return df


def train_svd_model():
    """Train the SVD model with current data"""
    global svd_model, user_factors, user_mapping, article_mapping, user_item_matrix, reverse_article_mapping

    print("🔄 Training SVD model...")

    df = load_training_data()

    if df.empty:
        print("⚠️ No training data available")
        return

    # Create mappings
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["article_idx"] = df["article_id"].astype("category").cat.codes

    # Create user-item matrix with interaction counts
    user_item_matrix = csr_matrix(
        (df["interaction_count"], (df["user_idx"], df["article_idx"]))
    )

    # Train model
    N_LATENT_FACTORS = min(50, min(user_item_matrix.shape) - 1)
    if N_LATENT_FACTORS < 2:
        N_LATENT_FACTORS = 2

    svd_model = TruncatedSVD(n_components=N_LATENT_FACTORS, random_state=42)
    user_factors = svd_model.fit_transform(user_item_matrix)

    # Create mappings
    user_mapping = (
        df[["user_idx", "user_id"]].drop_duplicates().set_index("user_id")["user_idx"]
    )
    article_mapping = (
        df[["article_idx", "article_id"]]
        .drop_duplicates()
        .set_index("article_idx")["article_id"]
    )

    # Create reverse mapping for article indices
    reverse_article_mapping = (
        df[["article_idx", "article_id"]]
        .drop_duplicates()
        .set_index("article_id")["article_idx"]
    )

    print(
        f"✅ Model trained with {len(user_mapping)} users and {len(article_mapping)} articles"
    )
    print(
        f"📈 Matrix shape: {user_item_matrix.shape}, Latent factors: {N_LATENT_FACTORS}"
    )


def get_popular_articles(limit: int = 10):
    """Get popular articles as fallback recommendations"""
    client = initialize_bigquery_client()

    sql = f"""
        SELECT
            a.article_id,
            COUNT(f.article_id) as click_count
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_ARTICLES_TABLE}` a
        LEFT JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}` f
            ON a.article_id = f.article_id
        WHERE DATE(a.published_at_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        GROUP BY a.article_id
        ORDER BY click_count DESC, a.published_at_ts DESC
        LIMIT {limit}
    """

    df = client.query(sql).to_dataframe()
    return df["article_id"].tolist()


# REMOVED: store_interaction_batch and process_interaction_buffer functions


@app.on_event("startup")
def initialize_api():
    """
    Runs ONCE when the API server starts.
    Loads data from BigQuery and trains the SVD model.
    """
    print("🚀 API starting up: Loading data and training model...")

    try:
        train_svd_model()
        print("✅ Model training completed successfully")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        print("🔄 API will start with fallback recommendations")

    print("✅ API startup completed")


@app.get("/")
def read_root():
    """Health check endpoint"""
    model_status = "trained" if svd_model is not None else "not trained"
    return {
        "status": "healthy",
        "model_status": model_status,
        "users_loaded": len(user_mapping) if user_mapping is not None else 0,
        "articles_loaded": len(article_mapping) if article_mapping is not None else 0,
    }


@app.post("/interactions")
async def record_interaction(interaction: dict):
    """Record user interactions for recommendations - IMPROVED VERSION"""
    try:
        client = initialize_bigquery_client()

        # Validate required fields
        if not interaction.get("user_id") or not interaction.get("article_id"):
            raise HTTPException(
                status_code=400, detail="user_id and article_id are required"
            )

        # Check if this interaction already exists (prevent duplicates)
        check_sql = f"""
            SELECT COUNT(*) as count
            FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}`
            WHERE user_id = @user_id
            AND article_id = @article_id
            AND DATE(click_timestamp) = CURRENT_DATE()
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "user_id", "INT64", interaction["user_id"]
                ),
                bigquery.ScalarQueryParameter(
                    "article_id", "STRING", interaction["article_id"]
                ),
            ]
        )

        existing_count = (
            client.query(check_sql, job_config=job_config)
            .to_dataframe()
            .iloc[0]["count"]
        )

        if existing_count == 0:
            # Insert new interaction
            insert_sql = f"""
                INSERT INTO `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}`
                (user_id, article_id, click_timestamp, interaction_type)
                VALUES (@user_id, @article_id, CURRENT_TIMESTAMP(), @interaction_type)
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "user_id", "INT64", interaction["user_id"]
                    ),
                    bigquery.ScalarQueryParameter(
                        "article_id", "STRING", interaction["article_id"]
                    ),
                    bigquery.ScalarQueryParameter(
                        "interaction_type",
                        "STRING",
                        interaction.get("interaction_type", "click"),
                    ),
                ]
            )

            # Execute the insert and wait for completion
            query_job = client.query(insert_sql, job_config=job_config)
            query_job.result()  # Wait for the query to complete

            print(
                f"✅ SUCCESS: Recorded interaction - User {interaction['user_id']} -> Article {interaction['article_id']}"
            )

            # Trigger model retraining in background after new interactions
            threading.Thread(target=retrain_model_with_new_data, daemon=True).start()

            return {
                "status": "success",
                "message": "Interaction recorded successfully",
                "action": "created",
            }
        else:
            print(
                f"⚠️ EXISTS: Interaction already exists - User {interaction['user_id']} -> Article {interaction['article_id']}"
            )
            return {
                "status": "success",
                "message": "Interaction already exists",
                "action": "exists",
            }

    except Exception as e:
        print(f"❌ ERROR recording interaction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record interaction: {str(e)}"
        )


@app.get("/debug/interactions/{user_id}")
def get_user_interactions_debug(user_id: int):
    """Debug endpoint to check user interactions"""
    client = initialize_bigquery_client()

    try:
        sql = f"""
            SELECT
                user_id,
                article_id,
                click_timestamp,
                interaction_type
            FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}`
            WHERE user_id = @user_id
            ORDER BY click_timestamp DESC
            LIMIT 10
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "INT64", user_id)
            ]
        )

        df = client.query(sql, job_config=job_config).to_dataframe()
        return {
            "user_id": user_id,
            "interaction_count": len(df),
            "interactions": df.to_dict("records") if not df.empty else [],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get user interactions: {str(e)}"
        )


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, limit: int = 5):
    """
    Takes a user_id and returns a list of recommended article_ids.
    """
    # Validate user_id
    try:
        user_id = int(user_id)
        if not (1 <= user_id <= 500):
            raise HTTPException(
                status_code=400, detail="User ID must be between 1 and 500"
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="User ID must be a valid integer")

    # Check if model is trained
    if svd_model is None or user_mapping is None:
        print("⚠️ Model not trained, returning popular articles as fallback")
        popular_articles = get_popular_articles(limit * 2)  # Get extra for filtering
        return {
            "user_id": user_id,
            "recommendations": popular_articles[:limit],
            "message": "Popular articles (model not ready)",
        }

    # Check if user exists in training data
    if user_id not in user_mapping.index:
        print(f"⚠️ User {user_id} not in training data, returning popular articles")
        popular_articles = get_popular_articles(limit * 2)
        return {
            "user_id": user_id,
            "recommendations": popular_articles[:limit],
            "message": "Popular articles (new user)",
        }

    try:
        # Get user index
        user_idx = user_mapping[user_id]
        user_profile = user_factors[user_idx]

        # Calculate scores for all articles
        scores = user_profile.dot(svd_model.components_)

        # Get articles the user has already clicked
        clicked_articles_indices = user_item_matrix[user_idx].nonzero()[1]

        # Apply penalties: reduce scores for clicked articles
        scores[clicked_articles_indices] = -10  # Strong penalty for clicked articles

        # Get top recommendations
        top_article_indices = np.argsort(scores)[::-1][
            : limit * 2
        ]  # Get extra for filtering

        # Filter out invalid indices and map to article IDs
        recommendations = []
        for idx in top_article_indices:
            if idx in article_mapping.index:
                article_id = article_mapping.loc[idx]
                if article_id not in recommendations:
                    recommendations.append(article_id)
                if len(recommendations) >= limit:
                    break

        # If we don't have enough recommendations, supplement with popular articles
        if len(recommendations) < limit:
            popular_articles = get_popular_articles(limit * 2)
            for article_id in popular_articles:
                if article_id not in recommendations:
                    recommendations.append(article_id)
                if len(recommendations) >= limit:
                    break

        return {
            "user_id": user_id,
            "recommendations": recommendations[:limit],
            "message": "Personalized recommendations",
        }

    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        # Fallback to popular articles
        popular_articles = get_popular_articles(limit)
        return {
            "user_id": user_id,
            "recommendations": popular_articles,
            "message": "Popular articles (error in personalization)",
        }


@app.post("/retrain")
async def retrain_model():
    """Manually trigger model retraining"""
    try:
        threading.Thread(target=train_svd_model, daemon=True).start()
        return {
            "status": "success",
            "message": "Model retraining started in background",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start retraining: {str(e)}"
        )


@app.get("/user/{user_id}/stats")
def get_user_stats(user_id: int):
    """Get user statistics and reading history"""
    client = initialize_bigquery_client()

    try:
        # Get user interaction count
        sql_interactions = f"""
            SELECT COUNT(*) as interaction_count
            FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}`
            WHERE user_id = @user_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "INT64", user_id)
            ]
        )
        interaction_df = client.query(
            sql_interactions, job_config=job_config
        ).to_dataframe()
        interaction_count = (
            interaction_df.iloc[0]["interaction_count"]
            if not interaction_df.empty
            else 0
        )

        # Get favorite categories
        sql_categories = f"""
            SELECT
                a.query_category as category,
                COUNT(*) as count
            FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}` f
            JOIN `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_ARTICLES_TABLE}` a
                ON f.article_id = a.article_id
            WHERE f.user_id = @user_id
            GROUP BY a.query_category
            ORDER BY count DESC
            LIMIT 5
        """
        category_df = client.query(sql_categories, job_config=job_config).to_dataframe()

        return {
            "user_id": user_id,
            "total_interactions": interaction_count,
            "favorite_categories": (
                category_df.to_dict("records") if not category_df.empty else []
            ),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get user stats: {str(e)}"
        )


def retrain_model_with_new_data():
    """Retrain model with new data (called in background)"""
    print("🔄 Retraining model with new data...")
    try:
        train_svd_model()
        print("✅ Model retraining completed successfully")
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")


# Add this endpoint for model information
@app.get("/model/info")
def get_model_info():
    """Get information about the current model state"""
    if svd_model is None:
        return {"status": "not_trained"}

    return {
        "status": "trained",
        "users": len(user_mapping) if user_mapping is not None else 0,
        "articles": len(article_mapping) if article_mapping is not None else 0,
        "matrix_shape": (
            user_item_matrix.shape if user_item_matrix is not None else None
        ),
        "components": svd_model.n_components if svd_model is not None else 0,
    }


# Add endpoint to check specific user interactions
@app.get("/user/{user_id}/interactions")
def get_user_interactions(user_id: int):
    """Get all interactions for a specific user"""
    client = initialize_bigquery_client()

    try:
        sql = f"""
            SELECT
                user_id,
                article_id,
                click_timestamp,
                interaction_type
            FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}`
            WHERE user_id = @user_id
            ORDER BY click_timestamp DESC
            LIMIT 50
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "INT64", user_id)
            ]
        )

        df = client.query(sql, job_config=job_config).to_dataframe()
        return {
            "user_id": user_id,
            "interactions": df.to_dict("records") if not df.empty else [],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get user interactions: {str(e)}"
        )
