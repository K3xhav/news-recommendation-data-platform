# /ml/train_model.py

import pandas as pd
from google.cloud import bigquery
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np

# --- Configuration ---
GCP_PROJECT_ID = "news-recommendation-473806"
BIGQUERY_DATASET = "news_recommender_dw"
BIGQUERY_FACT_TABLE = "fact_user_clicks"

# --- 1. Load Data from BigQuery ---
print("📚 Loading user click data from BigQuery...")
client = bigquery.Client(project=GCP_PROJECT_ID)
sql = f"SELECT user_id, article_id FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_FACT_TABLE}`"
df = client.query(sql).to_dataframe()
print(f"✅ Loaded {len(df)} click events.")

# --- 2. Create the User-Item Matrix ---
print("... Creating user-item interaction matrix...")
# Create unique integer IDs for users and articles for matrix indexing
df["user_idx"] = df["user_id"].astype("category").cat.codes
df["article_idx"] = df["article_id"].astype("category").cat.codes

# Create a sparse matrix: a grid where rows are users, columns are articles,
# and a '1' means the user clicked that article.
user_item_matrix = csr_matrix(
    (np.ones(df.shape[0]), (df["user_idx"], df["article_idx"]))
)
print("✅ Matrix created.")
print(
    f"Matrix shape: {user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} articles"
)

# --- 3. Train the SVD Model ---
print("🤖 Training SVD model...")
# This algorithm will discover 50 "hidden genres" or taste patterns from the click data.
N_LATENT_FACTORS = 50
svd = TruncatedSVD(n_components=N_LATENT_FACTORS, random_state=42)

# The 'fit_transform' command is where the model "learns" the user taste profiles.
user_factors = svd.fit_transform(user_item_matrix)
print("✅ Model training complete.")

# --- 4. Make a Sample Recommendation ---
print("\n---  SAMPLE RECOMMENDATION ---")
# Let's find recommendations for the first user (user_idx = 0)
target_user_idx = 0
target_user_profile = user_factors[target_user_idx]

# Calculate the similarity score between our target user and all articles
scores = target_user_profile.dot(svd.components_)

# Get the original article_ids back from their index
article_mapping = (
    df[["article_idx", "article_id"]].drop_duplicates().set_index("article_idx")
)

# Find the top 5 articles with the highest scores
top_5_article_indices = np.argsort(scores)[::-1][:5]
recommendations = article_mapping.loc[top_5_article_indices]

print(
    f"Top 5 recommendations for user_id: {df[df['user_idx'] == target_user_idx]['user_id'].iloc[0]}"
)
print(recommendations)
