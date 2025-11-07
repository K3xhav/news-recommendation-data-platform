# /ui/app.py

import streamlit as st
import requests
import pandas as pd
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta
import json
import hashlib
import time
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Tech News Portal",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
GCP_PROJECT_ID = "news-recommendation-473806"
BIGQUERY_LOCATION = "US"

# Your actual dataset and table names
DATASET_NAME = "news_recommender_dw"
ARTICLES_TABLE = "dim_articles"
INTERACTIONS_TABLE = "fact_user_clicks"

SERVICE_ACCOUNT_KEY_PATH = os.getenv("SERVICE_ACCOUNT_KEY_PATH")
API_BASE_URL = "http://127.0.0.1:8000"

# Updated category mapping with regex patterns
CATEGORY_MAPPING = {
    "Sports": [
        r'\b(sports|football|basketball|baseball|soccer|tennis|golf|nba|nfl|mlb|nhl|olympics|championship|game|match|player|team|coach|score|tournament)\b'
    ],
    "Politics": [
        r'\b(politics|government|election|congress|senate|white house|president|trump|biden|democrat|republican|vote|campaign|policy|law|legislation)\b'
    ],
    "Business": [
        r'\b(business|economy|stock|market|invest|finance|economic|company|corporation|earnings|profit|revenue|wall street|dow jones|nasdaq)\b'
    ],
    "Health & Science": [
        r'\b(health|medicine|medical|healthcare|hospital|doctor|patient|disease|treatment|vaccine|covid|virus|pandemic|fitness|nutrition|wellness|science|research|scientific|study|discovery)\b'
    ],
    "AI & Technology": [
        r'\b(artificial intelligence|ai\b|machine learning|deep learning|neural network|algorithm|chatgpt|gpt|openai|technology|tech|computer|software|hardware|internet|digital|app|application|startup|innovation)\b'
    ],
    "Entertainment": [
        r'\b(entertainment|movie|film|hollywood|celebrity|actor|actress|director|oscar|award|premiere|box office|streaming|netflix|disney)\b'
    ],
    "Gaming": [
        r'\b(gaming|video game|playstation|xbox|nintendo|steam|esports|gamer|game release|gameplay|console|pc gaming|mobile game)\b'
    ],
    "Environment": [
        r'\b(environment|climate|global warming|sustainability|renewable|solar|wind|energy|pollution|conservation|eco-friendly|green)\b'
    ]
}

# Standardized category names for consistency
STANDARD_CATEGORIES = [
    "Sports", "Politics", "Business", "Health & Science",
    "AI & Technology", "Entertainment", "Gaming", "Environment", "General"
]

# --- Service Account Setup ---
@st.cache_resource
def get_bigquery_client():
    """Initialize BigQuery client with error handling"""
    if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        st.error(f"❌ Service account key not found at: {SERVICE_ACCOUNT_KEY_PATH}")
        st.stop()
        return None

    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_PATH)
        client = bigquery.Client(
            project=GCP_PROJECT_ID,
            credentials=credentials,
            location=BIGQUERY_LOCATION,
        )
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize BigQuery client: {e}")
        st.stop()
        return None

client = get_bigquery_client()

# --- Session State ---
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'user_id': None,
        'user_interactions': [],
        'recommendations': [],
        'logged_in': False,
        'username': "",
        'articles_per_page': 12,
        'current_page': 1,
        'read_articles': set(),
        'selected_category': "All",
        'data_loaded': False,
        'user_history_loaded': False,
        'button_counter': 0,
        'clicked_articles': set(),  # Track clicked articles in this run
        'recommendation_source': ""  # Track how recommendations were generated
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Utility Functions ---
def clean_category_name(category):
    """Clean and standardize category names using pattern matching"""
    if not category or pd.isna(category):
        return "General"

    category = str(category).strip().lower()

    # Check each category pattern
    for standard_category, patterns in CATEGORY_MAPPING.items():
        for pattern in patterns:
            if re.search(pattern, category, re.IGNORECASE):
                return standard_category

    # Check for direct matches with standardized categories
    for standard_category in STANDARD_CATEGORIES:
        if standard_category.lower() in category:
            return standard_category

    # Default to General for unknown categories
    return "General"

def create_user_id(username):
    """Create consistent user ID from username"""
    return int(hashlib.sha256(username.encode()).hexdigest()[:8], 16) % 1000 + 1

def get_unique_button_key(article_id):
    """Generate unique button key to avoid duplicates"""
    st.session_state.button_counter += 1
    return f"btn_{article_id}_{st.session_state.button_counter}"

# --- Data Fetching Functions ---
@st.cache_data(ttl=1800)
def get_available_categories():
    """Get all available categories from database with cleaning"""
    try:
        sql = f"""
            SELECT DISTINCT query_category as category
            FROM `{GCP_PROJECT_ID}.{DATASET_NAME}.{ARTICLES_TABLE}`
            WHERE query_category IS NOT NULL
            AND query_category != ''
            ORDER BY category
        """
        df = client.query(sql).to_dataframe()

        if not df.empty:
            categories = [clean_category_name(cat) for cat in df['category'].tolist()]
            unique_categories = sorted(set(categories))
            return ["All"] + unique_categories
        else:
            return ["All"] + STANDARD_CATEGORIES

    except Exception as e:
        st.error(f"⚠️ Error fetching categories: {e}")
        return ["All"] + STANDARD_CATEGORIES

@st.cache_data(ttl=1800)
def get_articles_data(category="All", days_back=7, limit=200):
    """Get articles with proper error handling and category filtering"""
    try:
        base_sql = f"""
            SELECT
                article_id,
                title,
                url,
                query_category as original_category,
                published_at as published_date,
                description as summary,
                source,
                FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', published_at) as formatted_date
            FROM `{GCP_PROJECT_ID}.{DATASET_NAME}.{ARTICLES_TABLE}`
            WHERE DATE(published_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL @days_back DAY)
            AND title IS NOT NULL
            AND title != '[Removed]'
            AND url IS NOT NULL
        """

        params = [
            bigquery.ScalarQueryParameter("days_back", "INT64", days_back),
            bigquery.ScalarQueryParameter("limit", "INT64", limit)
        ]

        if category != "All":
            # Get both original category and cleaned category for filtering
            base_sql += " AND (LOWER(query_category) LIKE LOWER(@category_pattern) OR LOWER(query_category) LIKE LOWER(@category_pattern2))"
            params.append(bigquery.ScalarQueryParameter("category_pattern", "STRING", f"%{category}%"))
            params.append(bigquery.ScalarQueryParameter("category_pattern2", "STRING", f"%{category.lower()}%"))

        base_sql += " ORDER BY published_at DESC LIMIT @limit"

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        df = client.query(base_sql, job_config=job_config).to_dataframe()

        if not df.empty:
            df['category'] = df['original_category'].apply(clean_category_name)

            # If a specific category was requested, filter by cleaned category
            if category != "All":
                df = df[df['category'] == category]

            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Error fetching articles: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_trending_articles(limit=50):
    """Get trending articles based on user interactions"""
    try:
        sql = f"""
            SELECT
                a.article_id,
                a.title,
                a.url,
                a.query_category as original_category,
                a.published_at as published_date,
                a.description as summary,
                a.source,
                COUNT(f.article_id) as click_count,
                FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', a.published_at) as formatted_date
            FROM `{GCP_PROJECT_ID}.{DATASET_NAME}.{ARTICLES_TABLE}` a
            LEFT JOIN `{GCP_PROJECT_ID}.{DATASET_NAME}.{INTERACTIONS_TABLE}` f
                ON a.article_id = f.article_id
            WHERE DATE(a.published_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            AND a.title IS NOT NULL
            AND a.title != '[Removed]'
            GROUP BY
                a.article_id, a.title, a.url, a.query_category,
                a.published_at, a.description, a.source
            ORDER BY click_count DESC, a.published_at DESC
            LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        )

        df = client.query(sql, job_config=job_config).to_dataframe()

        if not df.empty:
            df['category'] = df['original_category'].apply(clean_category_name)
            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Error fetching trending articles: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_database_stats():
    """Get database statistics with error handling"""
    try:
        sql = f"""
            SELECT
                COUNT(*) as total_articles,
                COUNT(DISTINCT query_category) as total_categories,
                MIN(published_at) as oldest_article,
                MAX(published_at) as newest_article
            FROM `{GCP_PROJECT_ID}.{DATASET_NAME}.{ARTICLES_TABLE}`
            WHERE title IS NOT NULL AND title != '[Removed]'
        """
        stats_df = client.query(sql).to_dataframe()
        return stats_df.iloc[0] if not stats_df.empty else None
    except Exception as e:
        st.error(f"⚠️ Error fetching database stats: {e}")
        return None

# --- User Interaction Functions ---
def save_interaction_to_bigquery(user_id, article_id):
    """Directly save interaction to BigQuery"""
    try:
        # Check if interaction already exists
        check_sql = f"""
            SELECT COUNT(*) as count
            FROM `{GCP_PROJECT_ID}.{DATASET_NAME}.{INTERACTIONS_TABLE}`
            WHERE user_id = @user_id AND article_id = @article_id
        """

        check_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "INT64", user_id),
                bigquery.ScalarQueryParameter("article_id", "STRING", article_id)
            ]
        )

        check_result = client.query(check_sql, check_job_config).to_dataframe()

        if check_result.iloc[0]['count'] > 0:
            return "exists"  # Already exists in database

        # Insert new interaction
        insert_sql = f"""
            INSERT INTO `{GCP_PROJECT_ID}.{DATASET_NAME}.{INTERACTIONS_TABLE}`
            (user_id, article_id, click_timestamp)
            VALUES (@user_id, @article_id, CURRENT_TIMESTAMP())
        """

        insert_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "INT64", user_id),
                bigquery.ScalarQueryParameter("article_id", "STRING", article_id)
            ]
        )

        client.query(insert_sql, insert_job_config).result()
        return "saved"

    except Exception as e:
        st.error(f"❌ Error saving to BigQuery: {e}")
        return "error"

def record_article_read(user_id, article_id):
    """Record article read with proper database saving"""
    try:
        # Check if already recorded in this session
        if article_id in st.session_state.read_articles:
            return "already_read"

        # Store in session state immediately for UI feedback
        st.session_state.read_articles.add(article_id)
        st.session_state.user_interactions.append({
            'user_id': user_id,
            'article_id': article_id,
            'timestamp': datetime.now().isoformat()
        })

        # Try to save to BigQuery directly
        db_result = save_interaction_to_bigquery(user_id, article_id)

        if db_result == "saved":
            return "saved"
        elif db_result == "exists":
            return "exists"
        else:
            return "session_only"

    except Exception as e:
        st.error(f"❌ Error recording read: {e}")
        return "error"

def get_user_reading_history(user_id):
    """Get user's reading history from database"""
    try:
        sql = f"""
            SELECT
                a.article_id,
                a.title,
                a.query_category as original_category,
                a.source,
                a.published_at as article_date,
                f.click_timestamp as read_time,
                FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', f.click_timestamp) as formatted_time
            FROM `{GCP_PROJECT_ID}.{DATASET_NAME}.{INTERACTIONS_TABLE}` f
            JOIN `{GCP_PROJECT_ID}.{DATASET_NAME}.{ARTICLES_TABLE}` a
                ON f.article_id = a.article_id
            WHERE f.user_id = @user_id
            ORDER BY f.click_timestamp DESC
            LIMIT 100
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_id", "INT64", user_id)]
        )

        df = client.query(sql, job_config=job_config).to_dataframe()

        if not df.empty:
            df['category'] = df['original_category'].apply(clean_category_name)
            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"⚠️ Error fetching reading history: {e}")
        return pd.DataFrame()

def load_user_history_from_db(user_id):
    """Load user's read articles from database into session state"""
    try:
        sql = f"""
            SELECT DISTINCT article_id
            FROM `{GCP_PROJECT_ID}.{DATASET_NAME}.{INTERACTIONS_TABLE}`
            WHERE user_id = @user_id
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_id", "INT64", user_id)]
        )

        df = client.query(sql, job_config=job_config).to_dataframe()

        if not df.empty:
            db_read_articles = set(df['article_id'].tolist())
            st.session_state.read_articles.update(db_read_articles)
            return len(db_read_articles)
        return 0

    except Exception as e:
        st.error(f"⚠️ Error loading user history: {e}")
        return 0

# --- UI Components ---
def render_article_card_clean(article, user_id=None, index=0):
    """Render article card without debug information"""
    article_id = article['article_id']
    is_read = article_id in st.session_state.read_articles
    was_clicked = article_id in st.session_state.clicked_articles

    with st.container():
        # Card container with visual read indicator
        card_style = """
            <style>
            .article-card {
                border-left: 4px solid #4CAF50;
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 8px;
                background-color: #f8fff8;
                border: 1px solid #e0e0e0;
            }
            .article-card-unread {
                border-left: 4px solid #e0e0e0;
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 8px;
                background-color: white;
                border: 1px solid #f0f0f0;
            }
            </style>
        """
        st.markdown(card_style, unsafe_allow_html=True)

        card_class = "article-card" if is_read else "article-card-unread"
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

        # Header with title and category
        col1, col2 = st.columns([3, 1])
        with col1:
            read_indicator = "✅ " if is_read else ""
            st.markdown(f"### {read_indicator}{article['title']}")
        with col2:
            if 'category' in article and article['category']:
                # Color code categories with emojis
                category_colors = {
                    "Sports": "🏈",
                    "Politics": "🏛️",
                    "Business": "💼",
                    "Health & Science": "⚕️",
                    "AI & Technology": "🤖",
                    "Entertainment": "🎬",
                    "Gaming": "🎮",
                    "Environment": "🌍",
                    "General": "📰"
                }
                emoji = category_colors.get(article['category'], "📰")
                st.info(f"{emoji} {article['category']}")

        # Article details
        if 'summary' in article and pd.notna(article['summary']) and article['summary']:
            summary = str(article['summary'])
            if len(summary) > 200:
                summary = summary[:200] + "..."
            st.write(summary)

        # Metadata and actions
        meta_col1, meta_col2, meta_col3 = st.columns([2, 1, 1])

        with meta_col1:
            if 'formatted_date' in article and pd.notna(article['formatted_date']):
                st.caption(f"📅 {article['formatted_date']}")
            if 'source' in article and pd.notna(article['source']) and article['source']:
                st.caption(f"📰 {article['source']}")

        with meta_col2:
            st.markdown(f"[🔗 Read Full Article]({article['url']})", unsafe_allow_html=True)

        with meta_col3:
            # Create UNIQUE form key using article_id AND index
            form_key = f"form_{article_id}_{index}"

            if is_read or was_clicked:
                # Create a unique key for disabled buttons too
                disabled_key = f"disabled_{article_id}_{index}_{int(time.time() * 1000)}"
                st.button("✅ Read", key=disabled_key, disabled=True)
            else:
                with st.form(key=form_key):
                    submitted = st.form_submit_button("📖 Mark Read", use_container_width=True)

                    if submitted:
                        st.session_state.clicked_articles.add(article_id)
                        result = record_article_read(user_id, article_id)

                        if result == "saved":
                            st.success("✓ Article saved to your history!")
                        elif result == "exists":
                            st.info("✓ Already in your history!")
                        elif result == "session_only":
                            st.warning("✓ Saved in session")
                        elif result == "already_read":
                            st.info("✓ Already marked as read!")
                        else:
                            st.error("❌ Failed to mark as read")

                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with navigation and stats"""
    with st.sidebar:
        st.header("🔐 Account")

        if not st.session_state.logged_in:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                if st.form_submit_button("Login", use_container_width=True):
                    if username and password:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_id = create_user_id(username)
                        st.session_state.current_page = 1

                        # Load user's history from database
                        db_count = load_user_history_from_db(st.session_state.user_id)
                        if db_count > 0:
                            st.success(f"Loaded {db_count} previously read articles!")
                        st.rerun()
                    else:
                        st.error("Please enter username and password")
        else:
            st.success(f"👋 Welcome, {st.session_state.username}!")
            if st.button("Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        if st.session_state.logged_in:
            st.header("📂 Categories")
            categories = get_available_categories()
            selected = st.selectbox(
                "Choose category:",
                categories,
                index=categories.index(st.session_state.selected_category) if st.session_state.selected_category in categories else 0,
                key="category_selector"
            )
            st.session_state.selected_category = selected

            st.header("⚙️ Settings")
            st.session_state.articles_per_page = st.slider(
                "Articles per page:",
                min_value=6,
                max_value=24,
                value=st.session_state.articles_per_page,
                step=6
            )

            st.header("📊 Statistics")
            total_read = len(st.session_state.read_articles)
            st.metric("Articles Read", total_read)

            stats = get_database_stats()
            if stats is not None:
                st.metric("Total Articles", f"{stats['total_articles']:,}")
                if pd.notna(stats['newest_article']):
                    days_old = (datetime.now().date() - stats['newest_article'].date()).days
                    st.metric("Latest Article", f"{days_old}d ago")

            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.session_state.clicked_articles.clear()
                st.rerun()

# --- Main App ---
def main():
    st.title("📰 News Recommender")
    st.markdown("### Discover personalized news based on your interests")

    # Render sidebar
    render_sidebar()

    if not st.session_state.logged_in:
        show_login_screen()
        return

    # Main content for logged-in users
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Browse News", "🔥 Trending", "🎯 For You", "📚 My History"])

    with tab1:
        show_news_browse()

    with tab2:
        show_trending()

    with tab3:
        show_recommendations()

    with tab4:
        show_history()

def show_login_screen():
    """Show login screen for non-logged-in users"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Welcome to Your Personalized News Portal

        **Features:**
        - 📖 **Smart Article Browsing** - Browse by categories that matter to you
        - 🔥 **Trending News** - See what's popular in the community
        - 🎯 **Personalized Recommendations** - Get articles tailored to your interests
        - 📚 **Reading History** - Track what you've read

        **How it works:**
        1. Login with any username/password
        2. Browse articles by category
        3. Mark articles as read to improve recommendations
        4. Get personalized suggestions based on your reading habits
        """)

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2965/2965358.png", width=150)
        st.info("""
        **Demo Mode:**
        - Use any username/password
        - Your reading history is saved
        - Get better recommendations over time
        """)

def show_news_browse():
    """Show main news browsing interface"""
    st.header("📖 Browse News")

    if st.session_state.selected_category != "All":
        st.subheader(f"Category: {st.session_state.selected_category}")

    # Load articles
    with st.spinner("Loading articles..."):
        articles_df = get_articles_data(
            category=st.session_state.selected_category,
            days_back=7,
            limit=100
        )

    if articles_df.empty:
        st.warning("""
        No articles found. This could be because:
        - No recent articles in this category
        - News fetching needs to run
        - Try selecting "All" categories or check the Trending tab
        """)
        return

    # Pagination
    total_articles = len(articles_df)
    total_pages = max(1, (total_articles + st.session_state.articles_per_page - 1) // st.session_state.articles_per_page)

    # Pagination controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.write(f"**{total_articles} articles found**")
    with col2:
        if st.button("◀ Prev", disabled=st.session_state.current_page <= 1):
            st.session_state.current_page -= 1
            st.rerun()
    with col3:
        st.write(f"Page {st.session_state.current_page}/{total_pages}")
    with col4:
        if st.button("Next ▶", disabled=st.session_state.current_page >= total_pages):
            st.session_state.current_page += 1
            st.rerun()

    # Display articles
    start_idx = (st.session_state.current_page - 1) * st.session_state.articles_per_page
    end_idx = min(start_idx + st.session_state.articles_per_page, total_articles)
    current_articles = articles_df.iloc[start_idx:end_idx]

    # Read statistics
    read_count = sum(1 for article_id in current_articles['article_id']
                    if article_id in st.session_state.read_articles)
    if read_count > 0:
        st.info(f"📖 You've read {read_count} of {len(current_articles)} articles on this page")

    # Render articles
    for i, (_, article) in enumerate(current_articles.iterrows()):
        render_article_card_clean(article, st.session_state.user_id, i)

def show_trending():
    """Show trending articles"""
    st.header("🔥 Trending Articles")
    st.caption("Most popular articles based on community reads")

    with st.spinner("Loading trending articles..."):
        trending_df = get_trending_articles(limit=30)

    if trending_df.empty:
        st.warning("No trending articles found. This usually means:")
        st.info("""
        - Not enough user interactions yet
        - Articles are too new to have trending data
        - Try the Browse News tab for all available articles
        """)
        return

    st.success(f"Found {len(trending_df)} trending articles!")

    for i, (_, article) in enumerate(trending_df.iterrows()):
        render_article_card_clean(article, st.session_state.user_id, i)

def show_recommendations():
    """Show personalized recommendations with explanation"""
    st.header("🎯 Your Recommendations")
    st.caption("Articles tailored to your reading history")

    # Explanation of how recommendations work
    # with st.expander("ℹ️ How recommendations work"):
    #     st.markdown("""
    #     **Our Recommendation System:**

    #     - **For New Users**: We show trending articles that are popular with other readers
    #     - **For Returning Users**: We analyze your reading history to find articles in your favorite categories
    #     - **Smart Filtering**: We exclude articles you've already read
    #     - **Fresh Content**: We prioritize recent articles from the last 7 days

    #     **To get better recommendations:**
    #     1. Read articles from different categories
    #     2. Mark articles as read to build your profile
    #     3. The more you read, the smarter recommendations become
    #     """)

    # Check if user has reading history
    history_df = get_user_reading_history(st.session_state.user_id)
    has_history = not history_df.empty

    if has_history:
        # Show user's reading profile
        top_categories = history_df['category'].value_counts().head(3)
        st.write("**Your Reading Profile:**")
        for category, count in top_categories.items():
            st.write(f"• {category}: {count} articles")

    if st.button("🔄 Get Recommendations", type="primary"):
        with st.spinner("Analyzing your preferences..."):
            recommendation_source = ""

            if has_history:
                # Find top categories from history
                top_categories = history_df['category'].value_counts().head(3).index.tolist()
                recommendation_source = f"Based on your interest in: {', '.join(top_categories)}"

                # Get articles from preferred categories
                recommended_articles = []
                for category in top_categories:
                    category_articles = get_articles_data(category=category, limit=10)
                    if not category_articles.empty:
                        # Filter out already read articles and convert to dict
                        for _, article in category_articles.iterrows():
                            if article['article_id'] not in st.session_state.read_articles:
                                article_dict = article.to_dict()
                                recommended_articles.append(article_dict)

                # Remove duplicates by article_id using a set
                seen_ids = set()
                unique_articles = []
                for article in recommended_articles:
                    article_id = article['article_id']
                    if article_id not in seen_ids:
                        seen_ids.add(article_id)
                        unique_articles.append(article)

                # LIMIT TO 5 ARTICLES and ensure they're unique
                st.session_state.recommendations = unique_articles[:5]

            else:
                # Fallback: show trending articles for new users
                recommendation_source = "Based on trending articles (new user)"
                trending_df = get_trending_articles(limit=10)
                if not trending_df.empty:
                    # Remove duplicates using set
                    seen_ids = set()
                    unique_articles = []
                    for _, article in trending_df.iterrows():
                        article_id = article['article_id']
                        if article_id not in seen_ids:
                            seen_ids.add(article_id)
                            article_dict = article.to_dict()
                            unique_articles.append(article_dict)
                    # LIMIT TO 5 ARTICLES
                    st.session_state.recommendations = unique_articles[:5]
                else:
                    # Final fallback: show recent articles
                    recommendation_source = "Based on recent articles"
                    recent_df = get_articles_data(limit=10)
                    if not recent_df.empty:
                        # Remove duplicates using set
                        seen_ids = set()
                        unique_articles = []
                        for _, article in recent_df.iterrows():
                            article_id = article['article_id']
                            if article_id not in seen_ids:
                                seen_ids.add(article_id)
                                article_dict = article.to_dict()
                                unique_articles.append(article_dict)
                        # LIMIT TO 5 ARTICLES
                        st.session_state.recommendations = unique_articles[:5]
                    else:
                        st.session_state.recommendations = []

            # Store recommendation source for display
            st.session_state.recommendation_source = recommendation_source

    # Display recommendations
    if st.session_state.recommendations:
        # Show recommendation source
        if hasattr(st.session_state, 'recommendation_source'):
            st.info(f"💡 {st.session_state.recommendation_source}")

        st.success(f"Found {len(st.session_state.recommendations)} recommendations for you!")

        # Display recommendations - limited to 5
        for i, article in enumerate(st.session_state.recommendations):
            render_article_card_clean(article, st.session_state.user_id, i)
    else:
        st.info("""
        Click the button above to get personalized recommendations!

        **Tips for better recommendations:**
        - Read articles from different categories
        - Mark articles as read to build your profile
        - The more you read, the better recommendations get
        """)

def show_history():
    """Show user's reading history"""
    st.header("📚 Your Reading History")

    # Show both session and database history
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("This Session")
        if st.session_state.user_interactions:
            st.success(f"📖 Read {len(st.session_state.user_interactions)} articles this session")
            for interaction in reversed(st.session_state.user_interactions[-10:]):
                st.write(f"• Read at {interaction['timestamp'][11:19]}")
        else:
            st.info("No articles read in this session yet")

    with col2:
        st.subheader("Database History")
        history_df = get_user_reading_history(st.session_state.user_id)

        if not history_df.empty:
            total_db_reads = len(history_df)
            st.success(f"💾 {total_db_reads} articles in database")

            # Show most recent reads
            st.write("**Recently Read:**")
            for _, article in history_df.head(5).iterrows():
                st.write(f"• {article['title'][:50]}... - {article['formatted_time']}")
        else:
            st.info("No history in database yet")

    # Combined statistics
    st.subheader("📊 Reading Insights")

    total_read = len(st.session_state.read_articles)
    history_df = get_user_reading_history(st.session_state.user_id)

    if not history_df.empty:
        col1, col2, col3 = st.columns(3)

        with col1:
            favorite_category = history_df['category'].mode()[0] if not history_df.empty else "None"
            st.metric("Favorite Category", favorite_category)

        with col2:
            total_categories = history_df['category'].nunique()
            st.metric("Categories Read", total_categories)

        with col3:
            first_read = history_df['read_time'].min().strftime('%b %d') if not history_df.empty else "Never"
            st.metric("First Read", first_read)

        # Category distribution
        st.write("**Category Distribution:**")
        category_counts = history_df['category'].value_counts()
        for category, count in category_counts.items():
            st.write(f"• {category}: {count} articles")

    else:
        st.info("""
        Start reading articles to build your history!

        **Benefits of building your history:**
        - Better personalized recommendations
        - Track your reading habits over time
        - Discover your favorite topics

        Go to the **Browse News** tab to start reading!
        """)

# --- Footer ---
def render_footer():
    st.markdown("---")
    stats = get_database_stats()
    total_articles = stats['total_articles'] if stats is not None else "Unknown"

    # Get database read count
    db_read_count = 0
    if st.session_state.logged_in:
        history_df = get_user_reading_history(st.session_state.user_id)
        db_read_count = len(history_df) if not history_df.empty else 0

    st.caption(
        f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"👤 {st.session_state.username if st.session_state.logged_in else 'Not logged in'} | "
        f"📚 {total_articles} articles | "
        f"📖 {len(st.session_state.read_articles)} read (session) | "
        f"💾 {db_read_count} read (database)"
    )

# Run the app
if __name__ == "__main__":
    main()
    render_footer()
