# Setup Guide

## 15-Minute Complete Setup

### Step 1: Prerequisites Check

```bash
# Verify you have these installed
docker --version
python3 --version
git --version
```

**Required Accounts:**

- **Google Cloud Account** (Free $300 credit)
- **NewsAPI Key** (Free from https://newsapi.org/)

### Step 2: Clone & Initial Setup

```bash
# Clone the repository
git clone https://github.com/K3xhav/news-recommender-system-public
cd news-recommender-system-public

# Create Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Google Cloud Setup (5 minutes)

```bash
# Login to Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create cloud resources
cd infra
terraform init
terraform apply
# Type 'yes' when prompted
cd ..
```

### Step 4: Data Pipeline Setup

```bash
# Start Kestra (data orchestration)
cd kestra
docker-compose up -d

# Wait 30 seconds, then open:
# http://localhost:8080
```

**In Kestra UI:**

1. Go to **"Secrets"** tab
2. Click **"Add Secret"**
3. **Name:** `NEWS_API_KEY`
4. **Value:** [Your NewsAPI key]
5. Click **"Submit"**

### Step 5: Generate Sample Data

```bash
# Generate training data
python scripts/generate_clicks.py

# Build database tables
cd dbt/news_recommender
dbt run
cd ../..
```

### Step 6: Launch Application

**Terminal 1 - Start Backend API:**

```bash
cd api
uvicorn main:app --reload
```

**Wait for this message:** `"Model trained and ready"`

**Terminal 2 - Start Web Interface:**

```bash
cd ui
streamlit run app.py
```

**Automatically opens:** http://localhost:8501

## You're Done!

**Access Your Application:**

- **Web App:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Kestra UI:** http://localhost:8080

## Quick Verification

1. **Open** http://localhost:8501
2. **Login** with any username/password
3. **Browse** articles in different categories
4. **Click** "Mark Read" on a few articles
5. **Check** "For You" tab for personalized recommendations

## Common Issues & Fixes

**"GCP Credentials Error"**

```bash
gcloud auth application-default login
```

**"Port Already in Use"**

- Change ports in `kestra/docker-compose.yml`
- Or stop other applications using ports 8080, 8000, 8501

**"No Articles Showing"**

- Verify NewsAPI key in Kestra secrets
- Run data ingestion flow manually in Kestra UI

**"Model Not Training"**

- Wait for API to fully start
- Check that `generate_clicks.py` ran successfully

```

```
