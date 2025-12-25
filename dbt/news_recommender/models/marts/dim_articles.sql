{{
  config(
    materialized='table',
    unique_key='article_id'
  )
}}

SELECT
  article_id,
  title,
  url,
  published_at_ts as published_at,
  source,
  description,
  content,
  image_url,
  query_category,
  search_query,  
  LOWER(query_category) AS category_normalized,
  CASE
    WHEN LOWER(query_category) LIKE '%sports%' THEN 'Sports'
    WHEN LOWER(query_category) LIKE '%politics%' OR LOWER(query_category) LIKE '%government%' THEN 'Politics'
    WHEN LOWER(query_category) LIKE '%business%' OR LOWER(query_category) LIKE '%economy%' THEN 'Business'
    WHEN LOWER(query_category) LIKE '%health%' OR LOWER(query_category) LIKE '%science%' OR LOWER(query_category) LIKE '%medicine%' THEN 'Health & Science'
    WHEN LOWER(query_category) LIKE '%ai%' OR LOWER(query_category) LIKE '%technology%' OR LOWER(query_category) LIKE '%computer%' THEN 'Technology'
    WHEN LOWER(query_category) LIKE '%entertainment%' OR LOWER(query_category) LIKE '%movie%' OR LOWER(query_category) LIKE '%hollywood%' THEN 'Entertainment'
    WHEN LOWER(query_category) LIKE '%gaming%' OR LOWER(query_category) LIKE '%video game%' THEN 'Gaming'
    WHEN LOWER(query_category) LIKE '%environment%' OR LOWER(query_category) LIKE '%climate%' OR LOWER(query_category) LIKE '%sustainability%' THEN 'Environment'
    ELSE 'General News'
  END AS category_group
FROM {{ ref('stg_articles') }}
