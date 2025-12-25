{{
  config(
    materialized='view'
  )
}}

SELECT
  user_id,
  article_id,
  click_timestamp,  
  CURRENT_TIMESTAMP() as loaded_at
FROM {{ source('raw_news_data', 'raw_clicks') }}
WHERE user_id IS NOT NULL
  AND article_id IS NOT NULL
