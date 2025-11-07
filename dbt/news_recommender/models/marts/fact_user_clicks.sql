{{
    config(
        materialized='table'
    )
}}

select
    sc.user_id,
    da.article_id,
    sc.click_timestamp,
    da.published_at as published_date,
    da.query_category,
    da.search_query,  
    da.source,
    da.category_group,
    count(*) as click_count
from {{ ref('stg_clicks') }} as sc
inner join {{ ref('dim_articles') }} as da
    on sc.article_id = da.article_id
group by 1, 2, 3, 4, 5, 6, 7, 8
