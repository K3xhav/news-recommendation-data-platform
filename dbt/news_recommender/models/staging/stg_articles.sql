with raw_articles_with_row_num as (
    select
        *,
        row_number() over (partition by url order by publishedAt desc) as row_num
    from {{ source('raw_news_data', 'raw_articles') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['url', 'publishedAt']) }} as article_id,
    title,
    url,
    safe_cast(publishedAt as timestamp) as published_at_ts,
    source,
    description,
    content,
    urlToImage as image_url,
    query_category,
    search_query  

from raw_articles_with_row_num
where
    row_num = 1
    and title is not null
    and title != '[Removed]'
