-- Revenue and behavior rolled up per KMeans customer segment.
with clusters as (
    select
        "ClusterName"    as segment,
        "MonetaryValue"  as monetary,
        "Recency"        as recency,
        "Frequency"      as frequency
    from {{ source('raw', 'customer_clusters') }}
)
select
    segment,
    count(*)                                          as customers,
    round(sum(monetary), 2)                           as revenue,
    round(100.0 * sum(monetary) / sum(sum(monetary)) over (), 1) as revenue_pct,
    round(avg(recency), 1)                            as avg_recency,
    round(avg(frequency), 1)                          as avg_frequency
from clusters
group by segment
order by revenue desc
