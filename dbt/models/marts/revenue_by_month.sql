-- Monthly revenue, orders, and active customers (net of returns).
select
    cast(date_trunc('month', invoice_ts) as date) as month,
    count(distinct invoice_no)                    as orders,
    count(distinct customer_id)                   as customers,
    round(sum(total_price), 2)                    as revenue
from {{ ref('stg_transactions') }}
group by 1
order by 1
