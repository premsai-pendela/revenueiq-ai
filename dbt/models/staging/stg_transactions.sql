-- Typed, sale-only staging view over the raw transaction lines.
-- Returns are excluded so downstream revenue models are net of refunds.
select
    "InvoiceNo"                        as invoice_no,
    "StockCode"                        as stock_code,
    "Description"                      as description,
    cast("Quantity"   as integer)      as quantity,
    cast("InvoiceDate" as timestamp)   as invoice_ts,
    cast("UnitPrice"  as double)       as unit_price,
    "CustomerID"                       as customer_id,
    "Country"                          as country,
    cast("TotalPrice" as double)       as total_price
from {{ source('raw', 'transactions') }}
where "TotalPrice" is not null
  and coalesce("IsReturn", false) = false
