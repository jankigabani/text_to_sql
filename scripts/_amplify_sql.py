"""SQL used by 02_amplify.py, kept separate so the generator statements stay readable.

Performance note (this mattered): the obvious way to pick a random dimension row per
generated row is `OFFSET (g * k) % n LIMIT 1`, but OFFSET scans, so inserting a million
rows becomes O(n²) and never finishes. Instead we materialise each dimension into a dense
0..n-1 keyspace once, index it, and join on an arithmetic key — an index lookup per row.
"""
from __future__ import annotations

DIM_TABLES = """
DROP TABLE IF EXISTS dim_product_k, dim_customer_k, dim_person_k, dim_location_k;

CREATE TEMP TABLE dim_product_k AS
SELECT (row_number() OVER (ORDER BY productid) - 1)::bigint AS k,
       productid, COALESCE(listprice, 10.0)::numeric(12,4) AS listprice
  FROM production.product WHERE COALESCE(listprice, 0) > 0;
CREATE UNIQUE INDEX ON dim_product_k(k);

CREATE TEMP TABLE dim_customer_k AS
SELECT (row_number() OVER (ORDER BY customerid) - 1)::bigint AS k, customerid
  FROM sales.customer;
CREATE UNIQUE INDEX ON dim_customer_k(k);

CREATE TEMP TABLE dim_person_k AS
SELECT (row_number() OVER (ORDER BY businessentityid) - 1)::bigint AS k, businessentityid
  FROM person.person;
CREATE UNIQUE INDEX ON dim_person_k(k);

CREATE TEMP TABLE dim_location_k AS
SELECT (row_number() OVER (ORDER BY locationid) - 1)::bigint AS k, locationid
  FROM production.location;
CREATE UNIQUE INDEX ON dim_location_k(k);

ANALYZE dim_product_k; ANALYZE dim_customer_k; ANALYZE dim_person_k; ANALYZE dim_location_k;
"""

FACT_SALES = """
INSERT INTO warehouse.fact_sales
    (sale_id, order_date, product_id, customer_id, territory_id, channel_id,
     quantity, unit_price, discount_pct, line_total, is_returned, status)
SELECT g,
       DATE '2019-01-01' + (((g * 7919) %% 2190))::int,
       p.productid,
       c.customerid,
       (1 + ((g * 31) %% 10))::int,
       (1 + ((g * 7) %% 5))::smallint,
       (1 + ((g * 13) %% 9))::int,
       ROUND(p.listprice * (0.8 + ((g %% 40) / 100.0)), 4),
       ((g %% 5) * 0.05)::numeric(5,4),
       ROUND(p.listprice * (0.8 + ((g %% 40) / 100.0)) * (1 + ((g * 13) %% 9))
             * (1 - ((g %% 5) * 0.05)), 4),
       (g %% 37) = 0,
       CASE WHEN (g %% 53) = 0 THEN 'X' WHEN (g %% 11) = 0 THEN 'P' ELSE 'C' END
  FROM generate_series(%(start)s::bigint, %(end)s::bigint) AS g
  JOIN dim_product_k  p ON p.k = (g * 17) %% %(n_product)s
  JOIN dim_customer_k c ON c.k = (g * 29) %% %(n_customer)s;
"""

WEB_EVENT = """
INSERT INTO warehouse.web_event
    (event_id, occurred_at, person_id, product_id, event_type, session_id, device)
SELECT g,
       TIMESTAMPTZ '2023-01-01 00:00:00+00' + ((g %% 31536000) * INTERVAL '1 second'),
       pe.businessentityid,
       pr.productid,
       (ARRAY['view','view','view','add_to_cart','checkout','search'])[1 + (g %% 6)],
       ('00000000-0000-4000-8000-' || lpad(to_hex(g %% 281474976710656), 12, '0'))::uuid,
       (ARRAY['desktop','mobile','mobile','tablet'])[1 + (g %% 4)]
  FROM generate_series(%(start)s::bigint, %(end)s::bigint) AS g
  JOIN dim_person_k  pe ON pe.k = (g * 11) %% %(n_person)s
  JOIN dim_product_k pr ON pr.k = (g * 23) %% %(n_product)s;
"""

INVENTORY = """
INSERT INTO warehouse.inventory_snapshot
    (snapshot_id, taken_on, product_id, location_id, quantity_on_hand, reorder_flag)
SELECT g,
       DATE '2024-01-01' + (g %% 365)::int,
       pr.productid,
       lo.locationid,
       ((g * 37) %% 5000)::int,
       ((g * 37) %% 5000) < 50
  FROM generate_series(%(start)s::bigint, %(end)s::bigint) AS g
  JOIN dim_product_k  pr ON pr.k = (g * 19) %% %(n_product)s
  JOIN dim_location_k lo ON lo.k = (g * 3)  %% %(n_location)s;
"""
