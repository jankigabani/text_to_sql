"""Amplify the database to ~10 million rows, so the cost checks mean something.

AdventureWorks is realistic in *shape* (68 tables, 90 foreign keys) but small in *volume*
— about 120k order lines. At that size every query is cheap, so L4's plan-cost check and
the timeout in stage 5 would never fire, and the experiment could not show that bad SQL is
expensive. So we add a `warehouse` schema of large fact tables that reference the existing
AdventureWorks dimensions through real foreign keys:

    warehouse.fact_sales        -> production.product, sales.customer, sales.salesterritory
    warehouse.web_event         -> person.person, production.product
    warehouse.inventory_snapshot-> production.product, production.location

The result is one database with 70+ tables, genuine PK/FK structure, and enough rows that
a missing join condition is immediately visible in the plan.

    python scripts/02_amplify.py --target-rows 10000000
    python scripts/02_amplify.py --target-rows 1000000 --drop-first
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg   

from app.core.config import get_settings   
from scripts._amplify_sql import (   
    DIM_TABLES, FACT_SALES, INVENTORY, WEB_EVENT,
)

DDL = """
CREATE SCHEMA IF NOT EXISTS warehouse;

CREATE TABLE IF NOT EXISTS warehouse.sales_channel (
    channel_id   smallint PRIMARY KEY,
    channel_name text NOT NULL UNIQUE,
    is_online    boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS warehouse.fact_sales (
    sale_id        bigint PRIMARY KEY,
    order_date     date        NOT NULL,
    product_id     integer     NOT NULL REFERENCES production.product(productid),
    customer_id    integer     NOT NULL REFERENCES sales.customer(customerid),
    territory_id   integer     REFERENCES sales.salesterritory(territoryid),
    channel_id     smallint    NOT NULL REFERENCES warehouse.sales_channel(channel_id),
    quantity       integer     NOT NULL CHECK (quantity > 0),
    unit_price     numeric(12,4) NOT NULL,
    discount_pct   numeric(5,4)  NOT NULL DEFAULT 0,
    line_total     numeric(14,4) NOT NULL,
    is_returned    boolean     NOT NULL DEFAULT false,
    status         char(1)     NOT NULL DEFAULT 'C'   -- 'C' complete, 'P' pending, 'X' cancelled
);

CREATE TABLE IF NOT EXISTS warehouse.web_event (
    event_id     bigint PRIMARY KEY,
    occurred_at  timestamptz NOT NULL,
    person_id    integer REFERENCES person.person(businessentityid),
    product_id   integer REFERENCES production.product(productid),
    event_type   text    NOT NULL,   -- view | add_to_cart | checkout | search
    session_id   uuid    NOT NULL,
    device       text    NOT NULL    -- desktop | mobile | tablet
);

CREATE TABLE IF NOT EXISTS warehouse.inventory_snapshot (
    snapshot_id   bigint PRIMARY KEY,
    taken_on      date    NOT NULL,
    product_id    integer NOT NULL REFERENCES production.product(productid),
    location_id   smallint NOT NULL REFERENCES production.location(locationid),
    quantity_on_hand integer NOT NULL,
    reorder_flag  boolean NOT NULL DEFAULT false
);

INSERT INTO warehouse.sales_channel (channel_id, channel_name, is_online) VALUES
    (1, 'Retail Store', false), (2, 'Online Store', true), (3, 'Reseller', false),
    (4, 'Marketplace', true),   (5, 'Phone Order', false)
ON CONFLICT (channel_id) DO NOTHING;
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS fact_sales_date_idx     ON warehouse.fact_sales(order_date);
CREATE INDEX IF NOT EXISTS fact_sales_product_idx  ON warehouse.fact_sales(product_id);
CREATE INDEX IF NOT EXISTS fact_sales_customer_idx ON warehouse.fact_sales(customer_id);
CREATE INDEX IF NOT EXISTS web_event_time_idx      ON warehouse.web_event(occurred_at);
CREATE INDEX IF NOT EXISTS web_event_person_idx    ON warehouse.web_event(person_id);
CREATE INDEX IF NOT EXISTS inventory_day_idx       ON warehouse.inventory_snapshot(taken_on);
"""

COMMENTS = """
COMMENT ON TABLE warehouse.fact_sales IS
  'One row per sold line item. Synthetic volume layer over the AdventureWorks dimensions.';
COMMENT ON COLUMN warehouse.fact_sales.status IS
  'C = complete, P = pending, X = cancelled';
COMMENT ON TABLE warehouse.web_event IS 'Clickstream events: view, add_to_cart, checkout, search.';
COMMENT ON TABLE warehouse.inventory_snapshot IS 'Daily stock level per product and location.';
"""


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rows", type=int, default=10_000_000)
    ap.add_argument("--batch", type=int, default=500_000)
    ap.add_argument("--drop-first", action="store_true")
    a = ap.parse_args()

    # 70 % sales, 25 % events, 5 % inventory — a realistic fact-table mix.
    plan = [
        ("warehouse.fact_sales", FACT_SALES, int(a.target_rows * 0.70)),
        ("warehouse.web_event", WEB_EVENT, int(a.target_rows * 0.25)),
        ("warehouse.inventory_snapshot", INVENTORY, int(a.target_rows * 0.05)),
    ]

    with psycopg.connect(s.dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT (SELECT count(*) FROM production.product
                                     WHERE COALESCE(listprice,0) > 0),
                                  (SELECT count(*) FROM sales.customer),
                                  (SELECT count(*) FROM person.person),
                                  (SELECT count(*) FROM production.location)""")
            n_product, n_customer, n_person, n_location = cur.fetchone()
            dims = {"n_product": n_product, "n_customer": n_customer,
                    "n_person": n_person, "n_location": n_location}
            print(f"dimensions: {n_product} products, {n_customer} customers, "
                  f"{n_person} people, {n_location} locations")
            if min(dims.values()) == 0:
                print("[FAIL] AdventureWorks dimensions are empty — run "
                      "scripts/01_load_adventureworks.py first")
                return 1
            if a.drop_first:
                print("dropping warehouse schema")
                cur.execute("DROP SCHEMA IF EXISTS warehouse CASCADE")
            print("creating warehouse schema")
            cur.execute(DDL)
            cur.execute(COMMENTS)
            print("building dense dimension keys (temp tables)")
            cur.execute(DIM_TABLES)

            for table, stmt, want in plan:
                cur.execute(f"SELECT COALESCE(max(1), 0), count(*) FROM {table}")
                have = cur.fetchone()[1]
                if have >= want:
                    print(f"  {table}: already {have:,} rows (target {want:,}) — skipping")
                    continue
                print(f"  {table}: {have:,} -> {want:,} rows")
                t0 = time.perf_counter()
                start = have + 1
                while start <= want:
                    end = min(start + a.batch - 1, want)
                    cur.execute(stmt, {"start": start, "end": end, **dims})
                    done = end
                    pct = 100 * done / want
                    print(f"    {done:>12,} / {want:,}  ({pct:5.1f}%)", end="\r", flush=True)
                    start = end + 1
                print(f"\n    took {time.perf_counter() - t0:.1f}s")

            print("creating indexes")
            cur.execute(INDEXES)
            print("ANALYZE (needed for L4 cost estimates)")
            cur.execute("ANALYZE")

            cur.execute("""SELECT n.nspname || '.' || c.relname, c.reltuples::bigint
                             FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relkind = 'r' AND n.nspname NOT IN
                                  ('pg_catalog','information_schema')
                            ORDER BY c.reltuples DESC LIMIT 8""")
            print("\nlargest tables now:")
            for name, rows in cur.fetchall():
                print(f"  {name:<40} {rows:>14,}")

            cur.execute("""SELECT count(*) FROM pg_class c JOIN pg_namespace n
                             ON n.oid=c.relnamespace WHERE c.relkind='r'
                            AND n.nspname NOT IN ('pg_catalog','information_schema')""")
            print(f"\ntables in database: {cur.fetchone()[0]}")
    print("\nNext: python scripts/03_create_readonly_role.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
