import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st




SUPABASE_HOST = st.secrets["SUPABASE_HOST"]
SUPABASE_DB = "postgres"
SUPABASE_USER = "postgres.yrwwtrgshppxdspizvdg"
SUPABASE_PASSWORD = st.secrets["SUPABASE_PASSWORD"]
SUPABASE_PORT = 6543
TABLE_NAME = "investmentsdb"
pool_mode="transaction"



# Connection string for psycopg2
conn_str = f"host={SUPABASE_HOST} dbname={SUPABASE_DB} user={SUPABASE_USER} password={SUPABASE_PASSWORD} port={SUPABASE_PORT} sslmode=require"

def get_conn():
    return psycopg2.connect(conn_str)

def load_data():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {TABLE_NAME}")
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=[
                    'date', 'category', 'name', 'ticker', 'units', 'nav_at_purchase', 'current_nav'
                ])
            df = pd.DataFrame(rows)
            # Robust date parsing: try dayfirst, mixed formats
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
            return df

def save_data(df):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE_NAME}")
            for _, row in df.iterrows():
                cur.execute(
                    f"INSERT INTO {TABLE_NAME} (date, category, name, ticker, units, nav_at_purchase, current_nav) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        row['date'], row['category'], row['name'], row['ticker'],
                        row['units'], row['nav_at_purchase'], row['current_nav']
                    )
                )
        conn.commit()
