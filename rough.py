import bs4
import yfinance as yf

#ticker = yf.Ticker("EMBASSY.BO")

#info = ticker.info
#print(info)
#print(info.get('previousClose'))

#print(yf.Ticker("RELIANCE.NS").info.get('previousClose'))

#Fund,Equity quoteType

import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


load_dotenv()

SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_DB = os.getenv("SUPABASE_DB")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
SUPABASE_PORT = os.getenv("SUPABASE_PORT","5432")
TABLE_NAME = "investmentsdb"

# Connection string for psycopg2
conn_str = f"host={SUPABASE_HOST} dbname={SUPABASE_DB} user={SUPABASE_USER} password={SUPABASE_PASSWORD} port={SUPABASE_PORT} sslmode=require"

psycopg2.connect(conn_str)
print("Connected successfully")
