import bs4
import yfinance as yf

#ticker = yf.Ticker("EMBASSY.BO")

#info = ticker.info
#print(info)
#print(info.get('previousClose'))

#print(yf.Ticker("RELIANCE.NS").info.get('previousClose'))

#Fund,Equity quoteType

#import os
#import pandas as pd
#import psycopg2
#from psycopg2.extras import RealDictCursor
#from dotenv import load_dotenv
from supabase import create_client, Client


SUPABASE_URL = "https://yrwwtrgshppxdspizvdg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlyd3d0cmdzaHBweGRzcGl6dmRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE3MDk2NjQsImV4cCI6MjA3NzI4NTY2NH0.bmq_fUcqDs5QkOJgZlfuXCxJu5v8TJhAaOGjsAg5u10"


if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_KEY environment variables are not set.")
    exit()

try:
    # Initialize the client
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully!")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    supabase_client = None

if supabase_client:
    try:
        # Select all users from the 'users' table
        response = supabase_client.table('investmentsdb').select('*').execute()
        print(response)
       
        if response.data:
            print(response.data)
        else:
            print("No users found.")

         
            
    except Exception as e:
        print(f"Error reading data: {e}")


