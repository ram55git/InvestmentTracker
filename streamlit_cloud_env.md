# Streamlit Cloud Environment Variables for Supabase Postgres

To run your app on Streamlit Cloud with Supabase Postgres, set these environment variables in your app's secrets or environment settings:

- SUPABASE_HOST: The hostname of your Supabase Postgres instance
- SUPABASE_DB: The database name
- SUPABASE_USER: The database user
- SUPABASE_PASSWORD: The database password
- SUPABASE_PORT: The database port (default: 5432)

Example (in Streamlit Cloud secrets):
```
SUPABASE_HOST=your-host.supabase.co
SUPABASE_DB=your_db_name
SUPABASE_USER=your_db_user
SUPABASE_PASSWORD=your_db_password
SUPABASE_PORT=5432
```

Make sure your Supabase table `investments` exists with columns:
- date (timestamp/date)
- category (text)
- name (text)
- ticker (text)
- units (float)
- nav_at_purchase (float)
- current_nav (float)
