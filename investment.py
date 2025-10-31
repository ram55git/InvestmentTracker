import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from xirr import xirr
import yfinance as yf
import math
import numpy_financial as npf
from supabase_psycopg2_helper import load_data, save_data
from supabase_auth_helper import sign_in, sign_out, get_user

# Helper to safely trigger a rerun across Streamlit versions
def safe_rerun():
    try:
        # Preferred method when available
        st.experimental_rerun()
    except Exception:
        # Fallback: mutate query params to force a rerun and stop
        try:
            # Read query params using the supported API
            params = dict(st.query_params) if hasattr(st, 'query_params') else {}
            # Ensure values are lists (Streamlit expects list values)
            for k, v in list(params.items()):
                if not isinstance(v, (list, tuple)):
                    params[k] = [v]
            params['_refresh'] = [str(datetime.utcnow().timestamp())]
            # Prefer stable setter if available
            if hasattr(st, 'set_query_params'):
                st.set_query_params(**params)
            else:
                # fall back to experimental setter
                st.experimental_set_query_params(**params)
            st.stop()
        except Exception:
            # As a last resort, return and let Streamlit refresh on interaction
            return


def get_user_email(user):
    """Return an email string from various user object shapes (dict, pydantic model, simple object)."""
    if not user:
        return None
    try:
        if isinstance(user, dict):
            return user.get('email') or user.get('email_address')
        # pydantic BaseModel or other object with attribute
        if hasattr(user, 'email'):
            try:
                return getattr(user, 'email')
            except Exception:
                pass
        # object that supports dict-like get
        if hasattr(user, 'get'):
            try:
                return user.get('email')
            except Exception:
                pass
        # fallback to string representation
        return str(user)
    except Exception:
        return None


def show_app_ui():
    """Render the main investment UI. Separated so we can call it immediately after login without requiring a rerun."""
    GIT 
    st.title('Investment Tracker')

    df = load_data()

    st.sidebar.header('Add / Edit Investment')
    with st.sidebar.form('add_form'):
        category = st.selectbox('Category', ['Mutual Fund', 'Ind_Stock','Foreign_Stock','REIT','Commodity','Cash','FixedDeposit', 'Crypto', 'Other'])
        name = st.text_input('Name')
        ticker = st.text_input('Google Finance ticker (e.g. AAPL or 500325.BO)')
        date = st.date_input('Purchase date', value=datetime.today())
        units = st.number_input('Units purchased', min_value=0.0, value=0.0, format='%f')
        nav_at_purchase = st.number_input('NAV at purchase', min_value=0.0, value=0.0, format='%f')
        current_nav = st.number_input('Current NAV (optional, leave 0 to fetch)', min_value=0.0, value=0.0, format='%f')
        submitted = st.form_submit_button('Add investment')

    if submitted:
        if not name or units <= 0 or nav_at_purchase <= 0:
            st.error('Please provide name, units > 0 and NAV at purchase > 0')
        else:
            # Convert NAVs to INR for Foreign_Stock and Crypto
            nav_at_purchase_inr = nav_at_purchase
            current_nav_inr = current_nav
            if category in ['Foreign_Stock', 'Crypto']:
                fx = fetch_fx_rate('USD', 'INR')
                if fx:
                    nav_at_purchase_inr = nav_at_purchase * fx
                    if current_nav_inr != 0:
                        current_nav_inr = current_nav * fx
            if current_nav_inr == 0 and ticker:
                fetched = fetch_current_nav(ticker)
                if fetched:
                    # Convert fetched NAV to INR if needed
                    if category in ['Foreign_Stock', 'Crypto']:
                        fx = fetch_fx_rate('USD', 'INR')
                        if fx:
                            fetched = fetched * fx
                    current_nav_inr = fetched

            new = dict(date=pd.to_datetime(date), category=category, name=name, ticker=ticker, units=units, nav_at_purchase=nav_at_purchase_inr, current_nav=current_nav_inr)
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            save_data(df)
            st.success('Investment added')

    st.header('Portfolio')
    # Add refresh button
    if st.button('Refresh NAV data'):
        # Update current NAVs for all investments with tickers
        updated_df = df.copy()
        for idx, row in updated_df.iterrows():
            if row['ticker']:
                fetched_nav = fetch_current_nav(row['ticker'])
                if fetched_nav:
                    # Convert fetched NAV to INR for Foreign_Stock and Crypto
                    if row['category'] in ['Foreign_Stock', 'Crypto']:
                        fx = fetch_fx_rate('USD', 'INR')
                        if fx:
                            fetched_nav = fetched_nav * fx
                    updated_df.at[idx, 'current_nav'] = fetched_nav
        df = updated_df
        save_data(df)
        st.success('NAV data refreshed!')

    grouped, total_value = compute_metrics(df)

    st.metric('Total portfolio value', format_indian_currency(total_value))
    # show invested amount by category (use grouped values so FX conversion is respected)
    if not grouped.empty:
        # aggregate by category (respecting FX-converted grouped values)
        cat_totals = grouped.groupby('category', dropna=False).agg(
            amount_invested=('amount_invested', 'sum'),
            current_value=('current_value', 'sum')
        ).reset_index().sort_values('amount_invested', ascending=False)
        st.subheader('Invested by category')
        if not cat_totals.empty:
            total_invested = cat_totals['amount_invested'].sum()
            total_current = cat_totals['current_value'].sum()

            # allocations
            if total_invested != 0:
                cat_totals['alloc_invested_pct'] = cat_totals['amount_invested'] / total_invested
            else:
                cat_totals['alloc_invested_pct'] = 0.0

            if total_current != 0:
                cat_totals['alloc_current_pct'] = cat_totals['current_value'] / total_current
            else:
                cat_totals['alloc_current_pct'] = 0.0

            # absolute return and return percent
            cat_totals['absolute_return'] = cat_totals['current_value'] - cat_totals['amount_invested']
            def calc_return_pct(row):
                inv = row['amount_invested']
                cur = row['current_value']
                try:
                    if inv == 0:
                        return np.nan
                    return (cur - inv) / inv * 100.0
                except Exception:
                    return np.nan

            cat_totals['absolute_return_pct'] = cat_totals.apply(calc_return_pct, axis=1)

            # format functions
            def fmt_inr_roundup(x):
                try:
                    val = 0 if pd.isna(x) else math.ceil(float(x))
                    return f"₹{val:,}"
                except Exception:
                    return x

            def fmt_inr_two(x):
                try:
                    val = 0.0 if pd.isna(x) else float(x)
                    sign = '-' if val < 0 else ''
                    return f"{sign}₹{abs(val):,.2f}"
                except Exception:
                    return x

            cat_totals['Total Invested'] = cat_totals['amount_invested'].apply(fmt_inr_roundup)
            cat_totals['Total Current'] = cat_totals['current_value'].apply(fmt_inr_two)
            cat_totals['Absolute Return (%)'] = cat_totals['absolute_return_pct'].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else 'N/A')
            cat_totals['Allocation (Invested %)'] = cat_totals['alloc_invested_pct'].apply(lambda x: f"{x:.2%}")
            cat_totals['Allocation (Current %)'] = cat_totals['alloc_current_pct'].apply(lambda x: f"{x:.2%}")

            display_cat = cat_totals[['category', 'Total Invested', 'Total Current', 'Absolute Return (%)', 'Allocation (Invested %)', 'Allocation (Current %)']].rename(columns={'category': 'Category'})
            st.dataframe(display_cat, use_container_width=True)
        else:
            st.info('No category data to display')
    if not grouped.empty:
        st.subheader('Investments')
        display = grouped.copy()
        # compute percentage and round to 2 decimals
        display['xirr_pct'] = display['xirr'].apply(lambda r: round(r * 100, 2) if pd.notnull(r) else None)
        display = display[['category', 'name', 'ticker', 'units', 'amount_invested', 'current_value', 'xirr_pct']]
        display = display.rename(columns={'category': 'Category', 'name': 'Name', 'ticker': 'Ticker', 'units': 'Units', 'amount_invested': 'Amount Invested', 'current_value': 'Current Value', 'xirr_pct': 'XIRR (%)'})
        # format XIRR column as string with 2 decimals; show 'N/A' when not available
        display['XIRR (%)'] = display['XIRR (%)'].apply(lambda v: f"{v:.2f}" if pd.notnull(v) else 'N/A')
        # make the table taller/wider and scrollable
        st.dataframe(display, height=600, use_container_width=True)

        # --- Yearly invested table ---
        st.subheader('Yearly Invested Amount by Category')
        # Use the original df, but recompute amount_invested with FX applied
        df_year = df.copy()
        df_year['year'] = pd.to_datetime(df_year['date'], errors='coerce').dt.year
        df_year['amount_invested'] = df_year['units'] * df_year['nav_at_purchase'] * df_year.get('fx', 1.0)
        # Only keep rows with valid year
        df_year = df_year[df_year['year'].notna()]
        # Group by year and category
        yearly = df_year.groupby(['category', 'year'], dropna=False)['amount_invested'].sum().reset_index()
        # Pivot so category is rows, year is columns
        pivot_yearly = yearly.pivot(index='category', columns='year', values='amount_invested').fillna(0)
        # Format each cell as rupees, rounded up
        def fmt_inr_roundup(x):
            try:
                val = 0 if pd.isna(x) else math.ceil(float(x))
                return f"₹{val:,}"
            except Exception:
                return x
        display_pivot = pivot_yearly.applymap(fmt_inr_roundup)
        display_pivot.index.name = 'Category'
        st.dataframe(display_pivot, use_container_width=True)
    else:
        st.info('No investments yet. Add one from the sidebar.')

    st.title("Investment Return Calculator")

    # Input widgets for user data
    with st.expander("Calculate Future Value and Actual Worth", expanded=True): 
        st.subheader("Investment Inputs")
        initial_investment = st.number_input("Initial Investment", min_value=1.0, value=total_value, step=100000.0)
        annual_return_rate = st.slider("Annual Return Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
        inflation_rate = st.slider("Annual Inflation Rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)
        withdrawal_rate = st.slider("Annual Withdrawal Rate (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
        years = st.slider("Number of Years", min_value=1, max_value=50, value=5)

    if st.button("Calculate"):
        # Perform calculations
        nominal_value, actual_worth, _ = calculate_investment(
            initial_investment,
            annual_return_rate,
            inflation_rate,
            years,
            withdrawal_rate
        )

        # Display results
        st.subheader("Results")
        st.info(f"**Nominal Future Value:** {format_indian_currency(nominal_value)}")
        st.success(f"**Actual Worth (Adjusted for Inflation):** {format_indian_currency(actual_worth)}")

    with st.expander("Calculate Required Annual Investment", expanded=False):
        st.subheader("Corpus Goal Inputs")
        desired_corpus = st.number_input("Desired Corpus (in today's purchasing power)", min_value=1.0, value=50000000.0, step=100000.0)
        annual_return_rate_goal = st.slider("Expected Annual Return Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5, key="goal_return")
        inflation_rate_goal = st.slider("Expected Annual Inflation Rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.5, key="goal_inflation")
        years_goal = st.slider("Years to Reach Goal", min_value=1, max_value=50, value=20, key="goal_years")
        present_value_goal = st.number_input("Current Savings / Initial Investment (₹)", min_value=0.0, value=0.0, step=100000.0, key="goal_pv")

    if st.button("Calculate Annual Investment"):
        required_annual_investment, target_fv= calculate_annual_investment(desired_corpus, present_value_goal, annual_return_rate_goal, inflation_rate_goal,years_goal)

        st.subheader("Goal-Based Investment Plan")
        st.success(f"To reach a corpus with the purchasing power of ₹{desired_corpus:,.2f} in **{years_goal}** years, you need to invest approximately **₹{required_annual_investment:,.2f}** per year.")
        st.info(f"The actual target amount (nominal value) in {years_goal} years will be approximately **₹{target_fv:,.2f}**.")

def calculate_investment(initial_investment, annual_return_rate, inflation_rate, years,withdrawal_rate):
    
    # Convert percentages to decimals
    r = annual_return_rate / 100
    i = inflation_rate / 100
    w = withdrawal_rate / 100

    # Effective annual growth after withdrawal
    effective_growth = (1 + r) * (1 - w)

    # Nominal future value
    nominal_future_value = initial_investment * (effective_growth ** years)

    # Real (inflation-adjusted) future value
    actual_worth = nominal_future_value / ((1 + i) ** years)

    # Real rate of return per year
    real_rate = ((1 + r) / (1 + i)) - 1

    return nominal_future_value, actual_worth, real_rate

def calculate_annual_investment(target_corpus_today, current_savings,annual_return,inflation,years):
    # Step 1: Future value of current savings
    annual_return = annual_return / 100
    inflation = inflation / 100
    
    fv_current = current_savings * (1 + annual_return) ** years

    # Step 2: Inflation-adjusted (nominal) target corpus after 10 years
    fv_target = target_corpus_today * (1 + inflation) ** years

    # Step 3: Shortfall to be achieved via new annual investments
    shortfall = fv_target - fv_current

    # Step 4: Annual investment (PMT) required to cover shortfall
    fv_factor = ((1 + annual_return) ** years - 1) / annual_return
    annual_investment = shortfall / fv_factor
    
    return annual_investment,fv_target


def compute_metrics(df):
    if df.empty:
        return pd.DataFrame(), 0.0

    # Work on a copy and add FX conversion where needed
    df = df.copy()
    df['fx'] = 1.0
    # Convert USD to INR for Foreign_Stock and Crypto categories
    for cat in ['Foreign_Stock', 'Crypto']:
        if (df['category'] == cat).any():
            fx = fetch_fx_rate('USD', 'INR')
            if fx:
                df.loc[df['category'] == cat, 'fx'] = fx
    

    # compute amounts (apply FX conversion where applicable)
    df['amount_invested'] = df['units'] * df['nav_at_purchase']
    df['current_value'] = df['units'] * df['current_nav']
    df['absolute_return'] = df['current_value'] - df['amount_invested']

    # Group by name+category
    grouped = df.groupby(['category', 'name', 'ticker']).agg(
        units=('units', 'sum'),
        amount_invested=('amount_invested', 'sum'),
        current_value=('current_value', 'sum')
    ).reset_index()

    # compute XIRR per grouped investment using underlying transactions (converted amounts)
    xirr_list = []
    for _, row in grouped.iterrows():
        mask = (df['category'] == row['category']) & (df['name'] == row['name']) & (df['ticker'] == row['ticker'])
        txs = df[mask].sort_values('date')
        # cashflows: outflows (invested) negative, final value positive at today
        cashflows = []
        for _, tx in txs.iterrows():
            cashflows.append((tx['date'].date(), -tx['amount_invested']))
        # add current value as inflow today
        cashflows.append((datetime.today().date(), row['current_value']))
        try:
            rate = xirr(cashflows)
        except Exception:
            rate = np.nan
        xirr_list.append(rate)

    grouped['xirr'] = xirr_list

    total_value = grouped['current_value'].sum()
    return grouped, total_value


def fetch_current_nav(ticker):
    try:
        info=yf.Ticker(ticker).info
        if info.get('quoteType') in ['ETF','EQUITY']:
            return yf.Ticker(ticker).info.get('currentPrice')
            
        elif info.get('quoteType') in ['MUTUALFUND','CRYPTOCURRENCY']:
            return yf.Ticker(ticker).info.get('previousClose')
        else:
            return None
    except Exception:
        return None

def fetch_fx_rate(base='USD', quote='INR'):
    """Fetch the latest FX rate for base->quote using Yahoo Finance (e.g. USDINR=X)."""
    pair = f"{base}{quote}=X"
    try:
        tk = yf.Ticker(pair)
        info = tk.info
        # try a few likely keys
        rate = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        if rate:
            return float(rate)
        # fallback to recent history close
        hist = tk.history(period='5d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        return None
    return None

def format_indian_currency(amount):
        try:
            amount = float(amount)
            sign = '-' if amount < 0 else ''
            abs_amount = abs(amount)
            if abs_amount >= 1e7:
                # Crores
                val = abs_amount / 1e7
                return f"{sign}₹{val:,.2f} Cr"
            elif abs_amount >= 1e5:
                # Lakhs
                val = abs_amount / 1e5
                return f"{sign}₹{val:,.2f} Lakh"
            else:
                # Rupees
                return f"{sign}₹{abs_amount:,.2f}"
        except Exception:
            return str(amount)

def main():
    # Authentication: require user to sign in before showing the app
    user = st.session_state.get('user')
    if not user:
        st.title('Please sign in')
        # Use plain widgets (not a form) to avoid form-submit/rerun quirks
        email = st.text_input('Email', value=st.session_state.get('email', ''))
        password = st.text_input('Password', type='password', value=st.session_state.get('password', ''))
        if st.button('Sign in'):
            # persist entered values immediately
            st.session_state['email'] = email
            st.session_state['password'] = password
            try:
                session = sign_in(email, password)
                if not session:
                    st.error('Sign in did not return a session. Check credentials and Supabase settings.')
                    st.write('sign_in returned:', session)
                else:
                    st.session_state['user_session'] = session
                    def extract_session_info(s):
                        a_token = None
                        u = None
                        try:
                            if isinstance(s, dict):
                                data = s.get('data') or s
                                if isinstance(data, dict):
                                    sess = data.get('session') or data.get('ses')
                                    if isinstance(sess, dict):
                                        a_token = sess.get('access_token') or sess.get('accessToken') or sess.get('token')
                                        u = sess.get('user') or data.get('user') or sess.get('user_info')
                                    else:
                                        a_token = data.get('access_token') or data.get('accessToken') or data.get('token')
                                        u = data.get('user') or data.get('user_info')
                            for attr in ('data', 'session', 'user', 'access_token', 'accessToken', 'token'):
                                if getattr(s, attr, None):
                                    val = getattr(s, attr)
                                    if isinstance(val, dict):
                                        if not a_token:
                                            a_token = val.get('access_token') or val.get('accessToken') or val.get('token')
                                        if not u:
                                            u = val.get('user') or val.get('user_info')
                                    else:
                                        if attr in ('access_token', 'accessToken', 'token') and not a_token:
                                            a_token = val
                                        if attr == 'user' and not u:
                                            u = val
                        except Exception:
                            pass
                        if not u and a_token:
                            try:
                                u = get_user(a_token)
                            except Exception:
                                u = None
                        if not u:
                            try:
                                u = get_user(None)
                            except Exception:
                                u = None
                        return a_token, u

                    access_token, user = extract_session_info(session)
                    st.session_state['user'] = user
                    st.session_state['access_token'] = access_token
                    # Rerun so the app displays the authenticated UI immediately
                    try:
                        st.experimental_rerun()
                    except Exception:
                        # fallback to safe rerun
                        safe_rerun()
            except Exception as e:
                st.error(f"Sign in failed: {e}")
        return

    # If authenticated, show a sign-out button and then the app
    col1, col2 = st.columns([1, 3])
    with col1:
        signed = get_user_email(st.session_state.get('user'))
        st.write(f"Signed in: {signed if signed else 'Unknown'}")
    with col2:
        if st.button('Sign out'):
            try:
                sign_out()
            finally:
                st.session_state['user'] = None
                st.session_state['access_token'] = None
                st.session_state['user_session'] = None
                try:
                    st.experimental_rerun()
                except Exception:
                    safe_rerun()
            return
    show_app_ui()
    return
if __name__ == '__main__':
    main()
