import os
from supabase import create_client, Client
import streamlit as st


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def sign_in(email: str, password: str):
    """Try to sign in a user with email/password. Returns session dict or raises Exception."""
    # supabase-py has had breaking changes; attempt both common method names
    try:
        # v2 style
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        # res may contain 'data' with 'session' or directly a session
        if isinstance(res, dict) and res.get('error'):
            raise Exception(res.get('error'))
        return res
    except Exception:
        try:
            # v1 style
            res = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password
                    })
            return res
        except Exception as e:
            raise


def sign_out():
    try:
        supabase.auth.sign_out()
        # clear session keys if present
        for k in ['user_session', 'user', 'access_token']:
            try:
                st.session_state.pop(k, None)
            except Exception:
                pass
    except Exception:
        # best-effort
        pass


def get_user(access_token: str):
    try:
        user_res = supabase.auth.get_user(access_token)
        # v2 returns dict with 'data'
        if isinstance(user_res, dict):
            return user_res.get('data') or user_res
        return user_res
    except Exception:
        try:
            return supabase.auth.user(access_token)
        except Exception:
            return None
