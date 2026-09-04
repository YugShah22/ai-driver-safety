"""
Supabase server-side client for FastAPI.

Uses the SERVICE ROLE KEY — provides admin-level access to bypass RLS
for backend operations (video processing, ML inference writes, etc.).

IMPORTANT: This client must NEVER be exposed to the browser or frontend.
"""
import os
from functools import lru_cache
from supabase import create_client, Client
from app.core.config import settings


@lru_cache(maxsize=1)
def get_supabase_admin() -> Client:
    """
    Return a cached Supabase client with the service-role key.
    Use for backend write operations that need to bypass RLS.
    """
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the environment. "
            "Copy backend/.env.example → backend/.env and fill in your credentials."
        )
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_supabase_anon(jwt_token: str) -> Client:
    """
    Return a Supabase client scoped to the user's JWT token.
    This respects RLS — use for user-scoped operations triggered by API calls.
    """
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in the environment."
        )
    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    # Set the auth header so RLS policies are applied correctly
    client.postgrest.auth(jwt_token)
    return client
