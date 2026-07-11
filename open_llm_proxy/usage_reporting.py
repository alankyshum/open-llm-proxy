import fastapi
import logging
from datetime import datetime, timezone
from fastapi import Depends, HTTPException, status
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
from litellm.proxy.proxy_server import user_api_key_auth

logger = logging.getLogger("open_llm_proxy.usage_reporting")

def get_canonical_date(dt_or_str) -> str:
    if isinstance(dt_or_str, datetime):
        return dt_or_str.strftime("%Y-%m-%d")
    elif isinstance(dt_or_str, str):
        try:
            dt = datetime.fromisoformat(dt_or_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.strptime(dt_or_str[:19], "%Y-%m-%d %H:%M:%S")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                try:
                    dt = datetime.strptime(dt_or_str[:10], "%Y-%m-%d")
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    return str(dt_or_str)[:10]
    return str(dt_or_str)[:10]

def format_display_model(model: str, provider: str | None) -> str:
    if not model:
        return ""
    if not provider:
        return model
    p_lower = provider.lower()
    m_lower = model.lower()
    if m_lower.startswith(p_lower + "/"):
        return model
    return f"{provider}/{model}"

def format_date(dt_or_str) -> str:
    if isinstance(dt_or_str, datetime):
        return dt_or_str.strftime("%b %d")
    elif isinstance(dt_or_str, str):
        try:
            dt = datetime.fromisoformat(dt_or_str.replace("Z", "+00:00"))
            return dt.strftime("%b %d")
        except Exception:
            try:
                dt = datetime.strptime(dt_or_str[:19], "%Y-%m-%d %H:%M:%S")
                return dt.strftime("%b %d")
            except Exception:
                try:
                    dt = datetime.strptime(dt_or_str[:10], "%Y-%m-%d")
                    return dt.strftime("%b %d")
                except Exception:
                    return dt_or_str
    return str(dt_or_str)

def aggregate_and_normalize_spend_logs(db_rows) -> list:
    model_groups = {}
    
    for row in db_rows:
        if hasattr(row, "get"):
            r_get = lambda k, d=None: row.get(k, d)
        else:
            r_get = lambda k, d=None: getattr(row, k, d)
            
        model = r_get("model") or ""
        provider = r_get("custom_llm_provider") or ""
        date_val = r_get("date")
        api_requests = int(r_get("api_requests") or 0)
        prompt_tokens = int(r_get("prompt_tokens") or 0)
        completion_tokens = int(r_get("completion_tokens") or 0)
        total_tokens = int(r_get("total_tokens") or 0)
        
        display_model = format_display_model(model, provider)
        
        if display_model not in model_groups:
            model_groups[display_model] = {
                "model": display_model,
                "provider": provider,
                "sum_api_requests": 0,
                "sum_total_tokens": 0,
                "sum_prompt_tokens": 0,
                "sum_completion_tokens": 0,
                "daily_by_date": {}
            }
            
        group = model_groups[display_model]
        canonical_date = get_canonical_date(date_val)
        
        if canonical_date not in group["daily_by_date"]:
            group["daily_by_date"][canonical_date] = {
                "date": canonical_date,
                "api_requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
        day_entry = group["daily_by_date"][canonical_date]
        day_entry["api_requests"] += api_requests
        day_entry["prompt_tokens"] += prompt_tokens
        day_entry["completion_tokens"] += completion_tokens
        day_entry["total_tokens"] += total_tokens
        
        group["sum_api_requests"] += api_requests
        group["sum_prompt_tokens"] += prompt_tokens
        group["sum_completion_tokens"] += completion_tokens
        group["sum_total_tokens"] += total_tokens

    response = []
    for display_model, group in model_groups.items():
        # Sort by canonical date key (YYYY-MM-DD string sorts chronologically)
        sorted_daily = sorted(group["daily_by_date"].values(), key=lambda x: x["date"])
        
        # After sorting, format the date for display (UI compatibility)
        formatted_daily = []
        for item in sorted_daily:
            formatted_daily.append({
                "date": format_date(item["date"]),
                "api_requests": item["api_requests"],
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": item["completion_tokens"],
                "total_tokens": item["total_tokens"]
            })
        
        response.append({
            "model": group["model"],
            "provider": group["provider"],
            "daily_data": formatted_daily,
            "sum_api_requests": group["sum_api_requests"],
            "sum_total_tokens": group["sum_total_tokens"],
            "sum_prompt_tokens": group["sum_prompt_tokens"],
            "sum_completion_tokens": group["sum_completion_tokens"],
        })
        
    response = sorted(response, key=lambda x: x["sum_total_tokens"], reverse=True)[:10]
    return response

def install_usage_reporting(app):
    # Filter the main app.router.routes
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/global/activity/model"]
    
    # Filter any nested original_router routes (IncludedRouter)
    for r in list(app.router.routes):
        if hasattr(r, "original_router") and r.original_router:
            r.original_router.routes = [
                route for route in r.original_router.routes 
                if getattr(route, "path", None) != "/global/activity/model"
            ]
            if hasattr(r, "_effective_candidates_version"):
                r._effective_candidates_version = None
            if hasattr(r, "_effective_low_priority_routes_version"):
                r._effective_low_priority_routes_version = None
        if hasattr(r, "router") and r.router:
            r.router.routes = [
                route for route in r.router.routes 
                if getattr(route, "path", None) != "/global/activity/model"
            ]
            if hasattr(r, "_effective_candidates_version"):
                r._effective_candidates_version = None
            if hasattr(r, "_effective_low_priority_routes_version"):
                r._effective_low_priority_routes_version = None
            
    # Register the override route
    @app.get(
        "/global/activity/model",
        tags=["Budget & Spend Tracking"],
        dependencies=[Depends(user_api_key_auth)],
        include_in_schema=False,
    )
    async def get_global_activity_model(
        start_date: str | None = fastapi.Query(
            default=None,
            description="Time from which to start viewing spend",
        ),
        end_date: str | None = fastapi.Query(
            default=None,
            description="Time till which to view spend",
        ),
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        if start_date is None or end_date is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Please provide start_date and end_date"},
            )

        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"Invalid date format: {e}"},
            )

        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Database not connected."},
            )

        try:
            if (
                user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
                or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
            ):
                sql_query = """
                SELECT
                    model,
                    custom_llm_provider,
                    date_trunc('day', "startTime") AS date,
                    COUNT(*) AS api_requests,
                    SUM(prompt_tokens) AS prompt_tokens,
                    SUM(completion_tokens) AS completion_tokens,
                    SUM(total_tokens) AS total_tokens
                FROM "LiteLLM_SpendLogs"
                WHERE "startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
                  AND "startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
                  AND model IS NOT NULL AND model != '' AND total_tokens > 0
                  AND "user" = $3
                GROUP BY model, custom_llm_provider, date_trunc('day', "startTime")
                """
                user_id = user_api_key_dict.user_id
                if user_id is None:
                    raise HTTPException(status_code=500, detail={"error": "No user_id found"})
                db_response = await prisma_client.db.query_raw(sql_query, start_date_obj, end_date_obj, user_id)
            else:
                sql_query = """
                SELECT
                    model,
                    custom_llm_provider,
                    date_trunc('day', "startTime") AS date,
                    COUNT(*) AS api_requests,
                    SUM(prompt_tokens) AS prompt_tokens,
                    SUM(completion_tokens) AS completion_tokens,
                    SUM(total_tokens) AS total_tokens
                FROM "LiteLLM_SpendLogs"
                WHERE "startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
                  AND "startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
                  AND model IS NOT NULL AND model != '' AND total_tokens > 0
                GROUP BY model, custom_llm_provider, date_trunc('day', "startTime")
                """
                db_response = await prisma_client.db.query_raw(sql_query, start_date_obj, end_date_obj)

            if db_response is None:
                return []

            return aggregate_and_normalize_spend_logs(db_response)

        except HTTPException:
            raise
        except Exception as e:
            logging.getLogger(__name__).exception("Raw backend exception in usage reporting")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Internal server error"},
            )
