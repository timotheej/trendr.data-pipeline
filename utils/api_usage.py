import os
import datetime
import httpx

API_TYPE = "cse"

def today_utc():
    return datetime.datetime.utcnow().date().isoformat()

def get_caps():
    cap = int(os.getenv("CSE_DAILY_CAP", "1000"))
    return cap

def inc_api_usage(supabase_url: str, supabase_key: str, inc: int, daily_limit: int):
    """
    Use /rpc/inc_api_usage if exists, else fallback to upsert on /rest/v1/api_usage?on_conflict=date,api_type
    Return (used_today:int, cap:int, persisted:bool)
    """
    headers = {"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}", "Prefer":"resolution=merge-duplicates"}
    with httpx.Client(timeout=15.0) as s:
        # try RPC first
        rpc_url = f"{supabase_url}/rpc/inc_api_usage"
        payload = {"p_api_type": API_TYPE, "p_date": today_utc(), "p_inc": inc, "p_daily_limit": daily_limit}
        r = s.post(rpc_url, headers=headers, json=payload)
        if r.status_code == 200:
            data = r.json()
            # function returns table(queries_count,daily_limit)
            used = int(data[0]["queries_count"]) if data else 0
            cap = int(data[0]["daily_limit"]) if data else daily_limit
            return used, cap, True

        # fallback: upsert row then GET to read count (not perfectly atomic but acceptable for V1)
        upsert_url = f"{supabase_url}/rest/v1/api_usage?on_conflict=date,api_type"
        row = {"date": today_utc(), "api_type": API_TYPE, "queries_count": inc, "daily_limit": daily_limit}
        ur = s.post(upsert_url, headers=headers, json=row)
        if ur.status_code not in (200, 201, 204):
            return inc, daily_limit, False

        get_url = f"{supabase_url}/rest/v1/api_usage?select=queries_count,daily_limit&date=eq.{today_utc()}&api_type=eq.{API_TYPE}"
        gr = s.get(get_url, headers=headers)
        if gr.status_code == 200 and gr.json():
            used = int(gr.json()[0]["queries_count"])
            cap = int(gr.json()[0].get("daily_limit", daily_limit))
            return used, cap, True

        return inc, daily_limit, False