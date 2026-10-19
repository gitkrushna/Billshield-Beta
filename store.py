# Save each check as one row in the Supabase "checks" table, and read back "My checks".
#
# Supabase gives every table a REST address: POST /rest/v1/checks adds a row, GET /rest/v1/checks?... reads rows.
# We use that directly with httpx (already installed with openai and gradio), so no new package is needed.
# The service_role key lives only on the server (Vercel env vars). The table has Row Level Security on with no
# public policies, so nobody can read the checks from a browser.

import os

import httpx

SAVED_FIELDS = [
    "user_name", "product_facts", "bill_hsn", "bill_rate", "status", "verdict", "confidence", "checklist",
    "expected_rule_id", "expected_hsn", "expected_rate", "reason", "quotes", "conditions", "rivals",
    "missing_facts", "assumed_facts", "next_step", "retries_used", "raw_llm_output", "model",
    "prompt_version", "rules_version", "seconds_taken",
]


# Saving is switched on only when both settings exist, so the app still runs without Supabase
def supabase_settings():
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None
    return url + "/rest/v1/checks", {"apikey": key, "Authorization": "Bearer " + key}


# Returns "" when saved (or when there is no database yet), or a short message saying why not (the check result is shown either way)
def save_check(record):
    settings = supabase_settings()
    if not settings:
        return ""          # no database yet: say nothing, the result is still shown
    address, headers = settings
    row = {field: record.get(field) for field in SAVED_FIELDS}
    try:
        response = httpx.post(address, json=row, headers={**headers, "Prefer": "return=minimal"}, timeout=10)
        response.raise_for_status()
        return ""
    except Exception as error:
        return "Not saved: " + str(error)[:120]


# The last 20 checks saved under this name, newest first
def my_checks(user_name):
    user_name = (user_name or "").strip()
    if not user_name:
        return [], "Type your name above to see your checks."
    settings = supabase_settings()
    if not settings:
        return [], "The database isn't set up yet."
    address, headers = settings
    params = {
        "select": "created_at,product_facts,bill_hsn,bill_rate,verdict,confidence,expected_hsn,expected_rate",
        "user_name": "eq." + user_name,
        "order": "created_at.desc",
        "limit": "20",
    }
    try:
        response = httpx.get(address, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as error:
        return [], "Couldn't load your checks: " + str(error)[:120]
    rows = []
    for check in response.json():
        product = (check["product_facts"] or "").strip().splitlines()
        product = [line.replace("Product:", "").strip() for line in product if line.strip() and not line.startswith(("---", "FROM THE LABEL"))]
        rows.append([
            check["created_at"][:16].replace("T", " "),
            (product[0] if product else "")[:60],
            " · ".join(part for part in [check["bill_hsn"] or "", "" if check["bill_rate"] is None else str(check["bill_rate"]) + "%"] if part),
            (check["expected_hsn"] or "") + ("" if check["expected_rate"] is None else " · " + str(check["expected_rate"]) + "%"),
            check["verdict"] + (" (" + check["confidence"] + ")" if check["confidence"] else ""),
        ])
    return rows, ("" if rows else "No checks saved under this name yet.")
