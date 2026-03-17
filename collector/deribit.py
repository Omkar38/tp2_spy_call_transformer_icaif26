"""
collector/deribit.py — Clean Deribit API wrapper.
All network calls go through here. Nothing else touches requests directly.

Key API limits discovered (public endpoints):
  - get_last_trades_by_currency_and_time: max count=1000, no 'kind' filter, <7d history
  - get_last_trades_by_currency: max count=1000, supports 'kind' filter, recent only
  - get_book_summary_by_currency: no limits, full surface in one call ← our main source
"""

import time
import logging
import requests
from typing import Optional

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("deribit")


def _get(method: str, params: dict = None) -> Optional[dict]:
    """Single Deribit public API call with retry + backoff."""
    url = f"{cfg.DERIBIT_BASE}/{method}"
    params = params or {}

    for attempt in range(cfg.MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=cfg.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                log.error(f"API error [{method}]: {data['error']}")
                return None
            return data.get("result")

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            log.warning(f"HTTP {status} [{method}] attempt {attempt+1}/{cfg.MAX_RETRIES}")
            # 400 = bad params, no point retrying with same params
            if status == 400:
                log.error(f"400 Bad Request on [{method}] — check params: {params}")
                return None
        except requests.exceptions.ConnectionError:
            log.warning(f"Connection error [{method}] attempt {attempt+1}/{cfg.MAX_RETRIES}")
        except requests.exceptions.Timeout:
            log.warning(f"Timeout [{method}] attempt {attempt+1}/{cfg.MAX_RETRIES}")
        except Exception as e:
            log.error(f"Unexpected error [{method}]: {e}")
            return None

        if attempt < cfg.MAX_RETRIES - 1:
            time.sleep(cfg.RETRY_DELAY * (attempt + 1))

    log.error(f"All retries failed for [{method}]")
    return None


# ── Surface snapshot (main data source) ───────────────────────

def get_surface_snapshot() -> list:
    """
    Full BTC option surface in one call.
    Bid/ask/mark/IV/OI for every active instrument.
    This is the PRIMARY source — works perfectly on public API.
    """
    result = _get("get_book_summary_by_currency", {
        "currency": cfg.CURRENCY,
        "kind":     cfg.KIND,
    })
    if result is None:
        log.error("Surface snapshot returned None")
        return []
    log.info(f"Surface snapshot: {len(result)} instruments")
    return result


# ── Trade history (secondary, more limited) ────────────────────

def get_trades_page(start_ms: int, end_ms: int) -> dict:
    """
    Fetch BTC option trades in a time window.

    Public API constraints:
      - Max count = 1000 (not 10000 — this caused your 400 errors)
      - 'kind' param NOT supported on this endpoint (filter client-side)
      - Historical data > ~7 days returns 400 without auth key
      - Use get_recent_trades() for anything older than 3 days
    """
    result = _get("get_last_trades_by_currency_and_time", {
        "currency":        cfg.CURRENCY,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
        "count":           1000,      # ← public API hard cap
        "sorting":         "desc",
        # NOTE: no 'kind' here — filter client-side below
    })
    if result is None:
        return {"trades": [], "has_more": False}

    # Filter to options client-side
    if isinstance(result, dict) and "trades" in result:
        result["trades"] = [
            t for t in result["trades"]
            if "-C" in t.get("instrument_name", "")
            or "-P" in t.get("instrument_name", "")
        ]
        return result

    return {"trades": [], "has_more": False}


def get_recent_trades(count: int = 1000) -> list:
    """
    Most recent BTC option trades — no time range, just latest N.
    Reliable fallback when timed endpoint returns 400.
    """
    result = _get("get_last_trades_by_currency", {
        "currency": cfg.CURRENCY,
        "kind":     cfg.KIND,
        "count":    min(count, 1000),
    })
    if result is None:
        return []
    return result.get("trades", [])


def get_instrument_trades(instrument_name: str, count: int = 100) -> list:
    """
    Trades for a specific instrument (e.g. BTC-27MAR26-98000-C).
    Use this to track when a specific violation-involved option last traded.
    """
    result = _get("get_last_trades_by_instrument", {
        "instrument_name": instrument_name,
        "count": min(count, 1000),
        "sorting": "desc",
    })
    if result is None:
        return []
    return result.get("trades", [])


# ── Other helpers ──────────────────────────────────────────────

def get_instruments(expired: bool = False) -> list:
    result = _get("get_instruments", {
        "currency": cfg.CURRENCY,
        "kind":     cfg.KIND,
        "expired":  str(expired).lower(),
    })
    return result or []


def get_index_price() -> Optional[float]:
    result = _get("get_index_price", {"index_name": "btc_usd"})
    if result:
        return result.get("index_price")
    return None


def get_ticker(instrument_name: str) -> Optional[dict]:
    return _get("get_ticker", {"instrument_name": instrument_name})
