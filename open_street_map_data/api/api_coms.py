from datetime import datetime
import time
import random

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import requests
from parameter_settings import OVERPASS_URLS
from api_io import _log

def _overpass(query: str, timeout_s: int = 90, url: str = "") -> Dict[str, Any]:
    """Execute an Overpass QL query against a specific endpoint and return JSON.

    Args:
        query: Overpass QL query string.
        timeout_s: Request timeout in seconds.
        url: Overpass API endpoint; if empty, the caller should supply a default.

    Returns:
        Parsed JSON dict from the Overpass response.

    Raises:
        ValueError: If `query` is empty/whitespace.
        RuntimeError: If HTTP is not OK or response is not valid JSON.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Overpass query is empty/whitespace. Refusing to send request.")

    if not url:
        url = OVERPASS_URLS[0]

    r = requests.post(
        url,
        data={"data": q},
        timeout=timeout_s,
        headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
    )

    if not r.ok:
        snippet = (r.text or "")[:800]
        raise RuntimeError(
            f"Overpass HTTP {r.status_code} from {url}. QueryLen={len(q)}. ResponseSnippet={snippet!r}"
        )

    try:
        return r.json()
    except Exception:
        snippet = (r.text or "")[:800]
        raise RuntimeError(
            f"Overpass returned non-JSON from {url}. QueryLen={len(q)}. ResponseSnippet={snippet!r}"
        )


# Retry/fallback wrapper for Overpass queries
def _overpass_with_retries(
    query: str,
    timeout_s: int = 90,
    max_tries_per_server: int = 5,
    base_backoff_s: float = 1.0,
    max_backoff_s: float = 20.0,
) -> Dict[str, Any]:
    """Call Overpass with retries and server fallback.

    Tries each URL in `OVERPASS_URLS` up to `max_tries_per_server` times for
    transient failures (timeouts, 429/5xx), with exponential backoff + jitter.

    Args:
        query: Overpass QL query string.
        timeout_s: Request timeout in seconds.
        max_tries_per_server: Maximum attempts per server before switching.
        base_backoff_s: Initial backoff in seconds.
        max_backoff_s: Maximum backoff cap in seconds.

    Returns:
        Parsed JSON dict from the Overpass response.

    Raises:
        Exception: Re-raises the last encountered error after exhausting retries.
    """

    retryable_statuses = {408, 429, 500, 502, 503, 504}

    last_err: Optional[Exception] = None

    for server_idx, url in enumerate(OVERPASS_URLS, start=1):
        _log("INFO", f"Overpass using server {server_idx}/{len(OVERPASS_URLS)}: {url}")
        for attempt in range(1, max_tries_per_server + 1):
            try:
                return _overpass(query, timeout_s=timeout_s, url=url)
            except Exception as e:
                last_err = e

                # Decide whether to retry based on error content
                msg = str(e)
                is_retryable = any(f"HTTP {code}" in msg for code in retryable_statuses)

                # Also treat common gateway timeout strings as retryable
                if ("HTTP 504" in msg) or ("HTTP 429" in msg) or ("timed out" in msg.lower()):
                    is_retryable = True

                if not is_retryable:
                    # Non-transient: fail fast
                    raise

                # Backoff + jitter
                backoff = min(max_backoff_s, base_backoff_s * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, 0.5)
                sleep_s = backoff + jitter
                _log(
                    "WARN",
                    f"Overpass transient error (server {server_idx}/{len(OVERPASS_URLS)} attempt {attempt}/{max_tries_per_server}). "
                    f"Sleeping {sleep_s:.1f}s then retrying. Error={msg}"
                )
                time.sleep(sleep_s)

        if server_idx < len(OVERPASS_URLS):
            _log("WARN", f"Overpass exhausted retries on {url}. Switching to next server...")

    # Exhausted all servers/attempts
    red = "\033[91m"
    reset = "\033[0m"
    print(
        f"{red}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] "
        f"Overpass FAILED after {max_tries_per_server} tries on each server. Last error: {last_err}{reset}"
    )
    raise last_err if last_err is not None else RuntimeError("Overpass failed with unknown error")


def request_api_labels_multi(
    points: Sequence[Tuple[float, float]],
    radius_m: int = 20,
    timeout_s: int = 90,
    include_surface_features: bool = False,
) -> Dict[str, Any]:
    """Fetch nearby OSM ways for many points for many points in a single Overpass request.

    Builds a query that, for each (lon, lat), requests ways within `radius_m`.
    By default, it queries `highway=*` ways (which still returns surface/smoothness
    when present). Optionally, it also queries ways tagged with `surface` and
    `smoothness` explicitly.

    Args:
        points: Sequence of (lon, lat) tuples.
        radius_m: Search radius in metres for each point.
        timeout_s: Overpass query timeout in seconds.
        include_surface_features: If True, add extra way queries for `surface` and
            `smoothness` tags.

    Returns:
        Overpass JSON dict with an `elements` list.
    """
    points = list(points)
    if not points:
        return {"elements": []}

    _log("INFO", f"Building Overpass query for {len(points)} points (radius={radius_m}m, include_surface_features={include_surface_features})")

    blocks: List[str] = [
        f"[out:json][timeout:{int(timeout_s)}];",
        "(",
    ]

    for lon, lat in points:
        blocks.append(f'  way(around:{radius_m},{lat},{lon})["highway"];')

        if include_surface_features:
            blocks.append(f'  way(around:{radius_m},{lat},{lon})["surface"];')
            blocks.append(f'  way(around:{radius_m},{lat},{lon})["smoothness"];')

    blocks.append(");")
    blocks.append("out tags geom;")

    q = "\n".join(blocks)
    if not q.strip():
        raise ValueError("Overpass query is empty/whitespace. Refusing to send request.")
    _log("DEBUG", f"Overpass query length: {len(q)} characters")
    return _overpass_with_retries(q, timeout_s=timeout_s)
