"""Fetch company financial statements from SEC EDGAR's XBRL companyfacts API."""
from datetime import datetime
from typing import Dict, List, Optional

import requests

from merton_dd import config
from merton_dd.data.sec_tags import BROAD_FALLBACKS, TAG_MAP

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
PRIMARY_FORMS = ("10-Q", "10-K")


def lookup_cik(ticker: str) -> Optional[str]:
    resp = requests.get(TICKERS_URL, headers=config.SEC_HEADERS)
    resp.raise_for_status()
    ticker = ticker.upper().strip()
    return next(
        (str(v["cik_str"]).zfill(10) for v in resp.json().values() if v["ticker"] == ticker),
        None,
    )


def fetch_us_gaap_facts(cik: str) -> Dict:
    resp = requests.get(FACTS_URL.format(cik=cik), headers=config.SEC_HEADERS)
    resp.raise_for_status()
    return resp.json().get("facts", {}).get("us-gaap", {})


def _primary_filings(facts: Dict, tag: str, unit: str) -> List[Dict]:
    units = facts[tag].get("units", {})
    return [v for v in units.get(unit, []) if v.get("form") in PRIMARY_FORMS]


def _values_from_fallbacks(facts: Dict, tags: List[str]) -> List[Dict]:
    """Data points for the first tag in `tags` that the company reports."""
    for tag in tags:
        if tag in facts:
            units = facts[tag].get("units", {})
            for unit in ("USD", "shares", "USD/shares"):
                if unit in units:
                    return _primary_filings(facts, tag, unit)
    return []


def _fill_by_keyword_scan(periods: Dict[str, Dict], facts: Dict, field: str,
                          keywords, unit: str, default) -> None:
    missing = [d for d, row in periods.items() if row.get(field) is None]
    if not missing:
        return

    candidate_tags = [t for t in facts if all(k in t.lower() for k in keywords)]
    for tag in candidate_tags:
        for entry in _primary_filings(facts, tag, unit):
            row = periods.get(entry["end"])
            if row is not None and row.get(field) is None:
                row[field] = entry["val"]
                if field == "long_term_debt":
                    print(f"Filled {field} for {entry['end']} using {tag}")
        if all(periods[d].get(field) is not None for d in missing):
            break

    if default is not None:
        for d in missing:
            if periods[d].get(field) is None:
                periods[d][field] = default
                print(f"Set {field} to {default} for {d} (no data found)")


def get_company_financials(ticker: str, statement_type: str, years: int = 5) -> Optional[List[Dict]]:
    """
    Returns one dict per reporting period (newest first), each with
    'statement_date' plus every field in TAG_MAP[statement_type] (None if missing).
    """
    field_map = TAG_MAP.get(statement_type.lower(), {})
    try:
        cik = lookup_cik(ticker)
        if not cik:
            return None
        facts = fetch_us_gaap_facts(cik)

        # Collect every reported data point, keyed by period end date.
        periods: Dict[str, Dict] = {}
        for field, tags in field_map.items():
            for entry in _values_from_fallbacks(facts, tags):
                row = periods.setdefault(entry["end"], {
                    "statement_date": datetime.strptime(entry["end"], "%Y-%m-%d").date(),
                })
                row[field] = entry["val"]

        # Make every row have every field.
        for row in periods.values():
            for field in field_map:
                row.setdefault(field, None)

        # Broader searches for fields the model depends on.
        for field, keywords, unit, default in BROAD_FALLBACKS:
            if field in field_map:
                _fill_by_keyword_scan(periods, facts, field, keywords, unit, default)

        cutoff = datetime.now().year - years
        rows = sorted(periods.values(), key=lambda r: r["statement_date"], reverse=True)
        return [r for r in rows if r["statement_date"].year >= cutoff]

    except Exception as e:
        print(f"Error: {e}")
        return None
