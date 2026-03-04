# market_breadth.py — Market Breadth Dashboard (runs off cached sector data, no extra API calls)

import logging
from typing import Any, Dict, List

logger = logging.getLogger("market_breadth")

_NIFTY50_SECTOR = "NIFTY 50"


def get_market_breadth(sectors_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute market breadth from already-fetched sectors data.
    No new NSE/yfinance calls needed.

    Returns:
        advances, declines, unchanged, advance_decline_ratio,
        pct_above_vwap, pct_positive, nifty50_breadth,
        breadth_signal, sector_breadth, total_stocks
    """
    try:
        advances = 0
        declines = 0
        unchanged = 0
        above_vwap = 0
        total_with_vwap = 0

        # Unique stock tracking to avoid double-counting across sectors
        seen: Dict[str, Dict[str, Any]] = {}
        nifty50_symbols: set = set()
        sector_breadth: Dict[str, float] = {}

        # First pass: collect Nifty50 symbol set
        for sector in sectors_data:
            if sector.get("name") == _NIFTY50_SECTOR:
                for stock in sector.get("stocks", []):
                    sym = stock.get("symbol")
                    if sym:
                        nifty50_symbols.add(sym)

        # Second pass: compute breadth
        for sector in sectors_data:
            sector_name = sector.get("name", "")
            stocks = sector.get("stocks", [])
            if not stocks:
                continue

            sector_positive = 0
            sector_total = 0

            for stock in stocks:
                sym = stock.get("symbol")
                if not sym:
                    continue

                change_pct = float(stock.get("change_pct", 0) or 0)
                vwap = float(stock.get("vwap", 0) or 0)
                ltp = float(stock.get("ltp", 0) or 0)

                # Sector-level breadth counts all appearances (not deduplicated)
                sector_total += 1
                if change_pct > 0:
                    sector_positive += 1

                # Global counts are deduplicated by symbol
                if sym not in seen:
                    seen[sym] = stock
                    if change_pct > 0.05:
                        advances += 1
                    elif change_pct < -0.05:
                        declines += 1
                    else:
                        unchanged += 1

                    if vwap > 0 and ltp > 0:
                        total_with_vwap += 1
                        if ltp > vwap:
                            above_vwap += 1

            if sector_total > 0:
                sector_breadth[sector_name] = round(
                    sector_positive / sector_total * 100, 1
                )

        total = advances + declines + unchanged
        adr = round(advances / declines, 2) if declines > 0 else float(advances)
        pct_positive = round(advances / total * 100, 1) if total > 0 else 0.0
        pct_above_vwap = (
            round(above_vwap / total_with_vwap * 100, 1)
            if total_with_vwap > 0
            else 0.0
        )

        # Nifty50 breadth
        nifty50_positive = sum(
            1
            for sym, s in seen.items()
            if sym in nifty50_symbols and float(s.get("change_pct", 0) or 0) > 0
        )
        nifty50_breadth = (
            round(nifty50_positive / len(nifty50_symbols) * 100, 1)
            if nifty50_symbols
            else 0.0
        )

        # Overall signal
        ratio = advances / total if total > 0 else 0.0
        if ratio > 0.65:
            breadth_signal = "STRONG"
        elif ratio > 0.50:
            breadth_signal = "MODERATE"
        elif ratio > 0.35:
            breadth_signal = "WEAK"
        else:
            breadth_signal = "VERY_WEAK"

        # sector_breadth_list: sorted array version for easier frontend rendering
        sector_breadth_list = [
            {"name": k, "pct_positive": v}
            for k, v in sorted(sector_breadth.items(), key=lambda x: -x[1])
        ]

        return {
            "advances":             advances,
            "declines":             declines,
            "unchanged":            unchanged,
            "advance_decline_ratio": adr,
            "adr":                  adr,          # short-hand alias
            "pct_above_vwap":       pct_above_vwap,
            "pct_positive":         pct_positive,
            "nifty50_breadth":      nifty50_breadth,
            "breadth_signal":       breadth_signal,
            "sector_breadth":       sector_breadth,
            "sector_breadth_list":  sector_breadth_list,
            "total_stocks":         total,
        }

    except Exception as e:
        logger.error("get_market_breadth failed: %s", e, exc_info=True)
        return {
            "advances": 0,
            "declines": 0,
            "unchanged": 0,
            "advance_decline_ratio": 0.0,
            "adr": 0.0,
            "pct_above_vwap": 0.0,
            "pct_positive": 0.0,
            "nifty50_breadth": 0.0,
            "breadth_signal": "WEAK",
            "sector_breadth": {},
            "sector_breadth_list": [],
            "total_stocks": 0,
        }
