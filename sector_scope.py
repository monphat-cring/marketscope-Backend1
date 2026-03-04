# sector_scope.py — Intra-sector relative scoring for each stock

import logging
from statistics import mean
from typing import Any, Dict, List

logger = logging.getLogger("sector_scope")


def calculate_sector_scope(sectors_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds scope_score and rs_in_sector to every stock in every sector.

    Scoring components (all direction-neutral, 0–5 output scale):
      40% — Intra-sector relative strength  (vs sector avg change_pct)
      30% — Volume rank within sector       (vs sector avg volume_ratio)
      20% — Absolute momentum               (abs change_pct)
      10% — R-Factor rank within sector     (rfactor percentile)

    Args:
        sectors_data: list of sector dicts, each with a "stocks" list.

    Returns:
        The same list with scope_score and rs_in_sector added in-place.
    """
    for sector in sectors_data:
        stocks = sector.get("stocks", [])
        if not stocks:
            continue

        try:
            # ── Sector averages ──
            sector_avg_change = mean(s.get("change_pct", 0.0) for s in stocks)
            vol_values = [s.get("volume_ratio", 1.0) for s in stocks]
            sector_avg_vol = mean(vol_values) if vol_values else 1.0

            # ── Pre-sort stocks by rfactor for rank (highest first = rank 0) ──
            sorted_by_rf = sorted(
                range(len(stocks)),
                key=lambda i: stocks[i].get("rfactor", 0.0),
                reverse=True,
            )
            # Map original index → rfactor rank
            rfactor_rank_of: Dict[int, int] = {}
            for rank, orig_idx in enumerate(sorted_by_rf):
                rfactor_rank_of[orig_idx] = rank

            total = len(stocks)

            for idx, stock in enumerate(stocks):
                try:
                    # COMPONENT 1 — Intra-sector RS (40%)
                    rs_in_sector = stock.get("change_pct", 0.0) - sector_avg_change
                    rs_score = max(0.0, min(1.0, (rs_in_sector + 5.0) / 10.0))

                    # COMPONENT 2 — Volume rank in sector (30%)
                    vol_ratio = stock.get("volume_ratio", 1.0)
                    vol_rank = vol_ratio / sector_avg_vol if sector_avg_vol > 0 else 1.0
                    vol_score = min(1.0, vol_rank / 2.0)

                    # COMPONENT 3 — Absolute momentum (20%)
                    abs_score = min(1.0, abs(stock.get("change_pct", 0.0)) / 5.0)

                    # COMPONENT 4 — R-Factor rank in sector (10%)
                    rf_score = 1.0 - (rfactor_rank_of[idx] / total)

                    # FINAL scope score (0–5)
                    raw = (
                        rs_score  * 0.40
                        + vol_score * 0.30
                        + abs_score * 0.20
                        + rf_score  * 0.10
                    )
                    scope_score = round(raw * 5, 2)

                    stock["scope_score"] = scope_score
                    stock["rs_in_sector"] = round(rs_in_sector, 2)

                except Exception as e:
                    logger.warning(f"scope_score failed for {stock.get('symbol')}: {e}")
                    stock["scope_score"] = 0.0
                    stock["rs_in_sector"] = 0.0

        except Exception as e:
            logger.warning(f"sector_scope failed for sector {sector.get('name')}: {e}")
            for stock in stocks:
                stock.setdefault("scope_score", 0.0)
                stock.setdefault("rs_in_sector", 0.0)

    return sectors_data
