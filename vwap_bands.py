# vwap_bands.py — VWAP Standard Deviation Bands
#
# Uses day high/low as an intraday range proxy to estimate std_dev.
# Formula: std_dev ≈ (day_high - day_low) / 4
# Then bands: VWAP ± 1σ and VWAP ± 2σ

import logging
from typing import Any, Dict

logger = logging.getLogger("vwap_bands")


def calculate_vwap_bands(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds VWAP band fields to the stock dict in-place and returns it.

    Added fields:
        vwap_position   : EXTENDED_ABOVE | ABOVE | SLIGHTLY_ABOVE | AT_VWAP |
                          BELOW | EXTENDED_BELOW | UNKNOWN
        band_1_upper    : VWAP + 1σ
        band_1_lower    : VWAP - 1σ
        band_2_upper    : VWAP + 2σ
        band_2_lower    : VWAP - 2σ
        vwap_std_dev    : estimated intraday std_dev
    """
    try:
        ltp      = float(stock.get("ltp",      0) or 0)
        vwap     = float(stock.get("vwap",     0) or 0)
        day_high = float(stock.get("day_high", ltp) or ltp)
        day_low  = float(stock.get("day_low",  ltp) or ltp)

        if vwap <= 0 or ltp <= 0:
            stock["vwap_position"] = "UNKNOWN"
            return stock

        std_dev = (day_high - day_low) / 4.0
        if std_dev <= 0:
            std_dev = vwap * 0.005  # 0.5% of VWAP as minimum

        b1u = round(vwap + std_dev,       2)
        b1l = round(vwap - std_dev,       2)
        b2u = round(vwap + 2 * std_dev,   2)
        b2l = round(vwap - 2 * std_dev,   2)

        if ltp > b2u:
            pos = "EXTENDED_ABOVE"
        elif ltp > b1u:
            pos = "ABOVE"
        elif ltp > vwap:
            pos = "SLIGHTLY_ABOVE"
        elif ltp < b2l:
            pos = "EXTENDED_BELOW"
        elif ltp < b1l:
            pos = "BELOW"
        else:
            pos = "AT_VWAP"

        stock["vwap_position"] = pos
        stock["band_1_upper"]  = b1u
        stock["band_1_lower"]  = b1l
        stock["band_2_upper"]  = b2u
        stock["band_2_lower"]  = b2l
        stock["vwap_std_dev"]  = round(std_dev, 2)

    except Exception as e:
        logger.warning(
            "calculate_vwap_bands error for %s: %s",
            stock.get("symbol", "?"), e
        )
        stock["vwap_position"] = "UNKNOWN"

    return stock
