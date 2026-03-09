import math
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger("rfactor")


# ---------------------------------------------------------------------------
# Indicator calculators
# ---------------------------------------------------------------------------

def calculate_rsi(closes: pd.Series, period: int = 14) -> float:
    try:
        delta = closes.diff().dropna()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs    = gain / loss
        rsi   = 100 - (100 / (1 + rs))
        val   = float(rsi.dropna().iloc[-1])
        return round(val, 1) if not np.isnan(val) else 50.0
    except Exception:
        return 50.0


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> float:
    try:
        tp      = (df["High"] + df["Low"] + df["Close"]) / 3
        raw_mf  = tp * df["Volume"]
        pos_mf  = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf  = raw_mf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()
        mfr     = pos_sum / neg_sum.replace(0, np.nan)
        mfi     = 100 - (100 / (1 + mfr))
        val     = float(mfi.dropna().iloc[-1])
        return round(val, 1) if not np.isnan(val) else 50.0
    except Exception:
        return 50.0


def get_rfactor_color_tier(score: float) -> str:
    """Map 0–5 rfactor to a display tier."""
    if score >= 3.5: return "strong"
    if score >= 2.5: return "moderate"
    if score >= 1.5: return "weak"
    return "very_weak"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _volume_score(vol_ratio: float) -> float:
    """
    Logarithmic volume score so a 10x surge (1.0) is meaningfully higher
    than a 3x surge (0.58) rather than both capping at 1.0.
    log2(1+10) ≈ 3.46 used as normaliser so 10x = full 1.0.
    """
    return min(1.0, math.log2(1.0 + max(0.0, vol_ratio)) / math.log2(11.0))


def _oscillator_alignment_score(osc_val: float, change_pct: float) -> float:
    """
    Score an oscillator (RSI or MFI) based on how well it *aligns*
    with the price direction, rather than just measuring raw extremity.

    Bullish move (change_pct > 0):
        osc 60–70  → 1.00  ideal momentum confirmation
        osc 70–80  → 0.85  strong but approaching OB risk
        osc > 80   → 0.60  overbought — reversal risk
        osc 50–60  → 0.55  building, mild confirmation
        osc 40–50  → 0.25  weak / barely confirming
        osc < 40   → 0.00  divergence — RSI falling while price rising

    Bearish move (change_pct < 0): mirror image.
    Flat move (|change_pct| < 0.1): proximity to 50 = neutrality.
    """
    if abs(change_pct) < 0.1:
        # Flat stock — reward proximity to 50 (neutrally rangebound)
        return max(0.0, 1.0 - abs(osc_val - 50) / 50)

    if change_pct > 0:   # bullish
        if   osc_val >= 80: return 0.60
        elif osc_val >= 70: return 0.85
        elif osc_val >= 60: return 1.00
        elif osc_val >= 50: return 0.55
        elif osc_val >= 40: return 0.25
        else:               return 0.00   # bearish divergence
    else:                # bearish
        if   osc_val <= 20: return 0.60
        elif osc_val <= 30: return 0.85
        elif osc_val <= 40: return 1.00
        elif osc_val <= 50: return 0.55
        elif osc_val <= 60: return 0.25
        else:               return 0.00   # bullish divergence


def _bid_ask_alignment_score(ba_ratio: float, change_pct: float) -> float:
    """
    Score bid-ask ratio directionally:
      Bullish move → high bid_ask (buyers dominating) = good alignment
      Bearish move → low  bid_ask (sellers dominating) = good alignment
    Returns 0.0 when data is missing (ba_ratio == 1.0 default).
    """
    if ba_ratio == 1.0:
        return 0.0  # no data — contribute nothing
    if change_pct >= 0:
        # bid_ask > 1 means buy-side pressure; scale: 2.0 = full score
        return min(1.0, max(0.0, (ba_ratio - 1.0) / 1.0))
    else:
        # bid_ask < 1 means sell-side pressure
        return min(1.0, max(0.0, (1.0 - ba_ratio) / 1.0))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _compute_rfactor_snapshot(
    change_pct: float,
    vol_ratio: float,
    ltp_val: float,
    vwap_val: float,
    day_high: float,
    day_low: float,
    close_series: pd.Series,
    df_hist: pd.DataFrame,
    delivery_pct: float,
    ba_ratio: float,
    nifty_avg: float,
) -> Dict[str, float]:
    c1_volume = _volume_score(vol_ratio)

    c2_price = 0.0
    try:
        day_range = day_high - day_low
        if vwap_val > 0 and ltp_val > 0:
            vwap_pct = abs(ltp_val - vwap_val) / vwap_val * 100
            vwap_sub = min(1.0, vwap_pct / 2.0)
        else:
            vwap_sub = 0.0

        if day_range > 0 and ltp_val > 0:
            if change_pct >= 0:
                range_sub = (ltp_val - day_low) / day_range
            else:
                range_sub = (day_high - ltp_val) / day_range
            range_sub = _clamp(range_sub)
        else:
            range_sub = 0.5

        c2_price = vwap_sub * 0.5 + range_sub * 0.5
    except Exception:
        c2_price = 0.0

    rsi_val = 50.0
    c3_rsi = 0.0
    if len(close_series) >= 30:
        rsi_val = calculate_rsi(close_series)
        c3_rsi = _oscillator_alignment_score(rsi_val, change_pct)

    mfi_val = 50.0
    c4_mfi = 0.0
    if len(close_series) >= 30:
        mfi_val = calculate_mfi(df_hist)
        c4_mfi = _oscillator_alignment_score(mfi_val, change_pct)

    rs_val = round(change_pct - nifty_avg, 2)
    c5_rs = min(1.0, abs(rs_val) / 5.0)

    c6_trend = 0.0
    if len(close_series) >= 20:
        ma20 = float(close_series.tail(20).mean())
        ltp_close = float(close_series.iloc[-1])
        if ma20 > 0:
            pct_from_ma = (ltp_close - ma20) / ma20 * 100
            if change_pct >= 0 and ltp_close > ma20:
                c6_trend = min(1.0, pct_from_ma / 3.0)
            elif change_pct < 0 and ltp_close < ma20:
                c6_trend = min(1.0, abs(pct_from_ma) / 3.0)

    c7_delivery = min(1.0, delivery_pct / 70.0) if delivery_pct > 0 else 0.0
    c8_ba = _bid_ask_alignment_score(ba_ratio, change_pct)

    raw = (
        c1_volume * 0.25 +
        c2_price * 0.20 +
        c3_rsi * 0.15 +
        c4_mfi * 0.10 +
        c5_rs * 0.10 +
        c6_trend * 0.10 +
        c7_delivery * 0.05 +
        c8_ba * 0.05
    )

    return {
        "rfactor": round(raw * 5.0, 2),
        "rsi": round(rsi_val, 1),
        "mfi": round(mfi_val, 1),
        "relative_strength": rs_val,
    }


def _get_setup_stage(rfactor: float, trend_15m: float, extension_penalty: float) -> str:
    if rfactor >= 3.0 and (extension_penalty >= 0.55 or trend_15m <= 0.05):
        return "Extended"
    if rfactor >= 2.0 and trend_15m >= 0.20 and extension_penalty < 0.65:
        return "Triggering"
    if rfactor >= 1.2 and trend_15m >= 0.08:
        return "Building"
    return "Neutral"


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

def calculate_rfactor_for_all(
    sym_data: Dict[str, Any],
    intraday_data,          # unused (kept for API compatibility)
    data_15min=None,
    nse_data=None,
) -> Dict[str, Any]:
    """
    Adds rfactor, rsi, mfi, relative_strength to every stock in sym_data.

    Component weights (sum = 1.00):
        Volume score          0.25  — how unusual is today's volume?
        Price action          0.20  — VWAP deviation + candle structure
        RSI alignment         0.15  — does RSI confirm price direction?
        MFI alignment         0.10  — does money-flow confirm direction?
        Relative strength     0.10  — outperforming / underperforming Nifty?
        Trend context         0.10  — is price on the right side of the 20-MA?
        Delivery %            0.05  — institutional activity (0 if unavailable)
        Bid-ask alignment     0.05  — order-book pressure matches direction?
    """

    def _get_sym_df(data, symbol):
        try:
            if data is None:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                if symbol in data.columns.get_level_values(0):
                    df = data[symbol]
                    return df if not df.empty else None
                return None
            return data
        except Exception:
            return None

    def _get_latest_session_df(df: pd.DataFrame):
        try:
            if df is None or df.empty:
                return None
            latest_date = df.index[-1].date()
            session_df = df[df.index.date == latest_date].copy()
            return session_df if not session_df.empty else None
        except Exception:
            return None

    def _get_prev_session_close(df: pd.DataFrame, session_date, fallback: float) -> float:
        try:
            previous_rows = df[df.index.date < session_date]
            close_series = previous_rows["Close"].dropna()
            if not close_series.empty:
                return float(close_series.iloc[-1])
        except Exception:
            pass
        return fallback

    # ── Nifty-50 basket average for relative strength baseline ──────────────
    _NIFTY50 = [
        "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "SBIN",
        "BHARTIARTL", "HCLTECH", "WIPRO", "AXISBANK", "KOTAKBANK", "LT",
        "MARUTI", "NTPC", "ONGC", "POWERGRID", "BAJFINANCE", "M&M",
        "TITAN", "ADANIPORTS",
    ]
    nifty_changes = [sym_data[s]["change_pct"] for s in _NIFTY50 if s in sym_data]
    nifty_avg = float(np.mean(nifty_changes)) if nifty_changes else 0.0

    for clean_sym, stock in sym_data.items():
        try:
            symbol_ns  = clean_sym + ".NS"
            df_15      = _get_sym_df(data_15min, symbol_ns)
            change_pct = float(stock.get("change_pct", 0) or 0)

            # ── C1: Volume score (0.25) ─────────────────────────────────────
            vol_ratio    = float(stock.get("volume_ratio", 1.0) or 1.0)
            c1_volume    = _volume_score(vol_ratio)

            ltp_val  = float(stock.get("ltp", 0) or 0)
            vwap_nse = float(stock.get("vwap", 0) or 0)
            day_high = float(stock.get("day_high", ltp_val) or ltp_val)
            day_low  = float(stock.get("day_low", ltp_val) or ltp_val)

            # ── C2: Price action score (0.20) ───────────────────────────────
            # Uses NSE live fields (day_high, day_low, vwap, ltp) which are
            # always present — no candle dependency.
            c2_price = 0.0
            try:
                day_range = day_high - day_low

                # Sub-score A: VWAP deviation (price committed away from VWAP)
                if vwap_nse > 0 and ltp_val > 0:
                    vwap_pct   = abs(ltp_val - vwap_nse) / vwap_nse * 100
                    vwap_sub   = min(1.0, vwap_pct / 2.0)  # 2% away = full score
                else:
                    vwap_sub   = 0.0

                # Sub-score B: range extremity (near high on up day, near low on down day)
                if day_range > 0 and ltp_val > 0:
                    if change_pct >= 0:
                        # Bullish: % of range captured from low (higher = stronger)
                        range_sub = (ltp_val - day_low) / day_range
                    else:
                        # Bearish: % of range dropped from high (higher = stronger)
                        range_sub = (day_high - ltp_val) / day_range
                    range_sub = max(0.0, min(1.0, range_sub))
                else:
                    range_sub = 0.5

                c2_price = vwap_sub * 0.5 + range_sub * 0.5
            except Exception:
                c2_price = 0.0

            # ── C3: RSI alignment (0.15) ────────────────────────────────────
            rsi_val = 50.0
            c3_rsi  = 0.0
            if df_15 is not None and not df_15.empty:
                close_15 = df_15["Close"].dropna()
                if len(close_15) >= 30:
                    rsi_val = calculate_rsi(close_15)
                    c3_rsi  = _oscillator_alignment_score(rsi_val, change_pct)

            # ── C4: MFI alignment (0.10) ────────────────────────────────────
            mfi_val = 50.0
            c4_mfi  = 0.0
            if df_15 is not None and not df_15.empty:
                if len(df_15["Close"].dropna()) >= 30:
                    mfi_val = calculate_mfi(df_15)
                    c4_mfi  = _oscillator_alignment_score(mfi_val, change_pct)

            # ── C5: Relative strength vs Nifty (0.10) ───────────────────────
            rs_val  = round(change_pct - nifty_avg, 2)
            # Score both outperformers (rs > 0) and hard underperformers (rs < 0)
            c5_rs   = min(1.0, abs(rs_val) / 5.0)

            # ── C6: Trend context — price vs 20-period MA (0.10) ────────────
            # On a bullish day, price above the 20-MA confirms trend alignment.
            # On a bearish day, price below the 20-MA confirms trend alignment.
            # If MA unavailable → 0.0 (no free points).
            c6_trend = 0.0
            if df_15 is not None and not df_15.empty:
                close_15_all = df_15["Close"].dropna()
                if len(close_15_all) >= 20:
                    ma20  = float(close_15_all.tail(20).mean())
                    ltp_c = float(close_15_all.iloc[-1])
                    if ma20 > 0:
                        pct_from_ma = (ltp_c - ma20) / ma20 * 100
                        if change_pct >= 0 and ltp_c > ma20:
                            # Bullish and above MA — reward proportionally
                            c6_trend = min(1.0, pct_from_ma / 3.0)
                        elif change_pct < 0 and ltp_c < ma20:
                            # Bearish and below MA — reward proportionally
                            c6_trend = min(1.0, abs(pct_from_ma) / 3.0)
                        else:
                            # Price fighting against MA — weak trend context
                            c6_trend = 0.0

            # ── C7: Delivery % (0.05) ───────────────────────────────────────
            # Only scores when real data is available — no fake neutral fallback.
            nse          = nse_data.get(clean_sym, {}) if nse_data else {}
            delivery_pct = float(nse.get("delivery_pct", 0.0) or
                                 stock.get("delivery_pct", 0.0) or 0.0)
            c7_delivery  = min(1.0, delivery_pct / 70.0) if delivery_pct > 0 else 0.0

            # ── C8: Bid-ask alignment (0.05) ────────────────────────────────
            ba_ratio = float(nse.get("bid_ask_ratio", 1.0) or
                             stock.get("bid_ask_ratio", 1.0) or 1.0)
            c8_ba    = _bid_ask_alignment_score(ba_ratio, change_pct)

            # ── Final R-Factor (weighted sum → 0–5 scale) ───────────────────
            raw = (
                c1_volume    * 0.25 +
                c2_price     * 0.20 +
                c3_rsi       * 0.15 +
                c4_mfi       * 0.10 +
                c5_rs        * 0.10 +
                c6_trend     * 0.10 +
                c7_delivery  * 0.05 +
                c8_ba        * 0.05
            )
            rfactor = round(raw * 5.0, 2)

            stock["rfactor"]          = rfactor
            stock["rsi"]              = rsi_val
            stock["mfi"]              = mfi_val
            stock["relative_strength"]= rs_val
            stock["tier"]             = get_rfactor_color_tier(rfactor)

            recent_points = [rfactor]
            trend_15m = 0.0
            trend_acceleration = 0.0
            if df_15 is not None and not df_15.empty:
                session_df = _get_latest_session_df(df_15)
                if session_df is not None and len(session_df) >= 2:
                    session_date = session_df.index[-1].date()
                    session_open = float(session_df["Open"].dropna().iloc[0]) if not session_df["Open"].dropna().empty else ltp_val
                    prev_session_close = _get_prev_session_close(df_15, session_date, session_open)
                    cum_volume = session_df["Volume"].fillna(0.0).cumsum()
                    final_cum_volume = float(cum_volume.iloc[-1]) if not cum_volume.empty else 0.0
                    session_tp = (session_df["High"] + session_df["Low"] + session_df["Close"]) / 3
                    session_vwap = ((session_tp * session_df["Volume"].fillna(0.0)).cumsum() /
                                    cum_volume.replace(0, np.nan))

                    start_index = max(0, len(session_df) - 3)
                    points = []
                    for idx in range(start_index, len(session_df)):
                        hist_df = session_df.iloc[:idx + 1].copy()
                        close_series_hist = df_15.iloc[: df_15.index.get_loc(hist_df.index[-1]) + 1]["Close"].dropna()
                        ltp_hist = float(hist_df["Close"].dropna().iloc[-1]) if not hist_df["Close"].dropna().empty else ltp_val
                        change_hist = change_pct
                        if prev_session_close > 0:
                            change_hist = round(((ltp_hist - prev_session_close) / prev_session_close) * 100, 2)
                        progress = (float(cum_volume.iloc[idx]) / final_cum_volume) if final_cum_volume > 0 else 1.0
                        hist_snapshot = _compute_rfactor_snapshot(
                            change_pct=change_hist,
                            vol_ratio=max(0.0, vol_ratio * progress),
                            ltp_val=ltp_hist,
                            vwap_val=float(session_vwap.iloc[idx]) if not pd.isna(session_vwap.iloc[idx]) else 0.0,
                            day_high=float(hist_df["High"].max() or ltp_hist),
                            day_low=float(hist_df["Low"].min() or ltp_hist),
                            close_series=close_series_hist,
                            df_hist=df_15.iloc[: df_15.index.get_loc(hist_df.index[-1]) + 1],
                            delivery_pct=delivery_pct,
                            ba_ratio=ba_ratio,
                            nifty_avg=nifty_avg,
                        )
                        points.append(hist_snapshot["rfactor"])

                    if points:
                        points[-1] = rfactor
                        recent_points = [round(point, 2) for point in points]
                        if len(recent_points) >= 2:
                            trend_15m = round(recent_points[-1] - recent_points[-2], 2)
                        if len(recent_points) >= 3:
                            trend_acceleration = round(
                                (recent_points[-1] - recent_points[-2]) - (recent_points[-2] - recent_points[-3]),
                                2,
                            )

            extension_pct = abs(ltp_val - vwap_nse) / vwap_nse * 100 if vwap_nse > 0 else 0.0
            rsi_stretch = max(0.0, (rsi_val - 70.0) / 20.0) if change_pct >= 0 else max(0.0, (30.0 - rsi_val) / 20.0)
            extension_penalty = _clamp((extension_pct / 2.5) * 0.6 + _clamp(rsi_stretch) * 0.4)
            positive_trend = _clamp(trend_15m / 0.75)
            positive_acceleration = _clamp(trend_acceleration / 0.50)
            base_quality = _clamp(rfactor / 3.5)
            freshness = 1.0 if 1.5 <= rfactor <= 3.2 else (0.75 if rfactor < 1.5 else 0.45)
            opportunity_raw = (
                base_quality * 0.35 +
                positive_trend * 0.30 +
                positive_acceleration * 0.10 +
                (1.0 - extension_penalty) * 0.25
            ) * freshness
            opportunity_score = round(opportunity_raw * 100.0, 1)
            setup_stage = _get_setup_stage(rfactor, trend_15m, extension_penalty)

            stock["rfactor_trend_15m"] = trend_15m
            stock["rfactor_trend_acceleration"] = trend_acceleration
            stock["rfactor_trend_points"] = recent_points
            stock["opportunity_score"] = opportunity_score
            stock["setup_stage"] = setup_stage

            # Persist delivery / bid-ask only when we have real values
            if delivery_pct > 0:
                stock["delivery_pct"] = round(delivery_pct, 1)
            if ba_ratio != 1.0:
                stock["bid_ask_ratio"] = round(ba_ratio, 2)

        except Exception as e:
            logger.warning("R-Factor failed for %s: %s", clean_sym, e)
            stock.setdefault("rfactor",           0.0)
            stock.setdefault("rsi",               50.0)
            stock.setdefault("mfi",               50.0)
            stock.setdefault("relative_strength", 0.0)
            stock.setdefault("tier",              "very_weak")
            stock.setdefault("rfactor_trend_15m", 0.0)
            stock.setdefault("rfactor_trend_acceleration", 0.0)
            stock.setdefault("rfactor_trend_points", [stock.get("rfactor", 0.0)])
            stock.setdefault("opportunity_score", 0.0)
            stock.setdefault("setup_stage", "Neutral")

    return sym_data
