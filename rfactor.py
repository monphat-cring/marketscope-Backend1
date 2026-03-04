import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger("rfactor")

def calculate_rsi(closes: pd.Series, period: int = 14) -> float:
    try:
        delta = closes.diff().dropna()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.dropna().iloc[-1])
        return round(val, 1) if not np.isnan(val) else 50.0
    except:
        return 50.0

def calculate_mfi(df: pd.DataFrame, period: int = 14) -> float:
    try:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        raw_mf = tp * df["Volume"]
        pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
        val = float(mfi.dropna().iloc[-1])
        return round(val, 1) if not np.isnan(val) else 50.0
    except:
        return 50.0

def get_rfactor_color_tier(score: float) -> str:
    # Thresholds on 0-5 scale
    if score >= 3.5: return "strong"
    if score >= 2.5: return "moderate"
    if score >= 1.5: return "weak"
    return "very_weak"

def calculate_rfactor_for_all(sym_data: Dict[str, Any], intraday_data, data_15min=None, nse_data=None) -> Dict[str, Any]:
    """
    Adds rfactor, rsi, mfi, relative_strength to each stock in sym_data.

    sym_data:      dict keyed by clean symbol (e.g. "TIINDIA")
    intraday_data: yfinance 5-min multi-ticker dataframe (VWAP / price action)
    data_15min:    yfinance 15-min multi-ticker dataframe (RSI / MFI)
    nse_data:      dict from nse_fetcher (delivery%, bid-ask, OI) — optional
    """

    def get_sym_df(data, symbol):
        """Extract a single symbol\'s DataFrame from a (possibly multi-ticker) result."""
        try:
            if data is None:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                if symbol in data.columns.get_level_values(0):
                    df = data[symbol]
                    return df if not df.empty else None
                return None
            else:
                return data
        except Exception:
            return None
    
    # Calculate nifty50 average change_pct for relative strength
    nifty50_symbols = [
        "RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK","SBIN","BHARTIARTL",
        "HCLTECH","WIPRO","AXISBANK","KOTAKBANK","LT","MARUTI","NTPC",
        "ONGC","POWERGRID","BAJFINANCE","M&M","TITAN","ADANIPORTS"
    ]
    nifty_changes = [
        sym_data[s]["change_pct"] 
        for s in nifty50_symbols 
        if s in sym_data
    ]
    nifty_avg = float(np.mean(nifty_changes)) if nifty_changes else 0.0
    
    total_syms = len(sym_data)
    
    for clean_sym, stock in sym_data.items():
        try:
            symbol_ns = clean_sym + ".NS"

            # 5-min df — used for VWAP and price action
            df = get_sym_df(intraday_data, symbol_ns)

            # 15-min df — used for RSI and MFI (more candles, less noise)
            df_15 = get_sym_df(data_15min, symbol_ns)
            
            # ── COMPONENT 1: Volume Score (weight 0.30) ──
            vol_ratio = stock.get("volume_ratio", 1.0)
            volume_score = min(vol_ratio / 3.0, 1.0)  # 3x avg = full score

            # ── COMPONENT 2: Price Action Score (weight 0.25) ──
            price_action_score = 0.0
            if df is not None and not df.empty and len(df) >= 6:
                try:
                    close = df["Close"].dropna()
                    volume = df["Volume"].dropna()

                    # VWAP deviation — both sides count as momentum
                    vwap = (close * volume).cumsum() / volume.cumsum()
                    ltp_val = float(close.iloc[-1])
                    vwap_val = float(vwap.iloc[-1])
                    vwap_pct = abs(ltp_val - vwap_val) / vwap_val * 100 if vwap_val != 0 else 0.0
                    vwap_score = min(1.0, vwap_pct / 1.5)

                    # Higher highs OR lower lows in last 6 candles = directional momentum
                    last6_high = df["High"].dropna().tail(6).values
                    last6_low  = df["Low"].dropna().tail(6).values
                    hh = len(last6_high) >= 2 and last6_high[-1] > last6_high[0]
                    ll = len(last6_low)  >= 2 and last6_low[-1]  < last6_low[0]
                    momentum_score = 1.0 if (hh or ll) else 0.0

                    price_action_score = (vwap_score * 0.5) + (momentum_score * 0.5)
                except:
                    price_action_score = 0.0
            else:
                # No candle data — use NSE live fields stored in the stock dict
                try:
                    ltp_val  = float(stock.get("ltp",      0) or 0)
                    vwap_nse = float(stock.get("vwap",     0) or 0)
                    day_high = float(stock.get("day_high", ltp_val) or ltp_val)
                    day_low  = float(stock.get("day_low",  ltp_val) or ltp_val)
                    day_range = day_high - day_low
                    # VWAP deviation
                    if vwap_nse > 0 and ltp_val > 0:
                        vwap_pct  = abs(ltp_val - vwap_nse) / vwap_nse * 100
                        vwap_score = min(1.0, vwap_pct / 1.5)
                    else:
                        vwap_score = 0.0
                    # Range extremity as momentum proxy
                    if day_range > 0:
                        center = (day_high + day_low) / 2
                        momentum_score = min(1.0, abs(ltp_val - center) / (day_range / 2))
                    else:
                        momentum_score = 0.5
                    price_action_score = (vwap_score * 0.5) + (momentum_score * 0.5)
                except:
                    price_action_score = 0.0

            # ── COMPONENT 3: RSI Score (weight 0.20) ──
            # Use 15-min candles for better signal; need ≥30 candles
            rsi_val = 50.0
            rsi_score = 0.0
            if df_15 is not None and not df_15.empty:
                close_15 = df_15["Close"].dropna()
                if len(close_15) >= 30:
                    rsi_val = calculate_rsi(close_15)
                    rsi_score = abs(rsi_val - 50) / 50  # RSI 80 or 20 → 0.60; RSI 50 → 0.00
                    rsi_score = max(0.0, min(1.0, rsi_score))

            # ── COMPONENT 4: MFI Score (weight 0.15) ──
            # Use 15-min candles; need ≥30 candles
            mfi_val = 50.0
            mfi_score = 0.0
            if df_15 is not None and not df_15.empty and len(df_15["Close"].dropna()) >= 30:
                mfi_val = calculate_mfi(df_15)
                mfi_score = abs(mfi_val - 50) / 50
                mfi_score = max(0.0, min(1.0, mfi_score))

            # ── COMPONENT 5: Relative Strength (weight 0.10) ──
            # Absolute deviation from Nifty — large move either direction scores high
            rs_val = round(stock["change_pct"] - nifty_avg, 2)
            rs_score = min(1.0, abs(rs_val) / 5.0)

            # ── NSE data for this symbol ──
            nse = nse_data.get(clean_sym, {}) if nse_data else {}

            # ── COMPONENT 6: Delivery % Score (weight 0.10) ──
            # High delivery = institutional conviction; neutral (0.5) if data unavailable
            delivery_pct = nse.get("delivery_pct", 0.0)
            if delivery_pct > 0:
                delivery_score = min(1.0, delivery_pct / 70.0)
            else:
                delivery_score = 0.5

            # ── COMPONENT 7: Bid-Ask Ratio Score (weight 0.05) ──
            # Both extreme buyers and extreme sellers = high activity
            ba_ratio = nse.get("bid_ask_ratio", 1.0)
            ba_score = min(1.0, abs(ba_ratio - 1.0) / 1.0)

            # ── COMPONENT 8: OI Change Score (weight 0.05, F&O only) ──
            oi_change = abs(nse.get("oi_change_pct", 0.0))
            oi_score = min(1.0, oi_change / 20.0) if stock.get("fo") else 0.5

            # ── FINAL R-FACTOR (0–5 scale) ──
            raw = (
                volume_score       * 0.20 +
                price_action_score * 0.20 +
                rsi_score          * 0.15 +
                mfi_score          * 0.15 +
                rs_score           * 0.10 +
                delivery_score     * 0.10 +
                ba_score           * 0.05 +
                oi_score           * 0.05
            )
            rfactor = round(raw * 5, 2)
            
            stock["rfactor"] = rfactor
            stock["rsi"] = rsi_val
            stock["mfi"] = mfi_val
            stock["relative_strength"] = rs_val
            stock["tier"] = get_rfactor_color_tier(rfactor)
            # Only update delivery/bid-ask from nse_data if the value is non-zero
            # (keeps the value already written by fetcher.py if trade_info was slow)
            if delivery_pct > 0:
                stock["delivery_pct"] = round(delivery_pct, 1)
            if nse and ba_ratio != 1.0:
                stock["bid_ask_ratio"] = round(ba_ratio, 2)
            
        except Exception as e:
            logger.warning(f"R-Factor failed for {clean_sym}: {e}")
            stock["rfactor"] = 0.0
            stock["rsi"] = 50.0
            stock["mfi"] = 50.0
            stock["relative_strength"] = 0.0
            stock["tier"] = "very_weak"
    
    return sym_data
