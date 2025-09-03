import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import requests

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    import ccxt  # type: ignore
    import pandas as pd  # type: ignore
    import ta  # type: ignore
except Exception as e:  # pragma: no cover - dependencies may be missing at runtime
    raise SystemExit(f"Missing dependency: {e}")


@dataclass
class Position:
    symbol: str
    side: str
    entry: float
    size: float
    stop_loss: float
    take_profit: float


class Telegram:
    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    def send(self, msg: str) -> None:
        if not self.token or not self.chat_id:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.chat_id, "text": msg})
        except Exception:
            pass


def get_exchange() -> "ccxt.okx":
    api_key = os.getenv("OKX_API_KEY") or os.getenv("OKX_KEY")
    api_secret = os.getenv("OKX_API_SECRET") or os.getenv("OKX_SECRET")
    password = os.getenv("OKX_API_PASSPHRASE") or os.getenv("OKX_PASSWORD")
    exchange = ccxt.okx(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "password": password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    if os.getenv("OKX_DEMO") == "1":
        exchange.set_sandbox_mode(True)
    return exchange


def top_symbols(ex: "ccxt.okx", limit: int = 20) -> List[str]:
    tickers = ex.fetch_tickers()
    pairs = []
    for sym, t in tickers.items():
        if str(t.get("symbol", "")).endswith("/USDT:USDT"):
            vol = t.get("quoteVolume") or 0
            try:
                vol = float(vol)
            except (TypeError, ValueError):
                vol = 0.0
            pairs.append((sym, vol))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:limit]]


def fetch_df(ex: "ccxt.okx", symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


class Bot1:
    def __init__(self) -> None:
        self.ex = get_exchange()
        self.telegram = Telegram()
        self.max_positions = int(os.getenv("MAX_CONCURRENT", "2"))
        self.usdt_per_trade = float(os.getenv("POSITION_NOTIONAL_USD", "100"))
        self.margin_mode = os.getenv("MARGIN_MODE", "cross")
        self.leverage = float(os.getenv("LEVERAGE", "10"))
        self.positions: Dict[str, Position] = {}

    def analyze(self, symbol: str) -> Optional[Position]:
        df30 = fetch_df(self.ex, symbol, "30m", 800)
        df60 = fetch_df(self.ex, symbol, "1h", 200)

        ema_fast = ta.trend.EMAIndicator(df30["close"], window=9).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(df30["close"], window=21).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df30["high"], df30["low"], df30["close"], window=14).average_true_range()
        trend_h1_fast = ta.trend.EMAIndicator(df60["close"], window=20).ema_indicator()
        trend_h1_slow = ta.trend.EMAIndicator(df60["close"], window=50).ema_indicator()

        price = df30["close"].iloc[-1]
        atr_val = atr.iloc[-1]

        long_cond = ema_fast.iloc[-1] > ema_slow.iloc[-1] and trend_h1_fast.iloc[-1] > trend_h1_slow.iloc[-1]
        short_cond = ema_fast.iloc[-1] < ema_slow.iloc[-1] and trend_h1_fast.iloc[-1] < trend_h1_slow.iloc[-1]

        size = self.usdt_per_trade / price
        if long_cond:
            sl = price - atr_val * 1.5
            tp = price + atr_val * 3
            return Position(symbol, "long", price, size, sl, tp)
        if short_cond:
            sl = price + atr_val * 1.5
            tp = price - atr_val * 3
            return Position(symbol, "short", price, size, sl, tp)
        return None

    def open_trade(self, p: Position) -> None:
        try:
            inst_id = p.symbol.replace("/USDT:USDT", "-SWAP")
            resp = self.ex.private_post_trade_order(
                {
                    "instId": inst_id,
                    "tdMode": self.margin_mode,
                    "side": "buy" if p.side == "long" else "sell",
                    "ordType": "market",
                    "sz": p.size,
                    "lever": self.leverage,
                    "sl": p.stop_loss,
                    "tp": p.take_profit,
                }
            )
            if resp.get("code") != "0":
                raise Exception(resp.get("msg"))
        except Exception as e:
            self.telegram.send(f"Order failed for {p.symbol}: {e}")
            return
        self.positions[p.symbol] = p
        self.telegram.send(f"Opened {p.side} {p.symbol} @ {p.entry:.4f}")

    def manage_positions(self) -> None:
        for sym, pos in list(self.positions.items()):
            ticker = self.ex.fetch_ticker(sym)
            price = ticker["last"]
            if pos.side == "long":
                if price <= pos.stop_loss or price >= pos.take_profit:
                    self.positions.pop(sym, None)
                    self.telegram.send(f"Closed {sym} at {price:.4f}")
            else:
                if price >= pos.stop_loss or price <= pos.take_profit:
                    self.positions.pop(sym, None)
                    self.telegram.send(f"Closed {sym} at {price:.4f}")

    def run(self) -> None:
        symbols = top_symbols(self.ex)
        while True:
            self.manage_positions()
            if len(self.positions) < self.max_positions:
                for s in symbols:
                    if s in self.positions:
                        continue
                    pos = self.analyze(s)
                    if pos:
                        self.open_trade(pos)
                        if len(self.positions) >= self.max_positions:
                            break
            time.sleep(60)


if __name__ == "__main__":
    Bot1().run()
