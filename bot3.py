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
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception as e:
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
    api_key = os.getenv("OKX_API_KEY")
    api_secret = os.getenv("OKX_API_SECRET")
    password = os.getenv("OKX_API_PASSPHRASE")
    return ccxt.okx({
        "apiKey": api_key,
        "secret": api_secret,
        "password": password,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })


def top_symbols(ex: "ccxt.okx", limit: int = 20) -> List[str]:
    tickers = ex.fetch_tickers()
    pairs = []
    for sym, t in tickers.items():
        if t.get("symbol", "").endswith("/USDT:USDT"):
            vol = t.get("quoteVolume", 0)
            pairs.append((sym, vol))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:limit]]


def fetch_df(ex: "ccxt.okx", symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["return"] = d["close"].pct_change()
    d["rsi"] = ta.momentum.RSIIndicator(d["close"], window=14).rsi()
    d["ema"] = ta.trend.EMAIndicator(d["close"], window=20).ema_indicator()
    d["ema_diff"] = (d["close"] - d["ema"]) / d["ema"]
    d["target"] = (d["close"].shift(-1) > d["close"]).astype(int)
    return d.dropna()


class Bot3:
    def __init__(self) -> None:
        self.ex = get_exchange()
        self.telegram = Telegram()
        self.max_positions = 2
        self.usdt_per_trade = 100
        self.positions: Dict[str, Position] = {}

    def train_model(self, df: pd.DataFrame) -> LogisticRegression:
        feats = prepare_features(df)
        X = feats[["return", "rsi", "ema_diff"]].values
        y = feats["target"].values
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        return model

    def analyze(self, symbol: str) -> Optional[Position]:
        df30 = fetch_df(self.ex, symbol, "30m", 800)
        df60 = fetch_df(self.ex, symbol, "1h", 200)

        model = self.train_model(df30)
        feats = prepare_features(df30).iloc[-1]
        X_pred = [[feats["return"], feats["rsi"], feats["ema_diff"]]]
        prob_up = model.predict_proba(X_pred)[0][1]

        ema_fast = ta.trend.EMAIndicator(df60["close"], window=20).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(df60["close"], window=50).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df30["high"], df30["low"], df30["close"], window=14).average_true_range()

        price = df30["close"].iloc[-1]
        atr_val = atr.iloc[-1]
        size = self.usdt_per_trade / price

        if prob_up > 0.55 and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            sl = price - atr_val * 1.3
            tp = price + atr_val * 2.6
            return Position(symbol, "long", price, size, sl, tp)
        if prob_up < 0.45 and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            sl = price + atr_val * 1.3
            tp = price - atr_val * 2.6
            return Position(symbol, "short", price, size, sl, tp)
        return None

    def open_trade(self, p: Position) -> None:
        try:
            self.ex.private_post_trade_order({
                "instId": p.symbol.replace("/USDT:USDT", "-SWAP"),
                "tdMode": "cross",
                "side": "buy" if p.side == "long" else "sell",
                "ordType": "market",
                "sz": p.size,
                "lever": 10,
                "sl": p.stop_loss,
                "tp": p.take_profit,
            })
        except Exception:
            pass
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
    Bot3().run()
