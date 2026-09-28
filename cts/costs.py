"""Transaction cost model (docs/strategy_research.md §9.3). All rates are fractions of price."""
from __future__ import annotations

from dataclasses import dataclass

BP = 1e-4

# (half-spread bp, market slippage bp) by liquidity tier
TIERS = {
    "BTCUSDT": (0.5, 1.0), "ETHUSDT": (0.5, 1.0),
    "SOLUSDT": (1.0, 3.0), "BNBUSDT": (1.0, 3.0), "XRPUSDT": (1.0, 3.0), "DOGEUSDT": (1.0, 3.0),
}
DEFAULT_TIER = (2.0, 5.0)


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    half_spread: float = 2.0 * BP
    slippage: float = 5.0 * BP
    stop_slip_mult: float = 2.0
    shock_mult: float = 3.0
    multiplier: float = 1.0  # stress-test knob: 2.0 = double all costs

    @classmethod
    def for_symbol(cls, symbol: str, multiplier: float = 1.0) -> "CostModel":
        hs, sl = TIERS.get(symbol, DEFAULT_TIER)
        return cls(half_spread=hs * BP, slippage=sl * BP, multiplier=multiplier)

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, slippage=0.0)

    def market_impact(self, shock: bool = False) -> float:
        """Adverse price move (fraction) for a market order."""
        m = self.shock_mult if shock else 1.0
        return (self.half_spread + self.slippage * m) * self.multiplier

    def stop_impact(self, shock: bool = False) -> float:
        m = self.shock_mult if shock else 1.0
        return (self.half_spread + self.slippage * self.stop_slip_mult * m) * self.multiplier

    def taker(self) -> float:
        return self.taker_fee * self.multiplier

    def maker(self) -> float:
        return self.maker_fee * self.multiplier

    def round_trip(self) -> float:
        """Round-trip cost fraction for market-in / market-out (used by the cost filter)."""
        return 2.0 * (self.taker() + self.market_impact())
