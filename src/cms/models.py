"""
CMS Domain Models: Hypotheses, Lifecycle States, and Manual Trade Records.
Tracks trading ideas from inception to retirement, plus discretionary journal logs.
"""

from dataclasses import dataclass, field
import datetime as dt
from enum import Enum
from typing import Optional, List, Dict, Any


class HypothesisStatus(str, Enum):
    DRAFT = "DRAFT"
    IN_BACKTEST = "IN_BACKTEST"
    OPTIMIZING = "OPTIMIZING"
    PAPER_VALIDATION = "PAPER_VALIDATION"
    LIVE = "LIVE"
    RETIRED = "RETIRED"


@dataclass
class HypothesisCard:
    id: str
    title: str
    economic_rationale: str
    asset_symbol: str
    author: str
    status: HypothesisStatus = HypothesisStatus.DRAFT
    target_regimes: List[str] = field(default_factory=list)
    rules_summary: str = ""
    param_manifest: Dict[str, Any] = field(default_factory=dict)
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    retirement_reason: Optional[str] = None


@dataclass
class ManualTradeRecord:
    id: str
    hypothesis_id: str
    date: dt.date
    direction: str
    entry_time: dt.datetime
    entry_price: float
    units: float
    exit_time: Optional[dt.datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_net: float = 0.0
    execution_rating: int = 5  # 1 to 5
    notes: str = ""
    tags: List[str] = field(default_factory=list)
