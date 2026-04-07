"""
H2Wealth - Redis State Store
All live state: positions, signals, bot status, metrics.
"""
from __future__ import annotations
import json, logging, time
from typing import Dict, List, Optional
import redis.asyncio as aioredis
from core.config import Config, Position, Signal, BotStatus, PositionStatus

log = logging.getLogger("state")

POSITIONS_KEY = "h2w:positions"
SIGNALS_KEY   = "h2w:signals"
STATUS_KEY    = "h2w:status"
METRICS_KEY   = "h2w:metrics"
LOG_KEY       = "h2w:log"


class StateStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._r: Optional[aioredis.Redis] = None

    async def connect(self):
        self._r = aioredis.Redis(
            host=self.cfg.redis_host,
            port=self.cfg.redis_port,
            db=self.cfg.redis_db,
            decode_responses=True
        )
        await self._r.ping()
        log.info("Redis connected")

    async def disconnect(self):
        if self._r:
            await self._r.aclose()

    # ── Bot Status ───────────────────────────────────────────────────────────

    async def set_status(self, status: BotStatus):
        await self._r.set(STATUS_KEY, status.value)
        await self._publish("status", {"status": status.value})

    async def get_status(self) -> BotStatus:
        v = await self._r.get(STATUS_KEY)
        return BotStatus(v) if v else BotStatus.STOPPED

    # ── Signals ──────────────────────────────────────────────────────────────

    async def save_signal(self, sig: Signal):
        await self._r.hset(SIGNALS_KEY, sig.signal_id, json.dumps(sig.__dict__))
        await self._publish("signal", sig.__dict__)

    async def get_signals(self) -> List[Signal]:
        raw = await self._r.hgetall(SIGNALS_KEY)
        sigs = []
        for v in raw.values():
            try:
                d = json.loads(v)
                d["side"] = d["side"] if isinstance(d["side"], str) else d["side"]
                sigs.append(Signal(**d))
            except Exception as e:
                log.warning(f"Bad signal: {e}")
        return sorted(sigs, key=lambda s: s.score, reverse=True)

    async def remove_signal(self, signal_id: str):
        await self._r.hdel(SIGNALS_KEY, signal_id)

    async def clear_signals(self):
        await self._r.delete(SIGNALS_KEY)

    # ── Positions ────────────────────────────────────────────────────────────

    async def save_position(self, pos: Position):
        await self._r.hset(POSITIONS_KEY, pos.position_id, json.dumps(pos.__dict__))
        await self._publish("position", pos.__dict__)

    async def get_positions(self) -> List[Position]:
        raw = await self._r.hgetall(POSITIONS_KEY)
        positions = []
        for v in raw.values():
            try:
                d = json.loads(v)
                positions.append(Position(**d))
            except Exception as e:
                log.warning(f"Bad position: {e}")
        return positions

    async def get_open_positions(self) -> List[Position]:
        all_pos = await self.get_positions()
        return [p for p in all_pos if p.status in (PositionStatus.OPEN, PositionStatus.TP1_HIT, PositionStatus.TP2_HIT)]

    async def get_position(self, position_id: str) -> Optional[Position]:
        v = await self._r.hget(POSITIONS_KEY, position_id)
        if not v:
            return None
        return Position(**json.loads(v))

    async def remove_position(self, position_id: str):
        await self._r.hdel(POSITIONS_KEY, position_id)

    # ── Metrics ──────────────────────────────────────────────────────────────

    async def update_metrics(self, data: Dict):
        await self._r.hset(METRICS_KEY, mapping={k: json.dumps(v) for k, v in data.items()})
        await self._publish("metrics", data)

    async def get_metrics(self) -> Dict:
        raw = await self._r.hgetall(METRICS_KEY)
        return {k: json.loads(v) for k, v in raw.items()}

    # ── Log stream (last 500 lines) ──────────────────────────────────────────

    async def push_log(self, level: str, msg: str):
        entry = json.dumps({"t": round(time.time(), 2), "lvl": level, "msg": msg})
        await self._r.lpush(LOG_KEY, entry)
        await self._r.ltrim(LOG_KEY, 0, 499)
        await self._publish("log", {"lvl": level, "msg": msg})

    async def get_logs(self, n: int = 100) -> List[Dict]:
        raw = await self._r.lrange(LOG_KEY, 0, n - 1)
        return [json.loads(x) for x in raw]

    # ── Pub/Sub ──────────────────────────────────────────────────────────────

    async def _publish(self, channel: str, data: Dict):
        try:
            await self._r.publish(f"h2w:{channel}", json.dumps(data, default=str))
        except Exception:
            pass

    async def subscribe(self, *channels: str):
        pubsub = self._r.pubsub()
        await pubsub.subscribe(*[f"h2w:{c}" for c in channels])
        return pubsub

    async def ping(self) -> bool:
        try:
            return await self._r.ping()
        except Exception:
            return False
