"""
H2Wealth - FastAPI WebUI Backend
REST API + Server-Sent Events for live dashboard.
"""
from __future__ import annotations
import asyncio, json, logging, time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from core.config import Config
from core.orchestrator import Orchestrator
from core import journal

log = logging.getLogger("webui")
cfg = Config()
orch: Orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orch
    orch = Orchestrator(cfg)
    await orch.start()
    yield
    await orch.stop()


app = FastAPI(title="H2Wealth", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Control Endpoints ─────────────────────────────────────────────────────────

@app.post("/api/pause")
async def pause():
    await orch.pause()
    return {"ok": True}

@app.post("/api/resume")
async def resume():
    await orch.resume()
    return {"ok": True}

@app.post("/api/force_scan")
async def force_scan():
    asyncio.create_task(orch.force_scan())
    return {"ok": True}

@app.post("/api/close_all")
async def close_all():
    await orch.close_all()
    return {"ok": True}

@app.post("/api/close_position/{position_id}")
async def close_position(position_id: str):
    pos = await orch.state.get_position(position_id)
    if not pos:
        raise HTTPException(404, "Position not found")
    await orch.pos_mgr.close_position(pos, reason="manual_ui")
    return {"ok": True}

@app.post("/api/stop")
async def stop():
    asyncio.create_task(orch.stop())
    return {"ok": True}

@app.post("/api/restart")
async def restart():
    await orch.stop()
    await orch.start()
    return {"ok": True}


# ── Data Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/metrics")
async def get_metrics():
    return await orch.state.get_metrics()

@app.get("/api/positions")
async def get_positions():
    pos = await orch.state.get_open_positions()
    return [p.__dict__ for p in pos]

@app.get("/api/signals")
async def get_signals():
    sigs = await orch.state.get_signals()
    return [s.__dict__ for s in sigs]

@app.get("/api/logs")
async def get_logs(n: int = 100):
    return await orch.state.get_logs(n)

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    return journal.get_recent_trades(limit)

@app.get("/api/performance")
async def get_performance():
    return journal.get_performance_summary()


# ── SSE Live Stream ───────────────────────────────────────────────────────────

@app.get("/api/stream")
async def sse_stream(request: Request) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        pubsub = await orch.state.subscribe("metrics", "position", "signal", "log", "status")
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg:
                    channel = msg["channel"].split(":")[-1]
                    data    = msg["data"]
                    yield f"event: {channel}\ndata: {data}\n\n"
                else:
                    yield ": ping\n\n"
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H2Wealth Trading Bot</title>
<style>
  :root{--bg:#0d0d0d;--bg2:#151515;--bg3:#1c1c1c;--border:#2a2a2a;--text:#e0e0e0;--muted:#888;--green:#00d084;--red:#ff4757;--blue:#3d9cf5;--amber:#f5a623;--purple:#a855f7}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Courier New',monospace;font-size:13px;min-height:100vh}
  .topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;justify-content:space-between}
  .logo{color:var(--blue);font-size:18px;font-weight:bold;letter-spacing:2px}
  .status-badge{padding:4px 12px;border-radius:4px;font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:1px}
  .status-running{background:#003d22;color:var(--green);border:1px solid var(--green)}
  .status-paused{background:#3d2a00;color:var(--amber);border:1px solid var(--amber)}
  .status-stopped{background:#2a0000;color:var(--red);border:1px solid var(--red)}
  .controls{display:flex;gap:8px}
  .btn{padding:6px 14px;border-radius:4px;border:1px solid var(--border);background:var(--bg3);color:var(--text);cursor:pointer;font-size:12px;font-family:inherit}
  .btn:hover{border-color:var(--blue);color:var(--blue)}
  .btn-danger:hover{border-color:var(--red);color:var(--red)}
  .btn-green:hover{border-color:var(--green);color:var(--green)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;padding:16px}
  .metric-card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:14px}
  .metric-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
  .metric-value{font-size:22px;font-weight:bold}
  .green{color:var(--green)} .red{color:var(--red)} .blue{color:var(--blue)} .amber{color:var(--amber)}
  .panels{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 16px 16px}
  .panel{background:var(--bg2);border:1px solid var(--border);border-radius:6px;overflow:hidden}
  .panel-header{background:var(--bg3);padding:10px 14px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
  .panel-body{padding:0;max-height:300px;overflow-y:auto}
  table{width:100%;border-collapse:collapse}
  th{padding:8px 10px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);border-bottom:1px solid var(--border);background:var(--bg3)}
  td{padding:7px 10px;border-bottom:1px solid #111;font-size:12px}
  tr:hover td{background:var(--bg3)}
  .log-panel{grid-column:1/-1}
  .log-body{padding:10px 14px;max-height:200px;overflow-y:auto;font-size:11px;line-height:1.8}
  .log-entry{margin:2px 0}
  .log-INFO{color:var(--text)} .log-ERROR{color:var(--red)} .log-WARN{color:var(--amber)}
  .ttl-bar{background:var(--border);height:4px;border-radius:2px;overflow:hidden;width:80px}
  .ttl-fill{height:100%;background:var(--green);transition:width .5s}
  .ttl-fill.warning{background:var(--amber)} .ttl-fill.danger{background:var(--red)}
  .score-bar{display:inline-block;height:8px;border-radius:2px;background:var(--blue);margin-right:4px;vertical-align:middle}
  .close-btn{background:none;border:1px solid var(--border);color:var(--red);padding:2px 8px;border-radius:3px;cursor:pointer;font-size:10px}
  .close-btn:hover{background:var(--red);color:white}
  @media(max-width:768px){.panels{grid-template-columns:1fr}.log-panel{grid-column:1}}
</style>
</head>
<body>
<div class="topbar">
  <div class="logo">⟁ H2WEALTH</div>
  <div id="status-badge" class="status-badge status-stopped">STOPPED</div>
  <div class="controls">
    <button class="btn btn-green" onclick="api('resume')">▶ Resume</button>
    <button class="btn btn-danger" onclick="api('pause')">⏸ Pause</button>
    <button class="btn" onclick="api('force_scan')">⟳ Scan Now</button>
    <button class="btn btn-danger" onclick="if(confirm('Close ALL positions?'))api('close_all')">✕ Close All</button>
    <button class="btn" onclick="api('restart')">↺ Restart</button>
  </div>
</div>

<div class="grid">
  <div class="metric-card"><div class="metric-label">Equity (USDT)</div><div class="metric-value blue" id="m-equity">—</div></div>
  <div class="metric-card"><div class="metric-label">Open PNL</div><div class="metric-value" id="m-opnl">—</div></div>
  <div class="metric-card"><div class="metric-label">Total Closed PNL</div><div class="metric-value" id="m-tpnl">—</div></div>
  <div class="metric-card"><div class="metric-label">Open Positions</div><div class="metric-value" id="m-pos">—</div></div>
  <div class="metric-card"><div class="metric-label">Active Signals</div><div class="metric-value" id="m-sigs">—</div></div>
  <div class="metric-card"><div class="metric-label">Win Rate</div><div class="metric-value green" id="m-wr">—</div></div>
  <div class="metric-card"><div class="metric-label">Total Trades</div><div class="metric-value" id="m-trades">—</div></div>
  <div class="metric-card"><div class="metric-label">Pairs Watching</div><div class="metric-value amber" id="m-pairs">—</div></div>
</div>

<div class="panels">
  <div class="panel">
    <div class="panel-header">Open Positions <span id="pos-count">0</span></div>
    <div class="panel-body">
      <table><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>PNL</th><th>TTL</th><th>Status</th><th></th></tr></thead>
      <tbody id="pos-table"></tbody></table>
    </div>
  </div>
  <div class="panel">
    <div class="panel-header">Top Signals <span id="sig-count">0</span></div>
    <div class="panel-body">
      <table><thead><tr><th>Symbol</th><th>Side</th><th>Score</th><th>Reason</th><th>TTL</th></tr></thead>
      <tbody id="sig-table"></tbody></table>
    </div>
  </div>
  <div class="panel log-panel">
    <div class="panel-header">Live Log <button class="btn" onclick="clearLog()" style="padding:2px 8px;font-size:10px">Clear</button></div>
    <div class="log-body" id="log-body"></div>
  </div>
</div>

<script>
const fmtTime = ts => new Date(ts*1000).toLocaleTimeString();
const fmtN = (v,d=2) => v==null?'—':Number(v).toFixed(d);

async function api(action, method='POST') {
  const r = await fetch(`/api/${action}`, {method});
  const d = await r.json();
  if(!d.ok) alert('Error: '+JSON.stringify(d));
}

function updateMetrics(m) {
  document.getElementById('m-equity').textContent  = '$'+fmtN(m.equity);
  const opnl = m.open_pnl||0;
  const opnlEl = document.getElementById('m-opnl');
  opnlEl.textContent = (opnl>=0?'+':'')+fmtN(opnl,4)+' USDT';
  opnlEl.className = 'metric-value '+(opnl>=0?'green':'red');
  const tpnl = m.total_pnl_closed||0;
  const tpnlEl = document.getElementById('m-tpnl');
  tpnlEl.textContent = (tpnl>=0?'+':'')+fmtN(tpnl,4)+' USDT';
  tpnlEl.className = 'metric-value '+(tpnl>=0?'green':'red');
  document.getElementById('m-pos').textContent    = m.open_positions||0;
  document.getElementById('m-sigs').textContent   = m.active_signals||0;
  const total = m.total_trades||0, wins = m.win_trades||0;
  document.getElementById('m-wr').textContent    = total>0?(wins/total*100).toFixed(1)+'%':'—';
  document.getElementById('m-trades').textContent = total;
  document.getElementById('m-pairs').textContent  = m.pairs_watching||0;
  const badge = document.getElementById('status-badge');
  badge.textContent = (m.status||'stopped').toUpperCase();
  badge.className   = 'status-badge status-'+(m.status||'stopped');
}

function ttlBar(pct) {
  const cls = pct>75?'danger':pct>50?'warning':'';
  return `<div class="ttl-bar"><div class="ttl-fill ${cls}" style="width:${Math.min(pct,100)}%"></div></div>`;
}

function updatePositions(positions) {
  const now = Date.now()/1000;
  document.getElementById('pos-count').textContent = positions.length;
  const tbody = document.getElementById('pos-table');
  if(!positions.length){tbody.innerHTML='<tr><td colspan="7" style="color:var(--muted);text-align:center;padding:20px">No open positions</td></tr>';return;}
  tbody.innerHTML = positions.map(p=>{
    const pnl    = p.pnl_usdt||0;
    const pnlCls = pnl>=0?'green':'red';
    const total  = p.signal_expires_at - p.opened_at;
    const elapsed= now - p.opened_at;
    const ttlPct = total>0?Math.min(elapsed/total*100,100):100;
    return `<tr>
      <td><strong>${p.symbol}</strong></td>
      <td class="${p.side==='Buy'?'green':'red'}">${p.side}</td>
      <td>${fmtN(p.entry_price,4)}</td>
      <td class="${pnlCls}">${pnl>=0?'+':''}${fmtN(pnl,4)}</td>
      <td>${ttlBar(ttlPct)}</td>
      <td style="font-size:10px;color:var(--muted)">${p.status}</td>
      <td><button class="close-btn" onclick="closePos('${p.position_id}')">✕</button></td>
    </tr>`;
  }).join('');
}

async function closePos(id) {
  if(!confirm('Close this position?')) return;
  await fetch(`/api/close_position/${id}`, {method:'POST'});
}

function updateSignals(signals) {
  document.getElementById('sig-count').textContent = signals.length;
  const now = Date.now()/1000;
  const tbody = document.getElementById('sig-table');
  if(!signals.length){tbody.innerHTML='<tr><td colspan="5" style="color:var(--muted);text-align:center;padding:20px">No signals</td></tr>';return;}
  tbody.innerHTML = signals.slice(0,20).map(s=>{
    const total  = s.expires_at - s.created_at;
    const elapsed= now - s.created_at;
    const ttlPct = total>0?Math.min(elapsed/total*100,100):100;
    const barW   = Math.round(s.score||0);
    return `<tr>
      <td><strong>${s.symbol}</strong></td>
      <td class="${s.side==='Buy'?'green':'red'}">${s.side}</td>
      <td><span class="score-bar" style="width:${barW/2}px"></span>${fmtN(s.score,1)}</td>
      <td style="color:var(--muted);font-size:10px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s.reason||''}</td>
      <td>${ttlBar(ttlPct)}</td>
    </tr>`;
  }).join('');
}

function addLog(entry) {
  const el = document.getElementById('log-body');
  const line = document.createElement('div');
  line.className = `log-entry log-${entry.lvl||'INFO'}`;
  const t = entry.t ? new Date(entry.t*1000).toLocaleTimeString() : '—';
  line.textContent = `[${t}] ${entry.lvl||'INFO'}: ${entry.msg}`;
  el.prepend(line);
  while(el.children.length > 200) el.removeChild(el.lastChild);
}

function clearLog() { document.getElementById('log-body').innerHTML=''; }

// Initial load
async function loadAll() {
  const [metrics,positions,signals,logs] = await Promise.all([
    fetch('/api/metrics').then(r=>r.json()),
    fetch('/api/positions').then(r=>r.json()),
    fetch('/api/signals').then(r=>r.json()),
    fetch('/api/logs?n=50').then(r=>r.json()),
  ]);
  updateMetrics(metrics);
  updatePositions(positions);
  updateSignals(signals);
  logs.reverse().forEach(addLog);
}
loadAll();

// SSE live updates
const es = new EventSource('/api/stream');
es.addEventListener('metrics',  e => updateMetrics(JSON.parse(e.data)));
es.addEventListener('position', e => { fetch('/api/positions').then(r=>r.json()).then(updatePositions); });
es.addEventListener('signal',   e => { fetch('/api/signals').then(r=>r.json()).then(updateSignals); });
es.addEventListener('log',      e => addLog(JSON.parse(e.data)));
es.addEventListener('status',   e => { const d=JSON.parse(e.data); const b=document.getElementById('status-badge'); b.textContent=d.status.toUpperCase(); b.className='status-badge status-'+d.status; });
es.onerror = () => setTimeout(loadAll, 3000);
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


def run_webui(cfg: Config):
    uvicorn.run(app, host=cfg.webui_host, port=cfg.webui_port, log_level="warning")
