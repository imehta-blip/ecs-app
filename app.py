"""
ECS — Environmental Capital Score
Streamlit App
"""

import streamlit as st
import streamlit.components.v1 as components
import datetime
import zoneinfo
import json
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Suppress __main__ demo blocks ─────────────────────────────────────────────
_ECS_COLAB_EXEC = True  # noqa: F841 — read by __main__ guards in engine/data layer

# ── Optional: streamlit-js-eval for real GPS auto-detection ──────────────────
# pip install streamlit-js-eval
# Without it the app falls back to the URL query-param GPS approach (button)
try:
    from streamlit_js_eval import get_geolocation, streamlit_js_eval
    _HAS_JS_EVAL = True
except ImportError:
    _HAS_JS_EVAL = False

# ── Optional: streamlit-autorefresh for hourly auto-scoring ──────────────────
# pip install streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

# ── Load ECS modules ──────────────────────────────────────────────────────────
from ecs_engine import (
    compute_hourly_ECS, close_day, ecs_band,
    HourlyReading, UserState, WHO_THRESHOLDS,
)
from ecs_data_layer import (
    OpenWeatherMapClient, GooglePollenClient, PollutantAssembler,
    MockOWMClient, MockPollenClient, INFILTRATION_FACTORS,
    LocationZone, INDOOR_ZONES,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECS — Environmental Capital Score",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Score card */
.score-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 20px;
    padding: 32px 40px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.score-number {
    font-size: 96px;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}
.score-band {
    font-size: 22px;
    font-weight: 500;
    margin-top: 8px;
    opacity: 0.9;
}
.score-time {
    font-size: 13px;
    opacity: 0.5;
    margin-top: 6px;
}

/* Component bars */
.comp-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    gap: 10px;
}
.comp-label {
    width: 220px;
    font-size: 13px;
    color: #ccc;
    flex-shrink: 0;
}
.comp-bar-bg {
    flex: 1;
    background: #2a2a3e;
    border-radius: 6px;
    height: 14px;
    overflow: hidden;
}
.comp-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.comp-value {
    width: 48px;
    text-align: right;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
}

/* Zone buttons */
.stButton>button {
    border-radius: 20px !important;
    font-size: 13px !important;
    padding: 4px 16px !important;
    height: 36px !important;
}

/* Pollutant table */
.poll-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.poll-table th {
    text-align: left;
    padding: 6px 10px;
    color: #888;
    font-weight: 500;
    border-bottom: 1px solid #2a2a3e;
}
.poll-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #1e1e2e;
}
.above-who { color: #ff6b6b; font-weight: 600; }
.below-who { color: #51cf66; }

/* GPS badge */
.gps-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e2a1e;
    border: 1px solid #2d4a2d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    color: #69db7c;
    margin-bottom: 16px;
}
.gps-badge-warn {
    background: #2a1e1e;
    border-color: #4a2d2d;
    color: #ff8787;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TIME-BASED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _zone_from_time(dt: datetime.datetime) -> str:
    """
    Infer the most likely zone from time of day + weekday.
    Used as the default zone on first load and after each 07:00 reset.
    User can always override via the zone buttons.

    Heuristic:
      00:00–06:59  → HOME   (sleep, belongs to previous ECS day)
      07:00–08:29  → COMMUTE on weekdays, HOME on weekends
      08:30–17:29  → WORK on weekdays, HOME on weekends
      17:30–18:59  → COMMUTE on weekdays, HOME on weekends
      19:00–22:59  → HOME
      23:00–23:59  → HOME (sleep window)
    """
    h, m = dt.hour, dt.minute
    is_weekday = dt.weekday() < 5
    hm = h * 60 + m
    if hm < 7 * 60:
        return "HOME"
    elif hm < 8 * 60 + 30:
        return "COMMUTE" if is_weekday else "HOME"
    elif hm < 17 * 60 + 30:
        return "WORK" if is_weekday else "HOME"
    elif hm < 19 * 60:
        return "COMMUTE" if is_weekday else "HOME"
    else:
        return "HOME"

def _is_sleep_window(hour: int) -> bool:
    return hour >= 23 or hour < 7

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "gps_lat":            None,
        "gps_lon":            None,
        "gps_label":          None,
        "gps_pending":        False,
        "gps_fetched":        False,
        "zone":               "HOME",
        "zone_overridden":    False,
        "history":            [],
        "engine_state":       None,
        "last_scored":        None,
        "last_scored_hour":   None,   # local hour of last score — detects new hour
        "last_result":        None,
        "last_assembly":      None,
        "day_date":           None,   # set below after timezone is known
        "use_mock":           None,   # None = not yet initialised
        "last_refresh_count": 0,      # tracks autorefresh cycles
        "user_tz_str":        "UTC",  # filled from browser Intl API
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Timezone: resolve once from browser, persist in session_state ────────────
if _HAS_JS_EVAL and st.session_state.user_tz_str == "UTC":
    try:
        _tz_raw = streamlit_js_eval(
            js_expressions="Intl.DateTimeFormat().resolvedOptions().timeZone",
            key="tz_detect",
        )
        if _tz_raw and isinstance(_tz_raw, str):
            zoneinfo.ZoneInfo(_tz_raw)          # validate — raises if unknown
            st.session_state.user_tz_str = _tz_raw
    except Exception:
        pass   # keep UTC

try:
    _TZ = zoneinfo.ZoneInfo(st.session_state.user_tz_str)
except Exception:
    _TZ = zoneinfo.ZoneInfo("UTC")

def _local_now() -> datetime.datetime:
    return datetime.datetime.now(tz=_TZ)

_now = _local_now()

# ECS day: runs 07:00-06:59. Hours 00-06 belong to the previous calendar day.
_period_date = _now.date() if _now.hour >= 7 else (_now.date() - datetime.timedelta(days=1))
today = _period_date.isoformat()

# First run: seed day_date
if st.session_state.day_date is None:
    st.session_state.day_date = today

if st.session_state.day_date != today:
    st.session_state.history          = []
    st.session_state.engine_state     = None
    st.session_state.day_date         = today
    st.session_state.last_scored_hour = None
    st.session_state.zone_overridden  = False

# ── Reset GPS on every autorefresh cycle so coords stay current ──────────────
if _refresh_count != st.session_state.last_refresh_count:
    st.session_state.gps_fetched        = False
    st.session_state.last_refresh_count = _refresh_count

# ── Auto-GPS: fires once per session via streamlit-js-eval ───────────────────
# get_geolocation() calls navigator.geolocation.getCurrentPosition in the
# browser and returns the result directly to Python — no button needed.
# Runs silently on first page load; never runs again once gps_fetched=True.
if _HAS_JS_EVAL and not st.session_state.gps_fetched:
    with st.spinner("📍 Detecting your location…"):
        try:
            geo = get_geolocation()
            if geo and "coords" in geo:
                lat = geo["coords"].get("latitude")
                lon = geo["coords"].get("longitude")
                acc = geo["coords"].get("accuracy", "?")
                if lat is not None and lon is not None:
                    st.session_state.gps_lat   = round(lat, 6)
                    st.session_state.gps_lon   = round(lon, 6)
                    st.session_state.gps_label = (
                        f"{lat:.4f}, {lon:.4f}  ±{acc:.0f}m"
                    )
        except Exception:
            pass  # GPS denied or not available — falls back to manual sidebar
    st.session_state.gps_fetched = True

# ── Always-on autorefresh (must be called before any widget) ────────────────
# Fires a full rerun every 60 min regardless of toggle.
# We track the count so we know when a new cycle just fired.
_refresh_count = 0
if _HAS_AUTOREFRESH:
    _refresh_count = st_autorefresh(interval=3_600_000, limit=None, key="ecs_hourly")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — API KEYS + MODE
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🌿 ECS Settings")

    # Secrets — loaded once; never reset the toggle after first run
    _secrets    = getattr(st, "secrets", {})
    default_owm    = _secrets.get("OWM_API_KEY",    "")
    default_google = _secrets.get("GOOGLE_API_KEY", "")

    # First run: default mock OFF if keys are present in secrets, ON otherwise
    if st.session_state.use_mock is None:
        st.session_state.use_mock = not bool(default_owm)

    _mock_toggled = st.toggle(
        "Use mock data (no API keys needed)",
        value=st.session_state.use_mock,
    )
    st.session_state.use_mock = _mock_toggled
    use_mock = _mock_toggled

    if not use_mock:
        owm_key    = st.text_input("OpenWeatherMap API key", value=default_owm,    type="password")
        google_key = st.text_input("Google Pollen API key",  value=default_google, type="password")
        if not owm_key or not google_key:
            st.warning("Enter both keys or enable mock mode.")
            st.session_state.use_mock = True
            use_mock = True
    else:
        owm_key    = ""
        google_key = ""

    st.divider()
    st.markdown("**Location**")
    if st.session_state.gps_lat is not None:
        st.success(f"📍 {st.session_state.gps_label}")
        if st.button("🔄 Re-detect GPS"):
            st.session_state.gps_fetched = False
            st.rerun()
    else:
        if _HAS_JS_EVAL:
            st.info("GPS detecting on load. If denied, enter manually:")
        else:
            st.caption("Install `streamlit-js-eval` for auto GPS.")
    manual_lat = st.number_input("Lat (manual fallback)",  value=float(st.session_state.gps_lat or 51.5074), format="%.4f", step=0.0001)
    manual_lon = st.number_input("Lon (manual fallback)", value=float(st.session_state.gps_lon or -0.1278),  format="%.4f", step=0.0001)
    if st.button("Use manual coords"):
        st.session_state.gps_lat   = manual_lat
        st.session_state.gps_lon   = manual_lon
        st.session_state.gps_label = f"{manual_lat:.4f}, {manual_lon:.4f} (manual)"
        st.rerun()

    st.divider()
    st.markdown("**Auto-refresh**")
    if _HAS_AUTOREFRESH:
        mins_left = 60 - _now.minute
        st.success(f"✅ Scoring every 60 min automatically")
        st.caption(f"Next refresh in ~{mins_left} min · TZ: {st.session_state.user_tz_str}")
    else:
        st.warning("⚠ Install `streamlit-autorefresh` to enable auto-scoring.")
    if st.session_state.last_scored:
        st.caption(f"Last scored: {st.session_state.last_scored.strftime('%H:%M:%S')}")

    st.divider()
    if st.button("🗑 Clear today's history"):
        st.session_state.history      = []
        st.session_state.engine_state = None
        st.rerun()

    st.divider()
    st.markdown("""
<small style='color:#555'>
ECS v1.0 · DALY-weighted<br>
Data: OWM + Google Pollen<br>
Infiltration: Chen & Zhao 2011
</small>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GPS WIDGET
# ══════════════════════════════════════════════════════════════════════════════

GPS_HTML = """
<script>
function requestGPS() {
    if (!navigator.geolocation) {
        window.parent.postMessage({type:'ecs_gps', error:'Geolocation not supported'}, '*');
        return;
    }
    document.getElementById('gps-btn').disabled = true;
    document.getElementById('gps-btn').innerText = '⏳ Detecting…';
    navigator.geolocation.getCurrentPosition(
        function(pos) {
            window.parent.postMessage({
                type: 'ecs_gps',
                lat:  pos.coords.latitude,
                lon:  pos.coords.longitude,
                acc:  pos.coords.accuracy
            }, '*');
            document.getElementById('gps-btn').innerText = '✅ Got location';
        },
        function(err) {
            window.parent.postMessage({type:'ecs_gps', error: err.message}, '*');
            document.getElementById('gps-btn').disabled = false;
            document.getElementById('gps-btn').innerText = '📍 Detect my location';
        },
        {enableHighAccuracy: true, timeout: 10000}
    );
}
</script>
<button id="gps-btn" onclick="requestGPS()"
    style="background:#1e3a1e;border:1px solid #2d5a2d;color:#69db7c;
           border-radius:20px;padding:6px 18px;font-size:13px;cursor:pointer;">
    📍 Detect my location
</button>
"""

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

BAND_COLORS = {
    "Excellent":  "#51cf66",
    "Good":       "#94d82d",
    "Moderate":   "#fcc419",
    "Poor":       "#ff922b",
    "Very Poor":  "#ff6b6b",
    "Hazardous":  "#cc5de8",
}

COMP_COLORS = {
    "A": "#74c0fc",
    "B": "#63e6be",
    "C": "#ffa94d",
    "D": "#da77f2",
}

COMP_LABELS = {
    "A": "A — Instant air quality",
    "B": "B — 3h stability",
    "C": "C — 30-day load",
    "D": "D — Sleep environment",
}

ZONE_OPTIONS = ["HOME", "WORK", "OUTDOOR", "COMMUTE"]
ZONE_ICONS   = {"HOME": "🏠", "WORK": "🏢", "OUTDOOR": "🌳", "COMMUTE": "🚌"}

def _color_for_score(score: float) -> str:
    if score >= 90: return BAND_COLORS["Excellent"]
    if score >= 75: return BAND_COLORS["Good"]
    if score >= 60: return BAND_COLORS["Moderate"]
    if score >= 40: return BAND_COLORS["Poor"]
    if score >= 20: return BAND_COLORS["Very Poor"]
    return BAND_COLORS["Hazardous"]

def _get_assembler():
    if use_mock:
        return PollutantAssembler(owm_client=MockOWMClient(), pollen_client=MockPollenClient())
    return PollutantAssembler(
        owm_client    = OpenWeatherMapClient(api_key=owm_key),
        pollen_client = GooglePollenClient(api_key=google_key),
    )

def _get_or_init_engine_state() -> UserState:
    if st.session_state.engine_state is None:
        s = UserState()
        s.bio_baselines    = {"hrv": 55, "hr": 68, "spo2": 98}
        s.current_day_date = today
        st.session_state.engine_state = s
    return st.session_state.engine_state

def _effective_coords():
    if st.session_state.gps_lat is not None:
        return st.session_state.gps_lat, st.session_state.gps_lon
    return manual_lat, manual_lon

def _score_now():
    lat, lon     = _effective_coords()
    local_dt     = _local_now()
    hour         = local_dt.hour
    zone_str     = st.session_state.zone
    zone         = LocationZone(zone_str.lower())
    in_sleep     = _is_sleep_window(hour)
    assembler    = _get_assembler()
    engine_state = _get_or_init_engine_state()

    try:
        assembly = assembler.get(
            lat              = lat,
            lon              = lon,
            zone             = zone,
            indoor_overrides = None,   # no sensors — outdoor AQ x infiltration only
        )
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None, None

    reading = HourlyReading(
        timestamp            = hour,
        pollutants           = assembly.pollutants,
        biomarkers           = None,    # wearable integration added later
        in_sleep_window      = in_sleep,
        protection_confirmed = False,
    )

    result = compute_hourly_ECS(reading, engine_state)

    # Store raw API readings alongside scores so history charts have real data
    entry = {
        "hour":     hour,
        "ecs":      result["ECS"],
        "A":        result["A"],
        "B":        result["B"],
        "C":        result["C"],
        "D":        result["D"],
        "ts":       local_dt.strftime("%H:%M"),
        "zone":     zone_str,
        "offending": result.get("offending", []),
        "raw":      {**assembly.outdoor_raw, **assembly.pollen_raw},
        "indoor":   dict(assembly.pollutants),
    }
    history = [h for h in st.session_state.history if h["hour"] != hour]
    history.append(entry)
    history.sort(key=lambda x: x["hour"])
    st.session_state.history          = history
    st.session_state.last_result      = result
    st.session_state.last_assembly    = assembly
    st.session_state.last_scored      = local_dt
    st.session_state.last_scored_hour = hour

    return result, assembly


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Auto-score trigger ────────────────────────────────────────────────────────
# Score when: (a) no result yet, or (b) current hour differs from last scored hour
_need_score = (
    st.session_state.last_result is None
    or st.session_state.last_scored_hour != _now.hour
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("## 🌿 Environmental Capital Score")

# ── GPS row ───────────────────────────────────────────────────────────────────
col_gps, col_btn = st.columns([3, 1])
with col_gps:
    if st.session_state.gps_lat is not None:
        label = st.session_state.gps_label or f"{st.session_state.gps_lat:.4f}, {st.session_state.gps_lon:.4f}"
        st.markdown(
            f'<div class="gps-badge">📍 {label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="gps-badge gps-badge-warn">⚠️ No GPS — using manual coordinates from sidebar</div>',
            unsafe_allow_html=True,
        )

with col_btn:
    # GPS widget — renders a button; result is received via query_params workaround
    # Since Streamlit can't receive postMessage directly, we use a form + hidden input approach
    gps_result = components.html(GPS_HTML, height=44)

# GPS coordinates via URL query params (set by JS redirect after detection)
params = st.query_params
if "gps_lat" in params and "gps_lon" in params:
    try:
        new_lat = float(params["gps_lat"])
        new_lon = float(params["gps_lon"])
        if st.session_state.gps_lat != new_lat or st.session_state.gps_lon != new_lon:
            st.session_state.gps_lat   = new_lat
            st.session_state.gps_lon   = new_lon
            st.session_state.gps_label = f"{new_lat:.4f}, {new_lon:.4f}"
            st.rerun()
    except (ValueError, TypeError):
        pass

# ── Zone selector ─────────────────────────────────────────────────────────────
_time_zone = _zone_from_time(_now)
_override   = st.session_state.zone_overridden
_hint = f"⏰ Suggested from time ({_now.strftime('%H:%M')})" if not _override else "✏️ Manually set"
st.markdown(f"**Where are you right now?**  <small style='color:#666'>{_hint}</small>", unsafe_allow_html=True)

zone_cols = st.columns(4)
for i, z in enumerate(ZONE_OPTIONS):
    with zone_cols[i]:
        is_active   = st.session_state.zone == z
        is_suggest  = (z == _time_zone and not _override)
        btn_label   = f"{ZONE_ICONS[z]} {z}" + (" ●" if is_suggest else "")
        if st.button(btn_label, key=f"zone_{z}",
                     type="primary" if is_active else "secondary",
                     use_container_width=True):
            st.session_state.zone           = z
            st.session_state.zone_overridden = True
            st.rerun()

# Reset-to-suggestion link
if _override:
    if st.button(f"↺ Reset to time suggestion ({_time_zone})", key="zone_reset"):
        st.session_state.zone_overridden = False
        st.session_state.zone = _time_zone
        st.rerun()

st.markdown("")

# ── Score button ──────────────────────────────────────────────────────────────
col_score, col_time = st.columns([1, 2])
with col_score:
    score_now = st.button("⚡ Score this hour", type="primary", use_container_width=True)
with col_time:
    sleep_flag = "🌙 Sleep window" if _is_sleep_window(_now.hour) else ""
    st.markdown(
        f"<small style='color:#888'>{_now.strftime('%A %d %b · %H:%M')} {st.session_state.user_tz_str}  "
        f"{sleep_flag}  {'📡 MOCK' if use_mock else '🔴 LIVE'}</small>",
        unsafe_allow_html=True,
    )

# Score on: manual button press, first load, or start of a new hour
if score_now or _need_score:
    with st.spinner("Fetching air quality and pollen…"):
        result, assembly = _score_now()
else:
    result   = st.session_state.last_result
    assembly = st.session_state.last_assembly

# ── Score card ────────────────────────────────────────────────────────────────
if result is not None:
    ecs_score         = result["ECS"]
    band, icon        = ecs_band(ecs_score)
    color             = _color_for_score(ecs_score)
    last_scored_str   = st.session_state.last_scored.strftime("%H:%M:%S %Z") if st.session_state.last_scored else "—"

    st.markdown(f"""
<div class="score-card">
    <div class="score-number" style="color:{color}">{ecs_score}</div>
    <div class="score-band" style="color:{color}">{icon} {band}</div>
    <div class="score-time">Last scored {last_scored_str} · {st.session_state.zone} · {'Mock' if use_mock else 'Live'}</div>
</div>
""", unsafe_allow_html=True)

    # ── Components A/B/C/D ────────────────────────────────────────────────────
    st.markdown("#### Components")
    comp_html = ""
    for key in ["A", "B", "C", "D"]:
        val   = result[key]
        color = COMP_COLORS[key]
        label = COMP_LABELS[key]
        comp_html += f"""
<div class="comp-row">
  <div class="comp-label">{label}</div>
  <div class="comp-bar-bg">
    <div class="comp-bar-fill" style="width:{val}%;background:{color}"></div>
  </div>
  <div class="comp-value">{val:.1f}</div>
</div>"""
    st.markdown(comp_html, unsafe_allow_html=True)
    st.markdown("")

    # streak / penalty info
    streak  = result.get("exposure_streak", 0)
    penalty = result.get("penalty_pts", 0)
    bio     = result.get("bio_state", "neutral")
    offend  = result.get("offending", [])

    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("Exposure streak", f"{streak}h", help="Consecutive hours above WHO limits")
    with info_cols[1]:
        st.metric("Penalty this hour", f"-{penalty:.1f} pts")
    with info_cols[2]:
        st.metric("Body state", bio.replace("_", " ").title())

    if offend:
        st.warning(f"⚠️ Above WHO limits: **{', '.join(offend).upper()}**")
    elif result["A"] >= 98:
        st.success("✅ All pollutants within WHO guidelines")

    # co-exposure insights
    co_insights = result.get("co_insights", [])
    if co_insights:
        with st.expander(f"⚡ {len(co_insights)} co-exposure interaction(s)"):
            for ins in co_insights:
                st.markdown(f"- **{ins['pair']}** — {ins['dimension']}: {ins.get('note','')}")

    # ── Pollutant table ───────────────────────────────────────────────────────
    if assembly is not None:
        st.markdown("#### Pollutant breakdown")
        is_indoor = assembly.is_indoor

        DISPLAY_UNITS = {
            "pm25": "µg/m³", "pm10": "µg/m³", "no2": "µg/m³",
            "o3": "µg/m³", "co": "mg/m³",
            "pollen_tree": "grains/m³", "pollen_grass": "grains/m³", "pollen_weed": "grains/m³",
            "co2": "ppm", "radon": "Bq/m³", "humidity": "%", "temp": "°C",
        }
        DISPLAY_NAMES = {
            "pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "O₃", "co": "CO",
            "pollen_tree": "Tree pollen", "pollen_grass": "Grass pollen", "pollen_weed": "Weed pollen",
            "co2": "CO₂", "radon": "Radon", "humidity": "Humidity", "temp": "Temperature",
        }

        rows = ""
        for key, final_val in sorted(assembly.pollutants.items()):
            name    = DISPLAY_NAMES.get(key, key)
            unit    = DISPLAY_UNITS.get(key, "")
            # outdoor_raw holds AQ pollutants; pollen_raw holds pollen — check both
            outdoor = assembly.outdoor_raw.get(key) if assembly.outdoor_raw.get(key) is not None \
                      else assembly.pollen_raw.get(key)
            factor  = INFILTRATION_FACTORS.get(key)
            who     = WHO_THRESHOLDS.get(key)
            if isinstance(who, tuple): who = who[0]

            # For WHO comparison: use outdoor raw for pollen when indoors
            # (pollen is heavily attenuated indoors by ×0.10, but the health risk
            #  is from what you were exposed to outdoors — flag if outdoor raw is high)
            compare_val = final_val
            if is_indoor and outdoor is not None and key in ("pollen_tree", "pollen_grass", "pollen_weed"):
                compare_val = outdoor

            above   = who is not None and compare_val > who
            css_val = "above-who" if above else "below-who"
            who_str = f"{who}" if who is not None else "—"

            # Always show the outdoor raw number — it tells the user what the real exposure was
            if outdoor is not None:
                outdoor_str = f"{outdoor}"
            else:
                outdoor_str = "—"

            # Infiltration factor — only meaningful indoors
            if is_indoor and factor is not None and outdoor is not None:
                engine_str = f"{final_val}  <span style='color:#555;font-size:11px'>(×{factor})</span>"
            else:
                engine_str = f"{final_val}"

            # Status badge
            if above:
                badge = "⚠ HIGH"
                badge_css = "color:#ff6b6b;font-weight:700;font-size:11px"
            else:
                badge = "OK"
                badge_css = "color:#51cf66;font-size:11px"

            rows += f"""
<tr>
  <td>{name}</td>
  <td style="color:#888">{outdoor_str}</td>
  <td>{engine_str}</td>
  <td style="{badge_css}">{badge}</td>
  <td style="color:#555">{who_str}</td>
  <td style="color:#555">{unit}</td>
</tr>"""

        # Column headers depend on zone
        col2_header = "Outdoor raw" if is_indoor else "Value"
        col3_header = "Indoor (infiltrated)" if is_indoor else "—"

        st.markdown(f"""
<table class="poll-table">
  <thead>
    <tr>
      <th>Pollutant</th>
      <th>{col2_header}</th>
      <th>{col3_header}</th>
      <th>Status</th>
      <th>WHO limit</th>
      <th>Unit</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

        if is_indoor:
            st.markdown(
                f"<small style='color:#555'>Zone: **{assembly.zone.value.upper()}** (indoor) · "
                f"PM2.5 source: `{assembly.pm25_source}` · "
                f"Pollen WHO check uses outdoor raw (indoor pollen ×0.10 is not the exposure risk)</small>",
                unsafe_allow_html=True,
            )

        # ── Backend inspector ─────────────────────────────────────────────────
        with st.expander("🔬 Backend — raw data & assembly log"):
            st.markdown("**Outdoor API (raw)**")
            raw_cols = st.columns(len(assembly.outdoor_raw))
            for i, (k, v) in enumerate(assembly.outdoor_raw.items()):
                raw_cols[i].metric(DISPLAY_NAMES.get(k, k), f"{v}", help=f"{DISPLAY_UNITS.get(k,'')}")

            st.markdown("**Pollen API (raw)**")
            pol_cols = st.columns(len(assembly.pollen_raw))
            for i, (k, v) in enumerate(assembly.pollen_raw.items()):
                who_p = WHO_THRESHOLDS.get(k, "?")
                pol_cols[i].metric(
                    DISPLAY_NAMES.get(k, k),
                    f"{v}",
                    delta=f"WHO limit: {who_p}",
                    delta_color="off",
                    help=f"{DISPLAY_UNITS.get(k,'')}"
                )

            if assembly.source_notes:
                st.markdown("**Assembly log** — how every number was built")
                for note in assembly.source_notes:
                    st.markdown(f"`{note}`")

            if assembly.missing_outdoor:
                st.warning(f"Possible API gaps (returned 0): {', '.join(assembly.missing_outdoor)}")

            st.markdown("**Full pollutants dict sent to engine**")
            st.json(assembly.pollutants)


# ── History chart ─────────────────────────────────────────────────────────────
history = st.session_state.history
if len(history) >= 1:
    st.markdown("---")
    st.markdown("#### Today's ECS — hourly history")

    try:
        import plotly.graph_objects as go

        hours  = [h["hour"] for h in history]
        scores = [h["ecs"] for h in history]
        zones  = [h.get("zone", "?") for h in history]
        labels = [f"{h['ts']}  ECS {h['ecs']}  {h.get('zone','')}" for h in history]

        point_colors = [_color_for_score(s) for s in scores]

        fig = go.Figure()

        # Filled line
        fig.add_trace(go.Scatter(
            x=hours, y=scores,
            mode="lines+markers",
            line=dict(color="#51cf66", width=2),
            marker=dict(color=point_colors, size=10, line=dict(color="#1a1a2e", width=2)),
            text=labels,
            hovertemplate="%{text}<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(81,207,102,0.08)",
        ))

        # WHO reference bands
        fig.add_hrect(y0=75, y1=100, fillcolor="rgba(81,207,102,0.05)",  line_width=0)
        fig.add_hrect(y0=60, y1=75,  fillcolor="rgba(148,216,45,0.05)",  line_width=0)
        fig.add_hrect(y0=40, y1=60,  fillcolor="rgba(252,196,25,0.05)",  line_width=0)
        fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(255,107,107,0.05)", line_width=0)

        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#888", size=12),
            xaxis=dict(
                title="Hour", tickmode="linear", dtick=1,
                range=[-0.5, 23.5],
                gridcolor="#1e1e2e", zerolinecolor="#1e1e2e",
            ),
            yaxis=dict(
                title="ECS", range=[0, 100],
                gridcolor="#1e1e2e", zerolinecolor="#1e1e2e",
            ),
            margin=dict(l=40, r=20, t=20, b=40),
            height=280,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to st.line_chart if plotly not available
        import pandas as pd
        df = pd.DataFrame({"Hour": hours, "ECS": scores}).set_index("Hour")
        st.line_chart(df)

    # Component breakdown mini-table
    with st.expander("Component detail by hour"):
        import pandas as pd
        rows_data = []
        for h in history:
            rows_data.append({
                "Hour": f"{h['hour']:02d}:00",
                "ECS":  h["ecs"],
                "A":    h["A"],
                "B":    h["B"],
                "C":    h["C"],
                "D":    h["D"],
                "Zone": h.get("zone", "?"),
            })
        df = pd.DataFrame(rows_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

elif len(history) == 1:
    st.markdown("---")
    st.info("Score at least 2 hours to see the history chart. Come back next hour or click ⚡ again.")


# ── Auto-refresh handled at top of file via streamlit-autorefresh ─────────────


# ── GPS JS bridge ─────────────────────────────────────────────────────────────
# Real GPS detection: JS posts coords to query params, Streamlit picks them up on rerun
GPS_BRIDGE = """
<script>
(function() {
    function detect() {
        if (!navigator.geolocation) return;
        navigator.geolocation.getCurrentPosition(function(pos) {
            var lat = pos.coords.latitude.toFixed(6);
            var lon = pos.coords.longitude.toFixed(6);
            // Update URL query params so Streamlit reads them on next interaction
            var url = new URL(window.parent.location.href);
            url.searchParams.set('gps_lat', lat);
            url.searchParams.set('gps_lon', lon);
            window.parent.history.replaceState({}, '', url.toString());
            // Trigger Streamlit rerun by dispatching a click on the hidden rerun target
            window.parent.dispatchEvent(new Event('streamlit:rerun'));
        }, function(err) {
            console.warn('GPS error:', err.message);
        }, {enableHighAccuracy: true, timeout: 8000});
    }
    // Auto-detect on page load (once)
    if (!window._ecs_gps_init) {
        window._ecs_gps_init = true;
        detect();
    }
    // Also wire the visible GPS button
    window.requestGPS = detect;
})();
</script>
"""
components.html(GPS_BRIDGE, height=0)
