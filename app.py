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

# ── Display unit conversions ──────────────────────────────────────────────────
# Engine always stores µg/m³ (particles) or mg/m³ (CO) internally.
# NO2 and O3 displayed in ppb; CO displayed in ppm; PM2.5/PM10 stay µg/m³.
# Conversion at 25°C, 1 atm:
#   NO2:  1 µg/m³ = 1/1.88  ppb  (mol weight 46 g/mol)
#   O3:   1 µg/m³ = 1/1.99  ppb  (mol weight 48 g/mol)
#   CO:   1 mg/m³ = 1/1.145 ppm  (mol weight 28 g/mol)
UNIT_CONVERT = {
    "no2": {"factor": 1/1.88,  "unit": "ppb", "who_engine": 25.0},
    "o3":  {"factor": 1/1.99,  "unit": "ppb", "who_engine": 100.0},
    "co":  {"factor": 1/1.145, "unit": "ppm", "who_engine": 4.0},
}

def _display_val(key, engine_val):
    if key in UNIT_CONVERT:
        c = UNIT_CONVERT[key]
        dval  = round(engine_val * c["factor"], 2)
        who_d = round(c["who_engine"] * c["factor"], 1)
        return dval, c["unit"], who_d
    unit_map = {
        "pm25": "µg/m³", "pm10": "µg/m³",
        "pollen_tree": "gr/m³", "pollen_grass": "gr/m³", "pollen_weed": "gr/m³",
        "co2": "ppm", "radon": "Bq/m³", "humidity": "%", "temp": "°C",
    }
    from ecs_engine import WHO_THRESHOLDS as _WHO
    who_raw = _WHO.get(key)
    who_d   = who_raw[0] if isinstance(who_raw, tuple) else who_raw
    return engine_val, unit_map.get(key, ""), who_d

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
    """HOME during sleep/evening/night, OUTDOOR during daytime hours."""
    h = dt.hour
    if 8 <= h < 20:
        return "OUTDOOR"
    return "HOME"

def _is_sleep_window(hour: int) -> bool:
    return hour >= 23 or hour < 7

# ── Always-on autorefresh — must be before any widget ───────────────────────
# _refresh_count is always defined here (0 if package missing) so all code
# below can safely reference it without NameError.
_refresh_count = st_autorefresh(interval=3_600_000, limit=None, key="ecs_hourly") if _HAS_AUTOREFRESH else 0

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
            st.rerun()  # rerun so all time logic uses local tz from start
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
    # Properly close the previous ECS day — resets exposure streak
    if st.session_state.engine_state is not None:
        close_day(st.session_state.engine_state, st.session_state.day_date)
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

# ── Unit conversion — display only ──────────────────────────────────────────
# Engine stores all values in µg/m³ (particles) and mg/m³ (CO).
# WHO thresholds and weights all use these native units — DO NOT change them.
# Conversion is purely cosmetic: shown in pollutant tables and metric cards.
#
# At 25°C, 1 atm:
#   NO2: 1 ppb = 1.88 µg/m³   → ppb = µg/m³ ÷ 1.88
#   O3:  1 ppb = 1.96 µg/m³   → ppb = µg/m³ ÷ 1.96
#   CO:  1 ppb = 1.145 µg/m³  → ppb = mg/m³ × (1000 ÷ 1.145)
#
# PM2.5 and PM10 stay in µg/m³ (this is the standard display unit globally).
_NO2_PPB  = 1.88    # µg/m³ per ppb
_O3_PPB   = 1.96    # µg/m³ per ppb
_CO_PPB   = 1.145   # µg/m³ per ppb (engine stores CO in mg/m³ so ×1000 first)

def _to_display(key: str, engine_val: float) -> tuple:
    """Convert engine-unit value to display value and unit string.
    Returns (display_value, unit_string, who_display, who_unit).
    WHO display value is also converted to the same display unit.
    """
    WHO_ENG = {
        "pm25": 15.0, "pm10": 45.0, "no2": 25.0, "o3": 100.0, "co": 4.0,
        "co2": 1000.0, "radon": 100.0,
        "pollen_tree": 50.0, "pollen_grass": 50.0, "pollen_weed": 10.0,
    }
    who_eng = WHO_ENG.get(key)

    if key == "no2":
        val  = round(engine_val / _NO2_PPB, 1)
        who  = round(who_eng   / _NO2_PPB, 1) if who_eng else None
        return val, "ppb", who, "ppb"
    elif key == "o3":
        val  = round(engine_val / _O3_PPB,  1)
        who  = round(who_eng    / _O3_PPB,  1) if who_eng else None
        return val, "ppb", who, "ppb"
    elif key == "co":
        # Engine stores CO in mg/m³
        val  = round(engine_val * 1000.0 / _CO_PPB, 0)
        who  = round(who_eng    * 1000.0 / _CO_PPB, 0) if who_eng else None
        return int(val), "ppb", int(who) if who else None, "ppb"
    elif key == "pm25":
        return round(engine_val, 1), "µg/m³", who_eng, "µg/m³"
    elif key == "pm10":
        return round(engine_val, 1), "µg/m³", who_eng, "µg/m³"
    elif key == "co2":
        return round(engine_val, 0), "ppm", who_eng, "ppm"
    elif key == "radon":
        return round(engine_val, 1), "Bq/m³", who_eng, "Bq/m³"
    elif key in ("pollen_tree", "pollen_grass", "pollen_weed"):
        return round(engine_val, 1), "gr/m³", who_eng, "gr/m³"
    elif key == "temp":
        return round(engine_val, 1), "°C", None, ""
    elif key == "humidity":
        return round(engine_val, 1), "%", None, ""
    else:
        return round(engine_val, 3), "", None, ""


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
    "B": "#ff6b6b",   # B = penalty layer — red
    "C": "#ffa94d",
    "D": "#da77f2",
}

COMP_LABELS = {
    "A": "A — Instant air quality",
    "C": "C — 30-day chronic load",
    "D": "D — Sleep environment",
}

ZONE_OPTIONS = ["HOME", "OUTDOOR"]
ZONE_ICONS   = {"HOME": "🏠", "OUTDOOR": "🌳"}

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
        "hour":      hour,
        "ecs":       result["ECS"],
        "A":         result["A"],
        "B":         result["B"],
        "C":         result["C"],
        "D":         result["D"],
        "ts":        local_dt.strftime("%H:%M"),
        "zone":      zone_str,
        "offending": result.get("offending", []),
        "streak":    result.get("exposure_streak", 0),
        "penalty":   result.get("penalty_pts", 0),
        "raw":       {**assembly.outdoor_raw, **assembly.pollen_raw},
        "indoor":    dict(assembly.pollutants),
        "backfilled": False,
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


def _backfill_missing_hours():
    """
    Called on app open when hours have been missed (browser was closed).
    Uses OWM historical API to fetch AQ for each missed hour and scores them.
    Pollen is held constant at today's value (Google Pollen is daily only).
    Silently fills history so user sees a complete picture when they return.
    Only backfills within the current ECS day (07:00 boundary).
    """
    local_dt     = _local_now()
    current_hour = local_dt.hour
    last_hour    = st.session_state.last_scored_hour

    if last_hour is None or last_hour == current_hour:
        return

    day_start = 7
    if current_hour >= day_start:
        valid_hours = list(range(day_start, current_hour))
    else:
        valid_hours = list(range(day_start, 24)) + list(range(0, current_hour))

    scored_hours = {h["hour"] for h in st.session_state.history}
    missing      = [h for h in valid_hours if h not in scored_hours]
    if not missing:
        return

    lat, lon     = _effective_coords()
    assembler    = _get_assembler()
    engine_state = _get_or_init_engine_state()

    try:
        pollen_raw = assembler.pollen.fetch(lat, lon)
    except Exception:
        pollen_raw = {"pollen_tree": 0, "pollen_grass": 0, "pollen_weed": 0}

    def _hour_to_ts(h):
        d = local_dt.date()
        if current_hour < day_start and h >= day_start:
            d = d - datetime.timedelta(days=1)
        return int(datetime.datetime(d.year, d.month, d.day, h, 0, 0,
                                     tzinfo=_TZ).timestamp())

    start_ts = _hour_to_ts(missing[0])
    end_ts   = _hour_to_ts(missing[-1]) + 3600

    try:
        history_data = assembler.owm.fetch_history(lat, lon, start_ts, end_ts)
    except Exception:
        return

    hour_to_aq = {}
    for entry in history_data:
        h = datetime.datetime.fromtimestamp(entry["timestamp"], tz=_TZ).hour
        hour_to_aq[h] = entry

    zone_str  = st.session_state.zone
    zone      = LocationZone(zone_str.lower())
    zone_val  = zone.value
    is_indoor = zone_val in {z.value for z in INDOOR_ZONES}
    new_entries = []

    for h in sorted(missing):
        if h not in hour_to_aq:
            continue
        aq = hour_to_aq[h]
        outdoor_raw = {k: aq[k] for k in ("pm25","pm10","no2","o3","co")}

        if is_indoor:
            pollutants = {k: round(v * INFILTRATION_FACTORS[k], 3)
                          for k, v in outdoor_raw.items()}
            for pk, pv in pollen_raw.items():
                pollutants[pk] = round(pv * INFILTRATION_FACTORS.get(pk, 0.10), 3)
        else:
            pollutants = dict(outdoor_raw)
            pollutants.update(pollen_raw)

        in_sleep = _is_sleep_window(h)
        reading  = HourlyReading(
            timestamp=h, pollutants=pollutants,
            biomarkers=None, in_sleep_window=in_sleep,
            protection_confirmed=False,
        )
        result = compute_hourly_ECS(reading, engine_state)

        new_entries.append({
            "hour":       h,
            "ecs":        result["ECS"],
            "A":          result["A"],  "B": result["B"],
            "C":          result["C"],  "D": result["D"],
            "ts":         f"{h:02d}:00",
            "zone":       zone_str,
            "offending":  result.get("offending", []),
            "streak":     result.get("exposure_streak", 0),
            "penalty":    result.get("penalty_pts", 0),
            "raw":        {**outdoor_raw, **pollen_raw},
            "indoor":     dict(pollutants),
            "backfilled": True,
        })

    if new_entries:
        existing    = {e["hour"] for e in st.session_state.history}
        combined    = st.session_state.history + [e for e in new_entries if e["hour"] not in existing]
        combined.sort(key=lambda x: x["hour"])
        st.session_state.history      = combined
        st.session_state.engine_state = engine_state
        if combined:
            st.session_state.last_scored_hour = combined[-1]["hour"]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Backfill missed hours (silently, before rendering) ───────────────────────
# If the browser was closed and hours were missed, fetch OWM historical data
# and score each missed hour so history is complete when user returns.
if (st.session_state.last_scored_hour is not None
        and st.session_state.last_scored_hour != _now.hour):
    _backfill_missing_hours()

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

zone_cols = st.columns(2)
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

    # ── Components A, C, D as bars + B as penalty readout ───────────────────
    st.markdown("#### Components")
    streak  = result.get("exposure_streak", 0)
    penalty = result.get("penalty_pts", 0)
    bio     = result.get("bio_state", "neutral")
    offend  = result.get("offending", [])
    b_val   = result.get("B", 0)   # penalty pts — 0 when no streak

    comp_html = ""
    for key in ["A", "C", "D"]:
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

    # Component B — penalty layer (streak + bio) shown separately
    if streak == 0:
        b_label = "B — Exposure penalty"
        b_color = "#555"
        b_text  = "No penalty  (streak = 0)"
    elif streak <= 2:
        b_label = "B — Exposure penalty"
        b_color = "#fcc419"
        b_text  = f"⚠ Warning — streak {streak}h  (penalty starts at hour 3)"
    else:
        b_color = "#ff6b6b"
        b_label = "B — Exposure penalty"
        bio_tag = "  + bio-confirmed" if result.get("bio_confirmed") else ""
        b_text  = f"−{b_val:.1f} pts  (streak {streak}h{bio_tag})"

    comp_html += f"""
<div class="comp-row">
  <div class="comp-label" style="color:{b_color}">{b_label}</div>
  <div class="comp-bar-bg" style="background:#1a1a1a">
    <div style="padding:2px 8px;font-size:12px;color:{b_color}">{b_text}</div>
  </div>
  <div class="comp-value" style="color:{b_color}">{"−"+str(b_val) if b_val > 0 else "0"}</div>
</div>"""

    st.markdown(comp_html, unsafe_allow_html=True)
    st.markdown("")

    info_cols = st.columns(2)
    with info_cols[0]:
        st.metric("Body state", bio.replace("_", " ").title())
    with info_cols[1]:
        st.metric("Exposure streak", f"{streak}h", help="Consecutive hours above WHO limits")

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

        # Sensor-only pollutants — always show in table, blank if no sensor data
        SENSOR_ONLY_KEYS = ["co2", "radon", "humidity", "temp"]
        SENSOR_UNITS     = {"co2": "ppm", "radon": "Bq/m³", "humidity": "%", "temp": "°C"}

        rows = ""
        for key, final_val in sorted(assembly.pollutants.items()):
            name    = DISPLAY_NAMES.get(key, key)
            outdoor = assembly.outdoor_raw.get(key) if assembly.outdoor_raw.get(key) is not None \
                      else assembly.pollen_raw.get(key)
            factor  = INFILTRATION_FACTORS.get(key)

            # Convert to display units (ppb for NO2/O3, ppm for CO, unchanged for rest)
            disp_final, unit, who_disp = _display_val(key, final_val)
            disp_outdoor = round(outdoor * UNIT_CONVERT[key]["factor"], 2) if (outdoor is not None and key in UNIT_CONVERT) else outdoor
            who_str = f"{who_disp}" if who_disp is not None else "—"

            # For pollen: WHO comparison uses outdoor raw (indoor heavily attenuated)
            compare_val = disp_final
            if is_indoor and outdoor is not None and key in ("pollen_tree", "pollen_grass", "pollen_weed"):
                compare_val = disp_outdoor if disp_outdoor is not None else disp_final

            above       = who_disp is not None and compare_val > who_disp
            outdoor_str = f"{disp_outdoor}" if disp_outdoor is not None else "—"

            if is_indoor and factor is not None and outdoor is not None:
                engine_str = f"{disp_final}  <span style='color:#555;font-size:11px'>(×{factor})</span>"
            else:
                engine_str = f"{disp_final}"

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

        # Add sensor-only rows — always shown, marked blank if no sensor connected
        if is_indoor:
            for key in SENSOR_ONLY_KEYS:
                if key in assembly.pollutants:
                    continue  # already shown above
                name     = DISPLAY_NAMES.get(key, key)
                _, unit, who_disp = _display_val(key, 0)
                who_str  = f"{who_disp}" if who_disp is not None else "—"  if who is not None else "—"
                rows += f"""
<tr style="color:#555;font-style:italic">
  <td>{name}</td>
  <td style="color:#555">—</td>
  <td style="color:#555">—  <span style='font-size:10px'>(no sensor)</span></td>
  <td style="color:#555">—</td>
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

    hours  = [h["hour"] for h in history]
    scores = [h["ecs"]  for h in history]
    zones  = [h.get("zone", "?") for h in history]
    labels = [f"{h['ts']}  ECS {h['ecs']}  {h.get('zone','')}{'  ↩backfilled' if h.get('backfilled') else ''}"
              for h in history]

    try:
        import plotly.graph_objects as go

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

    except Exception as _chart_err:
        # Fallback to st.line_chart if plotly unavailable or fails
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
components.html(GPS_BRIDGE, height=0)# ── History — always visible ─────────────────────────────────────────────────
history = st.session_state.history
st.markdown("---")
st.markdown("#### 📋 Today's hourly history")

if not history:
    st.info("No readings yet today. Hit **⚡ Score this hour** to start, or come back after the first hour — missed readings are backfilled automatically.")
else:
    # ── ECS chart ──────────────────────────────────────────────────────────
    hours  = [h["hour"] for h in history]
    scores = [h["ecs"]  for h in history]
    zones  = [h.get("zone", "?") for h in history]
    labels = [f"{h['ts']}  ECS {h['ecs']}  {h.get('zone','')}{'  ↩' if h.get('backfilled') else ''}"
              for h in history]

    try:
        import plotly.graph_objects as go
        point_colors  = [_color_for_score(s) for s in scores]
        marker_symbols = ["circle-open" if h.get("backfilled") else "circle" for h in history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=scores,
            mode="lines+markers",
            line=dict(color="#51cf66", width=2),
            marker=dict(color=point_colors, size=10,
                        symbol=marker_symbols,
                        line=dict(color="#1a1a2e", width=2)),
            text=labels,
            hovertemplate="%{text}<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(81,207,102,0.08)",
        ))
        fig.add_hrect(y0=75, y1=100, fillcolor="rgba(81,207,102,0.05)",  line_width=0)
        fig.add_hrect(y0=60, y1=75,  fillcolor="rgba(148,216,45,0.05)",  line_width=0)
        fig.add_hrect(y0=40, y1=60,  fillcolor="rgba(252,196,25,0.05)",  line_width=0)
        fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(255,107,107,0.05)", line_width=0)
        fig.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#888", size=12),
            xaxis=dict(title="Hour", tickmode="linear", dtick=1,
                       range=[-0.5, 23.5],
                       gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
            yaxis=dict(title="ECS", range=[0, 100],
                       gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
            margin=dict(l=40, r=20, t=20, b=40),
            height=260, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        import pandas as pd
        df = pd.DataFrame({"Hour": hours, "ECS": scores}).set_index("Hour")
        st.line_chart(df)

    if any(h.get("backfilled") for h in history):
        st.caption("↩ = backfilled from OWM historical data while browser was closed")

    # ── Per-hour table — always visible, expandable detail per row ─────────
    DISPLAY_NAMES_SHORT = {
        "pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "O₃", "co": "CO",
        "pollen_tree": "Tree 🌲", "pollen_grass": "Grass 🌾", "pollen_weed": "Weed 🌿",
        "co2": "CO₂", "radon": "Radon", "humidity": "RH%", "temp": "Temp °C",
    }
    WHO_SHORT = {
        "pm25": 15, "pm10": 45, "no2": 25, "o3": 100, "co": 4,
        "pollen_tree": 50, "pollen_grass": 50, "pollen_weed": 10,
    }

    for entry in reversed(history):
        h        = entry["hour"]
        ecs      = entry["ecs"]
        color    = _color_for_score(ecs)
        offend   = entry.get("offending", [])
        streak   = entry.get("streak", "—")
        penalty  = entry.get("penalty", 0)
        bf_tag   = " ↩" if entry.get("backfilled") else ""
        alert_tag = f"  ⚠ {', '.join(o.upper() for o in offend)}" if offend else ""

        label = (
            f"**{h:02d}:00**  ·  "
            f"<span style='color:{color};font-weight:700'>{ecs}</span>  "
            f"· {entry.get('zone','?')}{bf_tag}{alert_tag}"
        )
        with st.expander(f"{h:02d}:00  ECS {ecs}  {entry.get('zone','?')}{bf_tag}{alert_tag}"):
            raw    = entry.get("raw", {})
            indoor = entry.get("indoor", {})

            # Show all known pollutants — blank if no data
            all_keys = ["pm25","pm10","no2","o3","co",
                        "pollen_tree","pollen_grass","pollen_weed",
                        "co2","radon","humidity","temp"]

            rows_html = ""
            for key in all_keys:
                name     = DISPLAY_NAMES_SHORT.get(key, key)
                raw_val  = raw.get(key)
                ind_val  = indoor.get(key)
                who_lim  = WHO_SHORT.get(key, "—")
                # Convert display units
                disp_raw, unit_str, who_disp = _display_val(key, raw_val if raw_val is not None else 0)
                disp_ind, _, _               = _display_val(key, ind_val if ind_val is not None else 0)
                who_lim  = who_disp

                raw_str = f"{disp_raw}" if raw_val is not None else "—"
                ind_str = f"{disp_ind}" if ind_val is not None else "—"

                # Highlight if above WHO
                above = (raw_val is not None and isinstance(who_lim, (int,float))
                         and disp_raw > who_lim)
                row_style = "color:#ff6b6b" if above else "color:#ccc"
                flag = " ⚠" if above else ""

                rows_html += f"""
<tr style="{row_style}">
  <td>{name}{flag}</td>
  <td>{raw_str}</td>
  <td>{ind_str}</td>
  <td style="color:#555">{who_lim}</td>
  <td style="color:#555">{unit_str}</td>
</tr>"""

            st.markdown(f"""
<table class="poll-table">
  <thead><tr>
    <th>Pollutant</th><th>Outdoor raw</th>
    <th>Engine (infiltrated)</th><th>WHO limit</th><th>Unit</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)

            # Component scores for this hour
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("ECS",         entry["ecs"])
            c2.metric("A — Instant", f"{entry['A']}")
            c3.metric("C — Chronic", f"{entry['C']}")
            c4.metric("D — Sleep",   f"{entry['D']}")
            b_entry = entry.get("B", entry.get("penalty", 0))
            sk      = entry.get("streak", 0)
            if sk and int(sk) >= 3:
                st.error(f"B — Penalty: −{b_entry} pts  |  Streak: {sk}h")
            elif sk and int(sk) >= 1:
                st.warning(f"B — Streak warning: {sk}h  (penalty fires at hour 3)")

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
