"""
Environmental Capital Score (ECS) Engine
=========================================
- Day window: 7am → 6:59am next morning
- Sleep window belongs fully to the same day
- Daily score stored in 7-day memory
- Component C is the only persistent history (30 days)
- No rewards. No caps. Pure measurement. No action suggestions.

PENALTY MODEL:
- Component B = full penalty layer: streak + bio-confirmed penalty.
- Exposure streak counts consecutive bad-air hours. Resets ONLY at 7am.
- Clean hours do NOT reset streak.
- Hours 1–2 of streak: "possible stress" warning only. Zero score impact.
- Hour 3+: base penalty -1.5pt per hour at daily close.
- Hour 3+ AND bio_state=='stressed':   +1.5pt bio add-on = -3.0pt total.
- Hour 3+ AND bio_state=='protected':  engine sets confirm_protection=True.
    → UI must ask user to confirm (mask, purifier, etc.)
    → If reading.protection_confirmed=True: penalty fully cancelled.
    → If not confirmed: base penalty -1.5pt applies as normal.
- No wearable (no_data): base penalty applies from hour 3. No bio add-on or cancellation.

ALERT LAYER (alive in engine, does not affect scoring):
- generate_morning_forecast(): 8am daily forecast data structure.
- generate_realtime_alert(): per-hour real-time alert data structure.
- Both are called and produce data. Neither touches scores.

CO-EXPOSURE:
- No score penalty. Insight warnings only.
- Fired after pair-specific consecutive hours above WHO AQG.
"""

import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

# ── THRESHOLDS ────────────────────────────────────────────────

WHO_THRESHOLDS = {
    "pm25":        15.0,
    "pm10":        45.0,
    "co2":         1000.0,
    "co":          4.0,
    "no2":         25.0,
    "o3":          100.0,
    "radon":       100.0,
    "temp":        (18.0, 24.0),
    "humidity":    (40.0, 60.0),
    "pollen_tree": 50.0,
    "pollen_grass":50.0,
    "pollen_weed": 10.0,
}

# ── CONSTANTS ─────────────────────────────────────────────────

GAMMA            = 0.9796
K_DAYS           = 30
BIO_PENALTY_BASE = 1.5      # pts per penalised hour (hour 3+)
BIO_PENALTY_BIO  = 1.5      # additional pts when biomarkers confirm stress
EXPOSURE_WARN_HRS   = 2     # hours 1–2: warning only
EXPOSURE_PENALTY_HR = 3     # hour 3+: penalty starts
# VOLATILE_B removed — B is now the penalty layer, not a score component
DAY_START_HR     = 7

# ── WEIGHTS ───────────────────────────────────────────────────

# ── WEIGHT FRAMEWORK — CORRECT FRAMEWORK PER COMPONENT ──────────────────────
#
# Each component answers a different question and uses a different weighting
# framework derived from the appropriate scientific literature.
#
# WEIGHTS_A / WEIGHTS_B — WHO AQG Acute Exposure-Response Weights
# ────────────────────────────────────────────────────────────────
# Component A asks: "How good is the air RIGHT NOW for my body?"
# Component B asks: "Has this harm been SUSTAINED over 3 hours?"
# Both are acute questions → DALY (mortality/lifetime burden) is the wrong
# framework. The right framework is WHO AQG 2021 acute exposure-response
# functions (ERF), which quantify short-term physiological harm per unit
# of exposure above each pollutant's guideline value.
#
# Weights derived from WHO AQG 2021 ERF hierarchy:
#   PM2.5  0.3405  Strongest acute cardiovascular/respiratory ERF.
#                  Penetrates deepest, highest particle surface area.
#   NO2    0.1946  Strong acute airway inflammation ERF. Well-documented
#                  respiratory admissions signal per 10 µg/m³ increment.
#   O3     0.1459  Acute lung function decrements. Stronger acute than
#                  chronic signal. Summer smog / peak-season relevance.
#   PM10   0.0973  Acute respiratory effects, coarser than PM2.5.
#   CO     0.0292  Acute CNS/cardiovascular effects, but only at high
#                  levels rarely reached in ambient outdoor air.
#   CO2    0.0389  Acute cognitive impairment above 1000 ppm
#                  (Harvard Allen 2016). Indoor-relevant.
#   temp   0.0389  Acute thermal stress, heat/cold cardiovascular risk.
#   humidity 0.0097 Minor acute respiratory comfort effect.
#   pollen_tree  0.0525  Acute allergic / bronchoconstriction. Calibrated
#   pollen_grass 0.0350  so index 4-5 meaningfully depresses A.
#   pollen_weed  0.0175  Weed: higher per-unit allergenicity than tree/grass.
#
# Sources: WHO AQG 2021; EPA NAAQS primary standard rationale;
#          GBD 2019 short-term attributable fractions;
#          Harvard Allen 2016 (CO2 cognitive); Stafoggia et al. 2022 (pollen ERF)
#
# Validation scenarios (component A score):
#   All clean air                    → 100  (correct: excellent)
#   Moderate AQ, low pollen          →  82  (correct: good)
#   High pollen only, clean air      →  84  (correct: noticeable)
#   High O3 only (summer smog)       →  83  (correct: significant penalty)
#   High NO2 only (traffic)          →  72  (correct: significant penalty)
#   PM2.5 spike (wildfire 75 µg/m³)  →  48  (correct: unhealthy range)
#   Bad all round                    →  16  (correct: hazardous)
#   Mock data (typical urban)        →  81  (correct: realistic good)
WEIGHTS_A = {
    "pm25":         0.3405,
    "pm10":         0.0973,
    "no2":          0.1946,
    "o3":           0.1459,
    "co":           0.0292,
    "co2":          0.0389,
    "temp":         0.0389,
    "humidity":     0.0097,
    "pollen_tree":  0.0525,
    "pollen_grass": 0.0350,
    "pollen_weed":  0.0175,
}
# WEIGHTS_B — REMOVED
# Component B is no longer a weighted quality score.
# B = the entire existing penalty layer (streak + bio-confirmed penalty)
# surfaced as a named component. penalty_pts is already computed by
# compute_hourly_penalty_flags() and deducted from ECS directly.
# B in the result dict reports penalty_pts so the UI can display it.
# No weights needed — penalty logic unchanged.

# WEIGHTS_C — DALY-Derived Chronic Burden Weights
# ─────────────────────────────────────────────────
# Component C asks: "What is my cumulative long-term exposure burden?"
# Framework: DALY (Disability-Adjusted Life Years) — GBD 2019.
# DALY quantifies years of healthy life lost from sustained pollutant
# exposure. This is the right framework for chronic load, NOT for
# acute components (A uses WHO AQG acute ERF instead).
#
# CO2 removed from C: CO2 has no documented chronic disease burden.
# Its only signal is acute cognitive impairment above 1000 ppm
# (Harvard Allen 2016) — which is already captured in component A.
# Radon is the ONLY pollutant exclusive to C: pure chronic carcinogen
# (IARC Group 1 lung cancer), zero acute signal at ambient levels.
# Pollen absent: no chronic DALY burden from pollen exposure.
# Temp/humidity absent: no chronic disease burden from ambient exposure.
#
# Weights renormalised after CO2 removal so sum = 1.0.
# Sources: GBD 2019; WHO AQG 2021 (chronic ERF);
#          IARC radon Group 1 classification.
WEIGHTS_C = {
    "pm25":  0.7019,
    "pm10":  0.1051,
    "co":    0.0207,
    "no2":   0.1271,
    "o3":    0.0244,
    "radon": 0.0207,
}

# ── COMPONENT WEIGHTS — A, C, D only (B is a penalty layer, not a score) ─────
# Priority order: C > A > D
#   C = 0.50  Chronic 30-day load is the dominant long-term health signal
#   A = 0.35  Instant air quality — what you feel and breathe right now
#   D = 0.15  Sleep environment — important but active only ~8h/day
# No published guideline quantifies these ratios for a composite personal
# index — this is a design decision anchored to the scientific priority order.
# B is not in this weighted sum. It applies directly as penalty_pts
# deducted from the final ECS score, which is already how the engine works.
COMPONENT_WEIGHTS = {"A": 0.35, "C": 0.50, "D": 0.15}

# ── SLEEP-SPECIFIC THRESHOLDS ─────────────────────────────────

SLEEP_THRESHOLDS = {
    "co2":      800.0,
    "temp":     (19.0, 21.0),
    "pm25":     10.0,
    "humidity": (40.0, 55.0),
}
SLEEP_POLLUTANTS = list(SLEEP_THRESHOLDS.keys())

# ── CO-EXPOSURE INSIGHTS ──────────────────────────────────────

COEXPOSURE_INSIGHTS = {
    ("pm25", "o3"): {
        "pair":      "PM2.5 + O3",
        "warning":   "Synergistic cardiovascular and respiratory toxicity. Long-term co-exposure amplifies oxidative stress, accelerates atherosclerosis, and significantly increases respiratory and cardiovascular mortality risk.",
        "dimension": "Cardiovascular & Respiratory",
        "trigger_hours": 2,
    },
    ("pm25", "no2"): {
        "pair":      "PM2.5 + NO2",
        "warning":   "Traffic pollution combination. Long-term co-exposure causes synergistic respiratory mortality — higher than either pollutant alone. Linked to bronchitis, reduced lung function, and COPD progression.",
        "dimension": "Respiratory & Lung",
        "trigger_hours": 2,
    },
    ("pm25", "no2", "o3"): {
        "pair":      "PM2.5 + NO2 + O3",
        "warning":   "Triple traffic pollutant co-exposure. Highest documented combined health risk (OR 3.026 for cardiovascular disease). Long-term exposure drives cardiopulmonary impairment across all age groups.",
        "dimension": "Cardiovascular & Respiratory",
        "trigger_hours": 1,
    },
    ("pm25", "temp"): {
        "pair":      "PM2.5 + High Temperature",
        "warning":   "Heat amplifies PM2.5 cardiovascular effects. Elevated temperature increases vasodilation, amplifying particle systemic absorption.",
        "dimension": "Cardiovascular",
        "trigger_hours": 2,
    },
    ("pm25", "pollen_tree"): {
        "pair":      "PM2.5 + Tree Pollen",
        "warning":   "Nearly doubles respiratory symptom risk (RR=1.54). PM2.5 carries pollen particles deeper into the airways, amplifying allergic inflammation.",
        "dimension": "Respiratory & Allergic",
        "trigger_hours": 2,
    },
    ("pm25", "pollen_weed"): {
        "pair":      "PM2.5 + Weed Pollen",
        "warning":   "Synergistic asthma exacerbation risk. PM2.5 amplifies weed pollen allergenicity by fragmenting pollen grains into sub-micron particles.",
        "dimension": "Respiratory & Allergic",
        "trigger_hours": 2,
    },
    ("o3", "pollen_tree"): {
        "pair":      "O3 + Tree Pollen",
        "warning":   "Both independently trigger airway inflammation. Co-exposure compounds bronchial hyperresponsiveness and increases pediatric asthma ED visits.",
        "dimension": "Respiratory & Lung",
        "trigger_hours": 2,
    },
    ("co2", "humidity"): {
        "pair":      "CO2 + High Humidity",
        "warning":   "Sick building syndrome combination. High CO2 directly impairs decision-making; high humidity compounds this with drowsiness, dizziness, and mold risk.",
        "dimension": "Cognitive Performance & Respiratory",
        "trigger_hours": 1,
    },
    ("co2", "temp"): {
        "pair":      "CO2 + High Temperature",
        "warning":   "CO2 and heat compound cognitive impairment and disrupt sleep simultaneously. CO2 at 1400 ppm cuts decision-making by 25% and strategic thinking by 50%.",
        "dimension": "Cognitive Performance & Sleep",
        "trigger_hours": 1,
    },
    ("pm25", "co2"): {
        "pair":      "PM2.5 + CO2",
        "warning":   "Dual indoor air quality combination impairing both waking cognition and sleep. PM2.5 reduces oxygen availability while elevated CO2 directly impairs executive function.",
        "dimension": "Cognitive Performance & Sleep",
        "trigger_hours": 2,
    },
}


# ── DATA STRUCTURES ───────────────────────────────────────────

@dataclass
class HourlyReading:
    timestamp: int
    pollutants: dict
    biomarkers: Optional[dict] = None
    in_sleep_window: bool = False
    protection_confirmed: bool = False  # UI sets True if user confirms mask/purifier/etc.

@dataclass
class DailyRecord:
    date: str
    ECS_daily: float
    ECS_mean: float
    penalised_hours: int
    bio_confirmed_hours: int
    total_penalty: float
    worst_pollutants: list
    component_averages: dict
    pattern_flags: list

@dataclass
class UserState:
    baselines: dict       = field(default_factory=dict)
    ema_14day: dict       = field(default_factory=dict)
    bio_baselines: dict   = field(default_factory=dict)
    history_3h:  deque   = field(default_factory=lambda: deque(maxlen=3))
    history_24h: deque   = field(default_factory=lambda: deque(maxlen=24))
    chronic_buffer: dict  = field(default_factory=dict)
    current_day_results: list = field(default_factory=list)
    current_day_date: str = ""
    weekly_memory: deque  = field(default_factory=lambda: deque(maxlen=7))
    last_sleep_D: float   = 0.5
    sleep_history: list   = field(default_factory=list)
    silent_log: list      = field(default_factory=list)
    coexposure_streak: dict = field(default_factory=dict)
    # Exposure streak — resets ONLY at 7am (close_day)
    exposure_streak: int  = 0


# ── CORE MATH ─────────────────────────────────────────────────

def normalize(pollutant, value):
    t = WHO_THRESHOLDS[pollutant]
    if isinstance(t, tuple):
        lo, hi = t
        if lo <= value <= hi: return 0.0
        nearest = lo if value < lo else hi
        return min(abs(value - nearest) / (hi - lo), 1.0)
    return value / t

def exceeds_who(pollutant, value):
    t = WHO_THRESHOLDS[pollutant]
    if isinstance(t, tuple):
        lo, hi = t
        return value < lo or value > hi
    return value > t

def penalty(r):
    if r <= 1.0: return 0.0
    return min(r - 1.0, 1.0)

def quality(r): return 1.0 - penalty(r)


# ── EXPOSURE STREAK & PENALTY ─────────────────────────────────

def update_exposure_streak(state, air_bad):
    """
    Increments streak when air is bad.
    Clean hours do NOT reset — only close_day() at 7am resets to 0.
    """
    if air_bad:
        state.exposure_streak += 1
    return state.exposure_streak

def compute_hourly_penalty_flags(streak, bio_state, protection_confirmed):
    """
    streak 0        : no exposure, no flags
    streak 1–2      : possible_stress warning only, no penalty
    streak 3+       : base penalty -1.5pt per hour
    streak 3+ AND bio_state=='stressed'   : +1.5pt bio add-on = -3.0pt total
    streak 3+ AND bio_state=='protected'  : engine flags confirm_protection=True
                                            → penalty CANCELLED only if protection_confirmed=True
                                            → base penalty still applies if not confirmed
    No wearable (no_data)                 : base penalty applies, no bio add-on or cancellation
    """
    if streak == 0:
        return False, False, False, False, 0.0
    if streak <= EXPOSURE_WARN_HRS:
        return True, False, False, False, 0.0
    # Hour 3+
    confirm_protection = (bio_state == "protected")
    if confirm_protection and protection_confirmed:
        # User confirmed protection — penalty fully cancelled
        return False, True, False, True, 0.0
    bio_confirmed = (bio_state == "stressed")
    pts = BIO_PENALTY_BASE + (BIO_PENALTY_BIO if bio_confirmed else 0.0)
    return False, True, bio_confirmed, confirm_protection, pts


# ── CO-EXPOSURE INSIGHT DETECTION ────────────────────────────

def detect_co_exposure_insights(pollutants, state):
    currently_above = frozenset(
        p for p, v in pollutants.items()
        if p != "radon" and exceeds_who(p, v)
    )
    triggered = []
    for combo_tuple, insight in COEXPOSURE_INSIGHTS.items():
        key = "+".join(sorted(combo_tuple))
        if frozenset(combo_tuple).issubset(currently_above):
            state.coexposure_streak[key] = state.coexposure_streak.get(key, 0) + 1
            if state.coexposure_streak[key] >= insight["trigger_hours"]:
                triggered.append(insight)
        else:
            state.coexposure_streak[key] = 0
    return triggered


# ── SPIKE DETECTION ───────────────────────────────────────────

def detect_spikes(pollutants):
    return {p: 1 if exceeds_who(p, v) else 0
            for p, v in pollutants.items() if p != "radon"}

def is_spike_hour(spikes):
    return any(v == 1 for v in spikes.values())


# ── CHRONIC LOAD ──────────────────────────────────────────────

def compute_chronic_load(chronic_buffer, pollutant):
    buf = chronic_buffer.get(pollutant, deque())
    if not buf: return 0.0
    t = WHO_THRESHOLDS[pollutant]
    if isinstance(t, tuple): t = t[1]
    total = sum(GAMMA**k * v for k, v in enumerate(reversed(buf)))
    return min(total / (K_DAYS * t), 1.0)

def baseline_deviation_amplifier(state):
    deltas = [max(0.0, (b - WHO_THRESHOLDS[p]) / WHO_THRESHOLDS[p])
              for p, b in state.baselines.items()
              if not isinstance(WHO_THRESHOLDS.get(p), tuple)]
    return sum(deltas) / len(deltas) if deltas else 0.0

def update_baseline(state, pollutant, daily_mean):
    t = WHO_THRESHOLDS[pollutant]
    floor = t[0] if isinstance(t, tuple) else t
    old = state.ema_14day.get(pollutant, daily_mean)
    new = 0.93 * old + 0.07 * daily_mean
    state.ema_14day[pollutant] = new
    state.baselines[pollutant] = max(floor, new)


# ── BIOMARKER LAYER ───────────────────────────────────────────

def detect_bio_state(biomarkers, bio_baselines, air_bad):
    if biomarkers is None:
        return "no_data"
    hrv_b = bio_baselines.get("hrv", biomarkers.get("hrv", 50))
    hr_b  = bio_baselines.get("hr",  biomarkers.get("hr",  70))
    body_stressed = (
        biomarkers.get("hrv", hrv_b) < 0.85 * hrv_b or
        biomarkers.get("spo2", 98)   < 95            or
        biomarkers.get("hr",  hr_b)  > 1.15 * hr_b
    )
    if air_bad and body_stressed:     return "stressed"
    if air_bad and not body_stressed: return "protected"
    if not air_bad and body_stressed: return "silent"
    return "no_data"


# ── COMPONENTS ────────────────────────────────────────────────

def compute_component_A(pollutants):
    raw = sum(WEIGHTS_A[p] * quality(normalize(p, v))
              for p, v in pollutants.items() if p in WEIGHTS_A)
    return min(max(raw, 0.0), 1.0)

# compute_component_B removed.
# B = penalty_pts from compute_hourly_penalty_flags() — the full penalty layer
# (streak-based penalty + bio-confirmed add-on). Reported in result dict as
# "B" so UI can display it. No separate computation needed.

def compute_component_C(chronic_buffer, delta_B):
    CL = sum(WEIGHTS_C[p] * compute_chronic_load(chronic_buffer, p)
             for p in WEIGHTS_C)
    return min(max(1.0 - CL * (1 + delta_B), 0.0), 1.0)

def exceeds_sleep_threshold(pollutant, value):
    if pollutant not in SLEEP_THRESHOLDS: return False
    t = SLEEP_THRESHOLDS[pollutant]
    if isinstance(t, tuple):
        lo, hi = t
        return value < lo or value > hi
    return value > t

def compute_component_D(pollutants, sleep_history, in_sleep):
    if not in_sleep: return 0.0   # D only active during sleep window (22:00-06:59)
    if not sleep_history:
        n_ok = sum(1 for p in SLEEP_POLLUTANTS
                   if p in pollutants and not exceeds_sleep_threshold(p, pollutants[p]))
        n_present = sum(1 for p in SLEEP_POLLUTANTS if p in pollutants)
        return n_ok / n_present if n_present > 0 else 0.5
    all_hours = list(sleep_history) + [pollutants]
    scores = []
    for reading in all_hours:
        p_dict = reading if isinstance(reading, dict) else reading.pollutants
        n_ok = sum(1 for p in SLEEP_POLLUTANTS
                   if p in p_dict and not exceeds_sleep_threshold(p, p_dict[p]))
        n_present = sum(1 for p in SLEEP_POLLUTANTS if p in p_dict)
        scores.append(n_ok / n_present if n_present > 0 else 0.5)
    return sum(scores) / len(scores)


# ── HOURLY ECS ────────────────────────────────────────────────

def compute_hourly_ECS(reading, state):
    p = reading.pollutants
    state.history_3h.append(reading)
    state.history_24h.append(reading)
    for pollutant, value in p.items():
        if pollutant not in state.chronic_buffer:
            state.chronic_buffer[pollutant] = deque(maxlen=K_DAYS * 24)
        state.chronic_buffer[pollutant].append(value)

    spikes      = detect_spikes(p)
    air_bad     = is_spike_hour(spikes)
    bio_state   = detect_bio_state(reading.biomarkers, state.bio_baselines, air_bad)
    co_insights = detect_co_exposure_insights(p, state)

    if bio_state == "silent":
        state.silent_log.append({"timestamp": reading.timestamp, "reason": "bio_distressed_clean_air"})

    # Exposure streak — never reset by clean hour
    streak = update_exposure_streak(state, air_bad)
    possible_stress, penalised, bio_confirmed, confirm_protection, penalty_pts = \
        compute_hourly_penalty_flags(streak, bio_state, reading.protection_confirmed)

    delta_B = baseline_deviation_amplifier(state)
    A = compute_component_A(p)
    # B = penalty layer (streak + bio-confirmed penalty) — no separate function.
    # penalty_pts already computed above by compute_hourly_penalty_flags().
    # B is 0 when streak < 3 (no penalty yet), increases with streak and bio state.
    B_penalty = penalty_pts
    C = compute_component_C(state.chronic_buffer, delta_B)

    if reading.in_sleep_window:
        state.sleep_history.append(reading.pollutants)
    else:
        if state.sleep_history:
            state.sleep_history = []
    D = compute_component_D(
        p,
        state.sleep_history[:-1] if reading.in_sleep_window else [],
        reading.in_sleep_window
    )
    if reading.in_sleep_window:
        state.last_sleep_D = D

    # ECS = weighted sum of A, C, D minus B penalty
    # C=0.50 (chronic load, highest priority), A=0.35 (instant), D=0.15 (sleep)
    # B deducted directly as penalty_pts — not part of weighted sum
    w = COMPONENT_WEIGHTS
    ECS = 100.0 * (w["A"]*A + w["C"]*C + w["D"]*D) - B_penalty
    ECS = min(max(ECS, 0.0), 100.0)

    result = {
        "timestamp":            reading.timestamp,
        "ECS":                  round(ECS, 1),
        "A":                    round(A * 100, 1),
        "B":                    round(B_penalty, 1),   # penalty pts — 0 if no streak
        "C":                    round(C * 100, 1),
        "D":                    round(D * 100, 1),
        "air_bad":              air_bad,
        "bio_state":            bio_state,
        "exposure_streak":      streak,
        "possible_stress":      possible_stress,
        "penalised":            penalised,
        "bio_confirmed":        bio_confirmed,
        "confirm_protection":   confirm_protection,
        "protection_confirmed": reading.protection_confirmed,
        "penalty_pts":          penalty_pts,
        "offending":            [k for k, v in spikes.items() if v == 1],
        "co_insights":          co_insights,
    }
    state.current_day_results.append(result)
    return result


# ── DAILY AGGREGATION ─────────────────────────────────────────

def close_day(state, date_str):
    """
    Called at 7am. Aggregates hourly results, applies accumulated penalties,
    stores DailyRecord in weekly memory, resets exposure streak to 0.
    """
    results = state.current_day_results
    if not results: return None

    scores            = [h["ECS"] for h in results]
    ECS_mean          = sum(scores) / len(scores)
    penalised_hours   = sum(1 for h in results if h["penalised"])
    bio_confirmed_hrs = sum(1 for h in results if h["bio_confirmed"])
    total_penalty     = sum(h["penalty_pts"] for h in results)
    ECS_daily         = round(min(max(ECS_mean - total_penalty, 0.0), 100.0), 1)

    offender_counts = {}
    for h in results:
        for pol in h["offending"]:
            offender_counts[pol] = offender_counts.get(pol, 0) + 1
    worst = sorted(offender_counts, key=offender_counts.get, reverse=True)[:3]

    comp_avgs = {
        "A": round(sum(h["A"] for h in results) / len(results), 1),
        "B": round(sum(h["B"] for h in results) / len(results), 1),
        "C": round(sum(h["C"] for h in results) / len(results), 1),
        "D": round(sum(h["D"] for h in results) / len(results), 1),
    }

    flags = check_patterns(state, list(state.weekly_memory))

    record = DailyRecord(
        date                = date_str,
        ECS_daily           = ECS_daily,
        ECS_mean            = round(ECS_mean, 1),
        penalised_hours     = penalised_hours,
        bio_confirmed_hours = bio_confirmed_hrs,
        total_penalty       = round(total_penalty, 1),
        worst_pollutants    = worst,
        component_averages  = comp_avgs,
        pattern_flags       = flags,
    )
    state.weekly_memory.append(record)
    state.current_day_results = []
    state.current_day_date    = date_str
    state.exposure_streak     = 0   # ← RESET AT 7AM ONLY
    return record


# ── PATTERN DETECTION ─────────────────────────────────────────

def check_patterns(state, daily_history):
    flags = []
    recent_6h = list(state.history_24h)[-6:]
    if sum(1 for h in recent_6h
           if is_spike_hour(detect_spikes(h.pollutants))) >= 4:
        flags.append({"type": "acute_exposure", "dimension": "Respiratory & Lung"})
    if sum(1 for d in daily_history[-7:] if d.ECS_daily < 65) >= 5:
        flags.append({"type": "sustained_low_quality", "dimension": "Chronic Disease Risk"})
    if len(state.silent_log[-72:]) >= 3:
        flags.append({"type": "unexplained_bio_distress", "dimension": "Investigate Further"})
    return flags


# ── WEEKLY OVERVIEW ───────────────────────────────────────────

def get_weekly_overview(state):
    records = list(state.weekly_memory)
    if not records: return None
    return [{
        "date":              r.date,
        "ECS_daily":         r.ECS_daily,
        "penalised_hours":   r.penalised_hours,
        "bio_confirmed_hrs": r.bio_confirmed_hours,
        "total_penalty":     r.total_penalty,
        "worst_pollutants":  r.worst_pollutants,
        "components":        r.component_averages,
        "flags":             [f["dimension"] for f in r.pattern_flags],
    } for r in records]


# ── ALERT LAYER ───────────────────────────────────────────────
# Separate from ECS score computation. Does not affect any scores.
# Call generate_morning_forecast() at 8am via scheduler.
# Call generate_realtime_alert() each hour after compute_hourly_ECS().

def generate_morning_forecast(date_str, forecast_pollutants, forecast_pollen):
    """
    8am daily forecast. Returns data structure for UI/notification layer.
    Engine is pure measurement — no action suggestions produced here.
    forecast_pollutants: dict of predicted peak values e.g. {"pm25": 28, "no2": 35}
    forecast_pollen:     dict e.g. {"pollen_tree": 60, "pollen_grass": 10, "pollen_weed": 12}
    """
    alerts = []

    for p, val in forecast_pollutants.items():
        if p in WHO_THRESHOLDS and exceeds_who(p, val):
            alerts.append({
                "type":      "pollution_forecast",
                "pollutant": p,
                "predicted": val,
                "who_limit": WHO_THRESHOLDS[p],
                "severity":  "high" if val > WHO_THRESHOLDS[p] * 1.5 else "moderate",
            })

    for p, val in forecast_pollen.items():
        if p in WHO_THRESHOLDS and exceeds_who(p, val):
            alerts.append({
                "type":      "pollen_forecast",
                "pollutant": p,
                "predicted": val,
                "who_limit": WHO_THRESHOLDS[p],
            })

    above_set = set(
        p for p, v in {**forecast_pollutants, **forecast_pollen}.items()
        if p in WHO_THRESHOLDS and exceeds_who(p, v)
    )
    for combo_tuple, insight in COEXPOSURE_INSIGHTS.items():
        if frozenset(combo_tuple).issubset(above_set):
            alerts.append({
                "type":      "co_exposure_forecast",
                "pair":      insight["pair"],
                "dimension": insight["dimension"],
                "warning":   insight["warning"],
            })

    return {
        "date":    date_str,
        "time":    "08:00",
        "alerts":  alerts,
        "summary": f"{len(alerts)} forecast concern(s) for today." if alerts else "Clean air day forecast.",
    }


def generate_realtime_alert(result, reading):
    """
    Per-hour real-time alert. Returns None if nothing to flag.
    Call after compute_hourly_ECS() each hour.
    """
    alerts = []

    for p in result["offending"]:
        alerts.append({
            "type":      "who_breach",
            "pollutant": p,
            "value":     reading.pollutants[p],
            "who_limit": WHO_THRESHOLDS[p],
        })

    for ins in result["co_insights"]:
        alerts.append({
            "type":      "co_exposure_realtime",
            "pair":      ins["pair"],
            "dimension": ins["dimension"],
            "warning":   ins["warning"],
        })

    if result["possible_stress"]:
        alerts.append({
            "type":    "possible_stress",
            "streak":  result["exposure_streak"],
            "message": f"Possible stress — {result['exposure_streak']}h consecutive above-WHO exposure. No penalty yet.",
        })

    if result["penalised"]:
        msg = f"Sustained exposure — {result['exposure_streak']}h streak. -{result['penalty_pts']:.1f}pt penalty this hour."
        if result["bio_confirmed"]:
            msg += " Biomarkers confirm stress — double penalty active."
        alerts.append({
            "type":    "sustained_exposure_penalty",
            "streak":  result["exposure_streak"],
            "penalty": result["penalty_pts"],
            "message": msg,
        })

    if not alerts:
        return None

    return {
        "timestamp": reading.timestamp,
        "hour":      f"{reading.timestamp:02d}:00",
        "alerts":    alerts,
    }


# ── CONSUMER OUTPUT ───────────────────────────────────────────

def ecs_band(score):
    if score >= 90: return "Excellent", "🟢"
    if score >= 75: return "Good",      "🟡"
    if score >= 60: return "Moderate",  "🟠"
    if score >= 45: return "Poor",      "🔴"
    return "Very Poor", "🔴"

def infer_protection(res, reading):
    if res["bio_state"] == "protected":
        return "🛡  Protection active — air is above safe limits but your body shows no stress response. No penalty applied."
    return None

def print_consumer_output(res, reading, state):
    score = res["ECS"]
    band, icon = ecs_band(score)
    offenders = res["offending"]
    sleep_tag = " · Sleep window" if reading.in_sleep_window else ""

    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  {icon}  ECS {score:5.1f} / 100  ·  {band:<10}{sleep_tag:<18}║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    if score >= 90:   meaning = "Air quality is excellent. No action needed."
    elif score >= 75: meaning = "Air quality is good. Minor pollutants present but within safe limits."
    elif score >= 60: meaning = "Air quality is moderate. Sensitive groups may notice effects."
    elif score >= 45: meaning = "Air quality is poor. Limit prolonged outdoor exposure."
    else:             meaning = "Air quality is very poor. Avoid outdoor exposure. Consider a mask."
    print(f"  {meaning}")

    if offenders:
        friendly = {
            "pm25": "Fine particles (PM2.5)", "pm10": "Coarse particles (PM10)",
            "no2":  "Nitrogen dioxide (NO2)", "o3":   "Ozone (O3)",
            "co":   "Carbon monoxide (CO)",   "co2":  "CO2 (indoor air)",
            "temp": "Temperature", "humidity": "Humidity",
            "pollen_tree": "Tree pollen", "pollen_grass": "Grass pollen",
            "pollen_weed": "Weed pollen", "radon": "Radon",
        }
        names = [friendly.get(p, p) for p in offenders[:3]]
        print(f"  ⚠  Above safe limits: {', '.join(names)}")
    else:
        print(f"  ✓  All pollutants within WHO safe limits.")

    streak = res["exposure_streak"]
    if res["possible_stress"]:
        print(f"  ⏱  Possible stress — {streak}h consecutive exposure. No penalty yet.")
    elif res["penalised"]:
        if res["protection_confirmed"]:
            print(f"  🛡  Protection confirmed by user — penalty cancelled for this hour.")
        elif res["confirm_protection"]:
            print(f"  ❓ Biomarkers calm during bad air — did you use protection? (streak {streak}h → -{BIO_PENALTY_BASE:.1f}pt if unconfirmed)")
        elif res["bio_confirmed"]:
            print(f"  🔴 Sustained exposure {streak}h — biomarkers confirm stress → -{res['penalty_pts']:.1f}pt (base + bio double penalty)")
        else:
            print(f"  🔴 Sustained exposure {streak}h → -{res['penalty_pts']:.1f}pt at daily close")

    drivers = []
    if res["A"] < 75:
        drivers.append(f"Air quality this hour: {res['A']:.0f}/100")
    if res["B"] < 60:
        drivers.append(f"Air stability (last 3h): {res['B']:.0f}/100 — readings have been fluctuating")
    if res["C"] < 85:
        drivers.append(f"30-day exposure load: {res['C']:.0f}/100 — sustained exposure building up")
    if reading.in_sleep_window and res["D"] < 75:
        drivers.append(f"Sleep environment: {res['D']:.0f}/100 — bedroom conditions disrupting recovery")
    if drivers:
        print(f"  📊 Contributing factors:")
        for d in drivers: print(f"     · {d}")

    protection = infer_protection(res, reading)
    if protection: print(f"  {protection}")

    if res["co_insights"]:
        print(f"  ⚡ Combination risks active this hour:")
        for ins in res["co_insights"]:
            brief = ins["warning"].split(".")[0]
            print(f"     · {ins['pair']} [{ins['dimension']}] — {brief}.")
    print()


# ── DISPLAY / RUN ─────────────────────────────────────────────

def run_day(label, date, readings, forecast_pollutants=None, forecast_pollen=None):
    state = UserState()
    state.baselines     = {p: WHO_THRESHOLDS[p] for p in WHO_THRESHOLDS
                           if not isinstance(WHO_THRESHOLDS[p], tuple)}
    state.bio_baselines = {"hrv": 55, "hr": 68, "spo2": 98}
    state.current_day_date = date

    print(f"\n{'='*62}")
    print(f"  {label}  [{date}]")
    print(f"{'='*62}")

    # 8am forecast — data only, zero score impact
    if forecast_pollutants or forecast_pollen:
        forecast = generate_morning_forecast(date, forecast_pollutants or {}, forecast_pollen or {})
        print(f"\n  ☀  08:00 DAILY FORECAST — {forecast['summary']}")
        for a in forecast["alerts"]:
            if a["type"] in ("pollution_forecast", "pollen_forecast"):
                print(f"  ⚠  {a['pollutant'].upper()} predicted at {a['predicted']} (WHO limit: {a['who_limit']})")
            elif a["type"] == "co_exposure_forecast":
                print(f"  ⚡ Co-exposure risk forecast: {a['pair']} [{a['dimension']}]")

    for r in readings:
        res = compute_hourly_ECS(r, state)
        above = [f"{p}={r.pollutants[p]}" for p in res["offending"]]
        sleep_str = " 🌙" if r.in_sleep_window else ""
        print(f"\n  Hour {r.timestamp:02d}:00{sleep_str}  [streak={res['exposure_streak']}h]")
        print(f"  {'⚠  Above WHO: ' + ', '.join(above) if above else '✓  All within WHO AQG'}")
        print(f"  ┌──────────────────────────────────────────────┐")
        print(f"  │  ECS = {res['ECS']:5.1f} / 100                         │")
        print(f"  ├──────────────────────────────────────────────┤")
        print(f"  │  A  Instant Quality    {res['A']:5.1f}                │")
        print(f"  │  B  Stability          {res['B']:5.1f}                │")
        print(f"  │  C  Chronic Load       {res['C']:5.1f}                │")
        print(f"  │  D  Recovery Env       {res['D']:5.1f}                │")
        print(f"  └──────────────────────────────────────────────┘")

        if res["possible_stress"]:
            print(f"  ⏱  Possible stress (streak {res['exposure_streak']}h) — warning only")
        elif res["penalised"]:
            if res["bio_confirmed"]:
                print(f"  🔴 Penalised (streak {res['exposure_streak']}h) — -1.5pt base + -1.5pt bio = -{res['penalty_pts']:.1f}pt")
            else:
                print(f"  🔴 Penalised (streak {res['exposure_streak']}h) — base -{res['penalty_pts']:.1f}pt")

        bio = res["bio_state"]
        if bio == "stressed":   print(f"  ⚡ Biomarkers: body confirms stress")
        elif bio == "protected":print(f"  🛡  Biomarkers: body calm during bad air → protection active")
        elif bio == "silent":   print(f"  ⚠  Biomarkers: body stressed during clean air → logged")

        for ins in res["co_insights"]:
            print(f"  ⚠  Co-exposure [{ins['pair']}] — {ins['dimension']}")

        # Real-time alert — data only, zero score impact
        rt = generate_realtime_alert(res, r)
        if rt:
            print(f"  🔔 REAL-TIME ALERT ({rt['hour']}): {len(rt['alerts'])} alert(s) — data only, no score impact")

        print_consumer_output(res, r, state)

    record = close_day(state, date)
    if record:
        print(f"\n  {'─'*50}")
        print(f"  DAY CLOSED AT 7AM — exposure streak reset to 0")
        print(f"  ┌──────────────────────────────────────────────┐")
        print(f"  │  DAILY ECS  =  {record.ECS_daily:5.1f} / 100                  │")
        print(f"  ├──────────────────────────────────────────────┤")
        print(f"  │  Hourly mean         {record.ECS_mean:5.1f}                  │")
        print(f"  │  Penalised hours     {record.penalised_hours:3d}  (streak hour 3+)     │")
        print(f"  │  Bio-confirmed hrs   {record.bio_confirmed_hours:3d}  (double penalty)     │")
        print(f"  │  Total penalty      -{record.total_penalty:4.1f} pts                │")
        print(f"  │  Worst actors   {', '.join(record.worst_pollutants) or 'none':30s}│")
        print(f"  │  Avg A/B/C/D    {record.component_averages['A']}/{record.component_averages['B']}/{record.component_averages['C']}/{record.component_averages['D']}                  │")
        print(f"  └──────────────────────────────────────────────┘")
        if record.pattern_flags:
            for f in record.pattern_flags:
                print(f"  🔔 Pattern flag: {f['dimension']}")

    overview = get_weekly_overview(state)
    if overview:
        print(f"\n  {'─'*50}")
        print(f"  WEEKLY MEMORY  ({len(overview)} day(s) stored)")
        print(f"  {'Date':<14} {'ECS':>6} {'PenHrs':>7} {'BioHrs':>7} {'Penalty':>8}  {'Worst':<20} Flags")
        print(f"  {'─'*78}")
        for d in overview:
            flags = ', '.join(d['flags']) if d['flags'] else '—'
            worst = ', '.join(d['worst_pollutants']) if d['worst_pollutants'] else '—'
            print(f"  {d['date']:<14} {d['ECS_daily']:>6} {d['penalised_hours']:>7} {d['bio_confirmed_hrs']:>7} {d['total_penalty']:>7.1f}  {worst:<20} {flags}")

