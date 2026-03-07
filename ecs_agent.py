"""
ECS Agent — Reasoning & Learning Layer
========================================
Sits alongside ecs_engine.py. Never modifies the engine.

SCOPE (current phase):
  Pure reasoning and learning only.
  No notifications. No UI messages. No action suggestions.
  No protection_confirmed logic.

  The agent's job right now:
    1. Learn the person — personal HRV/HR/SpO2 baselines per activity context
    2. Learn sensitivity — which pollutants this person's body responds to
    3. Understand the body — confounder-aware stress assessment
    4. Surface what it concluded and how confident it is

CALL PATTERN (with live data layer):
    from ecs_data_layer import PollutantAssembler, OpenWeatherMapClient, GooglePollenClient

    assembler = PollutantAssembler(
        owm_client    = OpenWeatherMapClient(api_key="YOUR_OWM_KEY"),
        pollen_client = GooglePollenClient(api_key="YOUR_GOOGLE_KEY"),
    )
    agent = ECSAgent(profile=UserProfile(age_band="31-45"), assembler=assembler)

    # Each hour — no manual HourlyReading needed:
    snapshot = EnvironmentSnapshot(
        timestamp        = 9,
        date             = "2024-03-15",
        location_zone    = LocationZone.HOME,
        gps_lat          = 51.5074,
        gps_lon          = -0.1278,
        activity         = ActivityLevel.REST,
        sleep_stage      = SleepStage.AWAKE,
        indoor_overrides = {
            "pm25":     12.0,  # indoor PM2.5 sensor reading
            "co2":       820,  # manual until CO2 sensor
            "radon":      50,
            "humidity":   50,
            "temp":       21,
        },
    )
    enriched = agent.enrich(None, snapshot)   # None = let agent fetch live data
    result   = compute_hourly_ECS(enriched, engine_state)
    output   = agent.post_process(result, enriched, snapshot)

CALL PATTERN (manual / testing — unchanged):
    snapshot  = EnvironmentSnapshot(...)
    enriched  = agent.enrich(raw_reading, snapshot)   # pass pre-built reading
    result    = compute_hourly_ECS(enriched, engine_state)
    output    = agent.post_process(result, enriched, snapshot)

    # output keys:
    #   learning_signal     dict   is_clean / is_elevated / dominant_pollutant
    #   confounder_score    float  0-1 (0=clean signal, 1=fully confounded)
    #   confounder_reasons  list
    #   stress_inferred     bool
    #   stress_confidence   float
    #   stress_evidence     list
    #   inferences          list   medium/low confidence inferences surfaced
    #   questions           list   PendingQuestion objects
    #   sensitivity_report  dict   current learned weights per pollutant
    #   personalised        bool
    #   days_of_data        int
    #   controllability     float

AT 7AM:
    brief = agent.pre_day_brief(lat, lon)
    agent.daily_close(date, daily_record)

USER ANSWERS:
    agent.record_response(question_id, "yes")

────────────────────────────────────────────────────────────
LEARNING SIGNAL — what the agent uses air quality data for

The engine already scores air quality (component A) and applies
the penalty model. The agent does NOT re-score or re-assess air
quality for any scoring purpose.

The agent uses air quality data for exactly three learning gates:

  1. Baseline gate (is_clean):
       Only update personal HRV/HR/SpO2 baselines from hours where
       air quality is genuinely clean. We do not want polluted-air
       readings teaching the agent what "normal" looks like.
       Gate: engine component A >= 0.90 (air near WHO limits or better)

  2. Sensitivity gate (is_elevated):
       Only accumulate pollutant→biomarker correlation data from hours
       where the pollutant is actually elevated above WHO. There is no
       correlation signal to learn from when exposure is near background.
       Gate: at least one pollutant exceeds WHO threshold in the reading

  3. Attribution context (dominant_pollutant):
       When the agent asks "was the body stressed because of air?",
       it needs to know which pollutant was dominant this hour so it
       can form the right question and accumulate against the right
       sensitivity track.

All three read directly from HourlyReading.pollutants — the same
data the engine already processed. No parallel scoring. No re-weighting.

────────────────────────────────────────────────────────────
BIOMARKER → PENALTY (engine, unchanged)

The engine's penalty model is untouched:
  Streak 1-2h: warning only
  Streak 3+h:  -1.5pt base
  bio_state=stressed: -3.0pt total

protection_confirmed is NOT set by agent. Engine receives False always.
Penalty is driven purely by biomarker confirmation from the engine.
The agent learns sensitivity over time — that personalises the story
of WHY the penalty happened, not whether it applies.
────────────────────────────────────────────────────────────
"""

import sqlite3
import json
import math
import time
import threading
import urllib.request
import urllib.parse
import datetime
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class ActivityLevel(Enum):
    REST    = "rest"
    LIGHT   = "light"
    INTENSE = "intense"
    UNKNOWN = "unknown"

class SleepStage(Enum):
    AWAKE   = "awake"
    LIGHT   = "light_sleep"
    DEEP    = "deep_sleep"
    REM     = "rem"
    UNKNOWN = "unknown"

class LocationZone(Enum):
    HOME    = "home"
    WORK    = "work"
    COMMUTE = "commute"
    OUTDOOR = "outdoor"
    UNKNOWN = "unknown"

class ConfidenceBand(Enum):
    HIGH   = "high"    # >= 0.80 → act silently, log only
    MEDIUM = "medium"  # 0.50–0.79 → log + flag in output
    LOW    = "low"     # < 0.50  → queue question for user


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CONF_HIGH   = 0.80
CONF_MEDIUM = 0.50

MIN_DAYS_PERSONAL_BASELINE = 14   # days before personal baselines replace priors
MIN_OBS_SENSITIVITY        = 20   # observations before weight shifts

EMA_ALPHA_RESTING    = 0.05   # REST + clean air — very slow (stable reference)
EMA_ALPHA_CONTEXTUAL = 0.10   # any other activity match — faster

OUTDOOR_CONTROLLABILITY = 0.70

# Activity multipliers for stress thresholds
ACTIVITY_HR_MULT: Dict[ActivityLevel, float] = {
    ActivityLevel.REST:    1.00,
    ActivityLevel.LIGHT:   1.15,
    ActivityLevel.INTENSE: 1.40,
    ActivityLevel.UNKNOWN: 1.00,
}
ACTIVITY_HRV_MULT: Dict[ActivityLevel, float] = {
    ActivityLevel.REST:    1.00,
    ActivityLevel.LIGHT:   0.90,
    ActivityLevel.INTENSE: 0.70,
    ActivityLevel.UNKNOWN: 1.00,
}

# Sleep stage HRV suppression multipliers
# Deep sleep HRV is ~30% lower than waking HRV — this is physiology, not stress
SLEEP_HRV_MULT: Dict[SleepStage, float] = {
    SleepStage.AWAKE:   1.00,
    SleepStage.LIGHT:   0.88,
    SleepStage.DEEP:    0.70,
    SleepStage.REM:     0.82,
    SleepStage.UNKNOWN: 1.00,
}

# Population priors by age band — used on day 1 before personal data exists
POPULATION_PRIORS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "18-30": {"hrv": (65.0, 18.0), "hr": (62.0, 8.0),  "spo2": (98.2, 0.8)},
    "31-45": {"hrv": (52.0, 16.0), "hr": (66.0, 9.0),  "spo2": (98.0, 0.9)},
    "46-60": {"hrv": (40.0, 14.0), "hr": (68.0, 10.0), "spo2": (97.8, 1.0)},
    "61+":   {"hrv": (30.0, 12.0), "hr": (70.0, 10.0), "spo2": (97.5, 1.1)},
}
ACTIVITY_PRIORS: Dict[str, Dict] = {
    "sedentary": {"hrv_factor": 1.00, "hr_delta": 0},
    "moderate":  {"hrv_factor": 0.85, "hr_delta": 10},
    "active":    {"hrv_factor": 0.70, "hr_delta": 20},
}

# DALY weights — from engine, used for graded AQ scoring + sensitivity profiling
DALY_WEIGHTS: Dict[str, float] = {
    "pm25":         0.6626,
    "pm10":         0.0992,
    "co2":          0.0112,
    "co":           0.0196,
    "no2":          0.1200,
    "o3":           0.0230,
    "temp":         0.0504,
    "humidity":     0.0056,
    "pollen_tree":  0.0056,
    "pollen_grass": 0.0018,
    "pollen_weed":  0.0010,
    "radon":        0.0000,  # no DALY weight but included in assessment
}

# WHO thresholds — used for normalising each pollutant's exceedance
# (same as engine — duplicated here so agent has no engine import at module level)
WHO_THRESHOLDS: Dict = {
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

# Sensitivity profiling
SENSITIVITY_POLLUTANTS = ["pm25", "no2", "o3", "co", "pollen_tree", "pollen_grass"]
# Primary biomarker for each pollutant — which bio responds most to this pollutant
SENSITIVITY_BIO_MAP: Dict[str, str] = {
    "pm25":         "hrv",   # PM2.5 suppresses HRV (best documented)
    "no2":          "hr",    # NO2 elevates HR
    "o3":           "spo2",  # O3 reduces SpO2
    "co":           "hr",    # CO elevates HR (displaces O2)
    "pollen_tree":  "hr",    # Pollen elevates HR (histamine/allergic response)
    "pollen_grass": "hr",
}
# Pearson r → weight multiplier
SENS_MULT_BANDS = [(0.10, 0.80), (0.30, 1.00), (0.50, 1.20), (1.00, 1.40)]
MAX_WEIGHT_DEVIATION = 0.40

# Location clustering
CLUSTER_RADIUS_M   = 150
MIN_CLUSTER_VISITS = 5

# Morning dip — HRV naturally ~10% lower in these hours regardless of air quality
MORNING_DIP_HOURS = {6, 7, 8}

TOOL_TIMEOUT_S      = 6
CLOUD_SYNC_ENDPOINT = "https://api.ecs-platform.com/v1/sync/daily"


# ══════════════════════════════════════════════════════════════════════════════
# I18N STUB
# ══════════════════════════════════════════════════════════════════════════════

_TRANSLATIONS: Dict[str, str] = {}

_ENGLISH: Dict[str, str] = {
    "q_bio_calm_bad_air":
        "Your body seemed calm during {hours}h of poor air (score {aq:.2f}). "
        "Were you in a filtered or ventilated space?",
    "q_stress_clean_air":
        "Your biomarkers showed stress even though air quality was good. "
        "Were you exercising, under stress, or unwell?",
    "q_sensitivity_spike":
        "You seemed more sensitive to {pollutant} than usual today. "
        "Did you notice any symptoms (headache, fatigue, eye irritation)?",
    "q_confounder_check":
        "Your HRV was lower than expected during {stage} — "
        "was this a particularly disrupted night?",
}

def _t(key: str, **kw) -> str:
    tmpl = _TRANSLATIONS.get(key, _ENGLISH.get(key, key))
    return tmpl.format(**kw) if kw else tmpl


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    age_band:          str       = "31-45"    # "18-30"|"31-45"|"46-60"|"61+"
    activity_level:    str       = "moderate" # "sedentary"|"moderate"|"active"
    health_conditions: List[str] = field(default_factory=list)

@dataclass
class EnvironmentSnapshot:
    """
    Everything the app knows about the user's situation this hour.

    Pollutant fields here are IN ADDITION to HourlyReading.pollutants.
    HourlyReading.pollutants is the primary reading (outdoor or primary sensor).
    EnvironmentSnapshot adds:
      - indoor_pm25   : separate indoor sensor reading
      - indoor_co2    : indoor CO2 (separate from primary reading)
      - indoor_co2_prev: previous hour's indoor CO2 (for ventilation trend)
      - outdoor_pm25  : confirmed outdoor station reading (if separate from primary)

    The LearningSignalClassifier merges all available signals into one graded score.
    Missing values are simply omitted from the weighted average — no imputation.
    """
    timestamp:       int
    date:            str
    location_zone:   LocationZone  = LocationZone.UNKNOWN
    gps_lat:         Optional[float] = None
    gps_lon:         Optional[float] = None
    is_outdoor:      bool            = False
    activity:        ActivityLevel   = ActivityLevel.UNKNOWN
    sleep_stage:     SleepStage      = SleepStage.AWAKE
    # Additional sensor readings beyond HourlyReading.pollutants
    indoor_pm25:     Optional[float] = None
    indoor_co2:      Optional[float] = None
    indoor_co2_prev: Optional[float] = None
    outdoor_pm25:    Optional[float] = None
    # Passed to PollutantAssembler when agent fetches live data.
    # Keys: pm25 (sensor reading), co2, radon, humidity, temp.
    # Only used when agent.enrich(None, snapshot) is called.
    indoor_overrides: Optional[Dict] = None

@dataclass
class LearningSignal:
    """
    Output of LearningSignalClassifier for one hour.
    Used only for gating the agent learning — not for scoring.

    is_clean:           True when engine component A >= CLEAN_A_THRESHOLD
                        → safe to update personal baselines this hour
    is_elevated:        True when any pollutant exceeds WHO in the reading
                        → worth accumulating sensitivity correlation data
    dominant_pollutant: pollutant with highest DALY-weighted exceedance
                        → used for attribution and sensitivity tracking
    exceeding:          list of pollutants above WHO this hour
    """
    is_clean:           bool
    is_elevated:        bool
    dominant_pollutant: Optional[str]
    exceeding:          List[str]

@dataclass
class InferenceRecord:
    timestamp:       int
    inference_type:  str    # "stress"|"sensitivity"|"confounder"|"air_quality"
    conclusion:      str
    confidence:      float
    evidence:        List[str]
    confidence_band: str = ""
    question_asked:  str = ""
    user_response:   Optional[str] = None

@dataclass
class PendingQuestion:
    id:             str
    timestamp:      int
    question_key:   str
    question_text:  str
    inference_type: str
    context:        dict
    priority:       int = 1

@dataclass
class AgentMemory:
    # Baselines per activity context: {activity_str: {metric: ema_value}}
    baselines:           Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline_obs_count:  Dict[str, int]              = field(default_factory=dict)
    # CO2 trend for ventilation detection
    co2_history:         deque = field(default_factory=lambda: deque(maxlen=6))
    # Sensitivity accumulators: {pollutant: {sum_x, sum_y, sum_xy, sum_x2, sum_y2, n}}
    sensitivity_acc:     Dict[str, dict] = field(default_factory=dict)
    # Computed personal weights (after MIN_OBS_SENSITIVITY obs per pollutant)
    personal_weights:    Dict[str, float] = field(default_factory=dict)
    # Today's state
    pending_questions:   List[PendingQuestion]  = field(default_factory=list)
    today_inferences:    List[InferenceRecord]  = field(default_factory=list)
    # Tracking
    days_of_data:        int         = 0
    profile:             UserProfile = field(default_factory=UserProfile)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — LEARNING SIGNAL CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class LearningSignalClassifier:
    """
    Classifies each hour's reading into learning signals for the agent.
    Reads from HourlyReading.pollutants — same data the engine already processed.
    Does NOT re-score air quality. Does NOT affect engine output.

    Answers three questions:

      1. is_clean (baseline gate)
           Is air quality clean enough this hour to update personal baselines?
           Uses engine component A from result — the engine already computed this.
           Gate: component A >= CLEAN_A_THRESHOLD (0.90 = near-WHO or better)
           Why: We only want to learn what "normal" HRV/HR looks like from
           hours where the body was genuinely not under air quality stress.

      2. is_elevated (sensitivity gate)
           Is at least one pollutant above WHO this hour?
           Gate: any pollutant in the reading exceeds its WHO threshold.
           Why: No correlation signal to learn from when exposure is background.
           Only accumulate sensitivity data when there was real exposure.

      3. dominant_pollutant (attribution context)
           Which pollutant had the highest DALY-weighted exceedance this hour?
           Used to form the right question ("was your HRV affected by PM2.5?")
           and to route correlation data to the right sensitivity track.
    """

    # Threshold for engine component A to count as "clean"
    # Engine returns component A on 0-100 scale (100 = all pollutants at WHO limit or below)
    # 85.0 means air quality is 85/100 — clean enough to learn personal baselines from
    CLEAN_A_THRESHOLD = 85.0

    def classify(self,
                 pollutants: dict,
                 engine_component_a: float) -> LearningSignal:
        """
        Returns LearningSignal from pollutant dict + engine's component A.
        Called in post_process() after engine result is available.

        engine_component_a: result["A"] from compute_hourly_ECS()
        """
        exceeding: List[str] = []
        dominant:  Optional[str] = None
        best_weighted_exc = 0.0

        for pollutant, threshold in WHO_THRESHOLDS.items():
            value = pollutants.get(pollutant)
            if value is None:
                continue
            exc = self._exceedance(value, threshold)
            if exc <= 0.0:
                continue
            exceeding.append(pollutant)
            # Track dominant by DALY-weighted exceedance
            w_exc = DALY_WEIGHTS.get(pollutant, 0.0) * exc
            if w_exc > best_weighted_exc:
                best_weighted_exc = w_exc
                dominant = pollutant

        return LearningSignal(
            is_clean           = engine_component_a >= self.CLEAN_A_THRESHOLD,
            is_elevated        = len(exceeding) > 0,
            dominant_pollutant = dominant,
            exceeding          = exceeding,
        )

    @staticmethod
    def _exceedance(value: float, threshold) -> float:
        """Returns exceedance ratio. 0.0 if at or below WHO. 1.0 at 2x WHO."""
        if isinstance(threshold, tuple):
            lo, hi = threshold
            if lo <= value <= hi:
                return 0.0
            return min(min(abs(value - lo), abs(value - hi)) / ((hi - lo) / 2), 1.0)
        else:
            if value <= threshold:
                return 0.0
            return min((value - threshold) / threshold, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — CONFOUNDER DETECTOR
# ══════════════════════════════════════════════════════════════════════════════# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — CONFOUNDER DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class ConfounderDetector:
    """
    Determines whether a biomarker deviation is likely caused by something
    other than air quality — before attributing it to air.

    Why this matters:
      Without confounder detection, the sensitivity profiler would accumulate
      correlations like: "PM2.5 was 35 AND HRV was 38ms → PM2.5 causes HRV drop"
      — but if the user was in deep sleep at the time, the HRV drop is
      physiological, not air-related. After 30 days this would produce a
      falsely inflated PM2.5 sensitivity weight.

    Confounders checked:
      1. Sleep stage — HRV suppression is expected and quantified per stage
      2. Activity level — intense exercise dominates HR/HRV signal
      3. Morning dip — HRV naturally ~10% lower at 06:00–08:00
      4. Light activity — minor bio changes expected, partial confounding

    confounder_score (0.0–1.0):
      0.0 = no confounders — bio change is unexplained, may be air-related
      1.0 = strong confounders — do not attribute bio change to air

    Downstream use:
      confounder_score >= 0.80 → skip stress attribution to air
      confounder_score >= 0.60 → skip sensitivity accumulation
      confounder_score >= 0.40 → skip baseline update (signal too noisy)
    """

    def assess(self,
               biomarkers:  Optional[dict],
               snapshot:    "EnvironmentSnapshot",
               baseline:    Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Returns (confounder_score: float, reasons: List[str]).
        """
        if not biomarkers:
            return 0.0, ["no_biomarkers_available"]

        score   = 0.0
        reasons = []

        # 1. Sleep stage — each stage has a known HRV suppression magnitude
        stage = snapshot.sleep_stage
        if stage == SleepStage.DEEP:
            # Deep sleep: HRV 30% lower than waking — fully confounded
            score = max(score, 0.85)
            reasons.append(
                "deep_sleep_hrv_suppression_~30pct_expected_not_air"
            )
        elif stage == SleepStage.REM:
            # REM: HRV 18% lower — strongly confounded
            score = max(score, 0.65)
            reasons.append(
                "rem_sleep_hrv_suppression_~18pct_expected_not_air"
            )
        elif stage == SleepStage.LIGHT:
            # Light sleep: HRV 12% lower — moderately confounded
            score = max(score, 0.45)
            reasons.append(
                "light_sleep_hrv_suppression_~12pct_expected_not_air"
            )

        # 2. Activity
        if snapshot.activity == ActivityLevel.INTENSE:
            # Intense exercise completely dominates HR and HRV
            score = max(score, 0.90)
            reasons.append(
                "intense_activity_hr_hrv_changes_physiology_not_air"
            )
        elif snapshot.activity == ActivityLevel.LIGHT:
            # Light activity: minor partial confounding
            score = max(score, 0.30)
            reasons.append(
                "light_activity_minor_hr_hrv_changes_expected"
            )

        # 3. Morning dip — HRV naturally ~10% lower at 06:00–08:00
        if snapshot.timestamp in MORNING_DIP_HOURS and stage == SleepStage.AWAKE:
            hrv   = biomarkers.get("hrv")
            b_hrv = baseline.get("hrv")
            if hrv is not None and b_hrv is not None:
                expected_morning_hrv = b_hrv * 0.90
                if hrv <= expected_morning_hrv * 1.05:
                    # HRV within range of expected morning dip
                    score = max(score, 0.40)
                    reasons.append(
                        f"morning_dip_hour={snapshot.timestamp}"
                        f"_hrv={hrv:.0f}_expected_morning_low=~{expected_morning_hrv:.0f}"
                    )

        if not reasons:
            reasons = ["no_confounders_detected_signal_attributable_to_air"]

        return round(min(score, 1.0), 3), reasons

    def is_attributable(self, score: float, threshold: float = 0.60) -> bool:
        """True if bio signal is clean enough to attribute to air quality."""
        return score < threshold


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — BASELINE PERSONALISER
# ══════════════════════════════════════════════════════════════════════════════

class BaselinePersonaliser:
    """
    Maintains personal HRV/HR/SpO2 baselines per activity context.

    EMA (Exponential Moving Average):
      new_baseline = old_baseline × (1 - alpha) + new_observation × alpha

      alpha = 0.05 (RESTING): REST + clean air
        5% weight to new reading. Takes ~14 obs to shift halfway to new true value.
        Very stable — resists noise, illness, bad days.

      alpha = 0.10 (CONTEXTUAL): any other activity match
        10% weight. Adapts faster to genuine fitness/health changes.

    Update gating — three conditions must ALL be met:
      1. confounder_score < 0.40  (signal clean enough to trust)
      2. air quality is clean     (only learn "normal" from clean-air hours)
      3. biomarker is available   (obviously)

    This prevents:
      · Polluted-air hours training the "normal" baseline lower
      · Sleep-confounded hours shifting the resting HRV baseline
      · Exercise hours inflating the resting HR baseline
    """

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self._seed_from_prior()

    def _seed_from_prior(self):
        p  = self.memory.profile
        pr = POPULATION_PRIORS.get(p.age_band, POPULATION_PRIORS["31-45"])
        am = ACTIVITY_PRIORS.get(p.activity_level, ACTIVITY_PRIORS["moderate"])
        for lvl in ActivityLevel:
            k = lvl.value
            if k not in self.memory.baselines:
                self.memory.baselines[k] = {
                    "hrv":  pr["hrv"][0]  * am["hrv_factor"],
                    "hr":   pr["hr"][0]   + am["hr_delta"],
                    "spo2": pr["spo2"][0],
                }
                self.memory.baseline_obs_count[k] = 0

    def get(self, activity: ActivityLevel) -> Dict[str, float]:
        self._seed_from_prior()
        return self.memory.baselines.get(
            activity.value,
            self.memory.baselines.get(
                "unknown", {"hrv": 52.0, "hr": 66.0, "spo2": 98.0}
            )
        )

    def update(self,
               biomarkers:       Optional[dict],
               activity:         ActivityLevel,
               signal:           LearningSignal,
               confounder_score: float):
        """
        Update EMA baseline.
        Gates: clean signal (low confounder) + engine says air was clean.
        """
        if not biomarkers:
            return
        if confounder_score >= 0.40:
            return   # signal too noisy — skip
        if not signal.is_clean:
            return   # engine component A says air was not clean this hour

        k     = activity.value
        alpha = (EMA_ALPHA_RESTING
                 if activity == ActivityLevel.REST
                 else EMA_ALPHA_CONTEXTUAL)

        cur = self.memory.baselines.get(k, {})
        for m in ("hrv", "hr", "spo2"):
            if m in biomarkers and biomarkers[m] is not None:
                old    = cur.get(m, biomarkers[m])
                cur[m] = old * (1 - alpha) + biomarkers[m] * alpha

        self.memory.baselines[k] = cur
        self.memory.baseline_obs_count[k] = \
            self.memory.baseline_obs_count.get(k, 0) + 1

    def is_personalised(self) -> bool:
        return self.memory.days_of_data >= MIN_DAYS_PERSONAL_BASELINE

    def assess_stress(self,
                      biomarkers:       Optional[dict],
                      activity:         ActivityLevel,
                      sleep_stage:      SleepStage,
                      confounder_score: float) -> Tuple[bool, float, List[str]]:
        """
        Assess whether biomarkers indicate physiological stress,
        adjusting for activity AND sleep stage.

        Threshold logic:
          HRV stress:
            threshold = baseline_hrv
                        × activity_hrv_mult  (0.70–1.00, reduces threshold for exercise)
                        × sleep_hrv_mult     (0.70–1.00, reduces threshold during sleep)
                        × 0.85               (stress = below 85% of adjusted baseline)

            A HRV of 38ms during deep sleep has a threshold of:
              baseline=62 × 0.70 × 0.85 = 36.9ms
              → 38ms is ABOVE threshold → NOT stressed → correct

            A HRV of 38ms while REST+AWAKE has a threshold of:
              baseline=62 × 1.00 × 0.85 = 52.7ms
              → 38ms is BELOW threshold → stressed → correct

          HR stress:
            threshold = baseline_hr × activity_hr_mult × 1.15
            (stress = above 115% of activity-adjusted baseline)

          SpO2:
            Hard threshold: spo2 < 95.0 regardless of activity/sleep

        Confidence is penalised:
          · 25% if not yet personalised (using population prior)
          · Proportional to confounder_score (confounded signal = lower confidence)

        Returns (stressed: bool, confidence: float, evidence: List[str]).
        """
        if not biomarkers:
            return False, 0.0, ["no_wearable_data"]
        if confounder_score >= 0.80:
            return False, 0.0, [
                f"signal_fully_confounded_score={confounder_score:.2f}"
                "_cannot_attribute_to_air"
            ]

        baseline    = self.get(activity)
        hrv_a_mult  = ACTIVITY_HRV_MULT.get(activity, 1.0)
        hr_a_mult   = ACTIVITY_HR_MULT.get(activity,  1.0)
        hrv_s_mult  = SLEEP_HRV_MULT.get(sleep_stage, 1.0)

        stressed   = False
        confidence = 0.0
        evidence   = []

        hrv = biomarkers.get("hrv")
        if hrv is not None:
            hrv_thresh = baseline["hrv"] * hrv_a_mult * hrv_s_mult * 0.85
            if hrv < hrv_thresh:
                stressed   = True
                drop_ratio = (hrv_thresh - hrv) / hrv_thresh
                confidence = max(confidence, min(drop_ratio * 2.0, 1.0))
                evidence.append(
                    f"hrv={hrv:.0f}ms < threshold={hrv_thresh:.0f}ms"
                    f" [baseline={baseline['hrv']:.0f}"
                    f" × act_mult={hrv_a_mult:.2f}"
                    f" × sleep_mult={hrv_s_mult:.2f}"
                    f" × 0.85]"
                )

        hr = biomarkers.get("hr")
        if hr is not None:
            hr_thresh = baseline["hr"] * hr_a_mult * 1.15
            if hr > hr_thresh:
                stressed   = True
                rise_ratio = (hr - hr_thresh) / hr_thresh
                confidence = max(confidence, min(rise_ratio * 2.0, 1.0))
                evidence.append(
                    f"hr={hr:.0f}bpm > threshold={hr_thresh:.0f}bpm"
                    f" [baseline={baseline['hr']:.0f}"
                    f" × act_mult={hr_a_mult:.2f}"
                    f" × 1.15]"
                )

        spo2 = biomarkers.get("spo2")
        if spo2 is not None and spo2 < 95.0:
            stressed   = True
            confidence = max(confidence, min((95.0 - spo2) / 5.0, 1.0))
            evidence.append(f"spo2={spo2:.1f}% below hard threshold 95.0%")

        if not stressed:
            confidence = 0.0
            evidence   = evidence or ["biomarkers_within_adjusted_thresholds"]

        # Penalise confidence if still on population prior
        if not self.is_personalised():
            confidence *= 0.75
            evidence.append(
                f"confidence_penalised_25pct_population_prior_"
                f"days={self.memory.days_of_data}/{MIN_DAYS_PERSONAL_BASELINE}"
            )

        # Scale confidence down by confounder presence
        # (partial confounders reduce confidence proportionally)
        if confounder_score > 0:
            confidence *= (1.0 - confounder_score * 0.5)
            evidence.append(
                f"confidence_reduced_by_confounder_score={confounder_score:.2f}"
            )

        return stressed, round(confidence, 3), evidence


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SENSITIVITY PROFILER
# ══════════════════════════════════════════════════════════════════════════════

class SensitivityProfiler:
    """
    Learns personal pollutant sensitivity via incremental Pearson correlation.

    Accumulation gates (both must pass):
      1. confounder_score < 0.60 — bio signal is plausibly air-related
         (exercise, deep sleep confounders excluded)
      2. signal.is_elevated (any pollutant above WHO threshold)
         — only learn from hours where the pollutant was actually elevated
         (no correlation signal when pollutant is near background level)

    Why incremental Pearson (not stored raw data):
      Instead of storing thousands of (pollutant_value, hrv_value) pairs,
      we maintain 6 accumulators per pollutant:
        sum_x, sum_y, sum_xy, sum_x2, sum_y2, n

      Pearson r = (n·Σxy - Σx·Σy) / √[(n·Σx² - (Σx)²)(n·Σy² - (Σy)²)]

      This gives exact Pearson r at any time, using O(1) memory per pollutant.
      No raw data stored after each hour.

    Minimum observations:
      20 qualifying observations before any weight shift.
      Below this: DALY weights only (no personal adjustment).

    Weight adjustment:
      personal_weight = DALY_weight × sensitivity_multiplier(r)
      Capped at ±40% from DALY baseline.
      Renormalised to sum=1.0 after all adjustments.

    Pollutant → biomarker pairing:
      PM2.5  → HRV (best documented in epidemiology)
      NO2    → HR
      O3     → SpO2
      CO     → HR (CO displaces O2, elevates HR)
      Pollen → HR (histamine/allergic response)
    """

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        for p in SENSITIVITY_POLLUTANTS:
            if p not in self.memory.sensitivity_acc:
                self.memory.sensitivity_acc[p] = {
                    "sum_x":  0.0,
                    "sum_y":  0.0,
                    "sum_xy": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                    "n":      0,
                }

    def update(self,
               pollutants:       dict,
               biomarkers:       Optional[dict],
               signal:    LearningSignal,
               confounder_score: float):
        """
        Accumulate one observation.
        Both gates must pass: low confounders + elevated air quality.
        """
        if not biomarkers:
            return
        if confounder_score >= 0.60:
            return   # bio signal dominated by non-air confounders
        if not signal.is_elevated:
            return   # no meaningful exposure to correlate against

        for p in SENSITIVITY_POLLUTANTS:
            # Only accumulate when this specific pollutant was above WHO
            if p not in signal.exceeding:
                continue
            p_val = pollutants.get(p)
            if p_val is None:
                continue
            bio_key = SENSITIVITY_BIO_MAP.get(p, "hrv")
            bio_val = biomarkers.get(bio_key)
            if bio_val is None:
                continue
            a = self.memory.sensitivity_acc[p]
            a["sum_x"]  += p_val
            a["sum_y"]  += bio_val
            a["sum_xy"] += p_val * bio_val
            a["sum_x2"] += p_val ** 2
            a["sum_y2"] += bio_val ** 2
            a["n"]      += 1

    def pearson_r(self, pollutant: str) -> Optional[float]:
        """
        Compute Pearson r from accumulators.
        Returns None if fewer than MIN_OBS_SENSITIVITY observations.
        Returns abs(r) — we care about magnitude, not direction
        (PM2.5 up → HRV down is a negative correlation but high sensitivity).
        """
        a = self.memory.sensitivity_acc.get(pollutant)
        if not a or a["n"] < MIN_OBS_SENSITIVITY:
            return None
        n   = a["n"]
        num = n * a["sum_xy"] - a["sum_x"] * a["sum_y"]
        d1  = math.sqrt(max(0.0, n * a["sum_x2"] - a["sum_x"] ** 2))
        d2  = math.sqrt(max(0.0, n * a["sum_y2"] - a["sum_y"] ** 2))
        den = d1 * d2
        return round(abs(num / den), 4) if den > 1e-10 else 0.0

    def compute_weights(self) -> Dict[str, float]:
        """
        Recompute personal DALY weight adjustments.
        Called at daily_close() after new data has been accumulated.
        Returns normalised weight dict — updates memory.personal_weights.
        """
        weights  = dict(DALY_WEIGHTS)
        adjusted = False

        for p in SENSITIVITY_POLLUTANTS:
            r = self.pearson_r(p)
            if r is None:
                continue   # insufficient data — keep DALY weight

            # Find multiplier band
            mult = next(m for t, m in SENS_MULT_BANDS if r < t)
            daly = DALY_WEIGHTS.get(p, 0.0)
            lo   = daly * (1 - MAX_WEIGHT_DEVIATION)
            hi   = daly * (1 + MAX_WEIGHT_DEVIATION)
            weights[p] = round(max(lo, min(hi, daly * mult)), 6)
            adjusted   = True

        if adjusted:
            total = sum(weights.values())
            if total > 0:
                weights = {k: round(v / total, 6) for k, v in weights.items()}

        self.memory.personal_weights = weights
        return weights

    def report(self) -> dict:
        """
        Full sensitivity state for debugging and status output.
        Shows which pollutants have been personalised vs still using DALY prior.
        """
        out = {}
        for p in SENSITIVITY_POLLUTANTS:
            r    = self.pearson_r(p)
            n    = self.memory.sensitivity_acc.get(p, {}).get("n", 0)
            daly = DALY_WEIGHTS.get(p, 0.0)
            pers = self.memory.personal_weights.get(p, daly)
            delta_pct = round((pers - daly) / daly * 100, 1) if daly else 0
            out[p] = {
                "pearson_r":       r,
                "observations":    n,
                "sufficient_data": n >= MIN_OBS_SENSITIVITY,
                "daly_weight":     daly,
                "personal_weight": pers,
                "delta_pct":       delta_pct,
                "bio_signal_used": SENSITIVITY_BIO_MAP.get(p, "hrv"),
                "status": (
                    f"personalised (r={r:.3f}, n={n})"
                    if r is not None
                    else f"accumulating ({n}/{MIN_OBS_SENSITIVITY} obs needed)"
                ),
            }
        return out


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — UNCERTAINTY MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyManager:
    """
    Routes each inference based on confidence band.

    HIGH (>= 0.80):  Act silently. Log inference. Nothing returned to app.
    MEDIUM (0.50–0.79): Act. Include inference in output with evidence.
    LOW (< 0.50):    Queue a question. Include in questions output.

    No cap on questions — asks whenever genuinely uncertain.
    Deduplicates same inference_type within one day so the same
    question isn't queued multiple times.
    """

    def __init__(self, memory: AgentMemory):
        self.memory       = memory
        self._asked_today: set = set()

    def band(self, confidence: float) -> ConfidenceBand:
        if confidence >= CONF_HIGH:   return ConfidenceBand.HIGH
        if confidence >= CONF_MEDIUM: return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW

    def handle(self,
               inf:      InferenceRecord,
               snapshot: "EnvironmentSnapshot") -> Optional[dict]:
        """
        Routes inference. Returns output dict or None (if silent).
        """
        b = self.band(inf.confidence)
        inf.confidence_band = b.value
        self.memory.today_inferences.append(inf)

        if b == ConfidenceBand.HIGH:
            return None   # silent — logged but not surfaced

        out = {
            "type":       inf.inference_type,
            "conclusion": inf.conclusion,
            "confidence": inf.confidence,
            "band":       b.value,
            "evidence":   inf.evidence,
            "question":   None,
        }

        if b == ConfidenceBand.LOW:
            if inf.inference_type not in self._asked_today:
                q = self._build_question(inf, snapshot)
                if q:
                    self.memory.pending_questions.append(q)
                    inf.question_asked = q.question_text
                    out["question"]    = q.question_text
                    self._asked_today.add(inf.inference_type)

        return out

    def _build_question(self,
                         inf:      InferenceRecord,
                         snapshot: "EnvironmentSnapshot") -> Optional[PendingQuestion]:
        key_map = {
            "stress":      "q_bio_calm_bad_air",
            "stress_clean":"q_stress_clean_air",
            "sensitivity": "q_sensitivity_spike",
            "confounder":  "q_confounder_check",
        }
        key  = key_map.get(inf.inference_type, "q_bio_calm_bad_air")
        pols = [e.split("=")[0] for e in inf.evidence if "=" in e]
        text = _t(
            key,
            hours    = 1,
            aq       = getattr(snapshot, "_aq_score", 0.0),
            pollutant= pols[0] if pols else "pm25",
            stage    = snapshot.sleep_stage.value,
        )
        return PendingQuestion(
            id             = _uuid.uuid4().hex[:8],
            timestamp      = snapshot.timestamp,
            question_key   = key,
            question_text  = text,
            inference_type = inf.inference_type,
            context        = {"pollutant": pols[0] if pols else "pm25"},
            priority       = 2 if "stress" in inf.inference_type else 1,
        )

    def get_pending(self) -> List[PendingQuestion]:
        return sorted(
            self.memory.pending_questions,
            key=lambda q: q.priority, reverse=True
        )

    def record_response(self,
                         qid:      str,
                         response: str,
                         pm:       "PatternMemory"):
        q = next((x for x in self.memory.pending_questions if x.id == qid), None)
        if q:
            pm.save_question(q, response)
            self.memory.pending_questions.remove(q)

    def reset_daily(self):
        self._asked_today = set()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — LOCATION INFERENCER
# ══════════════════════════════════════════════════════════════════════════════

class LocationInferencer:
    """
    GPS-based personal location map with time-of-day fallback.
    Learns named locations from repeated GPS visits.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS known_locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, zone TEXT NOT NULL,
        lat REAL NOT NULL, lon REAL NOT NULL,
        radius_m REAL DEFAULT 150, visit_count INTEGER DEFAULT 1,
        first_seen TEXT, last_seen TEXT
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._known: list = []
        with sqlite3.connect(db_path) as c:
            c.executescript(self.SCHEMA)
        self._load()

    def _load(self):
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute(
                "SELECT name, zone, lat, lon, radius_m, visit_count "
                "FROM known_locations"
            ).fetchall()
        self._known = [
            {"name": r[0], "zone": r[1], "lat": r[2],
             "lon": r[3], "radius_m": r[4], "visits": r[5]}
            for r in rows
        ]

    def register(self, name: str, zone: LocationZone, lat: float, lon: float):
        now = datetime.datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as c:
            c.execute("DELETE FROM known_locations WHERE name=?", (name,))
            c.execute(
                "INSERT INTO known_locations "
                "(name, zone, lat, lon, radius_m, visit_count, first_seen, last_seen)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (name, zone.value, lat, lon, CLUSTER_RADIUS_M, 1, now, now)
            )
        self._load()

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R  = 6_371_000
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    def infer(self,
              lat:     Optional[float],
              lon:     Optional[float],
              hour:    int,
              weekday: int) -> Tuple[LocationZone, Optional[str], float]:
        """
        Returns (zone, location_name, confidence).
        GPS match takes priority; time-of-day pattern is fallback.
        """
        if lat is not None and lon is not None:
            for loc in self._known:
                dist = self._haversine(lat, lon, loc["lat"], loc["lon"])
                if dist <= loc["radius_m"]:
                    conf = min(0.95, 0.70 + loc["visits"] / 100)
                    return LocationZone(loc["zone"]), loc["name"], round(conf, 3)

        # Time-of-day fallback
        wd = weekday < 5
        if 23 <= hour or hour < 7:
            return LocationZone.HOME,    None, 0.65
        if wd and 9 <= hour <= 17:
            return LocationZone.WORK,    None, 0.55
        if wd and hour in (7, 8, 18, 19):
            return LocationZone.COMMUTE, None, 0.50
        return LocationZone.UNKNOWN, None, 0.30


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — PATTERN MEMORY (SQLite)
# ══════════════════════════════════════════════════════════════════════════════

class PatternMemory:
    """
    SQLite persistence. Two tiers:
      Hot  — AgentMemory (in-process, updated each hour)
      Cold — SQLite (persisted after each hour, queried for trends)

    Stores the agent's full reasoning per hour:
      air_quality_score, confounder_score, stress inference,
      sensitivity accumulator state, baseline snapshots.

    Never stores raw pollutant or biomarker values for cloud sync
    (only scores and flags — preserves privacy).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS hourly_observations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_epoch        INTEGER, date TEXT, hour INTEGER,
        location_zone   TEXT, activity TEXT, sleep_stage TEXT, is_outdoor INTEGER,
        aq_score        REAL, dominant_pollutant TEXT, signal_count INTEGER,
        hrv             REAL, hr REAL, spo2 REAL,
        ecs             REAL, bio_state TEXT, streak INTEGER,
        penalised       INTEGER, penalty_pts REAL,
        confounder_score REAL,
        stress_inferred INTEGER, stress_confidence REAL,
        raw_result      TEXT
    );
    CREATE TABLE IF NOT EXISTS daily_summaries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        date            TEXT UNIQUE,
        ecs_daily       REAL, ecs_mean REAL,
        penalised_hours INTEGER, total_penalty REAL,
        avg_hrv         REAL, avg_hr REAL,
        dominant_zone   TEXT,
        avg_aq_score    REAL,
        notes           TEXT
    );
    CREATE TABLE IF NOT EXISTS baseline_snapshots (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        captured_at TEXT, activity TEXT,
        hrv REAL, hr REAL, spo2 REAL,
        obs_count INTEGER, personalised INTEGER
    );
    CREATE TABLE IF NOT EXISTS user_questions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        asked_at    TEXT, question_key TEXT, question_text TEXT,
        inference_type TEXT, user_response TEXT, responded_at TEXT
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as c:
            c.executescript(self.SCHEMA)

    def save_hourly(self,
                    snapshot:         EnvironmentSnapshot,
                    result:           dict,
                    signal:    LearningSignal,
                    confounder_score: float,
                    stress_inferred:  bool,
                    stress_conf:      float):
        bio = result.get("biomarkers_raw", {})
        with sqlite3.connect(self.db_path) as c:
            c.execute("""
                INSERT INTO hourly_observations (
                    ts_epoch, date, hour,
                    location_zone, activity, sleep_stage, is_outdoor,
                    aq_score, dominant_pollutant, signal_count,
                    hrv, hr, spo2,
                    ecs, bio_state, streak, penalised, penalty_pts,
                    confounder_score, stress_inferred, stress_confidence,
                    raw_result
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                int(time.time()), snapshot.date, snapshot.timestamp,
                snapshot.location_zone.value, snapshot.activity.value,
                snapshot.sleep_stage.value, int(snapshot.is_outdoor),
                result.get("A", 0.0),
                signal.dominant_pollutant,
                len(signal.exceeding),
                bio.get("hrv"), bio.get("hr"), bio.get("spo2"),
                result.get("ECS"), result.get("bio_state"),
                result.get("exposure_streak"),
                int(result.get("penalised", False)),
                result.get("penalty_pts", 0.0),
                confounder_score,
                int(stress_inferred), stress_conf,
                json.dumps({k: result.get(k) for k in
                            ("ECS","A","B","C","D","bio_state",
                             "penalised","penalty_pts")}),
            ))

    def save_daily(self, date: str, record: dict, memory: AgentMemory):
        with sqlite3.connect(self.db_path) as c:
            row = c.execute("""
                SELECT AVG(hrv), AVG(hr),
                       location_zone, COUNT(*) as cnt,
                       AVG(aq_score)
                FROM hourly_observations WHERE date=?
                GROUP BY location_zone ORDER BY cnt DESC LIMIT 1
            """, (date,)).fetchone()
            avg_hrv   = row[0] if row else None
            avg_hr    = row[1] if row else None
            zone      = row[2] if row else "unknown"
            avg_aq    = row[4] if row else None
            notes = json.dumps([
                i.conclusion for i in memory.today_inferences
                if i.confidence_band in ("high", "medium")
            ])
            c.execute("""
                INSERT OR REPLACE INTO daily_summaries
                (date, ecs_daily, ecs_mean, penalised_hours, total_penalty,
                 avg_hrv, avg_hr, dominant_zone, avg_aq_score, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                date,
                record.get("ECS_daily"), record.get("ECS_mean"),
                record.get("penalised_hours"), record.get("total_penalty"),
                avg_hrv, avg_hr, zone, avg_aq, notes,
            ))

    def save_baseline_snapshot(self, memory: AgentMemory):
        now  = datetime.datetime.now().isoformat()
        pers = int(memory.days_of_data >= MIN_DAYS_PERSONAL_BASELINE)
        rows = [
            (now, k, v.get("hrv"), v.get("hr"), v.get("spo2"),
             memory.baseline_obs_count.get(k, 0), pers)
            for k, v in memory.baselines.items()
        ]
        with sqlite3.connect(self.db_path) as c:
            c.executemany(
                "INSERT INTO baseline_snapshots "
                "(captured_at, activity, hrv, hr, spo2, obs_count, personalised)"
                " VALUES (?,?,?,?,?,?,?)",
                rows,
            )

    def save_question(self, q: PendingQuestion, response: Optional[str]):
        now = datetime.datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO user_questions "
                "(asked_at, question_key, question_text, inference_type,"
                " user_response, responded_at) VALUES (?,?,?,?,?,?)",
                (now, q.question_key, q.question_text, q.inference_type,
                 response, now if response else None),
            )

    def days_count(self) -> int:
        with sqlite3.connect(self.db_path) as c:
            r = c.execute(
                "SELECT COUNT(DISTINCT date) FROM hourly_observations"
            ).fetchone()
        return r[0] if r else 0

    def recent_summaries(self, n: int) -> List[dict]:
        try:
            with sqlite3.connect(self.db_path) as c:
                rows = c.execute("""
                    SELECT date, ecs_daily, ecs_mean,
                           penalised_hours, total_penalty,
                           avg_hrv, avg_hr, dominant_zone, avg_aq_score
                    FROM daily_summaries ORDER BY date DESC LIMIT ?
                """, (n,)).fetchall()
            return [
                {"date": r[0], "ecs_daily": r[1], "ecs_mean": r[2],
                 "penalised_hours": r[3], "total_penalty": r[4],
                 "avg_hrv": r[5], "avg_hr": r[6],
                 "dominant_zone": r[7], "avg_aq_score": r[8]}
                for r in rows
            ]
        except Exception:
            return []

    def cloud_sync(self, date: str, record: dict, memory: AgentMemory,
                   user_uuid: str, opted_in: bool):
        if not opted_in:
            return
        payload = {
            "user_uuid": user_uuid, "date": date,
            "age_band": memory.profile.age_band,
            "activity_level": memory.profile.activity_level,
            "health_conditions": memory.profile.health_conditions,
            "ecs_daily": record.get("ECS_daily"),
            "penalised_hours": record.get("penalised_hours"),
            "stress_hours": record.get("bio_confirmed_hours"),
        }
        def _post():
            try:
                body = json.dumps(payload).encode()
                req  = urllib.request.Request(
                    CLOUD_SYNC_ENDPOINT, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=10)
            except Exception:
                pass
        threading.Thread(target=_post, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — WEEKLY TREND ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

class WeeklyTrendAnalyser:
    """
    Reads last 14 days of daily summaries and detects multi-day patterns.
    Runs at each daily_close(). Results stored in agent._latest_trends.

    Detects:
      chronic_degradation  — ECS falling week-on-week
      hrv_trend_negative   — HRV declining across the week
      recovery_deficit     — ECS < 60 for 5+ of last 7 days
      aq_worsening         — average air quality score rising week-on-week
    """

    def __init__(self, pm: PatternMemory):
        self.pm = pm

    def analyse(self) -> List[dict]:
        summaries = self.pm.recent_summaries(14)
        if len(summaries) < 7:
            return []

        this = summaries[:7]
        last = summaries[7:]
        flags: List[dict] = []

        # ECS degradation
        te = [s["ecs_daily"] for s in this if s["ecs_daily"] is not None]
        le = [s["ecs_daily"] for s in last if s["ecs_daily"] is not None]
        if te and le:
            d = sum(te)/len(te) - sum(le)/len(le)
            if d < -5.0:
                flags.append({
                    "type":        "chronic_degradation",
                    "severity":    "high" if d < -10 else "moderate",
                    "description": f"ECS fell {abs(d):.1f}pts week-on-week",
                    "data":        {
                        "delta":     d,
                        "this_mean": sum(te)/len(te),
                        "last_mean": sum(le)/len(le),
                    },
                })

        # HRV decline
        th = [s["avg_hrv"] for s in this if s["avg_hrv"] is not None]
        lh = [s["avg_hrv"] for s in last if s["avg_hrv"] is not None]
        if th and lh:
            tm, lm = sum(th)/len(th), sum(lh)/len(lh)
            pct    = (lm - tm) / lm * 100 if lm else 0
            if pct > 10:
                flags.append({
                    "type":        "hrv_trend_negative",
                    "severity":    "high" if pct > 20 else "moderate",
                    "description": f"HRV dropped {pct:.0f}% week-on-week",
                    "data":        {
                        "pct_drop": pct,
                        "this_hrv": tm,
                        "last_hrv": lm,
                    },
                })

        # Recovery deficit
        low = sum(
            1 for s in this
            if s.get("ecs_daily") is not None and s["ecs_daily"] < 60
        )
        if low >= 5:
            flags.append({
                "type":        "recovery_deficit",
                "severity":    "high",
                "description": f"ECS below 60 for {low}/7 days",
                "data":        {"low_days": low},
            })

        # AQ worsening
        ta = [s["avg_aq_score"] for s in this if s.get("avg_aq_score") is not None]
        la = [s["avg_aq_score"] for s in last if s.get("avg_aq_score") is not None]
        if ta and la:
            t_aq = sum(ta)/len(ta)
            l_aq = sum(la)/len(la)
            if t_aq - l_aq > 0.10:
                flags.append({
                    "type":        "aq_worsening",
                    "severity":    "moderate",
                    "description": (
                        f"Average air quality score rose {t_aq-l_aq:.2f} "
                        f"this week ({t_aq:.2f} vs {l_aq:.2f})"
                    ),
                    "data": {"this_aq": t_aq, "last_aq": l_aq},
                })

        return flags


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 9 — AGENTIC TOOLS (pre_day_brief)
# ══════════════════════════════════════════════════════════════════════════════

def _http_get(url: str, timeout: int = TOOL_TIMEOUT_S) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ECSAgent/4.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


class _AQTool:
    def fetch(self, lat: float, lon: float) -> dict:
        data = _http_get(
            f"https://api.openaq.org/v2/latest"
            f"?limit=5&radius=5000&coordinates={lat},{lon}"
        )
        if not data:
            return {}
        out: dict = {}
        for res in data.get("results", [])[:3]:
            for m in res.get("measurements", []):
                p, v = m.get("parameter"), m.get("value")
                if p and v is not None and p not in out:
                    out[p] = v
        return out


class _PollenTool:
    def fetch(self, lat: float, lon: float) -> dict:
        params = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "hourly":   "birch_pollen,grass_pollen,mugwort_pollen",
            "forecast_days": 1,
        })
        data = _http_get(
            f"https://air-quality-api.open-meteo.com/v1/air-quality?{params}"
        )
        if not data or "hourly" not in data:
            return {}
        h = data["hourly"]
        return {
            "pollen_tree":  max((v or 0) for v in h.get("birch_pollen",   [0])[:24]),
            "pollen_grass": max((v or 0) for v in h.get("grass_pollen",   [0])[:24]),
            "pollen_weed":  max((v or 0) for v in h.get("mugwort_pollen", [0])[:24]),
        }


class _WeatherTool:
    def fetch(self, lat: float, lon: float) -> dict:
        params = urllib.parse.urlencode({
            "latitude":  lat, "longitude": lon,
            "hourly":    "temperature_2m,windspeed_10m,precipitation,surface_pressure",
            "forecast_days": 1, "timezone": "auto",
        })
        data = _http_get(f"https://api.open-meteo.com/v1/forecast?{params}")
        if not data or "hourly" not in data:
            return {}
        hr = datetime.datetime.now().hour
        h  = data["hourly"]
        def g(k):
            vals = h.get(k) or [None]
            return vals[min(hr, len(vals) - 1)]
        wind   = g("windspeed_10m")    or 99.0
        precip = g("precipitation")    or 0.0
        press  = g("surface_pressure") or 0.0
        return {
            "temp":           g("temperature_2m"),
            "wind_ms":        wind,
            "precip_mm":      precip,
            "pressure":       press,
            "inversion_risk": wind < 2.0 and precip < 0.1 and press > 1013,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ECS AGENT — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

class ECSAgent:
    """
    Reasoning and learning layer above the ECS engine.
    No notifications. No action suggestions. Pure inference and learning.

    QUICK START:
        from ecs_engine import HourlyReading, UserState, compute_hourly_ECS
        from ecs_agent  import ECSAgent, UserProfile, EnvironmentSnapshot
        from ecs_agent  import ActivityLevel, SleepStage, LocationZone

        agent = ECSAgent(profile=UserProfile(age_band="31-45"))
        state = UserState()
        agent.attach_engine_state(state)
        agent.register_location("home", LocationZone.HOME, 51.5074, -0.1278)

        # Each hour:
        snapshot = EnvironmentSnapshot(
            timestamp    = 9,
            date         = "2024-03-15",
            location_zone= LocationZone.COMMUTE,
            activity     = ActivityLevel.LIGHT,
            sleep_stage  = SleepStage.AWAKE,
            indoor_pm25  = None,       # not available outdoors
            outdoor_pm25 = 38.0,
            indoor_co2   = None,
        )
        enriched = agent.enrich(raw_reading, snapshot)
        result   = compute_hourly_ECS(enriched, state)
        output   = agent.post_process(result, enriched, snapshot)

        # output keys:
        #   air_quality_score   float   graded 0-1 from all signals
        #   signal       dict    per-pollutant breakdown
        #   confounder_score    float   0-1 (0=clean, 1=fully confounded)
        #   confounder_reasons  list    what confounders were detected
        #   stress_inferred     bool    did agent detect stress this hour
        #   stress_confidence   float
        #   stress_evidence     list
        #   inferences          list    medium-confidence inferences surfaced
        #   questions           list    PendingQuestion objects
        #   sensitivity_report  dict    current learned weights
        #   personalised        bool
        #   days_of_data        int
        #   controllability     float

        # At 7am:
        brief = agent.pre_day_brief(lat, lon)
        agent.daily_close(date, daily_record)

        # User answers:
        agent.record_response(question_id, "yes")
    """

    def __init__(self,
                 profile:        Optional[UserProfile] = None,
                 db_path:        str  = "ecs_agent.db",
                 user_uuid:      str  = "local",
                 cloud_opted_in: bool = False,
                 assembler:      Optional[object] = None):
        # assembler: optional PollutantAssembler from ecs_data_layer.
        # When provided, agent.enrich(None, snapshot) fetches live data.
        # When None, a pre-built HourlyReading must be passed to enrich().
        self.memory         = AgentMemory()
        self.memory.profile = profile or UserProfile()
        self.db_path        = db_path
        self.user_uuid      = user_uuid
        self.cloud_opted_in = cloud_opted_in
        self.assembler      = assembler   # PollutantAssembler | None

        self.pm          = PatternMemory(db_path)
        self.location    = LocationInferencer(db_path)
        self.learning_classifier = LearningSignalClassifier()
        self.personaliser= BaselinePersonaliser(self.memory)
        self.confounder  = ConfounderDetector()
        self.sensitivity = SensitivityProfiler(self.memory)
        self.uncertainty = UncertaintyManager(self.memory)
        self.trends      = WeeklyTrendAnalyser(self.pm)

        self._engine_ref      = None
        self._latest_trends:  List[dict] = []
        # Stored between enrich() and post_process() within same hour
        self._last_c_score:   float = 0.0
        self._last_c_reasons: List[str] = []

        self.memory.days_of_data = self.pm.days_count()

    # ── Configuration ─────────────────────────────────────────────────────────

    def attach_engine_state(self, engine_state):
        """Hold reference to engine UserState. Agent syncs personal baselines into it."""
        self._engine_ref = engine_state
        self._sync_baselines()

    def register_location(self, name: str, zone: LocationZone,
                           lat: float, lon: float):
        """Register a named GPS location (home, work, etc.) at onboarding."""
        self.location.register(name, zone, lat, lon)

    def load_translations(self, translations: Dict[str, str]):
        """Load i18n string dict. All user-facing text routes through _t()."""
        global _TRANSLATIONS
        _TRANSLATIONS = translations

    def _sync_baselines(self):
        """Push personal baselines into engine's bio_baselines each hour."""
        if self._engine_ref is None:
            return
        b = self.personaliser.get(ActivityLevel.REST)
        self._engine_ref.bio_baselines = {
            "hrv":  b.get("hrv",  52.0),
            "hr":   b.get("hr",   66.0),
            "spo2": b.get("spo2", 98.0),
        }

    # ── enrich() ─────────────────────────────────────────────────────────────

    def enrich(self, reading, snapshot: EnvironmentSnapshot):
        """
        Pre-engine enrichment. Returns a HourlyReading ready for the engine.

        TWO MODES:
          Live mode:   reading=None  + self.assembler set
                       Agent fetches outdoor AQ and pollen from APIs,
                       applies infiltration factors and indoor_overrides
                       from snapshot, builds the HourlyReading itself.
          Manual mode: reading=HourlyReading (pre-built)
                       Agent uses it as-is. No API calls. Backward compatible.

        Steps (in order):
          0. [Live mode only] Fetch + assemble pollutants via data layer
          1. Resolve location zone from GPS if available
          2. Determine is_outdoor from zone
          3. Detect confounders in biomarker signal
          4. Sync personal baselines into engine UserState

        Note: protection_confirmed is NOT set. Engine default (False) is used.
        Penalty is driven by biomarker confirmation only.
        """
        import copy

        # ── Step 0: Live data fetch (only when reading is None + assembler set) ──
        if reading is None:
            if self.assembler is None:
                raise ValueError(
                    "enrich() called with reading=None but no assembler is attached. "
                    "Either pass a pre-built HourlyReading, or attach a "
                    "PollutantAssembler via ECSAgent(assembler=...)."
                )
            if snapshot.gps_lat is None or snapshot.gps_lon is None:
                raise ValueError(
                    "enrich() called with reading=None but snapshot has no GPS "
                    "coordinates. Set snapshot.gps_lat and snapshot.gps_lon."
                )

            # Resolve zone before assembler call so infiltration logic is correct
            _zone = snapshot.location_zone

            # Merge indoor_pm25 shortcut into indoor_overrides for assembler
            _overrides = dict(snapshot.indoor_overrides or {})
            if snapshot.indoor_pm25 is not None and "pm25" not in _overrides:
                _overrides["pm25"] = snapshot.indoor_pm25
            if snapshot.indoor_co2 is not None and "co2" not in _overrides:
                _overrides["co2"] = snapshot.indoor_co2

            assembly = self.assembler.get(
                lat              = snapshot.gps_lat,
                lon              = snapshot.gps_lon,
                zone             = _zone,
                indoor_overrides = _overrides or None,
            )

            # Determine sleep window from snapshot
            _in_sleep = (snapshot.sleep_stage is not None and
                         snapshot.sleep_stage.value != "awake")

            from ecs_engine import HourlyReading as _HR
            reading = _HR(
                timestamp            = snapshot.timestamp,
                pollutants           = assembly.pollutants,
                biomarkers           = None,   # caller sets biomarkers separately
                in_sleep_window      = _in_sleep,
                protection_confirmed = False,
            )
            # Store assembly metadata on snapshot for downstream inspection
            snapshot._assembly = assembly

        enriched = copy.copy(reading)

        # ── Step 1: Location inference
        if snapshot.gps_lat is not None and snapshot.gps_lon is not None:
            now  = datetime.datetime.now()
            zone, _, conf = self.location.infer(
                snapshot.gps_lat, snapshot.gps_lon,
                snapshot.timestamp, now.weekday(),
            )
            if conf >= 0.60:
                snapshot.location_zone = zone

        # 2. Outdoor resolution
        snapshot.is_outdoor = snapshot.location_zone in (
            LocationZone.OUTDOOR, LocationZone.COMMUTE
        )

        # 3. Confounder detection — runs before engine so baseline is ready
        baseline = self.personaliser.get(snapshot.activity)
        c_score, c_reasons = self.confounder.assess(
            reading.biomarkers, snapshot, baseline
        )
        self._last_c_score   = c_score
        self._last_c_reasons = c_reasons

        # 4. Sync personal baselines → engine uses them for bio_state assessment
        #    (baseline update happens in post_process after we have component A)
        self._sync_baselines()

        return enriched

    # ── post_process() ────────────────────────────────────────────────────────

    def post_process(self,
                     result:   dict,
                     reading,
                     snapshot: EnvironmentSnapshot) -> dict:
        """
        Post-engine reasoning. Does NOT modify engine result.

        Returns structured output for the app — pure data, no display logic.
        """
        result["biomarkers_raw"] = reading.biomarkers or {}

        c_score   = self._last_c_score
        c_reasons = self._last_c_reasons

        # ── Learning signal — reads engine component A ─────────────────────
        # Component A is the engine's own weighted air quality score.
        # We use it directly to gate our learning — no re-scoring.
        component_a = result.get("A", 0.0)
        signal = self.learning_classifier.classify(reading.pollutants, component_a)

        # ── Update baselines (now we have component A) ─────────────────────
        # Gated: clean signal (low confounder) + engine says air is clean
        self.personaliser.update(
            reading.biomarkers,
            snapshot.activity,
            signal,
            c_score,
        )

        # ── Update sensitivity accumulators ────────────────────────────────
        # Gated: low confounder + at least one pollutant above WHO
        self.sensitivity.update(
            reading.pollutants,
            reading.biomarkers,
            signal,
            c_score,
        )

        # ── Stress assessment (confounder-gated) ─────────────────────────
        stressed, stress_conf, stress_ev = self.personaliser.assess_stress(
            reading.biomarkers,
            snapshot.activity,
            snapshot.sleep_stage,
            c_score,
        )

        # ── Inference routing ────────────────────────────────────────────
        inferences_out: List[dict] = []

        # Case 1: air elevated + body calm → low sensitivity or clean zone
        if signal.is_elevated and not stressed and reading.biomarkers:
            # Confidence scales with how elevated the air was
            # (mild elevation + calm body = weak signal, strong elevation + calm = interesting)
            exc_count = len(signal.exceeding)
            conf = min(0.30 + exc_count * 0.15, 0.75)
            inf = InferenceRecord(
                timestamp      = snapshot.timestamp,
                inference_type = "stress",
                conclusion     = (
                    f"body_calm_despite_elevated_air_"
                    f"dominant={signal.dominant_pollutant}_"
                    f"exceeding={signal.exceeding}"
                ),
                confidence     = conf,
                evidence       = stress_ev + [
                    f"dominant_pollutant={signal.dominant_pollutant}",
                    f"exceeding={signal.exceeding}",
                    f"confounder_score={c_score:.2f}",
                ],
            )
            out = self.uncertainty.handle(inf, snapshot)
            if out:
                inferences_out.append(out)

        # Case 2: air clean + body stressed → silent stress (non-air cause)
        if signal.is_clean and stressed and stress_conf >= CONF_MEDIUM:
            inf = InferenceRecord(
                timestamp      = snapshot.timestamp,
                inference_type = "stress_clean",
                conclusion     = "silent_stress_body_stressed_air_clean",
                confidence     = stress_conf,
                evidence       = stress_ev + [
                    f"engine_component_a={component_a:.3f}",
                    f"confounder_score={c_score:.2f}",
                ],
            )
            out = self.uncertainty.handle(inf, snapshot)
            if out:
                inferences_out.append(out)

        # Case 3: air elevated + stressed + confounder present → uncertain attribution
        if signal.is_elevated and stressed and c_score >= 0.40:
            inf = InferenceRecord(
                timestamp      = snapshot.timestamp,
                inference_type = "confounder",
                conclusion     = (
                    f"stress_with_elevated_air_but_confounder_present_"
                    f"c_score={c_score:.2f}_attribution_uncertain"
                ),
                confidence     = c_score * 0.80,
                evidence       = c_reasons + [
                    f"dominant_pollutant={signal.dominant_pollutant}",
                    f"stress_conf={stress_conf:.2f}",
                ],
            )
            out = self.uncertainty.handle(inf, snapshot)
            if out:
                inferences_out.append(out)

        # ── Persist ──────────────────────────────────────────────────────
        self.pm.save_hourly(
            snapshot, result, signal, c_score, stressed, stress_conf
        )

        return {
            # Learning signal (what agent uses air data for — NOT scoring)
            "learning_signal": {
                "is_clean":           signal.is_clean,
                "is_elevated":        signal.is_elevated,
                "dominant_pollutant": signal.dominant_pollutant,
                "exceeding":          signal.exceeding,
                "engine_component_a": component_a,
            },
            # Confounder
            "confounder_score":     c_score,
            "confounder_reasons":   c_reasons,
            # Stress
            "stress_inferred":      stressed,
            "stress_confidence":    stress_conf,
            "stress_evidence":      stress_ev,
            # Inferences (medium/low confidence surfaced)
            "inferences":           inferences_out,
            # Questions
            "questions":            self.uncertainty.get_pending(),
            # Sensitivity
            "sensitivity_report":   self.sensitivity.report(),
            # Context
            "controllability":  OUTDOOR_CONTROLLABILITY if snapshot.is_outdoor else 1.0,
            "personalised":     self.personaliser.is_personalised(),
            "days_of_data":     self.memory.days_of_data,
        }

    # ── pre_day_brief() ───────────────────────────────────────────────────────

    def pre_day_brief(self,
                      lat: Optional[float] = None,
                      lon: Optional[float] = None) -> dict:
        """
        Called at 7am. Fetches outdoor AQ, pollen, weather concurrently.
        Returns structured brief — no notifications, raw data only.
        """
        brief: dict = {
            "date":           datetime.date.today().isoformat(),
            "generated_at":   datetime.datetime.now().isoformat(),
            "outdoor_aq":     {},
            "pollen":         {},
            "weather":        {},
            "inversion_risk": False,
            "alerts":         [],
        }

        if lat is None or lon is None:
            home = next(
                (loc for loc in self.location._known
                 if loc.get("zone") == LocationZone.HOME.value),
                None,
            )
            if home:
                lat, lon = home["lat"], home["lon"]

        if lat is None:
            brief["alerts"].append("location_unknown_forecast_unavailable")
            return brief

        results: dict = {}

        def _fetch(key, fn, *args):
            results[key] = fn(*args)

        threads = [
            threading.Thread(target=_fetch, args=("aq",     _AQTool().fetch,     lat, lon)),
            threading.Thread(target=_fetch, args=("pollen", _PollenTool().fetch,  lat, lon)),
            threading.Thread(target=_fetch, args=("weather",_WeatherTool().fetch, lat, lon)),
        ]
        for t in threads: t.start()
        for t in threads: t.join(timeout=TOOL_TIMEOUT_S + 1)

        brief["outdoor_aq"] = results.get("aq",      {})
        brief["pollen"]     = results.get("pollen",  {})
        brief["weather"]    = results.get("weather", {})

        if brief["weather"].get("inversion_risk"):
            brief["inversion_risk"] = True
            brief["alerts"].append(
                "temperature_inversion_pollution_will_persist_longer"
            )

        return brief

    # ── daily_close() ─────────────────────────────────────────────────────────

    def daily_close(self, date: str, daily_record: dict) -> List[dict]:
        """
        Called at 7am alongside engine's close_day().

        1. Save daily summary + baseline snapshot to SQLite
        2. Recompute personal sensitivity weights
        3. Sync updated weights to engine (if personal_weights_A supported)
        4. Run weekly trend analysis
        5. Cloud sync (anonymised, opt-in, background thread)
        6. Reset daily tracking state

        Returns list of trend flags detected this close.
        """
        self.pm.save_daily(date, daily_record, self.memory)
        self.pm.save_baseline_snapshot(self.memory)

        # Recompute sensitivity weights from all accumulated data
        self.sensitivity.compute_weights()

        # Inject into engine if engine supports personal weights
        if (self._engine_ref is not None
                and hasattr(self._engine_ref, "personal_weights_A")):
            self._engine_ref.personal_weights_A = self.memory.personal_weights

        # Weekly trend analysis
        self._latest_trends = self.trends.analyse()

        # Cloud sync
        self.pm.cloud_sync(
            date, daily_record, self.memory,
            self.user_uuid, self.cloud_opted_in,
        )

        # Reset daily state
        self.memory.days_of_data     += 1
        self.memory.today_inferences  = []
        self.uncertainty.reset_daily()
        self._sync_baselines()

        return self._latest_trends

    # ── record_response() ────────────────────────────────────────────────────

    def record_response(self, question_id: str, response: str):
        """Store user's answer to a pending question. Updates pattern memory."""
        self.uncertainty.record_response(question_id, response, self.pm)

    # ── status() ─────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Full agent state snapshot — for debugging and testing."""
        rest = self.personaliser.get(ActivityLevel.REST)
        return {
            "days_of_data":            self.memory.days_of_data,
            "personalised":            self.personaliser.is_personalised(),
            "resting_baseline":        rest,
            "contextual_baselines":    dict(self.memory.baselines),
            "baseline_obs_counts":     dict(self.memory.baseline_obs_count),
            "pending_questions":       len(self.memory.pending_questions),
            "today_inferences":        len(self.memory.today_inferences),
            "known_locations":         len(self.location._known),
            "sensitivity_report":      self.sensitivity.report(),
            "personal_weights_active": bool(self.memory.personal_weights),
            "latest_trends":           self._latest_trends,
            "profile": {
                "age_band":            self.memory.profile.age_band,
                "activity_level":      self.memory.profile.activity_level,
                "health_conditions":   self.memory.profile.health_conditions,
            },
        }
