"""
ECS Data Layer
==============
Fetches real outdoor air quality and pollen data and assembles
the pollutants dict the ECS Engine expects.

THREE COMPONENTS:
  1. AirNowClient        — outdoor AQ via EPA AirNow /aq/data/ endpoint
                           pm25, pm10, no2, o3, co — US ground station network
  2. GooglePollenClient  — pollen: tree, grass, weed
  3. PollutantAssembler  — combines outdoor + indoor sources based on
                           location zone, applying infiltration factors
                           when the person is indoors.

DATA SOURCE — AirNow (EPA):
  Over 2,500 EPA ground-station monitors across the US.
  Same data used by Apple Weather, AirVisual, and the US government
  for official AQ reporting. Updated hourly (~35 min past the hour).
  Free API key from https://docs.airnowapi.org/

  Uses the /aq/data/ endpoint which returns raw concentrations:
    PM2.5, PM10  → µg/m³   (stored as-is, engine expects µg/m³)
    NO2, O3      → ppb      (converted to µg/m³ for engine)
    CO           → ppm      (converted to mg/m³ for engine)

  Conversion factors at 25°C, 1 atm:
    NO2: 1 ppb = 1.88 µg/m³   (MW 46 g/mol)
    O3:  1 ppb = 1.96 µg/m³   (MW 48 g/mol)
    CO:  1 ppm = 1.145 mg/m³  (MW 28 g/mol)

  AirNow only reports pollutants measured at nearby stations.
  If a pollutant has no station within distance_miles, it is omitted
  and the assembler treats it as 0 (missing_outdoor will flag it).

LOCATION LOGIC:
  OUTDOOR / COMMUTE zone → outdoor API values only — PM2.5, PM10, NO2, O3,
                           CO, pollen. No indoor pollutants.
  HOME / WORK zone       → ALL outdoor pollutants × infiltration factor
                           (PM2.5, PM10, NO2, O3, CO, pollen all infiltrated)
                           If indoor PM2.5 sensor provided, its reading ADDS
                           to infiltrated outdoor PM2.5 (does not replace it)
                           CO2, radon, humidity, temp from manual overrides

INFILTRATION FACTORS (I/O ratios from building science literature):
  PM2.5:  0.50  Fine particles pass through gaps and ventilation
                (+ indoor sensor reading if provided — both stack)
  PM10:   0.30  Coarser particles filtered more by building envelope
  NO2:    0.60  Gas — penetrates well through ventilation
  O3:     0.20  Reacts with surfaces, degrades quickly indoors
  CO:     0.70  Gas — penetrates freely
  Pollen: 0.10  Large particles, mostly blocked by windows/doors

Sources:
  Nazaroff 2004 — Inhalation exposure to particles indoors
  Chen & Zhao 2011 — Review of I/O ratios for urban residential
  WHO 2010 — WHO guidelines for indoor air quality

USAGE:
  airnow = AirNowClient(api_key="YOUR_AIRNOW_KEY")
  pollen = GooglePollenClient(api_key="YOUR_GOOGLE_KEY")
  assembler = PollutantAssembler(owm_client=airnow, pollen_client=pollen)

  # One call — returns engine-ready dict
  result = assembler.get(
      lat=37.7664, lon=-122.3990,
      zone=LocationZone.HOME,
      indoor_overrides={
          "pm25":     12.0,  # indoor PM2.5 sensor (stacks with outdoor infiltration)
          "co2":       820,  # manual until CO2 sensor arrives
          "radon":      50,  # manual
          "humidity":   50,  # manual
          "temp":       21,  # manual
      }
  )
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum


# ── Location zone enum (mirrors agent) ───────────────────────────────────────

class LocationZone(Enum):
    HOME    = "home"
    WORK    = "work"
    COMMUTE = "commute"
    OUTDOOR = "outdoor"
    UNKNOWN = "unknown"

INDOOR_ZONES  = {LocationZone.HOME, LocationZone.WORK}
OUTDOOR_ZONES = {LocationZone.COMMUTE, LocationZone.OUTDOOR, LocationZone.UNKNOWN}


# ── Infiltration factors ──────────────────────────────────────────────────────

# Infiltration factors applied to ALL outdoor pollutants when indoors.
# PM2.5 uses 0.50 infiltration AND stacks with any indoor sensor reading.
INFILTRATION_FACTORS = {
    "pm25":         0.50,  # fine particles pass through gaps and ventilation
    "pm10":         0.30,  # coarser particles filtered more by building envelope
    "no2":          0.60,  # gas — penetrates well through ventilation
    "o3":           0.20,  # reacts with surfaces, degrades quickly indoors
    "co":           0.70,  # gas — penetrates freely
    "pollen_tree":  0.10,  # large particles, mostly blocked by windows/doors
    "pollen_grass": 0.10,
    "pollen_weed":  0.10,
}

# Indoor-only pollutants — only present when indoors.
# pm25 is listed here because indoors it comes from the sensor, not the API.
INDOOR_ONLY_POLLUTANTS = {"co2", "radon", "humidity", "temp"}

# Pollutants supplied by indoor sensor (replaces outdoor API value when indoors)
INDOOR_SENSOR_POLLUTANTS = {"pm25"}

# Outdoor-only pollutants — only scored when outdoors
OUTDOOR_ONLY_POLLUTANTS = {"pollen_tree", "pollen_grass", "pollen_weed"}

# Default indoor values — ONLY applied when an explicit sensor reading is passed in.
# If no sensor reading is provided, these pollutants are NOT included in scoring.
# We do not fabricate numbers we haven't measured.
DEFAULT_INDOOR_OVERRIDES = {
    # No defaults. Radon, CO2, temp, humidity only scored when sensor data is provided.
    # pm25 falls back to outdoor infiltration (outdoor_pm25 * 0.50) when no sensor.
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. AirNow Air Quality Client (EPA)
# ══════════════════════════════════════════════════════════════════════════════

# Unit conversion constants (at 25°C, 1 atm)
_NO2_UGM3_PER_PPB  = 1.88    # NO2: 1 ppb = 1.88 µg/m³  (MW 46 g/mol)
_O3_UGM3_PER_PPB   = 1.96    # O3:  1 ppb = 1.96 µg/m³  (MW 48 g/mol)
_CO_MGM3_PER_PPM   = 1.145   # CO:  1 ppm = 1.145 mg/m³ (MW 28 g/mol)

# AirNow ParameterName strings → engine keys
_AIRNOW_PARAM_MAP = {
    "PM2.5": "pm25",
    "PM10":  "pm10",
    "NO2":   "no2",
    "OZONE": "o3",
    "CO":    "co",
}


class AirNowClient:
    """
    Fetches current outdoor air quality from the EPA AirNow /aq/data/ API.
    Returns: pm25, pm10, no2, o3, co as a dict with engine-compatible units.

    Data source: 2,500+ EPA ground-station monitors across the US.
    Same data used by Apple Weather and official US AQ reporting.
    Updated hourly (~35 minutes past the hour).

    API key: free registration at https://docs.airnowapi.org/
    Rate limit: 500 requests/hour (free tier)

    Endpoint used: /aq/data/
    Returns raw concentrations with units — not AQI.
    Units from AirNow:
        PM2.5, PM10  → µg/m³    stored as-is (engine expects µg/m³)
        NO2, O3      → ppb      converted → µg/m³ for engine
        CO           → ppm      converted → mg/m³ for engine
    """

    DATA_URL    = "https://www.airnowapi.org/aq/data/"
    HISTORY_URL = "https://www.airnowapi.org/aq/data/"

    def __init__(self, api_key: str, distance_miles: int = 50):
        self.api_key        = api_key
        self.distance_miles = distance_miles   # radius to search for monitoring stations

    def _parse_records(self, records: list) -> dict:
        """
        Parse a list of AirNow /aq/data/ records into engine-compatible dict.
        Each record: {"Parameter": "PM2.5", "Value": 12.3, "Unit": "UG/M3", ...}

        Returns dict with keys: pm25, pm10, no2, o3, co
        Missing pollutants are omitted (no station in range) — caller handles gaps.
        """
        result = {}
        for rec in records:
            param = rec.get("Parameter", "").strip().upper()
            key   = _AIRNOW_PARAM_MAP.get(param)
            if key is None:
                continue
            try:
                raw_val = float(rec.get("Value", 0.0))
            except (TypeError, ValueError):
                continue

            # AirNow may return negative values for very clean air (instrument drift).
            # Clamp to 0 — negative concentrations have no physical meaning for scoring.
            raw_val = max(0.0, raw_val)

            unit = rec.get("Unit", "").strip().upper()

            if key == "no2":
                # AirNow returns NO2 in ppb → convert to µg/m³ for engine
                val = round(raw_val * _NO2_UGM3_PER_PPB, 2)
            elif key == "o3":
                # AirNow returns O3 in ppb → convert to µg/m³ for engine
                val = round(raw_val * _O3_UGM3_PER_PPB, 2)
            elif key == "co":
                # AirNow returns CO in ppm → convert to mg/m³ for engine
                val = round(raw_val * _CO_MGM3_PER_PPM, 3)
            else:
                # PM2.5, PM10 — already in µg/m³
                val = round(raw_val, 2)

            # Keep the highest reading if multiple stations report same pollutant
            if key not in result or val > result[key]:
                result[key] = val

        return result

    def fetch(self, lat: float, lon: float) -> dict:
        """
        Fetch the most recent hour of AQ data for a GPS location.

        Returns a dict with keys: pm25, pm10, no2, o3, co
        All values in engine-compatible units (µg/m³ or mg/m³).
        Pollutants with no nearby station are omitted.

        Raises:
            ValueError   if API key is not set
            RuntimeError if API call fails
        """
        if self.api_key in ("YOUR_AIRNOW_KEY", "", None):
            raise ValueError(
                "AirNow API key not set. "
                "Register free at https://docs.airnowapi.org/"
            )

        # AirNow /aq/data/ needs an explicit time window.
        # Use a 2-hour window ending now to guarantee we catch the latest reading
        # (data is published ~35 min past the hour, so current hour may not exist yet).
        now_utc   = datetime.datetime.utcnow()
        end_dt    = now_utc.replace(minute=0, second=0, microsecond=0)
        start_dt  = end_dt - datetime.timedelta(hours=2)

        params = urllib.parse.urlencode({
            "startDateTimeISO": start_dt.strftime("%Y-%m-%dT%H:%M"),
            "endDateTimeISO":   end_dt.strftime("%Y-%m-%dT%H:%M"),
            "parameters":       "PM2.5,PM10,NO2,OZONE,CO",
            "BBOX":             self._bbox(lat, lon),
            "dataType":         "C",           # C = concentrations (not AQI)
            "format":           "application/json",
            "verbose":          0,             # minimal fields
            "nowcastonly":      0,
            "includerawconcentrations": 1,
            "API_KEY":          self.api_key,
        })
        url = f"{self.DATA_URL}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise RuntimeError(
                    "AirNow API error 400 — check your API key and parameters. "
                    "Key must be activated (check email confirmation)."
                )
            if e.code == 429:
                raise RuntimeError("AirNow API rate limit hit (500 req/hour). Try again shortly.")
            raise RuntimeError(f"AirNow API error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error reaching AirNow API: {e.reason}")

        if not isinstance(data, list) or len(data) == 0:
            raise RuntimeError(
                f"AirNow returned no data for lat={lat}, lon={lon}. "
                f"No monitoring stations within {self.distance_miles} miles?"
            )

        return self._parse_records(data)

    def fetch_history(self, lat: float, lon: float,
                      start_ts: int, end_ts: int) -> list:
        """
        Fetch historical hourly AQ from AirNow for a time range.
        Used to backfill missing hours when the app was closed.

        Args:
            start_ts: Unix timestamp of first hour to fetch (inclusive)
            end_ts:   Unix timestamp of last hour to fetch (inclusive)

        Returns list of dicts: [{timestamp, pm25, pm10, no2, o3, co}, ...]
        Sorted ascending by timestamp.
        AirNow supports history for the past 2 months.
        """
        if self.api_key in ("YOUR_AIRNOW_KEY", "", None):
            raise ValueError("AirNow API key not set.")

        start_dt = datetime.datetime.utcfromtimestamp(start_ts)
        end_dt   = datetime.datetime.utcfromtimestamp(end_ts)

        params = urllib.parse.urlencode({
            "startDateTimeISO": start_dt.strftime("%Y-%m-%dT%H:%M"),
            "endDateTimeISO":   end_dt.strftime("%Y-%m-%dT%H:%M"),
            "parameters":       "PM2.5,PM10,NO2,OZONE,CO",
            "BBOX":             self._bbox(lat, lon),
            "dataType":         "C",
            "format":           "application/json",
            "verbose":          0,
            "nowcastonly":      0,
            "includerawconcentrations": 1,
            "API_KEY":          self.api_key,
        })
        url = f"{self.DATA_URL}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"AirNow history API error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error reaching AirNow history API: {e.reason}")

        if not isinstance(data, list):
            return []

        # AirNow returns one record per station per pollutant per hour.
        # Group by UTC hour, parse each group into one readings dict.
        from collections import defaultdict
        by_hour = defaultdict(list)
        for rec in data:
            # UTC datetime from record, e.g. "2026-03-12T14:00"
            utc_str = rec.get("UTC", "") or rec.get("DateObserved", "")
            if not utc_str:
                continue
            by_hour[utc_str].append(rec)

        results = []
        for utc_str, recs in sorted(by_hour.items()):
            try:
                dt = datetime.datetime.strptime(utc_str[:16], "%Y-%m-%dT%H:%M")
                ts = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
            except ValueError:
                continue
            pollutants = self._parse_records(recs)
            if pollutants:
                entry = {"timestamp": ts}
                entry.update(pollutants)
                results.append(entry)

        results.sort(key=lambda x: x["timestamp"])
        return results

    def _bbox(self, lat: float, lon: float) -> str:
        """
        Build a bounding box string around a lat/lon point.
        AirNow /aq/data/ requires BBOX as "minLon,minLat,maxLon,maxLat".
        Uses distance_miles converted to approximate degrees.
        1 degree lat ≈ 69 miles; 1 degree lon ≈ 69 * cos(lat) miles.
        """
        import math
        delta_lat = self.distance_miles / 69.0
        delta_lon = self.distance_miles / (69.0 * math.cos(math.radians(lat)))
        min_lon = round(lon - delta_lon, 4)
        max_lon = round(lon + delta_lon, 4)
        min_lat = round(lat - delta_lat, 4)
        max_lat = round(lat + delta_lat, 4)
        return f"{min_lon},{min_lat},{max_lon},{max_lat}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Google Pollen Client
# ══════════════════════════════════════════════════════════════════════════════

class GooglePollenClient:
    """
    Fetches daily pollen forecast from Google Pollen API.
    Returns: pollen_tree, pollen_grass, pollen_weed as grains/m³ equivalents.

    API docs: https://developers.google.com/maps/documentation/pollen/overview
    Note: Google Pollen API returns an index (0-5) not grains/m³.
    We map the index to approximate grains/m³ using WHO reference ranges
    so the engine can apply its thresholds consistently.

    Index → grains/m³ mapping (conservative midpoints):
      0 (None)    →   0
      1 (Very Low)→   5
      2 (Low)     →  20
      3 (Moderate)→  50   ← WHO tree/grass threshold
      4 (High)    → 100
      5 (Very High)→ 200

    For weed pollen (WHO limit = 10 grains/m³), the mapping is tighter:
      0 → 0, 1 → 2, 2 → 5, 3 → 10, 4 → 20, 5 → 50
    """

    BASE_URL = "https://pollen.googleapis.com/v1/forecast:lookup"

    # Index → approximate grains/m³ for tree and grass pollen
    TREE_GRASS_MAP = {0: 0, 1: 5, 2: 20, 3: 50, 4: 100, 5: 200}
    # Index → approximate grains/m³ for weed pollen (lower WHO threshold)
    WEED_MAP       = {0: 0, 1: 2, 2: 5,  3: 10, 4: 20,  5: 50}

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, lat: float, lon: float) -> dict:
        """
        Fetch today's pollen forecast for a GPS location.
        Returns: {pollen_tree, pollen_grass, pollen_weed}
        """
        if self.api_key in ("YOUR_GOOGLE_KEY", "", None):
            raise ValueError(
                "Google API key not set. "
                "See Section 1 of the notebook for setup instructions."
            )

        params = urllib.parse.urlencode({
            "key":      self.api_key,
            "location.longitude": lon,
            "location.latitude":  lat,
            "days":     1,
        })
        url = f"{self.BASE_URL}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise RuntimeError(
                    "Google Pollen API key rejected (403). "
                    "Check the key has Pollen API enabled in Google Cloud Console."
                )
            raise RuntimeError(f"Google Pollen API error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error reaching Google Pollen API: {e.reason}")

        # Parse response
        # IMPORTANT: Google has TWO arrays in dailyInfo:
        #   - pollenTypeInfo: aggregated TREE / GRASS / WEED  <-- this is what we need
        #   - plantInfo / plantsInfo: individual species (BIRCH, OAK, etc.) <-- NOT used here
        # The original code read from plantInfo which never contains TREE/GRASS/WEED codes,
        # causing pollen to always return 0.
        daily = data.get("dailyInfo", [{}])[0]
        pollen_type_info = {p["code"]: p for p in daily.get("pollenTypeInfo", [])}

        def index_value(code, pollen_map):
            entry = pollen_type_info.get(code, {})
            # indexInfo is omitted entirely when plant is out of season — default to 0
            idx = entry.get("indexInfo", {}).get("value", 0)
            try:
                return pollen_map.get(int(idx), 0)
            except (TypeError, ValueError):
                return 0

        # Google codes in pollenTypeInfo: TREE, GRASS, WEED
        return {
            "pollen_tree":  index_value("TREE",  self.TREE_GRASS_MAP),
            "pollen_grass": index_value("GRASS", self.TREE_GRASS_MAP),
            "pollen_weed":  index_value("WEED",  self.WEED_MAP),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Pollutant Assembler
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AssemblerResult:
    """
    Full assembled pollutants dict plus metadata about how it was built.
    """
    pollutants:       dict           # engine-ready dict
    zone:             LocationZone
    is_indoor:        bool
    outdoor_raw:      dict           # raw outdoor API values (before infiltration)
    pollen_raw:       dict           # raw pollen values (before infiltration)
    indoor_overrides: dict           # manual/sensor indoor values that were applied
    infiltrated:      dict           # outdoor values after infiltration factor
    pm25_source:      str            # "sensor_plus_infiltration" | "infiltration_only" | "outdoor_api"
    missing_outdoor:  list           # pollutants that defaulted to 0 (API gap)
    source_notes:     list           # human-readable explanation of each value's source


class PollutantAssembler:
    """
    Combines outdoor AQ + pollen + indoor overrides into the pollutants dict
    the ECS Engine expects. Applies infiltration factors when indoors.

    Usage:
        assembler = PollutantAssembler(owm_client, pollen_client)
        result = assembler.get(
            lat=51.5074, lon=-0.1278,
            zone=LocationZone.HOME,
            indoor_overrides={"co2": 820, "radon": 50, "humidity": 50, "temp": 21}
        )
        # Pass result.pollutants to HourlyReading(pollutants=...)
    """

    def __init__(self,
                 owm_client,   # AirNowClient or MockAirNowClient
                 pollen_client: GooglePollenClient):
        self.owm    = owm_client
        self.pollen = pollen_client

    def get(self,
            lat: float,
            lon: float,
            zone: LocationZone,
            indoor_overrides: Optional[Dict] = None) -> AssemblerResult:
        """
        Fetch and assemble all pollutants for the given location and zone.

        Args:
            lat, lon:         GPS coordinates
            zone:             LocationZone — determines indoor vs outdoor logic
            indoor_overrides: Dict of manual/sensor indoor values.
                              For HOME/WORK:
                                "pm25"     — from indoor PM2.5 sensor (replaces outdoor)
                                "co2"      — from CO2 sensor or manual
                                "radon"    — manual
                                "humidity" — manual
                                "temp"     — manual
                              Defaults applied for any missing keys except pm25
                              (pm25 falls back to outdoor × 0.50 if not provided).
        """
        # Compare by value string, not enum identity.
        # When ecs_data_layer and ecs_agent are both exec'd into the same
        # Colab namespace, they each define their own LocationZone class.
        # The `in` set check uses object identity and fails across classes.
        # Using .value (a plain string) is identity-independent.
        _zone_val = zone.value if hasattr(zone, 'value') else str(zone)
        is_indoor = _zone_val in {z.value for z in INDOOR_ZONES}
        overrides = dict(DEFAULT_INDOOR_OVERRIDES)
        if indoor_overrides:
            overrides.update(indoor_overrides)

        # ── Fetch outdoor data ────────────────────────────────────────────────
        outdoor_raw  = self.owm.fetch(lat, lon)   # AirNow EPA ground stations
        pollen_raw   = self.pollen.fetch(lat, lon)
        outdoor_full = {**outdoor_raw, **pollen_raw}

        # ── Build pollutants dict ─────────────────────────────────────────────
        pollutants   = {}
        infiltrated  = {}
        missing      = []
        source_notes = []
        pm25_source  = "outdoor_api"

        if is_indoor:
            # ── PM2.5: sensor reading + outdoor infiltration both stack ───────
            # The sensor measures particles already in the room.
            # Outdoor PM2.5 continuously infiltrates on top — both contribute.
            pm25_factor      = INFILTRATION_FACTORS["pm25"]
            outdoor_pm25     = outdoor_full.get("pm25", 0.0)
            infiltrated_pm25 = round(outdoor_pm25 * pm25_factor, 3)

            if "pm25" in overrides and overrides["pm25"] is not None:
                # Sensor reading + outdoor infiltration contribution (both stack)
                sensor_pm25 = round(float(overrides["pm25"]), 3)
                total_pm25  = round(sensor_pm25 + infiltrated_pm25, 3)
                pollutants["pm25"]  = total_pm25
                infiltrated["pm25"] = infiltrated_pm25
                pm25_source = "sensor_plus_infiltration"
                source_notes.append(
                    f"pm25: {sensor_pm25} (indoor sensor) + {infiltrated_pm25} "
                    f"(outdoor {outdoor_pm25} x {pm25_factor} infiltration)"
                    f" = {total_pm25}"
                )
            else:
                # No sensor — infiltration only
                pollutants["pm25"]  = infiltrated_pm25
                infiltrated["pm25"] = infiltrated_pm25
                pm25_source = "infiltration_only"
                source_notes.append(
                    f"pm25: {outdoor_pm25} (outdoor) x {pm25_factor} "
                    f"infiltration = {infiltrated_pm25}  "
                    f"[no indoor sensor — add indoor_overrides['pm25'] to include sensor reading]"
                )

            # ── All other outdoor pollutants × infiltration factor ────────────
            # pm25 is skipped — already handled above with sensor blending logic
            for key, factor in INFILTRATION_FACTORS.items():
                if key == "pm25":
                    continue
                raw = outdoor_full.get(key, 0.0)
                val = round(raw * factor, 3)
                infiltrated[key] = val
                pollutants[key]  = val
                source_notes.append(
                    f"{key}: {raw} (outdoor) × {factor} infiltration = {val}"
                )

            # ── Indoor-only pollutants — only if sensor data was passed in ─────
            # We never fabricate radon, CO2, temp, or humidity.
            # They are only included when explicitly provided in indoor_overrides.
            for key in INDOOR_ONLY_POLLUTANTS:
                if indoor_overrides and key in indoor_overrides and indoor_overrides[key] is not None:
                    val = indoor_overrides[key]
                    pollutants[key] = val
                    source_notes.append(f"{key}: {val} (sensor reading)")

        else:
            # ── Outdoors — raw API values, no indoor pollutants ───────────────
            for key in outdoor_full:
                if key not in INDOOR_ONLY_POLLUTANTS:
                    pollutants[key] = outdoor_full[key]
                    source_notes.append(f"{key}: {outdoor_full[key]} (outdoor API)")

            # Check for zero values that might be API gaps
            for key in ["pm25", "pm10", "no2", "o3", "co"]:
                if pollutants.get(key, 0) == 0:
                    missing.append(key)

        return AssemblerResult(
            pollutants       = pollutants,
            zone             = zone,
            is_indoor        = is_indoor,
            outdoor_raw      = outdoor_raw,
            pollen_raw       = pollen_raw,
            indoor_overrides = overrides if is_indoor else {},
            infiltrated      = infiltrated,
            pm25_source      = pm25_source,
            missing_outdoor  = missing,
            source_notes     = source_notes,
        )

    def print_assembly(self, result: AssemblerResult):
        """Pretty-print the assembly result for debugging."""
        zone_str = "INDOOR" if result.is_indoor else "OUTDOOR"
        print(f"\n{'='*60}")
        print(f"  POLLUTANT ASSEMBLY — {result.zone.value.upper()} ({zone_str})")
        print(f"{'='*60}")
        print(f"\n  Outdoor API (raw):")
        for k, v in result.outdoor_raw.items():
            print(f"    {k:<15} {v}")
        print(f"\n  Pollen API (raw):")
        for k, v in result.pollen_raw.items():
            print(f"    {k:<15} {v}")
        if result.is_indoor:
            pm25_label = {
                "sensor_plus_infiltration": "  <- sensor + outdoor infiltration blended",
                "infiltration_only":        "  <- no sensor, outdoor x 0.50 only",
                "outdoor_api":              "  <- outdoor only (not indoors)",
            }.get(result.pm25_source, "")
            print(f"\n  PM2.5 source: {result.pm25_source}{pm25_label}")
            print(f"\n  Infiltrated outdoor pollutants (outdoor × factor):")
            for k, v in result.infiltrated.items():
                factor = INFILTRATION_FACTORS.get(k, 0.50)
                print(f"    {k:<15} {v}  (×{factor})")
            print(f"\n  Manual indoor overrides:")
            shown = set()
            for k in list(INDOOR_ONLY_POLLUTANTS) + ["radon"]:
                if k not in shown:
                    v = result.indoor_overrides.get(k, "—")
                    print(f"    {k:<15} {v}")
                    shown.add(k)
            if result.pm25_source == "sensor_plus_infiltration":
                outdoor_pm25 = result.outdoor_raw.get("pm25", 0.0)
                infiltrated  = round(outdoor_pm25 * INFILTRATION_FACTORS["pm25"], 3)
                sensor_val   = round(result.pollutants.get("pm25", 0.0) - infiltrated, 3)
                print(f"\n  PM2.5 blend:")
                print(f"    indoor sensor      {sensor_val}")
                print(f"    outdoor infiltration {infiltrated}  ({outdoor_pm25} x {INFILTRATION_FACTORS['pm25']})")
                print(f"    total (engine sees) {result.pollutants.get('pm25', 0.0)}")
        print(f"\n  Final pollutants dict (engine input):")
        for k, v in sorted(result.pollutants.items()):
            print(f"    {k:<15} {v}")
        if result.missing_outdoor:
            print(f"\n  ⚠  Possible API gaps (returned 0): {result.missing_outdoor}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Mock client for testing without real API keys
# ══════════════════════════════════════════════════════════════════════════════

class MockAirNowClient:
    """
    Returns realistic outdoor AQ values without hitting the AirNow API.
    Values approximate a typical urban US day (comparable to SF Bay Area).
    All values in engine-compatible units (µg/m³ or mg/m³) — same as AirNowClient.

    Mock values and their real-world equivalents:
        pm25:  8.0 µg/m³    (good — AQI ~33)
        pm10: 18.0 µg/m³    (good)
        no2:  54.0 µg/m³    (≈28.7 ppb — moderate urban)
        o3:  117.6 µg/m³    (≈60 ppb   — moderate afternoon)
        co:    0.69 mg/m³   (≈0.6 ppm  — clean urban)

    Note: pm25 here is the OUTDOOR value. When indoors, the assembler
    uses outdoor × 0.50 infiltration factor unless a sensor reading
    is provided in indoor_overrides["pm25"].
    """
    # Mock values already in engine units (µg/m³ / mg/m³)
    _BASE = {"pm25": 8.0, "pm10": 18.0, "no2": 54.0, "o3": 117.6, "co": 0.69}

    def __init__(self):
        self.api_key        = "MOCK"
        self.distance_miles = 50

    def fetch(self, lat: float, lon: float) -> dict:
        return dict(self._BASE)

    def fetch_history(self, lat: float, lon: float,
                      start_ts: int, end_ts: int) -> list:
        """Mock historical data — one entry per hour with slight variation."""
        import math
        results = []
        ts = start_ts
        i  = 0
        while ts <= end_ts:
            factor = 1.0 + 0.15 * math.sin(i * 0.8)
            results.append({
                "timestamp": ts,
                "pm25": round(self._BASE["pm25"] * factor, 2),
                "pm10": round(self._BASE["pm10"] * factor, 2),
                "no2":  round(self._BASE["no2"]  * factor, 2),
                "o3":   round(self._BASE["o3"]   * factor, 2),
                "co":   round(self._BASE["co"]   * factor, 3),
            })
            ts += 3600
            i  += 1
        return results

# Alias so existing code that references MockOWMClient still works
MockOWMClient = MockAirNowClient


class MockPollenClient:
    """
    Returns moderate pollen values without hitting the API.
    """
    def __init__(self):
        self.api_key = "MOCK"

    def fetch(self, lat: float, lon: float) -> dict:
        return {"pollen_tree": 45, "pollen_grass": 20, "pollen_weed": 5}

