"""
ECS Data Layer
==============
Fetches real outdoor air quality and pollen data and assembles
the pollutants dict the ECS Engine expects.

THREE COMPONENTS:
  1. GoogleAirQualityClient — outdoor AQ via Google Air Quality API
                             pm25, pm10, no2, o3, co — 500×500m resolution, global
  2. GooglePollenClient  — pollen: tree, grass, weed
  3. PollutantAssembler  — combines outdoor + indoor sources based on
                           location zone, applying infiltration factors
                           when the person is indoors.

DATA SOURCE — Google Air Quality API:
  CAMS (Copernicus Atmosphere Monitoring Service) model, same source
  used by Apple Weather. 500×500m resolution globally, updated hourly.
  Same GOOGLE_API_KEY as Pollen API — enable 'Air Quality API' in
  Google Cloud Console. Free tier: 10,000 calls/month.

  Returns raw concentrations via POLLUTANT_CONCENTRATION computation:
    PM2.5, PM10  → µg/m³   (stored as-is, engine expects µg/m³)
    NO2, O3      → ppb      (converted to µg/m³ for engine)
    CO           → ppm      (converted to mg/m³ for engine)

  Conversion factors at 25°C, 1 atm:
    NO2: 1 ppb = 1.88 µg/m³   (MW 46 g/mol)
    O3:  1 ppb = 1.96 µg/m³   (MW 48 g/mol)
    CO:  1 ppm = 1.145 mg/m³  (MW 28 g/mol)

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
  aq     = GoogleAirQualityClient(api_key="YOUR_GOOGLE_KEY")
  pollen = GooglePollenClient(api_key="YOUR_GOOGLE_KEY")
  assembler = PollutantAssembler(owm_client=aq, pollen_client=pollen)

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
# 1. Google Air Quality Client
# ══════════════════════════════════════════════════════════════════════════════

# Unit conversion constants (at 25°C, 1 atm)
_NO2_UGM3_PER_PPB  = 1.88    # NO2: 1 ppb = 1.88 µg/m³  (MW 46 g/mol)
_O3_UGM3_PER_PPB   = 1.96    # O3:  1 ppb = 1.96 µg/m³  (MW 48 g/mol)
_CO_MGM3_PER_PPM   = 1.145   # CO:  1 ppm = 1.145 mg/m³ (MW 28 g/mol)

# Google Air Quality API pollutant codes → engine keys
_GAQ_PARAM_MAP = {
    "pm25":  "pm25",
    "pm10":  "pm10",
    "no2":   "no2",
    "o3":    "o3",
    "co":    "co",
}

# Google concentration units → whether we need to convert
# Google returns: µg/m³ for PM, ppb for NO2/O3, ppm for CO
_GAQ_UNIT_NEEDS_CONVERT = {"ppb", "ppm"}


class GoogleAirQualityClient:
    """
    Fetches outdoor air quality from the Google Air Quality API.
    Returns: pm25, pm10, no2, o3, co as a dict in engine-compatible units.

    Data source: CAMS (Copernicus Atmosphere Monitoring Service) model,
    same source used by Apple Weather. 500×500 m resolution globally.
    Updated hourly.

    API key: same GOOGLE_API_KEY used for the Pollen API.
    Enable "Air Quality API" in Google Cloud Console.
    Free tier: 10,000 calls/month (~$0 for our ~720 calls/month usage).

    Endpoints (HTTP POST with JSON body):
        Current:  /v1/currentConditions:lookup
        History:  /v1/history:lookup

    extraComputations: POLLUTANT_CONCENTRATION — returns raw concentration
    values alongside AQI. Without this, only AQI is returned.

    Units from Google:
        PM2.5, PM10  → µg/m³   stored as-is
        NO2, O3      → ppb     converted → µg/m³ for engine
        CO           → ppm     converted → mg/m³ for engine
    """

    BASE_URL    = "https://airquality.googleapis.com/v1"
    CURRENT_URL = "/currentConditions:lookup"
    HISTORY_URL = "/history:lookup"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _parse_pollutants(self, pollutants_list: list) -> dict:
        """
        Parse Google Air Quality API pollutants array into engine-compatible dict.

        Each item in pollutants_list:
        {
            "code": "no2",
            "displayName": "NO2",
            "fullName": "Nitrogen Dioxide",
            "concentration": {"value": 14.9, "units": "ppb"},
            "additionalInfo": {...}
        }
        """
        result = {}
        for p in pollutants_list:
            code = p.get("code", "").lower()
            key  = _GAQ_PARAM_MAP.get(code)
            if key is None:
                continue
            conc = p.get("concentration", {})
            try:
                raw_val = float(conc.get("value", 0.0))
            except (TypeError, ValueError):
                continue

            raw_val = max(0.0, raw_val)   # clamp negatives
            unit    = conc.get("units", "").lower()

            if key == "no2":
                val = round(raw_val * _NO2_UGM3_PER_PPB, 2)
            elif key == "o3":
                val = round(raw_val * _O3_UGM3_PER_PPB, 2)
            elif key == "co":
                val = round(raw_val * _CO_MGM3_PER_PPM, 3)
            else:
                val = round(raw_val, 2)   # PM2.5, PM10 already µg/m³

            result[key] = val

        return result

    def _post(self, endpoint: str, body: dict) -> dict:
        """Make a POST request to the Google Air Quality API."""
        import urllib.request as _req
        url     = f"{self.BASE_URL}{endpoint}?key={self.api_key}"
        payload = json.dumps(body).encode("utf-8")
        request = _req.Request(
            url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with _req.urlopen(request, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_txt = ""
            try:
                body_txt = e.read().decode()
            except Exception:
                pass
            if e.code == 400:
                raise RuntimeError(
                    f"Google Air Quality API 400 error. "
                    f"Check your API key has 'Air Quality API' enabled in Google Cloud Console. "
                    f"Details: {body_txt[:200]}"
                )
            if e.code == 403:
                raise RuntimeError(
                    "Google Air Quality API 403 — key rejected or billing not enabled. "
                    "Enable billing in Google Cloud Console (free tier covers our usage)."
                )
            raise RuntimeError(f"Google Air Quality API error {e.code}: {e.reason} — {body_txt[:200]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error reaching Google Air Quality API: {e.reason}")

    def fetch(self, lat: float, lon: float) -> dict:
        """
        Fetch current hourly AQ for a GPS location.

        Returns dict with keys: pm25, pm10, no2, o3, co
        All values in engine-compatible units (µg/m³ or mg/m³).
        500×500 m resolution — true micro-level data.

        Raises:
            ValueError   if API key not set
            RuntimeError if API call fails
        """
        if self.api_key in ("YOUR_GOOGLE_KEY", "", None):
            raise ValueError(
                "Google API key not set. "
                "Enable 'Air Quality API' in Google Cloud Console."
            )

        body = {
            "location": {"latitude": lat, "longitude": lon},
            "extraComputations": ["POLLUTANT_CONCENTRATION"],
            "languageCode": "en",
        }
        data = self._post(self.CURRENT_URL, body)

        pollutants = data.get("pollutants", [])
        if not pollutants:
            raise RuntimeError(
                f"Google Air Quality API returned no pollutant data for "
                f"lat={lat:.4f}, lon={lon:.4f}. "
                "Check the location is in a supported country."
            )

        result = self._parse_pollutants(pollutants)
        if not result:
            raise RuntimeError(
                "Google Air Quality API returned pollutants but none matched "
                "known keys (pm25, pm10, no2, o3, co)."
            )
        return result

    def fetch_history(self, lat: float, lon: float,
                      start_ts: int, end_ts: int) -> list:
        """
        Fetch historical hourly AQ from Google Air Quality API.
        Used to backfill missed hours when the app was closed.
        Supports up to 30 days of history.

        Returns list of dicts: [{timestamp, pm25, pm10, no2, o3, co}, ...]
        Sorted ascending by timestamp.
        """
        if self.api_key in ("YOUR_GOOGLE_KEY", "", None):
            raise ValueError("Google API key not set.")

        start_dt = datetime.datetime.utcfromtimestamp(start_ts)
        end_dt   = datetime.datetime.utcfromtimestamp(end_ts)

        body = {
            "location": {"latitude": lat, "longitude": lon},
            "period": {
                "startTime": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "extraComputations": ["POLLUTANT_CONCENTRATION"],
            "languageCode": "en",
            "pageSize": 720,   # max hours in 30 days
        }

        results = []
        while True:
            data = self._post(self.HISTORY_URL, body)
            for hour in data.get("hoursInfo", []):
                dt_str = hour.get("dateTime", "")
                try:
                    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
                    ts = int(dt.replace(
                        tzinfo=datetime.timezone.utc).timestamp())
                except ValueError:
                    continue
                pollutants = hour.get("pollutants", [])
                parsed = self._parse_pollutants(pollutants)
                if parsed:
                    entry = {"timestamp": ts}
                    entry.update(parsed)
                    results.append(entry)

            # Handle pagination
            next_token = data.get("nextPageToken")
            if not next_token:
                break
            body["pageToken"] = next_token

        results.sort(key=lambda x: x["timestamp"])
        return results


# Keep AirNowClient name as alias for backward compatibility
AirNowClient = GoogleAirQualityClient



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

class MockGoogleAQClient:
    """
    Returns realistic outdoor AQ values without hitting the Google Air Quality API.
    Values approximate a clean urban day (SF Bay Area level).
    All values in engine-compatible units (µg/m³ or mg/m³).

    Mock values and their real-world equivalents:
        pm25:  8.0 µg/m³    (good — AQI ~33)
        pm10: 18.0 µg/m³    (good)
        no2:  54.0 µg/m³    (≈28.7 ppb — moderate urban, matches Apple Weather SF)
        o3:  117.6 µg/m³    (≈60 ppb   — moderate afternoon)
        co:    0.69 mg/m³   (≈0.6 ppm  — clean urban)
    """
    _BASE = {"pm25": 8.0, "pm10": 18.0, "no2": 54.0, "o3": 117.6, "co": 0.69}

    def __init__(self):
        self.api_key = "MOCK"

    def fetch(self, lat: float, lon: float) -> dict:
        return dict(self._BASE)

    def fetch_history(self, lat: float, lon: float,
                      start_ts: int, end_ts: int) -> list:
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

# Aliases for backward compatibility
MockAirNowClient = MockGoogleAQClient
MockOWMClient    = MockGoogleAQClient


class MockPollenClient:
    """
    Returns moderate pollen values without hitting the API.
    """
    def __init__(self):
        self.api_key = "MOCK"

    def fetch(self, lat: float, lon: float) -> dict:
        return {"pollen_tree": 45, "pollen_grass": 20, "pollen_weed": 5}

