"""Indicative weather-based disease risk and spray timing from the free Open-Meteo forecast.

This is a prototype, not a validated disease forecast. Weather only changes the risk level
and spray-timing advice; it never changes the verified chemical, dose or rule status.
"""
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
FORECAST_DAYS = 4

# Hours at or above this humidity, or with rain, are treated as leaf wetness
WET_RH = 90

# Temperature range (°C) and wet hours in one day that favour infection. Based on the
# favouring conditions in the knowledge base and commonly cited infection criteria;
# re-check them against ICAR guidance before relying on them.
DISEASE_WEATHER = {
    "maize_common_rust": (15, 25, 6, "cool, humid weather with long leaf wetness"),
    "maize_gray_leaf_spot": (22, 30, 11, "warm, humid weather with very long leaf wetness"),
    "maize_northern_leaf_blight": (18, 27, 6, "moderate temperatures with long leaf wetness"),
    "potato_early_blight": (20, 30, 8, "warm weather with wet periods"),
    "rice_brown_spot": (22, 30, 8, "warm, humid weather"),
    "rice_leaf_blast": (20, 28, 10, "humid, cloudy weather with long leaf wetness"),
    "rice_neck_blast": (20, 28, 10, "humid weather with long leaf wetness around heading"),
    "wheat_brown_rust": (15, 25, 4, "mild temperatures with dew or high humidity"),
    "wheat_yellow_rust": (5, 15, 6, "cool, moist weather with dew or fog"),
}

# Potato late blight uses the Hutton criteria (Dancey et al., 2017): a day with a minimum
# temperature of at least 10 °C and at least 6 hours of humidity >= 90 %. The Hutton
# criteria were developed in the UK, so a mean-temperature ceiling is added for the warm
# Indian plains, where late blight slows down above about 25 °C (project adaptation)
HUTTON_MIN_TEMP = 10
HUTTON_WET_HOURS = 6
LATE_BLIGHT_MAX_MEAN_TEMP = 25

# Spray-window limits
MAX_WIND_KMH = 15
MAX_TEMP = 35
DRY_HOURS_AFTER = 6
MAX_RAIN_MM = 0.2
MAX_RAIN_PROBABILITY = 40


def geocode(name):
    # Indian places matching the typed name, best match first
    response = requests.get(GEOCODE_URL, params={
        "name": name, "count": 10, "countryCode": "IN", "language": "en"
    }, timeout=10)
    response.raise_for_status()

    places = []
    for item in response.json().get("results", []):
        if item.get("country_code") != "IN":
            continue
        parts = [item.get("name"), item.get("admin2"), item.get("admin1")]
        places.append({
            "label": ", ".join(dict.fromkeys(part for part in parts if part)),
            "latitude": item["latitude"],
            "longitude": item["longitude"],
        })
    return places[:5]


def fetch_forecast(latitude, longitude):
    response = requests.get(FORECAST_URL, params={
        "latitude": latitude, "longitude": longitude, "timezone": TIMEZONE,
        "forecast_days": FORECAST_DAYS,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                  "precipitation_probability,wind_speed_10m",
    }, timeout=10)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    return [
        {
            "time": datetime.fromisoformat(time),
            "temp": hourly["temperature_2m"][i],
            "rh": hourly["relative_humidity_2m"][i] or 0,
            "rain": hourly["precipitation"][i] or 0,
            "rain_probability": hourly["precipitation_probability"][i] or 0,
            "wind": hourly["wind_speed_10m"][i] or 0,
        }
        for i, time in enumerate(hourly["time"])
        if hourly["temperature_2m"][i] is not None
    ]


def _by_day(hours):
    days = {}
    for hour in hours:
        days.setdefault(hour["time"].date(), []).append(hour)
    return days


def _is_wet(hour):
    return hour["rh"] >= WET_RH or hour["rain"] > 0


def disease_risk(disease_key, hours):
    """Low / Moderate / High risk for one disease over the forecast, or None if unsupported."""
    days = _by_day(hours)

    if disease_key == "potato_late_blight":
        qualifying = [
            min(h["temp"] for h in day) >= HUTTON_MIN_TEMP
            and sum(h["temp"] for h in day) / len(day) <= LATE_BLIGHT_MAX_MEAN_TEMP
            and sum(h["rh"] >= WET_RH for h in day) >= HUTTON_WET_HOURS
            for day in days.values()
        ]
        consecutive = any(a and b for a, b in zip(qualifying, qualifying[1:]))
        level = "High" if consecutive else "Moderate" if any(qualifying) else "Low"
        return {
            "disease_key": disease_key,
            "level": level,
            "days": [(date, "favourable" if ok else "") for date, ok in zip(days, qualifying)],
            "reason": (
                "Two days in a row are mild and humid enough for late blight to spread."
                if consecutive else
                "One day is mild and humid enough for late blight to spread." if any(qualifying)
                else "The forecast is not mild and humid enough for late blight to spread."
            ),
            "basis": f"Hutton criteria: minimum temperature ≥ {HUTTON_MIN_TEMP} °C and "
                     f"≥ {HUTTON_WET_HOURS} h at humidity ≥ {WET_RH} % on two consecutive days "
                     f"(plus a daily mean ≤ {LATE_BLIGHT_MAX_MEAN_TEMP} °C for Indian conditions)",
        }

    if disease_key not in DISEASE_WEATHER:
        return None

    low, high, needed, conditions = DISEASE_WEATHER[disease_key]
    counts = {
        date: sum(_is_wet(h) and low <= h["temp"] <= high for h in day)
        for date, day in days.items()
    }
    peak = max(counts.values(), default=0)
    level = "High" if peak >= needed else "Moderate" if peak >= math.ceil(needed / 2) else "Low"

    return {
        "disease_key": disease_key,
        "level": level,
        "days": [(date, f"{count} h") for date, count in counts.items()],
        "reason": f"Up to {peak} hours a day of {conditions} ({low}–{high} °C) are forecast; "
                  f"{needed} or more hours favour infection.",
        "basis": f"Hours at {low}–{high} °C with humidity ≥ {WET_RH} % or rain; "
                 f"{needed} h in one day = high risk",
    }


def spray_window(hours, now=None, horizon_hours=48):
    """First daytime slot of at least two hours that is calm, not too hot and dry afterwards."""
    now = now or datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    upcoming = [h for h in hours if now <= h["time"] < now + timedelta(hours=horizon_hours)]

    def suitable(i):
        hour = upcoming[i]
        after = upcoming[i:i + DRY_HOURS_AFTER]
        return (
            6 <= hour["time"].hour < 18
            and hour["wind"] < MAX_WIND_KMH
            and hour["temp"] < MAX_TEMP
            and len(after) == DRY_HOURS_AFTER
            and all(h["rain"] < MAX_RAIN_MM and h["rain_probability"] < MAX_RAIN_PROBABILITY
                    for h in after)
        )

    window = None
    run_start = None
    for i in range(len(upcoming)):
        if suitable(i):
            run_start = i if run_start is None else run_start
            if i - run_start + 1 >= 2:
                end = i
                while end + 1 < len(upcoming) and suitable(end + 1):
                    end += 1
                window = (upcoming[run_start]["time"], upcoming[end]["time"] + timedelta(hours=1))
                break
        else:
            run_start = None

    next_day = upcoming[:24]
    return {
        "window": window,
        "rain_next_24h": any(h["rain"] >= 0.5 or h["rain_probability"] >= 60 for h in next_day),
        "max_wind_next_24h": max((h["wind"] for h in next_day), default=0),
    }


def summary_for_prompt(place, risks, spray):
    # Short plain-text summary that Gemini may use for timing only
    lines = [f"Location: {place}"]
    for risk in risks:
        lines.append(f"{risk['disease_key']}: {risk['level']} weather risk. {risk['reason']}")
    if spray:
        if spray["window"]:
            start, end = spray["window"]
            lines.append(f"Next suitable spray window: {start:%a %d %b %H:%M}–{end:%H:%M}.")
        else:
            lines.append("No suitable spray window in the next 48 hours.")
        if spray["rain_next_24h"]:
            lines.append("Rain is likely in the next 24 hours.")
    return "\n".join(lines)
