"""
FlyOnTime — Intégration AeroDataBox (via RapidAPI) pour récupérer de vrais
vols programmés dans les 72h à venir, par aéroport.

✅ Structure de champs confirmée avec une vraie clé API (chaque vol a une
clé "movement" unique, scheduledTime.utc au format "2026-08-28 06:05Z").
La forme exacte du niveau supérieur de la réponse (objet avec clés
"departures"/"arrivals", ou liste brute) reste tolérée dans les deux cas
par get_upcoming_flights ci-dessous, faute d'avoir vu un payload complet
non tronqué.

Documentation RapidAPI : https://rapidapi.com/aedbx-aedbx/api/aerodatabox
Endpoint utilisé : Flight Schedules By Airport
  GET /flights/airports/icao/{icao}/{fromLocal}/{toLocal}
Limite connue : la fenêtre fromLocal→toLocal est plafonnée (souvent 12h
sur les plans d'entrée de gamme) — on découpe donc l'horizon de 72h en
plusieurs appels successifs.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import requests

AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"
AERODATABOX_BASE_URL = f"https://{AERODATABOX_HOST}"
CHUNK_HOURS = 12  # remonté (était 6h) : moins de requêtes au total (6 au lieu
                   # de 12 pour couvrir 72h) — le rate-limit RapidAPI s'est
                   # révélé plus contraignant que le risque de timeout sur les
                   # gros aéroports, déjà couvert par le timeout à 25s ci-dessous
REQUEST_TIMEOUT = 25  # augmenté (était 10s) pour les fenêtres plus larges
REQUEST_DELAY_SECONDS = 1.5  # augmenté (était 0.6s) — 0.6s ne suffisait pas à
                              # éviter le rate-limit RapidAPI en pratique
MAX_RETRIES_ON_429 = 3  # augmenté (était 2), avec un backoff plus généreux

_cache: dict = {}
CACHE_TTL_SECONDS = 15 * 60  # 15 min — évite de re-consommer le quota à chaque rechargement dashboard


def _get_api_key() -> str | None:
    return os.getenv("AERODATABOX_API_KEY")


def _to_iso_utc(aerodatabox_time: str) -> str:
    """Convertit le format AeroDataBox ('2026-08-28 06:05Z', sans secondes,
    espace au lieu de 'T') en ISO 8601 standard ('2026-08-28T06:05:00Z'),
    exploitable par datetime.fromisoformat côté dashboard et par Pydantic
    côté API."""
    s = aerodatabox_time.strip()
    tz = ""
    if s.endswith("Z"):
        s, tz = s[:-1], "Z"
    s = s.replace(" ", "T")
    if len(s) == 16:  # 'YYYY-MM-DDTHH:MM' -> ajoute les secondes manquantes
        s += ":00"
    return s + tz


def _parse_flight(item: dict, direction: str) -> dict | None:
    """Transforme un enregistrement AeroDataBox en format interne au
    projet. Retourne None si les champs essentiels manquent (vol ignoré
    plutôt que de planter toute la liste).

    Structure réelle observée (confirmée en direct) : chaque élément a une
    clé unique `movement` (pas `departure`/`arrival` séparées) qui décrit
    l'aéroport de contrepartie (destination si on interroge les départs,
    origine si on interroge les arrivées) et l'horaire programmé."""
    try:
        movement = item.get("movement", {})
        scheduled_raw = movement.get("scheduledTime", {}).get("utc")
        if not scheduled_raw:
            return None
        scheduled = _to_iso_utc(scheduled_raw)

        airline = item.get("airline", {}).get("name", "Compagnie inconnue")
        number = item.get("number", "")
        other_airport = movement.get("airport", {}).get("icao")

        label = f"{airline} {number} — {scheduled[11:16]} UTC"
        if direction == "departure" and other_airport:
            label += f" → {other_airport}"
        elif direction == "arrival" and other_airport:
            label += f" (venant de {other_airport})"

        return {
            "flight_number": number,
            "airline": airline,
            "scheduled_utc": scheduled,
            "destination_icao": other_airport if direction == "departure" else None,
            "origin_icao": other_airport if direction == "arrival" else None,
            "label": label,
        }
    except (KeyError, TypeError, IndexError):
        return None


def _fetch_window_with_retry(url: str, headers: dict, params: dict) -> list:
    """Un appel HTTP avec relance automatique en cas de 429 (rate-limit
    RapidAPI), en respectant l'en-tête Retry-After s'il est fourni. Retourne
    une liste d'items JSON, ou une liste vide si toutes les tentatives échouent."""
    for attempt in range(MAX_RETRIES_ON_429 + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if attempt < MAX_RETRIES_ON_429:
                    wait = float(resp.headers.get("Retry-After", 4.0)) * (attempt + 1)
                    print(f"[warn] AeroDataBox rate-limit (429) sur {url} — "
                          f"nouvelle tentative dans {wait:.1f}s ({attempt + 1}/{MAX_RETRIES_ON_429})")
                    time.sleep(wait)
                    continue
                print(f"[warn] AeroDataBox rate-limit (429) persistant sur {url} après {MAX_RETRIES_ON_429} tentatives")
                return []
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data
        except requests.exceptions.Timeout:
            print(f"[warn] AeroDataBox timeout sur {url} — réponse trop longue (aéroport à fort trafic ?)")
            return []
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"[warn] AeroDataBox indisponible sur {url} : {e}")
            return []
    return []


def get_upcoming_flights(icao: str, flight_type: str, hours_ahead: int = 72) -> list[dict]:
    """Récupère les vols réels programmés pour l'aéroport donné dans les
    `hours_ahead` heures à venir. Retourne une liste vide (jamais une
    exception) en cas d'échec — c'est à l'appelant de gérer le repli sur
    des vols réalistes de substitution."""
    api_key = _get_api_key()
    if not api_key:
        return []

    cache_key = f"{icao}:{flight_type}:{hours_ahead}"
    cached = _cache.get(cache_key)
    if cached and (datetime.now(timezone.utc) - cached["fetched_at"]).total_seconds() < CACHE_TTL_SECONDS:
        return cached["flights"]

    direction_param = "Departure" if flight_type == "departure" else "Arrival"
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": AERODATABOX_HOST}

    flights = []
    now = datetime.now(timezone.utc)
    elapsed = 0
    first_request = True
    while elapsed < hours_ahead:
        if not first_request:
            time.sleep(REQUEST_DELAY_SECONDS)  # espace les appels pour éviter le rate-limit
        first_request = False

        window_start = now + timedelta(hours=elapsed)
        window_end = min(now + timedelta(hours=elapsed + CHUNK_HOURS), now + timedelta(hours=hours_ahead))
        url = (
            f"{AERODATABOX_BASE_URL}/flights/airports/icao/{icao}/"
            f"{window_start.strftime('%Y-%m-%dT%H:%M')}/{window_end.strftime('%Y-%m-%dT%H:%M')}"
        )
        params = {
            "direction": direction_param,
            "withLeg": "false",
            "withCancelled": "false",
            "withCodeshared": "false",
            "withCargo": "false",
            "withPrivate": "false",
        }

        data = _fetch_window_with_retry(url, headers, params)
        items = data if isinstance(data, list) else data.get("departures" if flight_type == "departure" else "arrivals", [])
        for item in items:
            parsed = _parse_flight(item, flight_type)
            if parsed:
                flights.append(parsed)

        elapsed += CHUNK_HOURS

    flights.sort(key=lambda f: f["scheduled_utc"])
    _cache[cache_key] = {"flights": flights, "fetched_at": datetime.now(timezone.utc)}
    return flights