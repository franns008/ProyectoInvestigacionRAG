"""
acquire_sofascore.py
Descarga estadísticas de jugadores desde Sofascore usando el endpoint
de estadísticas de liga, que devuelve todos los jugadores paginados.

En lugar de iterar equipo → jugador (~650 requests por liga),
usa un solo endpoint paginado (~7 requests por liga).

Instalación:
    pip install requests pandas
"""

import time
import random
import pandas as pd
import requests
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "rawdata/sofascore"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.sofascore.com/api/v1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

# IDs verificados directamente contra la API de Sofascore
LEAGUE_TOURNAMENT_IDS = {
    "liga_arg":       155,
    "brasileirao":    325,
    "premier_league": 17,
    "la_liga":        8,
    "bundesliga":     35,
    "serie_a":        23,
    "ligue_1":        34,
}

LEAGUE_DISPLAY_NAMES = {
    "liga_arg":       "Liga Profesional Argentina",
    "brasileirao":    "Brasileirão Betano",
    "premier_league": "Premier League",
    "la_liga":        "LaLiga",
    "bundesliga":     "Bundesliga",
    "serie_a":        "Serie A",
    "ligue_1":        "Ligue 1",
}

# Grupos de stats disponibles en el endpoint de liga
STAT_GROUPS = ["summary", "attack", "defence", "passing", "duels"]

PAGE_SIZE = 100  # Máximo que acepta el endpoint

# ── API ────────────────────────────────────────────────────────────────────────

def api_get(endpoint: str, retries: int = 3) -> dict | None:
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 60 + random.uniform(10, 30)
                print(f"  Rate limited. Esperando {wait:.0f}s...")
                time.sleep(wait)
            elif r.status_code == 404:
                return None
            else:
                print(f"  HTTP {r.status_code}: {url}")
                return None
        except requests.RequestException as e:
            print(f"  Error intento {attempt + 1}: {e}")
            time.sleep(5)
    return None


def get_latest_season_id(tournament_id: int) -> tuple[int, str] | None:
    """Temporada más reciente de un torneo."""
    data = api_get(f"unique-tournament/{tournament_id}/seasons")
    if not data or not data.get("seasons"):
        return None
    latest = data["seasons"][0]
    return (latest["id"], latest["name"])


def fetch_league_stats_page(
    tournament_id: int,
    season_id: int,
    group: str,
    offset: int,
) -> list[dict]:
    """
    Una página de stats de liga.
    Devuelve lista de registros con stats + info de jugador y equipo.
    """
    endpoint = (
        f"unique-tournament/{tournament_id}/season/{season_id}/statistics"
        f"?limit={PAGE_SIZE}&offset={offset}"
        f"&order=-rating&accumulation=total&group={group}"
    )
    data = api_get(endpoint)
    if not data:
        return []
    return data.get("results", [])


def fetch_all_pages(
    tournament_id: int,
    season_id: int,
    group: str,
) -> list[dict]:
    """Descarga todas las páginas de un grupo de stats."""
    all_results = []
    offset = 0

    while True:
        page = fetch_league_stats_page(tournament_id, season_id, group, offset)
        if not page:
            break
        all_results.extend(page)
        print(f"    offset={offset}: {len(page)} jugadores")

        if len(page) < PAGE_SIZE:
            break  # Última página

        offset += PAGE_SIZE
        time.sleep(random.uniform(1.0, 2.0))

    return all_results


# ── Procesamiento ──────────────────────────────────────────────────────────────

def flatten_record(record: dict, group: str) -> dict:
    """
    Aplana un registro del endpoint de stats de liga.
    Estructura: {stat1: val, stat2: val, ..., "player": {...}, "team": {...}}
    """
    flat = {}

    # Stats del grupo (todo lo que no sea player/team)
    for key, value in record.items():
        if key not in ("player", "team"):
            flat[f"{group}_{key}"] = value

    # Info del jugador
    player = record.get("player", {})
    flat["sofascore_id"] = player.get("id")
    flat["name"] = player.get("name")
    flat["short_name"] = player.get("shortName", player.get("name"))

    # Info del equipo
    team = record.get("team", {})
    flat["team_id"] = team.get("id")
    flat["team_name"] = team.get("name")

    return flat


def merge_stat_groups(groups_data: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Mergea todos los grupos de stats por sofascore_id.
    Cada grupo es un DataFrame con columnas prefijadas por el nombre del grupo.
    """
    dfs = []
    for group, records in groups_data.items():
        if not records:
            continue
        df = pd.DataFrame([flatten_record(r, group) for r in records])
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Join por sofascore_id, empezando por summary que tiene la info base
    merged = dfs[0]
    for df in dfs[1:]:
        # Columnas a no duplicar
        id_cols = ["sofascore_id", "name", "short_name", "team_id", "team_name"]
        stat_cols = [c for c in df.columns if c not in id_cols]
        merged = merged.merge(
            df[["sofascore_id"] + stat_cols],
            on="sofascore_id",
            how="outer",
        )

    return merged


def add_per90_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Versiones por 90 minutos para métricas numéricas."""
    minutes_col = next(
        (c for c in df.columns if "minutes" in c.lower()),
        None
    )
    if not minutes_col:
        return df

    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in ["float64", "int64"]
        and c != minutes_col
        and not c.endswith("_p90")
        and c not in ["sofascore_id", "team_id"]
    ]

    for col in numeric_cols:
        df[col + "_p90"] = df.apply(
            lambda row: (row[col] / row[minutes_col]) * 90
            if pd.notna(row.get(minutes_col)) and row.get(minutes_col, 0) > 0
            else None,
            axis=1,
        )

    return df


# ── Runner ─────────────────────────────────────────────────────────────────────

def fetch_league(league_key: str) -> pd.DataFrame | None:
    tournament_id = LEAGUE_TOURNAMENT_IDS[league_key]
    display_name  = LEAGUE_DISPLAY_NAMES[league_key]

    season_result = get_latest_season_id(tournament_id)
    if not season_result:
        print(f"  No se encontró temporada para {display_name}")
        return None

    season_id, season_name = season_result
    print(f"  Temporada: {season_name} (id={season_id})")

    groups_data = {}
    for group in STAT_GROUPS:
        print(f"  Grupo: {group}")
        records = fetch_all_pages(tournament_id, season_id, group)
        groups_data[group] = records
        print(f"    Total: {len(records)} jugadores")
        time.sleep(random.uniform(2.0, 3.0))

    df = merge_stat_groups(groups_data)
    if df.empty:
        return None

    df = add_per90_stats(df)
    df["league"] = league_key
    df["league_name"] = display_name
    df["season_name"] = season_name
    df["tournament_id"] = tournament_id
    df["season_id"] = season_id

    return df


def run(league_keys: list[str] | None = None):
    targets = league_keys or list(LEAGUE_TOURNAMENT_IDS.keys())
    all_dfs = []

    for league_key in targets:
        print(f"\n{'='*50}")
        print(f"Liga: {LEAGUE_DISPLAY_NAMES[league_key]}")

        df = fetch_league(league_key)

        if df is not None and not df.empty:
            all_dfs.append(df)
            checkpoint = OUTPUT_DIR / f"checkpoint_{league_key}.parquet"
            df.to_parquet(checkpoint, index=False)
            print(f"  ✓ {len(df)} jugadores guardados")
        else:
            print(f"  Sin datos")

        time.sleep(random.uniform(5.0, 10.0))

    if not all_dfs:
        print("\nNo se obtuvo ningún dato.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_parquet(OUTPUT_DIR / "players_sofascore_raw.parquet", index=False)
    full_df.to_csv(OUTPUT_DIR / "players_sofascore_raw.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Total jugadores: {len(full_df)}")
    print(f"Columnas: {len(full_df.columns)}")
    print(f"\nDistribución por liga:")
    print(full_df["league"].value_counts().to_string())


if __name__ == "__main__":
    # Testear con una liga primero:
    # run(league_keys=["liga_arg"])
    run()