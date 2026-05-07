"""
acquire_fbref.py
Descarga estadísticas de jugadores de FBref para las Big 5 europeas.
Usa la librería soccerdata que abstrae el scraping con rate limiting.

Instalación:
    pip install soccerdata pandas
"""

import time
import pandas as pd
import soccerdata as sd
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────

LEAGUES = ["Big 5 European Leagues Combined"]

# Temporadas en formato "XXYY" donde XX=año inicio, YY=año fin
SEASONS = ["2324", "2425"]

OUTPUT_DIR = Path(__file__).parent / "rawdata/fbref"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tipos de stats disponibles en FBref.
# Cada uno genera un DataFrame distinto que después normalizamos.
STAT_TYPES = [
    "standard",      # goles, asistencias, minutos, tarjetas
    "keeper",        # stats de portero
    "shooting",      # xG, tiros, tiros al arco
    "playing_time",  # minutos, partidos, rotaciones
    "misc",          # duelos aéreos, faltas, recuperaciones
]

# ── Funciones ──────────────────────────────────────────────────────────────────

def fetch_stat_type(fbref: sd.FBref, stat_type: str) -> pd.DataFrame | None:
    """
    Descarga un tipo de stat específico.
    Devuelve None si falla (FBref a veces da timeout o 429).
    """
    try:
        df = fbref.read_player_season_stats(stat_type=stat_type)
        print(f"  ✓ {stat_type}: {len(df)} filas")
        return df
    except Exception as e:
        print(f"  ✗ {stat_type}: {e}")
        return None


def merge_stat_types(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Mergea todos los DataFrames por jugador + temporada + equipo.
    FBref usa un MultiIndex en columnas (stat_type, metric), lo aplanamos.
    """
    merged = None

    KEY_COLS = ["player", "team", "season", "league", "nation", "pos", "age"]

    for _, df in dfs.items():
        if df is None:
            continue

        # Traer índice (player, team, season, league…) como columnas regulares
        df = df.reset_index()

        # Aplanar MultiIndex de columnas si quedaron columnas anidadas
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns]

        key_cols = [c for c in KEY_COLS if c in df.columns]

        if merged is None:
            merged = df
        else:
            right_cols = key_cols + [c for c in df.columns if c not in merged.columns]
            merged = merged.merge(df[right_cols], on=key_cols, how="outer")

    return merged


def run():
    print("Iniciando descarga de FBref...")
    print(f"Ligas: {', '.join(LEAGUES)}")
    print(f"Temporadas: {', '.join(SEASONS)}\n")

    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS)

    stat_dfs = {}
    for stat_type in STAT_TYPES:
        print(f"Descargando: {stat_type}")
        stat_dfs[stat_type] = fetch_stat_type(fbref, stat_type)
        time.sleep(3)  # Rate limiting entre tipos

    print("\nMergeando stat types...")
    full_df = merge_stat_types(stat_dfs)

    if full_df is None or full_df.empty:
        print("No se pudo obtener datos. Revisá conexión o rate limits.")
        return

    output_path = OUTPUT_DIR / "players_raw.parquet"
    full_df.to_parquet(output_path, index=False)
    print(f"\nGuardado: {output_path} ({len(full_df)} jugadores)")


if __name__ == "__main__":
    run()
