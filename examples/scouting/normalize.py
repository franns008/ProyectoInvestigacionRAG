"""
normalize.py
Toma los datos raw de FBref, Transfermarkt y Sofascore y produce
un DataFrame unificado con un ID canónico por jugador.

El problema central es la resolución de identidad: cada fuente
usa nombres ligeramente distintos para el mismo jugador.
Este script usa fuzzy matching + reglas de negocio para el join.

Instalación:
    pip install pandas thefuzz python-Levenshtein unidecode
"""

import re
import pandas as pd
from pathlib import Path
from unidecode import unidecode
from thefuzz import process as fuzz_process

# ── Configuración ──────────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent / "rawdata"
OUTPUT_DIR = Path(__file__).parent / "rawdata/normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FBREF_PATH         = RAW_DIR / "fbref/players_raw.parquet"
TRANSFERMARKT_PATH = RAW_DIR / "transfermarkt/players_market_raw.parquet"
SOFASCORE_PATH     = RAW_DIR / "sofascore/players_sofascore_raw.parquet"

# Umbral de similitud para fuzzy match (0-100)
# 85 es conservador: mejor falsos negativos que falsos positivos
FUZZY_THRESHOLD = 85

# ── Normalización de nombres ───────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Normaliza un nombre para comparación:
    - Quita acentos y caracteres especiales
    - Lowercase
    - Quita puntos y guiones
    - Ordena tokens alfabéticamente para manejar orden de apellidos distinto
    """
    if not isinstance(name, str):
        return ""
    # Quitar acentos (ñ → n, é → e, etc.)
    name = unidecode(name)
    # Lowercase
    name = name.lower()
    # Quitar caracteres no alfanuméricos excepto espacios
    name = re.sub(r"[^a-z0-9\s]", "", name)
    # Colapsar espacios múltiples
    name = re.sub(r"\s+", " ", name).strip()
    # Ordenar tokens para manejar "Lionel Messi" vs "Messi Lionel"
    tokens = sorted(name.split())
    return " ".join(tokens)


def normalize_position(pos: str | None) -> str:
    """
    Mapea las distintas nomenclaturas de posición a un estándar común.
    FBref usa "MF", "DF", "FW", "GK".
    Transfermarkt y Sofascore usan strings descriptivos.
    """
    if not isinstance(pos, str):
        return "UNKNOWN"

    pos = pos.upper().strip()

    POSITION_MAP = {
        # Porteros
        "GK": "GK", "GOALKEEPER": "GK", "PORTERO": "GK",
        # Defensores
        "DF": "DF", "CB": "DF", "LB": "DF", "RB": "DF",
        "CENTRE-BACK": "DF", "LEFT BACK": "DF", "RIGHT BACK": "DF",
        "DEFENDER": "DF", "DEFENSA": "DF",
        # Mediocampistas
        "MF": "MF", "CM": "MF", "DM": "MF", "AM": "MF",
        "MIDFIELDER": "MF", "CENTRAL MIDFIELD": "MF",
        "DEFENSIVE MIDFIELD": "MF", "ATTACKING MIDFIELD": "MF",
        "MEDIOCAMPISTA": "MF", "VOLANTE": "MF",
        # Delanteros
        "FW": "FW", "ST": "FW", "LW": "FW", "RW": "FW",
        "FORWARD": "FW", "CENTRE-FORWARD": "FW", "LEFT WINGER": "FW",
        "RIGHT WINGER": "FW", "DELANTERO": "FW", "EXTREMO": "FW",
    }

    # Buscar match exacto
    if pos in POSITION_MAP:
        return POSITION_MAP[pos]

    # Buscar si alguna key está contenida en el string
    for key, value in POSITION_MAP.items():
        if key in pos:
            return value

    return "UNKNOWN"


def normalize_market_value(value) -> float | None:
    """Asegura que el valor de mercado sea float o None."""
    if pd.isna(value) or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ── Carga de fuentes ───────────────────────────────────────────────────────────

def load_fbref() -> pd.DataFrame:
    """Carga FBref y estandariza columnas clave."""
    if not FBREF_PATH.exists():
        print(f"  FBref no encontrado en {FBREF_PATH}")
        return pd.DataFrame()

    df = pd.read_parquet(FBREF_PATH)
    print(f"  FBref cargado: {len(df)} filas, {len(df.columns)} columnas")

    key_rename = {
        "player": "name",
        "squad":  "team",
        "comp":   "league_raw",
        "pos":    "position_raw",
        "nation": "nationality_raw",
    }
    df = df.rename(columns={k: v for k, v in key_rename.items() if k in df.columns})

    # Renombrar stats al esquema esperado por build_unified_profile / generate_markdown
    stat_rename = {
        "Performance_Gls":  "standard_Gls",
        "Performance_Ast":  "standard_Ast",
        "Playing Time_Min": "standard_Min",
        "Playing Time_MP":  "standard_MP",
        "Standard_SoT":     "shooting_SoT",
        "Performance_TklW": "defense_Tkl",
        "Performance_Int":  "defense_Int",
    }
    df = df.rename(columns={k: v for k, v in stat_rename.items() if k in df.columns})

    df["name_norm"] = df["name"].apply(normalize_name)

    if "position_raw" in df.columns:
        df["position_canonical"] = df["position_raw"].apply(normalize_position)

    df["source_fbref"] = True
    return df


def load_transfermarkt() -> pd.DataFrame:
    """Carga Transfermarkt y estandariza columnas clave."""
    if not TRANSFERMARKT_PATH.exists():
        print(f"  Transfermarkt no encontrado en {TRANSFERMARKT_PATH}")
        return pd.DataFrame()

    df = pd.read_parquet(TRANSFERMARKT_PATH)
    print(f"  Transfermarkt cargado: {len(df)} filas")

    df["name_norm"] = df["name"].apply(normalize_name)
    df["position_canonical"] = df["position"].apply(normalize_position)
    df["market_value_eur"] = df["market_value_eur"].apply(normalize_market_value)
    df["source_transfermarkt"] = True

    return df


def load_sofascore() -> pd.DataFrame:
    """Carga Sofascore y estandariza columnas clave."""
    if not SOFASCORE_PATH.exists():
        print(f"  Sofascore no encontrado en {SOFASCORE_PATH}")
        return pd.DataFrame()

    df = pd.read_parquet(SOFASCORE_PATH)
    print(f"  Sofascore cargado: {len(df)} filas")

    df["name_norm"] = df["name"].apply(normalize_name)

    # Sofascore no provee posición individual
    if "position_group" in df.columns:
        df["position_canonical"] = df["position_group"].apply(normalize_position)
    else:
        df["position_canonical"] = "UNKNOWN"

    if "team_name" in df.columns:
        df = df.rename(columns={"team_name": "current_club"})

    stat_rename = {
        "summary_goals":              "stat_goals",
        "summary_assists":            "stat_assists",
        "summary_rating":             "stat_rating",
        "summary_successfulDribbles": "stat_successfulDribbles",
        "summary_tackles":            "stat_tackles",
        "attack_totalShots":          "stat_shots",
        "defence_interceptions":      "stat_interceptions",
        "passing_accuratePasses":     "stat_accuratePasses",
        "passing_keyPasses":          "stat_keyPasses",
        "passing_bigChancesCreated":  "stat_bigChancesCreated",
    }
    df = df.rename(columns={k: v for k, v in stat_rename.items() if k in df.columns})

    df["source_sofascore"] = True
    return df


# ── Resolución de identidad ────────────────────────────────────────────────────

def match_players_fuzzy(
    base_df: pd.DataFrame,
    other_df: pd.DataFrame,
    other_name: str,
) -> pd.DataFrame:
    """
    Hace fuzzy match entre base_df y other_df usando nombre normalizado
    + posición como filtro secundario.

    Estrategia:
    1. Join exacto por nombre normalizado (rápido, cubre la mayoría)
    2. Fuzzy match para los que no matchearon (lento, para el resto)
    3. Filtro por posición para reducir falsos positivos
    """
    print(f"\n  Matching con {other_name}...")

    # ── Paso 1: join exacto ──────────────────────────────────────────────────
    exact_matches = base_df.merge(
        other_df,
        on="name_norm",
        how="inner",
        suffixes=("", f"_{other_name}"),
    )
    print(f"    Exact matches: {len(exact_matches)}")

    matched_base_names = set(exact_matches["name_norm"])
    unmatched_base = base_df[~base_df["name_norm"].isin(matched_base_names)].copy()
    unmatched_other = other_df[~other_df["name_norm"].isin(matched_base_names)].copy()

    # ── Paso 2: fuzzy match para los no matcheados ───────────────────────────
    fuzzy_rows = []
    other_names = unmatched_other["name_norm"].tolist()

    for _, row in unmatched_base.iterrows():
        if not other_names:
            break

        result = fuzz_process.extractOne(
            row["name_norm"],
            other_names,
            score_cutoff=FUZZY_THRESHOLD,
        )

        if result is None:
            continue

        matched_name, score = result[0], result[1]

        # Filtro adicional: verificar posición si está disponible en ambos
        other_row = unmatched_other[
            unmatched_other["name_norm"] == matched_name
        ].iloc[0]

        if (
            "position_canonical" in row.index
            and "position_canonical" in other_row.index
            and row["position_canonical"] != "UNKNOWN"
            and other_row["position_canonical"] != "UNKNOWN"
            and row["position_canonical"] != other_row["position_canonical"]
        ):
            # Mismo nombre pero posiciones incompatibles → descartar
            continue

        combined = {**row.to_dict()}
        for col in other_row.index:
            if col not in combined and col != "name_norm":
                combined[col] = other_row[col]
        combined[f"fuzzy_score_{other_name}"] = score
        fuzzy_rows.append(combined)

    fuzzy_df = pd.DataFrame(fuzzy_rows) if fuzzy_rows else pd.DataFrame()
    print(f"    Fuzzy matches: {len(fuzzy_df)}")

    # ── Combinar exact + fuzzy ───────────────────────────────────────────────
    if fuzzy_df.empty:
        return exact_matches

    return pd.concat([exact_matches, fuzzy_df], ignore_index=True)


# ── Construcción del perfil unificado ─────────────────────────────────────────

def build_unified_profile(df: pd.DataFrame) -> pd.DataFrame:
    unified = pd.DataFrame()

    # Identidad
    unified["name"] = df.get("name", df.get("name_transfermarkt", "Unknown"))
    unified["name_norm"] = df["name_norm"]
    unified["tm_id"] = df.get("tm_id", None)
    unified["sofascore_id"] = df.get("sofascore_id", None)

    # Datos demográficos (Transfermarkt es la fuente más confiable)
    unified["age"] = df.get("age", df.get("age_transfermarkt", None))
    unified["nationality"] = df.get("nationality", df.get("nationality_raw", None))
    unified["date_of_birth"] = df.get("date_of_birth", None)
    unified["height_cm"] = df.get("height_cm", df.get("height", None))
    unified["foot"] = df.get("foot", df.get("preferred_foot", None))

    # Posición (FBref tiene la nomenclatura más precisa para Big 5)
    unified["position"] = df.get("position_canonical", None)

    # Club y liga
    unified["current_club"] = df.get("current_club", df.get("team", None))
    unified["league"] = df.get("league", df.get("league_raw", None))
    unified["season"] = df.get("season", None)

    # Mercado (solo Transfermarkt)
    unified["market_value_eur"] = df.get("market_value_eur", None)
    unified["contract_expiry"] = df.get("contract_expiry", None)

    # Stats avanzadas FBref (Big 5)
    fbref_stat_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in [
            "standard_", "shooting_", "passing_", "defense_", "possession_"
        ]
    )]
    for col in fbref_stat_cols:
        unified[col] = df[col]

    # Stats Sofascore (principalmente sudamérica)
    sofascore_stat_cols = [c for c in df.columns if c.startswith("stat_")]
    for col in sofascore_stat_cols:
        unified[col] = df[col]

    # Flags de cobertura (útiles para el RAG para saber qué tan completo es un perfil)
    unified["has_advanced_stats"]   = df["source_fbref"].fillna(False)          if "source_fbref"          in df.columns else False
    unified["has_market_data"]      = df["source_transfermarkt"].fillna(False)  if "source_transfermarkt"  in df.columns else False
    unified["has_sofascore_stats"]  = df["source_sofascore"].fillna(False)      if "source_sofascore"      in df.columns else False

    return unified


# ── Runner principal ───────────────────────────────────────────────────────────

def run():
    print("Cargando fuentes raw...")
    fbref_df = load_fbref()
    tm_df = load_transfermarkt()
    ss_df = load_sofascore()

    sources_available = [
        df for df in [fbref_df, tm_df, ss_df] if not df.empty
    ]

    if not sources_available:
        print("No hay datos disponibles. Corré primero los scripts de adquisición.")
        return

    # ── Estrategia de merge ──────────────────────────────────────────────────
    # Usamos FBref como base para Big 5 y Sofascore como base para sudamérica.
    # Luego mergeamos Transfermarkt sobre ambos.

    print("\nMergeando fuentes...")

    if not fbref_df.empty and not tm_df.empty:
        big5_merged = match_players_fuzzy(fbref_df, tm_df, "transfermarkt")
    elif not fbref_df.empty:
        big5_merged = fbref_df
    else:
        big5_merged = pd.DataFrame()

    if not ss_df.empty and not tm_df.empty:
        # Para sudamérica, filtrar solo ligas no cubiertas por FBref
        sa_leagues = [
            "liga_arg", "brasileirao", "liga_chile",
            "liga_colombia", "liga_uruguay"
        ]
        ss_sa = ss_df[ss_df["league"].isin(sa_leagues)].copy()
        tm_sa = tm_df[tm_df["league"].isin(sa_leagues)].copy() if not tm_df.empty else pd.DataFrame()

        if not tm_sa.empty:
            sa_merged = match_players_fuzzy(ss_sa, tm_sa, "transfermarkt")
        else:
            sa_merged = ss_sa
    else:
        sa_merged = pd.DataFrame()

    # Concatenar Big 5 + Sudamérica
    dfs_to_concat = [df for df in [big5_merged, sa_merged] if not df.empty]
    if not dfs_to_concat:
        print("No se pudo generar el DataFrame unificado.")
        return

    full_df = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"\nTotal antes de deduplicar: {len(full_df)}")

    # Deduplicar: si un mismo jugador aparece dos veces por el mismo período,
    # quedarse con el que tiene más columnas no-nulas
    full_df["_completeness"] = full_df.notna().sum(axis=1)
    full_df = (
        full_df
        .sort_values("_completeness", ascending=False)
        .drop_duplicates(subset=["name_norm", "league", "season"], keep="first")
        .drop(columns=["_completeness"])
    )
    print(f"Total después de deduplicar: {len(full_df)}")

    # Construir perfil unificado
    unified = build_unified_profile(full_df)

    # ── Guardar ──────────────────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "players_unified.parquet"
    unified.to_parquet(output_path, index=False)

    print(f"\nGuardado: {output_path}")
    print(f"Jugadores totales: {len(unified)}")
    print(f"\nCobertura por fuente:")
    print(f"  Con stats avanzadas (FBref):      {unified['has_advanced_stats'].sum()}")
    print(f"  Con datos de mercado (TM):         {unified['has_market_data'].sum()}")
    print(f"  Con stats Sofascore:               {unified['has_sofascore_stats'].sum()}")
    print(f"\nDistribución por liga:")
    print(unified["league"].value_counts().to_string())
    print(f"\nDistribución por posición:")
    print(unified["position"].value_counts().to_string())

    return unified


if __name__ == "__main__":
    run()
