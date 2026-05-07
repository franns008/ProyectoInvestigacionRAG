"""
generate_markdown.py
Genera un archivo .md por liga a partir del DataFrame unificado.
Cada jugador es una sección dentro del archivo de su liga.

La estructura del markdown está pensada para chunking posterior en RAG:
cada jugador tiene un header H2 con su nombre para que sea fácil
dividir el archivo por sección.

Instalación:
    pip install pandas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Configuración ──────────────────────────────────────────────────────────────

UNIFIED_PATH = Path(__file__).parent / "rawdata/normalized/players_unified.parquet"
OUTPUT_DIR   = Path(__file__).parent / "rawdata/markdown"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nombres legibles por clave de liga
LEAGUE_DISPLAY_NAMES = {
    "premier_league": "Premier League",
    "la_liga":        "La Liga",
    "bundesliga":     "Bundesliga",
    "serie_a":        "Serie A",
    "ligue_1":        "Ligue 1",
    "liga_arg":       "Liga Profesional Argentina",
    "brasileirao":    "Brasileirão Série A",
    "liga_chile":     "Primera División Chile",
    "liga_colombia":  "Liga BetPlay Colombia",
    "liga_uruguay":   "Primera División Uruguay",
}

# Stats FBref a mostrar, con etiqueta legible
FBREF_STATS = {
    "standard_Gls":        "Goles",
    "standard_Ast":        "Asistencias",
    "standard_xG":         "xG",
    "standard_xAG":        "xA",
    "standard_Min":        "Minutos jugados",
    "standard_MP":         "Partidos",
    "standard_PrgP":       "Pases progresivos",
    "standard_PrgC":       "Conducciones progresivas",
    "shooting_SoT":        "Tiros al arco",
    "shooting_npxG":       "npxG (sin penales)",
    "passing_KP":          "Pases clave",
    "passing_xA":          "xA (pases)",
    "passing_Cmp%":        "% pases completados",
    "defense_Tkl":         "Tackles",
    "defense_Int":         "Intercepciones",
    "defense_Press":       "Presiones",
    "defense_Succ%":       "% éxito en presiones",
    "possession_Succ":     "Regates exitosos",
    "possession_Carries":  "Conducciones",
}

# Stats Sofascore a mostrar
SOFASCORE_STATS = {
    "stat_goals":                  "Goles",
    "stat_assists":                "Asistencias",
    "stat_minutesPlayed":          "Minutos jugados",
    "stat_rating":                 "Rating Sofascore",
    "stat_shots":                  "Tiros",
    "stat_shotsOnTarget":          "Tiros al arco",
    "stat_keyPasses":              "Pases clave",
    "stat_successfulDribbles":     "Regates exitosos",
    "stat_tackles":                "Tackles",
    "stat_interceptions":          "Intercepciones",
    "stat_duelsWon":               "Duelos ganados",
    "stat_aerialDuelsWon":         "Duelos aéreos ganados",
    "stat_accuratePasses":         "Pases precisos",
    "stat_goals_p90":              "Goles p90",
    "stat_assists_p90":            "Asistencias p90",
    "stat_keyPasses_p90":          "Pases clave p90",
    "stat_tackles_p90":            "Tackles p90",
}

# ── Helpers de formato ─────────────────────────────────────────────────────────

def fmt_value(value) -> str:
    """Formatea un valor numérico o None para mostrar en markdown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def fmt_market_value(value) -> str:
    """Convierte entero de euros a string legible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    value = int(value)
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"€{value / 1_000:.0f}K"
    return f"€{value}"


def safe(row, col, default="—"):
    """Lee una columna del row de forma segura."""
    if col not in row.index:
        return default
    val = row[col]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val).strip() or default


# ── Generación de sección por jugador ─────────────────────────────────────────

def build_player_section(row: pd.Series) -> str:
    """
    Genera el bloque markdown de un jugador.
    Estructura:
        ## Nombre
        ### Perfil
        ### Estadísticas
        ### Mercado
    """
    lines = []

    name   = safe(row, "name", "Jugador desconocido")
    pos    = safe(row, "position")
    club   = safe(row, "current_club")
    league = safe(row, "league")
    season = safe(row, "season")
    age    = safe(row, "age")
    nat    = safe(row, "nationality")
    foot   = safe(row, "foot", safe(row, "preferred_foot"))
    height = safe(row, "height_cm", safe(row, "height"))

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append(f"## {name}")
    lines.append("")

    # ── Perfil ──────────────────────────────────────────────────────────────
    lines.append("### Perfil")
    lines.append("")
    lines.append(f"- **Posición:** {pos}")
    lines.append(f"- **Club:** {club}")
    lines.append(f"- **Liga:** {league}")
    if season != "—":
        lines.append(f"- **Temporada:** {season}")
    lines.append(f"- **Edad:** {age}")
    lines.append(f"- **Nacionalidad:** {nat}")
    if foot != "—":
        lines.append(f"- **Pie dominante:** {foot}")
    if height != "—":
        lines.append(f"- **Altura:** {height} cm")
    lines.append("")

    # ── Estadísticas ────────────────────────────────────────────────────────
    has_advanced = safe(row, "has_advanced_stats") == "True"
    has_sofascore = safe(row, "has_sofascore_stats") == "True"

    if has_advanced:
        lines.append("### Estadísticas (FBref / Opta)")
        lines.append("")

        stat_lines = []
        for col, label in FBREF_STATS.items():
            if col in row.index:
                val = fmt_value(row[col])
                if val != "—":
                    stat_lines.append(f"- **{label}:** {val}")

        if stat_lines:
            lines.extend(stat_lines)
        else:
            lines.append("- Sin datos disponibles")
        lines.append("")

    elif has_sofascore:
        lines.append("### Estadísticas (Sofascore)")
        lines.append("")

        stat_lines = []
        for col, label in SOFASCORE_STATS.items():
            if col in row.index:
                val = fmt_value(row[col])
                if val != "—":
                    stat_lines.append(f"- **{label}:** {val}")

        if stat_lines:
            lines.extend(stat_lines)
        else:
            lines.append("- Sin datos disponibles")
        lines.append("")

    else:
        lines.append("### Estadísticas")
        lines.append("")
        lines.append("- Sin datos estadísticos disponibles para esta temporada")
        lines.append("")

    # ── Mercado ─────────────────────────────────────────────────────────────
    has_market = safe(row, "has_market_data") == "True"
    mv = fmt_market_value(row["market_value_eur"]) if "market_value_eur" in row.index else "—"
    contract = safe(row, "contract_expiry")

    if has_market or mv != "—":
        lines.append("### Mercado")
        lines.append("")
        lines.append(f"- **Valor de mercado:** {mv}")
        lines.append(f"- **Vencimiento de contrato:** {contract}")
        lines.append("")

    # Separador entre jugadores
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


# ── Generación de archivo por liga ────────────────────────────────────────────

def build_league_file(league_key: str, players_df: pd.DataFrame) -> str:
    """
    Genera el contenido completo del archivo markdown de una liga.
    """
    display_name = LEAGUE_DISPLAY_NAMES.get(league_key, league_key)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []

    # Header del archivo
    lines.append(f"# {display_name}")
    lines.append("")
    lines.append(f"**Jugadores:** {len(players_df)}  ")
    lines.append(f"**Generado:** {generated_at}  ")

    # Indicar cobertura de datos
    has_advanced = players_df.get("has_advanced_stats", pd.Series(dtype=bool))
    has_sofascore = players_df.get("has_sofascore_stats", pd.Series(dtype=bool))
    coverage_note = "Estadísticas avanzadas (FBref/Opta)" if has_advanced.any() else "Estadísticas básicas (Sofascore)"
    lines.append(f"**Cobertura estadística:** {coverage_note}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sección por jugador, ordenados por posición y luego por nombre
    position_order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3, "UNKNOWN": 4}
    players_df = players_df.copy()
    players_df["_pos_order"] = players_df["position"].map(position_order).fillna(4)
    players_df = players_df.sort_values(["_pos_order", "name"]).drop(columns=["_pos_order"])

    for _, row in players_df.iterrows():
        lines.append(build_player_section(row))

    return "\n".join(lines)


# ── Runner principal ───────────────────────────────────────────────────────────

def run():
    if not UNIFIED_PATH.exists():
        print(f"No se encontró el archivo unificado en {UNIFIED_PATH}")
        print("Corré normalize.py primero.")
        return

    print(f"Cargando {UNIFIED_PATH}...")
    df = pd.read_parquet(UNIFIED_PATH)
    print(f"{len(df)} jugadores, {df['league'].nunique()} ligas")

    leagues = df["league"].dropna().unique()

    for league_key in sorted(leagues):
        league_df = df[df["league"] == league_key].copy()
        display_name = LEAGUE_DISPLAY_NAMES.get(league_key, league_key)

        print(f"  Generando {display_name} ({len(league_df)} jugadores)...")

        content = build_league_file(league_key, league_df)

        output_path = OUTPUT_DIR / f"{league_key}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"  Guardado: {output_path} ({len(content):,} chars)")

    print(f"\nMarkdowns generados en {OUTPUT_DIR}/")
    print(f"Archivos:")
    for f in sorted(OUTPUT_DIR.glob("*.md")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    run()
