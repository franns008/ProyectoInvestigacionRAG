"""
acquire_transfermarkt.py
Descarga datos de jugadores desde Transfermarkt usando el endpoint
de valores de mercado por liga, paginado.

En lugar de iterar liga → equipos → jugadores → perfil,
usa /marktwerte/wettbewerb/{ID}/page/{N} que lista todos los
jugadores de la liga directamente (~4 páginas por liga).

Instalación:
    pip install requests beautifulsoup4 lxml pandas
"""

import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "rawdata/transfermarkt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.transfermarkt.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://www.transfermarkt.com/",
}

# comp_id y saison_id verificados contra las URLs de Transfermarkt
COMPETITIONS = {
    "premier_league": {"comp_id": "GB1",  "saison_id": "2024"},
    "la_liga":        {"comp_id": "ES1",  "saison_id": "2024"},
    "bundesliga":     {"comp_id": "L1",   "saison_id": "2024"},
    "serie_a":        {"comp_id": "IT1",  "saison_id": "2024"},
    "ligue_1":        {"comp_id": "FR1",  "saison_id": "2024"},
    "liga_arg":       {"comp_id": "ARG1", "saison_id": "2024"},
    "brasileirao":    {"comp_id": "BRA1", "saison_id": "2025"},  # Brasil usa año calendario
}

# ── Scraping ───────────────────────────────────────────────────────────────────

def get_soup(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return BeautifulSoup(r.content, "lxml")
            elif r.status_code == 429:
                wait = 30 + random.uniform(5, 15)
                print(f"  Rate limited. Esperando {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {r.status_code}: {url}")
                return None
        except requests.RequestException as e:
            print(f"  Error intento {attempt + 1}: {e}")
            time.sleep(5)
    return None


def parse_market_value(raw: str | None) -> int | None:
    """Convierte '€45.00m' o '€800Th.' a entero en euros."""
    if not raw:
        return None
    raw = raw.replace("€", "").replace(",", ".").strip()
    try:
        if "m" in raw.lower():
            return int(float(raw.lower().replace("m", "")) * 1_000_000)
        elif "th." in raw.lower() or "k" in raw.lower():
            val = raw.lower().replace("th.", "").replace("k", "").strip()
            return int(float(val) * 1_000)
        return int(float(raw))
    except ValueError:
        return None


def parse_page(soup: BeautifulSoup, league_key: str) -> list[dict]:
    """
    Parsea una página de /marktwerte y devuelve lista de jugadores.
    Cada fila tiene: nombre, posición, nacionalidad, edad, club, valor de mercado.
    """
    table = soup.find("table", {"class": "items"})
    if not table:
        return []

    records = []
    rows = table.find_all("tr", {"class": ["odd", "even"]})

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        record = {"league": league_key}

        # Nombre y ID del jugador (en inline-table dentro de la segunda celda)
        inline = row.find("table", {"class": "inline-table"})
        if inline:
            name_link = inline.find("a", {"class": None, "title": True})
            if name_link:
                record["name"] = name_link.text.strip()
                href = name_link.get("href", "")
                record["tm_id"] = href.split("/")[-1] if href else None
                record["tm_profile_url"] = BASE_URL + href if href else None

            # Posición (segunda fila del inline-table)
            pos_td = inline.find_all("td")
            if len(pos_td) > 2:
                record["position"] = pos_td[-1].text.strip()

        # Nacionalidad (imagen de bandera con title)
        flag = row.find("img", {"class": "flaggenrahmen"})
        if flag:
            record["nationality"] = flag.get("title", "").strip()

        # Edad
        age_cell = row.find("td", {"class": "zentriert"})
        age_cells = row.find_all("td", {"class": "zentriert"})
        # La edad suele estar en la 4ta celda centrada
        for cell in age_cells:
            text = cell.text.strip()
            if text.isdigit() and 14 <= int(text) <= 45:
                record["age"] = int(text)
                break

        # Club actual
        club_links = row.find_all("a", href=lambda h: h and "/startseite/verein/" in h)
        if club_links:
            record["current_club"] = club_links[0].get("title", club_links[0].text).strip()
            club_href = club_links[0]["href"]
            record["club_tm_id"] = club_href.split("/startseite/verein/")[-1].split("/")[0]

        # Valor de mercado (última celda con clase "rechts hauptlink")
        mv_cell = row.find("td", {"class": "rechts hauptlink"})
        if mv_cell:
            record["market_value_raw"] = mv_cell.text.strip()
            record["market_value_eur"] = parse_market_value(mv_cell.text.strip())

        if record.get("name"):
            records.append(record)

    return records


def get_total_pages(soup: BeautifulSoup) -> int:
    """Detecta el número total de páginas de la paginación."""
    pager = soup.find("ul", {"class": "tm-pagination"})
    if not pager:
        return 1
    page_links = pager.find_all("a")
    pages = []
    for link in page_links:
        text = link.text.strip()
        if text.isdigit():
            pages.append(int(text))
    return max(pages) if pages else 1


# ── Runner ─────────────────────────────────────────────────────────────────────

def fetch_league(league_key: str) -> pd.DataFrame | None:
    comp = COMPETITIONS[league_key]
    comp_id   = comp["comp_id"]
    saison_id = comp["saison_id"]

    # Primera página para detectar total de páginas
    url_page1 = f"{BASE_URL}/x/marktwerte/wettbewerb/{comp_id}/saison_id/{saison_id}/page/1"
    soup = get_soup(url_page1)
    if not soup:
        return None

    total_pages = get_total_pages(soup)
    print(f"  {total_pages} páginas")

    all_records = parse_page(soup, league_key)
    print(f"  Página 1: {len(all_records)} jugadores")

    for page in range(2, total_pages + 1):
        url = f"{BASE_URL}/x/marktwerte/wettbewerb/{comp_id}/saison_id/{saison_id}/page/{page}"
        soup = get_soup(url)
        if not soup:
            break
        records = parse_page(soup, league_key)
        all_records.extend(records)
        print(f"  Página {page}: {len(records)} jugadores")
        time.sleep(random.uniform(2.0, 4.0))

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    df["saison_id"] = saison_id
    return df


def run(competition_keys: list[str] | None = None):
    targets = competition_keys or list(COMPETITIONS.keys())
    all_dfs = []

    for league_key in targets:
        print(f"\n{'='*50}")
        print(f"Liga: {league_key}")

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
    full_df.to_parquet(OUTPUT_DIR / "players_market_raw.parquet", index=False)
    full_df.to_csv(OUTPUT_DIR / "players_market_raw.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Total jugadores: {len(full_df)}")
    print(f"\nDistribución por liga:")
    print(full_df["league"].value_counts().to_string())


if __name__ == "__main__":
    # Testear con una liga primero:
    # run(competition_keys=["premier_league"])
    run()