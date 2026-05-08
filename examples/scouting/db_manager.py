"""
Gestor interactivo de bases de datos PostgreSQL / pgvector.
Permite listar y eliminar tablas e índices sin entrar al SQL.
"""
from pathlib import Path
import re
import sys

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import psycopg
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from rich.console import Console
from rich.table import Table
from rich import box

DB_CONNECTION = "postgresql://avdbuser:avdbpass@localhost:5433/pgvdb"

console = Console()


# --- DB helpers ---

def connect() -> psycopg.Connection:
    try:
        return psycopg.connect(DB_CONNECTION, row_factory=dict_row)
    except Exception as e:
        console.print(f"[red]No se pudo conectar a la base de datos:[/red] {e}")
        sys.exit(1)


def fetch(conn: psycopg.Connection, query: str, params: tuple = ()) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def execute(conn: psycopg.Connection, query, params: tuple = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(query, params)
    conn.commit()


def get_databases(conn: psycopg.Connection) -> list[str]:
    rows = fetch(conn, "SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname")
    return [r["datname"] for r in rows]


def get_current_db(conn: psycopg.Connection) -> str:
    return fetch(conn, "SELECT current_database() AS db")[0]["db"]


def get_tables(conn: psycopg.Connection) -> list[dict]:
    rows = fetch(conn, """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    result = []
    for r in rows:
        name = r["table_name"]
        count_row = fetch(conn, SQL("SELECT COUNT(*) AS cnt FROM {}").format(Identifier(name)))
        count = count_row[0]["cnt"]

        emb_row = fetch(conn, """
            SELECT pg_catalog.format_type(atttypid, atttypmod) AS etype
            FROM pg_catalog.pg_attribute
            WHERE attrelid = %s::regclass AND attname = 'embedding' AND attnum > 0
        """, (name,))
        etype = emb_row[0]["etype"] if emb_row else "—"

        result.append({"name": name, "count": count, "embedding": etype})
    return result


def get_indexes(conn: psycopg.Connection) -> list[dict]:
    rows = fetch(conn, """
        SELECT indexname, tablename, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY tablename, indexname
    """)
    result = []
    for r in rows:
        defn = r["indexdef"] or ""
        if "USING gin" in defn:
            kind = "gin"
            # extraer language del tsvector si está presente
            m = re.search(r"to_tsvector\('(\w+)'", defn)
            lang = m.group(1) if m else ""
            detail = f"tsvector('{lang}')" if lang else "gin"
        elif "USING hnsw" in defn:
            kind = "hnsw"
            m = re.search(r"\((\w+)\s+\w+\)", defn)
            detail = m.group(0) if m else "hnsw"
        elif "USING btree" in defn:
            kind = "btree"
            detail = "primary key"
        else:
            kind = "?"
            detail = defn[:40]

        result.append({
            "name":    r["indexname"],
            "table":   r["tablename"],
            "kind":    kind,
            "detail":  detail,
            "indexdef": defn,
        })
    return result


# --- Display ---

def show_overview(conn: psycopg.Connection) -> tuple[list[dict], list[dict]]:
    current_db = get_current_db(conn)
    databases  = get_databases(conn)
    tables     = get_tables(conn)
    indexes    = get_indexes(conn)

    console.rule("[bold cyan]DB Manager[/bold cyan]")

    # Bases de datos
    dbs_str = "  ".join(
        f"[bold]{db}[/bold][dim] ←[/dim]" if db == current_db else db
        for db in databases
    )
    console.print(f"\n[bold]Bases de datos:[/bold]  {dbs_str}\n")

    # Tablas
    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    tbl.add_column("#",         style="dim",  width=4)
    tbl.add_column("Tabla",     style="bold")
    tbl.add_column("Documentos", justify="right")
    tbl.add_column("Embedding")

    for i, t in enumerate(tables, 1):
        tbl.add_row(str(i), t["name"], f"{t['count']:,}", t["embedding"])

    console.print("[bold]Tablas:[/bold]")
    console.print(tbl)

    # Índices
    idx_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    idx_tbl.add_column("#",       style="dim",  width=4)
    idx_tbl.add_column("Índice",  style="bold")
    idx_tbl.add_column("Tabla")
    idx_tbl.add_column("Tipo",    justify="center")
    idx_tbl.add_column("Detalle", style="dim")

    for i, idx in enumerate(indexes, 1):
        kind_color = {"gin": "yellow", "hnsw": "green", "btree": "dim"}.get(idx["kind"], "white")
        idx_tbl.add_row(
            str(i),
            idx["name"],
            idx["table"],
            f"[{kind_color}]{idx['kind']}[/{kind_color}]",
            idx["detail"],
        )

    console.print("[bold]Índices:[/bold]")
    console.print(idx_tbl)

    return tables, indexes


# --- Actions ---

def pick_from(label: str, items: list[dict], name_key: str) -> dict | None:
    console.print(f"\n[bold cyan]{label}[/bold cyan]")
    for i, item in enumerate(items, 1):
        console.print(f"  [cyan]{i}[/cyan]  {item[name_key]}")
    console.print("  [cyan]0[/cyan]  Cancelar\n")

    while True:
        raw = console.input("[bold cyan]Opción:[/bold cyan] ").strip()
        if raw == "0" or raw == "":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1]
        console.print(f"[yellow]Ingresá un número entre 0 y {len(items)}.[/yellow]")


def confirm(msg: str) -> bool:
    raw = console.input(f"[bold red]{msg}[/bold red] [dim](s/N):[/dim] ").strip().lower()
    return raw in ("s", "si", "sí", "y", "yes")


def action_drop_index(conn: psycopg.Connection, indexes: list[dict]) -> None:
    idx = pick_from("Seleccioná el índice a eliminar", indexes, "name")
    if not idx:
        return
    console.print(f"\n  [dim]{idx['indexdef']}[/dim]")
    if confirm(f"¿Eliminar índice '{idx['name']}'?"):
        execute(conn, SQL("DROP INDEX IF EXISTS {}").format(Identifier(idx["name"])))
        console.print(f"[green]Índice '{idx['name']}' eliminado.[/green]")
    else:
        console.print("[dim]Cancelado.[/dim]")


def action_drop_table(conn: psycopg.Connection, tables: list[dict]) -> None:
    tbl = pick_from("Seleccioná la tabla a eliminar", tables, "name")
    if not tbl:
        return
    console.print(
        f"\n  [bold red]ATENCIÓN:[/bold red] se eliminarán "
        f"[bold]{tbl['count']:,}[/bold] documentos y todos los índices asociados."
    )
    if confirm(f"¿Eliminar tabla '{tbl['name']}' y todos sus datos?"):
        execute(conn, SQL("DROP TABLE IF EXISTS {} CASCADE").format(Identifier(tbl["name"])))
        console.print(f"[green]Tabla '{tbl['name']}' eliminada.[/green]")
    else:
        console.print("[dim]Cancelado.[/dim]")


# --- Main loop ---

def main() -> None:
    conn = connect()

    while True:
        console.print()
        tables, indexes = show_overview(conn)

        console.print("[bold]Acciones:[/bold]")
        console.print("  [cyan]1[/cyan]  Eliminar un índice")
        console.print("  [cyan]2[/cyan]  Eliminar una tabla  [dim](y todos sus documentos)[/dim]")
        console.print("  [cyan]3[/cyan]  Refrescar")
        console.print("  [cyan]0[/cyan]  Salir")
        console.print()

        raw = console.input("[bold cyan]Acción:[/bold cyan] ").strip()

        if raw == "0" or raw.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break
        elif raw == "1":
            if not indexes:
                console.print("[yellow]No hay índices.[/yellow]")
            else:
                action_drop_index(conn, indexes)
        elif raw == "2":
            if not tables:
                console.print("[yellow]No hay tablas.[/yellow]")
            else:
                action_drop_table(conn, tables)
        elif raw == "3":
            pass  # loop vuelve a llamar show_overview
        else:
            console.print("[yellow]Opción inválida.[/yellow]")

    conn.close()


if __name__ == "__main__":
    main()
