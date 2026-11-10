#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wf_lint.py - Pruefung von ComfyUI-Workflows VOR dem Speichern.

Zwei Ebenen:
  1. comfy-lint (pip install comfy-lint) fuer API-Format-Workflows, gegen das
     /object_info eines laufenden ComfyUI oder gegen einen Schema-Cache.
  2. Eigene Struktur-Pruefung fuer UI-Format-Workflows (das Format, das dieses
     Projekt erzeugt: "nodes" + "links"): doppelte IDs, kaputte Links,
     unbekannte Node-Typen (sobald ein Schema-Cache vorhanden ist).

Speichern wird NICHT blockiert, es gibt nur eine Warnung im Log.
  WF_LINT_STRICT=1  -> bei Fehlern wird WorkflowLintError geworfen
  WF_LINT=0         -> Pruefung komplett aus

Schema-Cache (fuer die Pruefung auf unbekannte Nodes) einmalig anlegen,
waehrend ComfyUI laeuft:
  python wf_lint.py --refresh [--server http://127.0.0.1:8188]

Direkt benutzen:
  python wf_lint.py mein_workflow.json [weitere.json ...]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SCHEMA_CACHE = HERE / ".object_info_cache.json"
FRONTEND_ONLY = {"Reroute", "PrimitiveNode", "Note", "MarkdownNote", "Group"}
MAX_LINES = 15


class WorkflowLintError(RuntimeError):
    """Wird nur bei WF_LINT_STRICT=1 geworfen."""


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def is_ui_format(wf) -> bool:
    return isinstance(wf, dict) and isinstance(wf.get("nodes"), list)


def is_api_format(wf) -> bool:
    return (
        isinstance(wf, dict) and bool(wf)
        and all(isinstance(v, dict) and "class_type" in v for v in wf.values())
    )


def _comfy_lint_exe() -> str | None:
    exe_dir = Path(sys.executable).parent
    for cand in (
        PROJECT / "env" / "Scripts" / "comfy-lint.exe",
        exe_dir / "Scripts" / "comfy-lint.exe",
        exe_dir / "comfy-lint.exe",
        exe_dir / "comfy-lint",
    ):
        if cand.is_file():
            return str(cand)
    return shutil.which("comfy-lint")


def _load_schema() -> dict | None:
    try:
        if SCHEMA_CACHE.is_file():
            data = json.loads(SCHEMA_CACHE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) and data else None
    except (OSError, ValueError):
        pass
    return None


def refresh_schema(server: str = "http://127.0.0.1:8188") -> int:
    """Laedt /object_info vom laufenden ComfyUI in den Schema-Cache."""
    url = server.rstrip("/") + "/object_info"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[lint] Schema nicht geladen ({url}): {e}")
        return 1
    SCHEMA_CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[lint] Schema-Cache gespeichert: {SCHEMA_CACHE} ({len(data)} Node-Typen)")
    return 0


def _iter_links(links):
    """Liefert Links einheitlich als Dict, egal ob Listen- oder Dict-Format."""
    if isinstance(links, dict):
        links = list(links.values())
    for l in links or []:
        if isinstance(l, (list, tuple)) and len(l) >= 5:
            yield {"id": l[0], "origin_id": l[1], "target_id": l[3]}
        elif isinstance(l, dict) and "id" in l:
            yield {"id": l["id"], "origin_id": l.get("origin_id"), "target_id": l.get("target_id")}


def check_ui(wf: dict, schema: dict | None) -> tuple[list[str], list[str]]:
    """Struktur-Pruefung eines UI-Format-Workflows -> (Fehler, Warnungen)."""
    errors: list[str] = []
    warnings: list[str] = []
    nodes = wf.get("nodes") or []
    by_id: dict = {}
    for n in nodes:
        if not isinstance(n, dict):
            errors.append("Eintrag in nodes ist kein Objekt")
            continue
        nid = n.get("id")
        if nid is None:
            errors.append("Node ohne id")
            continue
        if nid in by_id:
            errors.append(f"Doppelte Node-ID {nid}")
        by_id[nid] = n
        if not n.get("type"):
            errors.append(f"Node {nid} hat keinen type")

    links = {}
    for l in _iter_links(wf.get("links")):
        links[l["id"]] = l
        if l["origin_id"] not in by_id:
            errors.append(f"Link {l['id']}: Quell-Node {l['origin_id']} fehlt")
        if l["target_id"] not in by_id:
            errors.append(f"Link {l['id']}: Ziel-Node {l['target_id']} fehlt")

    for nid, n in by_id.items():
        ntype = n.get("type")
        for inp in n.get("inputs") or []:
            lid = inp.get("link") if isinstance(inp, dict) else None
            if lid is None:
                continue
            if lid not in links:
                errors.append(f"Node {nid} ({ntype}): Input '{inp.get('name')}' zeigt auf fehlenden Link {lid}")
            elif links[lid]["target_id"] != nid:
                warnings.append(f"Node {nid} ({ntype}): Link {lid} gehoert zu Node {links[lid]['target_id']}")
        for out in n.get("outputs") or []:
            for lid in (out.get("links") if isinstance(out, dict) else None) or []:
                if lid not in links:
                    warnings.append(f"Node {nid} ({ntype}): Output '{out.get('name')}' zeigt auf fehlenden Link {lid} (Frontend raeumt das beim Laden auf)")

    int_ids = [i for i in by_id if isinstance(i, int)]
    last_node = wf.get("last_node_id")
    if int_ids and isinstance(last_node, int) and max(int_ids) > last_node:
        warnings.append(f"last_node_id ({last_node}) kleiner als hoechste Node-ID ({max(int_ids)})")
    int_links = [i for i in links if isinstance(i, int)]
    last_link = wf.get("last_link_id")
    if int_links and isinstance(last_link, int) and max(int_links) > last_link:
        warnings.append(f"last_link_id ({last_link}) kleiner als hoechste Link-ID ({max(int_links)})")

    if schema:
        unknown: dict = {}
        for nid, n in by_id.items():
            t = n.get("type")
            if not t or t in schema or t in FRONTEND_ONLY:
                continue
            if isinstance(t, str) and len(t) == 36 and t.count("-") == 4:
                continue  # Subgraph-Node (UUID)
            unknown.setdefault(t, []).append(nid)
        for t, ids in unknown.items():
            warnings.append(
                f"Unbekannter Node-Typ '{t}' (Node {', '.join(map(str, ids[:5]))}) - Custom-Node nicht installiert?"
            )
    return errors, warnings


def run_comfy_lint(path: Path) -> tuple[bool | None, str]:
    """comfy-lint auf eine API-Format-Datei. None = nicht ausfuehrbar."""
    exe = _comfy_lint_exe()
    if not exe:
        return None, "comfy-lint nicht installiert (pip install comfy-lint)"
    cmd = [exe, "--schema-cache", str(SCHEMA_CACHE), "--timeout", "3", "--color", "never", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=90)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"comfy-lint nicht ausfuehrbar: {e}"
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode == 0:
        return True, out
    if r.returncode == 1:
        return False, out
    return None, "uebersprungen (kein ComfyUI-Server und kein Schema-Cache): " + out[:200]


def check(workflow, path=None, label: str = "") -> bool:
    """Prueft einen Workflow vor dem Speichern. True = keine Fehler."""
    if not _flag("WF_LINT", True):
        return True
    name = Path(path).name if path else (label or "workflow")
    errors: list[str] = []
    warnings: list[str] = []
    kind = "unbekanntes Format"
    try:
        json.dumps(workflow, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        errors.append(f"nicht als JSON speicherbar: {e}")

    try:
        if is_api_format(workflow):
            kind = "API-Format, comfy-lint"
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / (name if name.lower().endswith(".json") else name + ".json")
                tmp.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
                ok, out = run_comfy_lint(tmp)
            if ok is False:
                errors.extend(l.strip() for l in out.splitlines() if l.strip())
            elif ok is None:
                warnings.append(out)
        elif is_ui_format(workflow):
            schema = _load_schema()
            kind = f"UI-Format, {len(workflow.get('nodes') or [])} Nodes"
            kind += ", Schema geprueft" if schema else ", ohne Schema"
            e, w = check_ui(workflow, schema)
            errors.extend(e)
            warnings.extend(w)
        else:
            warnings.append("Workflow-Format nicht erkannt (weder UI- noch API-Format)")
    except Exception as ex:  # noqa: BLE001 - Lint darf das Speichern nie verhindern
        warnings.append(f"Lint-Fehler (ignoriert): {ex}")

    if not errors and not warnings:
        print(f"[lint] {name}: OK ({kind})")
        return True
    print(f"[lint] {name}: {len(errors)} Fehler, {len(warnings)} Warnungen ({kind})")
    for prefix, items in (("FEHLER ", errors), ("Warnung", warnings)):
        for line in items[:MAX_LINES]:
            print(f"  - {prefix}: {line}")
        if len(items) > MAX_LINES:
            print(f"  - ... und {len(items) - MAX_LINES} weitere")
    if errors and _flag("WF_LINT_STRICT", False):
        raise WorkflowLintError(f"{name}: {len(errors)} Fehler (WF_LINT_STRICT=1)")
    return not errors


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--refresh":
        server = "http://127.0.0.1:8188"
        if "--server" in argv:
            i = argv.index("--server")
            if i + 1 < len(argv):
                server = argv[i + 1]
        return refresh_schema(server)
    if not argv:
        print(__doc__)
        return 2
    rc = 0
    for f in argv:
        try:
            wf = json.loads(Path(f).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            print(f"[lint] {f}: nicht lesbar: {e}")
            rc = 1
            continue
        if not check(wf, f):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
