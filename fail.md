PR REVIEW SUMMARY

Bitte folgende Punkte prüfen und umsetzen. CodeRabbit meldet noch mehrere echte Probleme.

==================================================
1. run.bat: Python-Aufruf korrigieren
==================================================

Aktuell:

"%PYTHON%" immo.py!GERICHTE!

ändern zu:

"%PYTHON%" immo.py !GERICHTE!

Sonst wird bei Auswahl aller Gerichte z.B.

immo.pyAachen

aufgerufen statt

immo.py Aachen

==================================================
2. run.bat: Gerichtsnummern korrekt auflösen
==================================================

Die aktuelle Nummernauflösung für Eingaben wie

5 22

ist fehlerhaft.

Zusätzlich bindet das else aktuell an das falsche if.

Eingaben wie:

0 5

markieren die 0 nicht korrekt als ungültig.

Beispiel für robuste Logik:

for %%i in (%CHOICE%) do (
    set "OK="
    if %%i geq 1 if %%i leq 38 set "OK=1"

    if defined OK (
        call set "GERICHTE=%%GERICHTE%% %%G%%i%%"
    ) else (
        set "BAD=1"
    )
)

Erwartung:

5 22

→ Bonn Koeln

==================================================
3. DeepSeek API-Key nicht an alle Endpunkte senden
==================================================

Derzeit wird der Authorization-Header an alle gefundenen Endpunkte übertragen:

- localhost
- 127.0.0.1
- beliebige LLM_ENDPOINTS
- *.env Einträge

Dadurch kann der DeepSeek-Key versehentlich offengelegt werden.

Ziel:

- DeepSeek-Key nur für DeepSeek-Endpunkte
- nur über HTTPS
- keine Credentials an lokale Endpunkte

Beispiel:

from urllib.parse import urlparse

def key_for(base: str):
    p = urlparse(base)

    if (
        p.scheme == "https"
        and p.hostname
        and p.hostname.endswith("deepseek.com")
    ):
        return DEEPSEEK_API_KEY

    return ""

Sowohl in:

- _probe()
- llm_chat()

anwenden.

==================================================
4. Separate Connect- und Read-Timeouts
==================================================

Aktuell:

LLM_TIMEOUT = 15

Problem:

Lokale Modelle (20B+)
größere JSON-Antworten
lange Analysen

laufen regelmäßig in Timeouts.

Empfehlung:

LLM_TIMEOUT = (5, 180)

oder

LLM_CONNECT_TIMEOUT = 5
LLM_READ_TIMEOUT = 180

und dann:

requests.post(
    url,
    json=payload,
    timeout=LLM_TIMEOUT
)

Umsetzen in:

- app.py
- build_index.py

==================================================
5. Fehlgeschlagene Analysen nicht dauerhaft cachen
==================================================

Aktuell:

obj_id, data = fut.result()

results.append(data)
cache[obj_id] = data
save_cache(cache)

Problem:

Wenn Analyse fehlschlägt, wird trotzdem ein Dummy gespeichert.

Danach gilt:

obj_id in cache

und das Objekt wird nie erneut analysiert.

Lösung:

analyze_one() soll zusätzlich einen Erfolgsstatus zurückgeben.

Beispiel:

obj_id, success, data = fut.result()

results.append(data)

if success:
    cache[obj_id] = data
    save_cache(cache)

Fehlerobjekte nur rendern, nicht persistieren.

==================================================
6. Defekten Cache sauber behandeln
==================================================

Aktuell:

raise SystemExit(...)

Problem:

SystemExit ist BaseException und kann Gradio-Callbacks unsauber beenden.

Besser:

raise RuntimeError(...)

und im Pipeline-Code:

try:
    cache = load_cache()
except RuntimeError as e:
    return f"ERROR: {e}"

==================================================
7. XSS-Härtung im HTML-Renderer
==================================================

Folgende Felder werden laut Review ungefiltert in HTML eingefügt:

- d.wfl
- d.rooms
- d.bj
- d.risk

Vor dem Rendern escapen:

esc(d.wfl)
esc(d.rooms)
esc(d.bj)

Für risk nur erlauben:

low
med
high

Beispiel:

risk = str(d.risk).lower()

if risk not in ("low", "med", "high"):
    risk = "med"

Erst danach im CSS-Classnamen verwenden:

class="risk-badge risk-{risk}"

==================================================
PRIORITÄT
==================================================

1. run.bat Argument-Space-Fix
2. run.bat Gerichtsnummern-Fix
3. API-Key-Leak schließen
4. Connect/Read-Timeouts trennen
5. Fehleranalysen nicht cachen
6. RuntimeError statt SystemExit
7. XSS-Härtung

Nach Umsetzung erneut CodeRabbit ausführen und verbleibende Findings prüfen.
