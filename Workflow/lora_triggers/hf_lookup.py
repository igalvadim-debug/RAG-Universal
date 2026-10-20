import os, re, json, time, urllib.request, urllib.parse, urllib.error

BASE = r"D:\claudeAgent\RAG\workflow\lora_triggers"
ENV = r"D:\claudeAgent\Coder\setting_json\.env"
RAW = os.path.join(BASE, "loras_raw.json")
OUT = os.path.join(BASE, "hf_raw.json")

def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)

def get_token():
    txt = open(ENV, encoding="utf-8").read()
    m = re.search(r"hf_[A-Za-z0-9]{30,}", txt)
    return m.group(0) if m else ""

TOKEN = get_token()

def http(url, as_json=True, timeout=30):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + TOKEN, "User-Agent": "lora-hf-lookup/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                return json.loads(data.decode("utf-8")) if as_json else data.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            if e.code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(2)
    return None

def queries(name):
    stem = re.sub(r"\.(safetensors|pt|pth|ckpt)$", "", name, flags=re.I)
    spaced = re.sub(r"[_\-.\s]+", " ", stem).strip()
    toks = spaced.split()
    qs = [stem, spaced]
    if len(toks) > 3:
        qs.append(" ".join(toks[:3]))
        qs.append(" ".join(toks[:2]))
    out = []
    for q in qs:
        if q and q not in out:
            out.append(q)
    return out

def candidate_repos(name):
    repos = []
    for q in queries(name):
        res = http("https://huggingface.co/api/models?limit=8&search=" + urllib.parse.quote(q))
        for r in res or []:
            rid = r.get("id") or r.get("modelId")
            if rid and rid not in repos:
                repos.append(rid)
        time.sleep(0.2)
    return repos[:16]

def tree_has_sha(repo, sha):
    url = "https://huggingface.co/api/models/%s/tree/main?recursive=1&limit=1000" % repo
    res = http(url)
    for it in res or []:
        lfs = it.get("lfs") or {}
        if lfs.get("oid") == sha:
            return it.get("path")
    return None

def triggers_from_readme(repo):
    txt = http("https://huggingface.co/%s/resolve/main/README.md" % repo, as_json=False)
    if not txt:
        return []
    found = []
    for m in re.finditer(r"(?im)^\s*(?:instance_prompt|trigger_words?|activation_text)\s*:\s*(.+)$", txt):
        found.append(m.group(1).strip().strip("'\""))
    for m in re.finditer(r"(?im)^[\s>*#-]*\**(?:trigger(?:\s*words?)?|activation(?:\s*words?)?|prompt\s*trigger)\**\s*[:\-]\s*(.+)$", txt):
        found.append(m.group(1).strip().strip("*`'\""))
    out = []
    for f in found:
        f = re.sub(r"\s+", " ", f)[:220]
        if f and f not in out:
            out.append(f)
    return out[:4]

def main():
    raw = json.load(open(RAW, encoding="utf-8"))
    targets = [e for e in raw if not (e.get("civitai") or {}).get("modelId")]
    log("%d Dateien ohne CivitAI-Treffer, Token vorhanden: %s" % (len(targets), bool(TOKEN)))
    results = {}
    for i, e in enumerate(targets, 1):
        name = e["file"]
        hit = None
        for repo in candidate_repos(name):
            path = tree_has_sha(repo, e["sha256"])
            time.sleep(0.2)
            if path:
                hit = {"repo": repo, "path": path, "url": "https://huggingface.co/" + repo, "triggers": triggers_from_readme(repo)}
                break
        results[name] = hit
        log("[%d/%d] %s -> %s" % (i, len(targets), name, (hit["repo"] + " " + json.dumps(hit["triggers"], ensure_ascii=False)) if hit else "kein HF-Treffer"))
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("FERTIG")

if __name__ == "__main__":
    main()
