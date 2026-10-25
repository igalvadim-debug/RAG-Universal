import os, sys, json, time, hashlib, struct, urllib.request, urllib.error

BASE = r"D:\claudeAgent\RAG\workflow\lora_triggers"
ENV = r"D:\claudeAgent\Coder\setting_json\.env"
DIRS = [r"D:\ComfyUiCuda\models\loras", r"C:\ai_diffusion\models\loras"]
EXT = (".safetensors", ".pt", ".pth", ".ckpt")
HASH_CACHE = os.path.join(BASE, "hashes.json")
OUT = os.path.join(BASE, "loras_raw.json")

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def load_env():
    d = {}
    with open(ENV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.replace("export ", "").strip()
            d[k] = v.strip().strip('"').strip("'")
    return d

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def st_meta(path):
    if not path.lower().endswith(".safetensors"):
        return {}
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 200 * 1024 * 1024:
                return {}
            hdr = json.loads(f.read(n))
        meta = hdr.get("__metadata__", {}) or {}
        out = {}
        for k in ("modelspec.trigger_phrase", "trigger_word", "trigger_words", "ss_output_name", "modelspec.title", "ss_base_model_version"):
            if k in meta:
                out[k] = meta[k]
        tf = meta.get("ss_tag_frequency")
        if tf:
            try:
                tfd = json.loads(tf) if isinstance(tf, str) else tf
                agg = {}
                for ds in tfd.values():
                    for t, c in ds.items():
                        agg[t] = agg.get(t, 0) + c
                out["top_tags"] = [t for t, _ in sorted(agg.items(), key=lambda x: -x[1])[:6]]
            except Exception:
                pass
        return out
    except Exception:
        return {}

def civitai(hash_, key):
    url = "https://civitai.com/api/v1/model-versions/by-hash/" + hash_
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key, "User-Agent": "lora-trigger-fetch/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            return {"error": "HTTP %d" % e.code}
        except Exception as e:
            time.sleep(3)
    return {"error": "failed"}

def main():
    os.makedirs(BASE, exist_ok=True)
    env = load_env()
    key = env.get("Civitai_API_KEY", "")
    files = []
    for d in DIRS:
        for root, _, names in os.walk(d):
            for n in names:
                if n.lower().endswith(EXT):
                    files.append(os.path.join(root, n))
    files.sort()
    log("%d Dateien gefunden" % len(files))

    cache = {}
    if os.path.exists(HASH_CACHE):
        cache = json.load(open(HASH_CACHE, encoding="utf-8"))
    results = []
    for i, p in enumerate(files, 1):
        size = os.path.getsize(p)
        c = cache.get(p)
        if not c or c.get("size") != size:
            log("[%d/%d] hashe %s (%.1f GB)" % (i, len(files), os.path.basename(p), size / 1e9))
            c = {"size": size, "sha256": sha256_file(p)}
            cache[p] = c
            json.dump(cache, open(HASH_CACHE, "w", encoding="utf-8"))
        entry = {"path": p, "file": os.path.basename(p), "size": size, "sha256": c["sha256"], "meta": st_meta(p)}
        data = civitai(c["sha256"], key)
        if data is None:
            entry["civitai"] = None
        elif "error" in data:
            entry["civitai"] = {"error": data["error"]}
        else:
            m = data.get("model", {}) or {}
            entry["civitai"] = {
                "modelId": data.get("modelId"),
                "modelName": m.get("name"),
                "versionName": data.get("name"),
                "baseModel": data.get("baseModel"),
                "trainedWords": data.get("trainedWords", []),
                "url": "https://civitai.com/models/%s?modelVersionId=%s" % (data.get("modelId"), data.get("id")),
            }
        cv = entry["civitai"]
        log("[%d/%d] %s -> %s" % (i, len(files), entry["file"], ("CIVITAI " + json.dumps(cv.get("trainedWords"))) if cv and "trainedWords" in cv else "kein Treffer"))
        results.append(entry)
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        time.sleep(0.4)
    log("FERTIG")

if __name__ == "__main__":
    main()
