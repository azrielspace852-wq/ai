import json, os, sys, zipfile, urllib.request

DATASET_DIR = 'downloads/datasets'
MODEL_DIR   = 'downloads/model'

def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def safe_extract(zip_path, target):
    with zipfile.ZipFile(zip_path, 'r') as z:
        abs_t = os.path.abspath(target)
        for member in z.namelist():
            dest = os.path.abspath(os.path.join(target, member))
            if not (dest == abs_t or dest.startswith(abs_t + os.sep)):
                print(f"[warn] unsafe path skipped: {member}")
                continue
            z.extract(member, target)

def extract_recursive(initial_zips, target_root, max_depth=10):
    """Antisipasi zip bersarang: BFS + visited set + depth guard."""
    os.makedirs(target_root, exist_ok=True)
    queue   = [(os.path.abspath(p), 0) for p in initial_zips]
    visited = set()
    counter = 0
    while queue:
        zpath, depth = queue.pop(0)
        if zpath in visited:
            continue
        visited.add(zpath)
        if depth > max_depth:
            print(f"[warn] max depth reached: {zpath}")
            continue
        try:
            subdir = os.path.join(target_root, f"_x{counter:04d}")
            counter += 1
            os.makedirs(subdir, exist_ok=True)
            safe_extract(zpath, subdir)
            for root, _, files in os.walk(subdir):
                for f in files:
                    full = os.path.join(root, f)
                    if f.lower().endswith('.zip'):
                        queue.append((full, depth + 1))
        except Exception as e:
            print(f"[error] extract {zpath}: {e}")

def find_files(root, ext):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ext):
                out.append(os.path.join(r, f))
    return out

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'pretrain'
    if mode not in ('pretrain', 'finetune'):
        sys.exit(f"invalid mode: {mode}")

    if not os.path.exists('link.json'):
        sys.exit("link.json tidak ditemukan di root repo")

    cfg          = json.load(open('link.json'))
    ds_urls      = cfg.get('link-dataset', []) or []
    model_url    = cfg.get('link-model', '')  or ''
    if isinstance(ds_urls, str):
        ds_urls = [ds_urls]

    if not ds_urls:
        sys.exit("'link-dataset' kosong di link.json")
    if mode == 'finetune' and not model_url:
        sys.exit("mode finetune wajib mengisi 'link-model'")

    # --- datasets ---
    os.makedirs(DATASET_DIR, exist_ok=True)
    zips = []
    for i, url in enumerate(ds_urls):
        p = os.path.join(DATASET_DIR, f"src_{i}.zip")
        try:
            download(url, p); zips.append(p)
        except Exception as e:
            sys.exit(f"[error] download {url}: {e}")

    extract_recursive(zips, os.path.join(DATASET_DIR, "_extracted"))

    jsonl = find_files(DATASET_DIR, ".jsonl")
    if not jsonl:
        sys.exit("tidak ada .jsonl di dalam zip dataset")
    open('jsonl_files.txt', 'w').write('\n'.join(jsonl))
    print(f"[ok] {len(jsonl)} jsonl:")
    for p in jsonl: print("  -", p)

    # --- model ---
    if mode == 'finetune':
        os.makedirs(MODEL_DIR, exist_ok=True)
        zp = os.path.join(MODEL_DIR, "src.zip")
        download(model_url, zp)
        extract_recursive([zp], os.path.join(MODEL_DIR, "_extracted"))

        bins = find_files(MODEL_DIR, ".bin")
        if not bins:
            sys.exit("tidak ada file .bin di zip model")
        pref = [b for b in bins if os.path.basename(b).lower() == 'model.bin']
        chosen = pref[0] if pref else bins[0]
        open('model_path.txt', 'w').write(chosen)
        print(f"[ok] model: {chosen}")
    else:
        if os.path.exists('model_path.txt'):
            os.remove('model_path.txt')

if __name__ == '__main__':
    main()
