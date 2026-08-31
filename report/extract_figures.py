import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "notebooks", "projet_tarification_anomalies_MAIRLOT_Antony.ipynb")
OUT_DIR = os.path.join(ROOT, "report", "figures")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

os.makedirs(OUT_DIR, exist_ok=True)

count = 0
manifest = []
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    first_line = source.strip().split("\n")[0][:80] if source.strip() else ""
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            count += 1
            fname = f"fig_{count:02d}_cell{idx}.png"
            fpath = os.path.join(OUT_DIR, fname)
            with open(fpath, "wb") as img_f:
                img_f.write(base64.b64decode(data["image/png"]))
            manifest.append((fname, idx, first_line))

print(f"Extracted {count} images to {OUT_DIR}")
for fname, idx, first_line in manifest:
    print(f"{fname}\tcell {idx}\t{first_line}")
