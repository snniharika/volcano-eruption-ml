from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
DATA.mkdir(parents=True, exist_ok=True)

FILES = {
    "puuoo_earthquakes.csv": "https://raw.githubusercontent.com/bmullet/PEEVED/master/puuoo_earthquakes.csv",
    "PuuOo.csv": "https://raw.githubusercontent.com/bmullet/PEEVED/master/PuuOo.csv",
}

for name, url in FILES.items():
    destination = DATA / name
    print(f"Downloading {name} ...")
    urlretrieve(url, destination)
    print(f"Saved: {destination}")

print("\nData download complete.")
