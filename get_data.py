import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA = Path("data")

DATA.mkdir(exist_ok=True)
target = DATA / "scifact.zip"
if not target.exists():
    print("downloading...")
    urllib.request.urlretrieve(URL, target)
with zipfile.ZipFile(target) as z:
    z.extractall(DATA)
print("files:", sorted(p.name for p in (DATA / "scifact").iterdir()))