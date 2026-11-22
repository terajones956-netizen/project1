import csv, io, zipfile, urllib.request
import pandas as pd

URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"   # original LIAR download
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))

frames = []
for name in z.namelist():
    if name.endswith((".tsv")) and any(k in name for k in ("train", "valid", "test")):
        frames.append(pd.read_csv(z.open(name), sep="\t", header=None, usecols=[0, 1, 2],
                                  names=["id", "label", "statement"], quoting=csv.QUOTE_NONE))
df = pd.concat(frames, ignore_index=True)
print(df.label.value_counts())
df[["statement", "label"]].to_csv(r"data\politifact.csv", index=False)
print("Wrote data\\politifact.csv:", len(df), "rows")