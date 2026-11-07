"""Download the three public datasets into data/raw/.

Run:  python data/download_data.py
Please cite the original authors (see README.md) if you reuse the data.
"""

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).with_name("raw")

SOURCES = {
    # Mishra & Soni (2022), SMS Phishing Dataset for Machine Learning and Pattern Recognition, Mendeley Data
    "english": "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/edb361de-918d-469f-9106-e84823830665/file_downloaded",
    # Zhang et al. (2020), Lies in the Air: Characterizing Fake-base-station Spam Ecosystem in China, ACM CCS
    "chinese_fbs": "https://github.com/Cypher-Z/FBS_SMS_Dataset/archive/refs/heads/master.zip",
    # Chen & Kan (2013), Creating a Live, Public Short Message Service Corpus: The NUS SMS Corpus
    "chinese_nus": "https://raw.githubusercontent.com/kite1988/nus-sms-corpus/master/smsCorpus_zh_xml_2015.03.09.zip",
}


def fetch_zip(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (prayat-student-project)"})
    with urllib.request.urlopen(request) as r:
        return zipfile.ZipFile(io.BytesIO(r.read()))


def main():
    RAW.mkdir(exist_ok=True)

    with fetch_zip(SOURCES["english"]) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        (RAW / "en_sms_phishing_5971.csv").write_bytes(z.read(name))

    fbs = RAW / "zh_fbs"
    shutil.rmtree(fbs, ignore_errors=True)
    fbs.mkdir()
    with fetch_zip(SOURCES["chinese_fbs"]) as z:
        for name in z.namelist():
            base = name.split("/", 1)[-1]
            if base and "/" not in base and not name.endswith("/"):
                (fbs / base).write_bytes(z.read(name))

    with fetch_zip(SOURCES["chinese_nus"]) as z:
        name = next(n for n in z.namelist() if n.endswith(".xml"))
        (RAW / "zh_nus_sms.xml").write_bytes(z.read(name))

    for p in sorted(RAW.rglob("*")):
        if p.is_file():
            print(f"{p.relative_to(RAW)}: {p.stat().st_size / 1024:,.0f} KB")


if __name__ == "__main__":
    main()
