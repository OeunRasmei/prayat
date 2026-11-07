"""Text normalisation and character n-grams shared by training and the web app.

docs/js/textprep.js is a line-by-line port of this file. Any change here must be
made there too; the notebook checks that both give identical output.
"""

import json
import re
import unicodedata
from pathlib import Path

# Placeholders (Unicode private-use characters, never typed by people)
URL, EMAIL, PHONE = "", "", ""

T2S = {ord(k): v for k, v in json.loads((Path(__file__).with_name("t2s.json")).read_text(encoding="utf-8")).items()}
KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")

ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
URL_RE = re.compile(
    r"(?:https?://|www\.)[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+"
    r"|[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:com|net|org|info|top|xyz|club|cn|kh|io|ly|me|co|vip|cc|site"
    r"|online|link|app|shop|do|gl|to|biz|asia|store|live)(?:/[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)?"
)
PHONE_RE = re.compile(r"\+?[0-9][0-9 \-()]{6,}[0-9]")
DIGITS_RE = re.compile(r"[0-9]+")
SPACE_RE = re.compile(r"\s+")
CJK_GAP_RE = re.compile(r"([一-鿿]) (?=[一-鿿])")

NGRAM_RANGE = (1, 4)


def _drop_symbols(text):
    """Punctuation and symbols become spaces, except currency signs ($ £ ¥ ៛ ...)."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        out.append(" " if cat[0] in "PS" and cat != "Sc" else ch)
    return "".join(out)


def normalize(text):
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = t.translate(T2S)              # traditional → simplified Chinese
    t = t.translate(KHMER_DIGITS)     # ០-៩ → 0-9
    t = ZERO_WIDTH_RE.sub("", t)
    t = EMAIL_RE.sub(f" {EMAIL} ", t)
    t = URL_RE.sub(f" {URL} ", t)
    t = PHONE_RE.sub(f" {PHONE} ", t)
    t = _drop_symbols(t)
    t = DIGITS_RE.sub("0", t)         # every number becomes 0
    t = SPACE_RE.sub(" ", t).strip()
    t = CJK_GAP_RE.sub(r"\1", t)      # Chinese has no spaces between words
    return t


def char_ngrams(normalized):
    chars = list(normalized)
    lo, hi = NGRAM_RANGE
    return ["".join(chars[i:i + n]) for n in range(lo, hi + 1) for i in range(len(chars) - n + 1)]


def analyze(text):
    return char_ngrams(normalize(text))


def detect_language(text):
    """Script-based: Khmer letters → km, Chinese characters → zh, otherwise en."""
    km = sum("ក" <= c <= "៿" for c in text)
    zh = sum("一" <= c <= "鿿" for c in text)
    if km == 0 and zh == 0:
        return "en"
    return "km" if km >= zh else "zh"
