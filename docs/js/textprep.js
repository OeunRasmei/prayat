// Port of src/textprep.py. Training and the browser must normalise text identically;
// tests/check_parity.mjs compares the two on the reference messages.

export const URL = "";
export const EMAIL = "";
export const PHONE = "";

let T2S = {};
export function setT2S(map) {
  T2S = map;
}

const ZERO_WIDTH_RE = /[​‌‍⁠﻿️]/g;
// Emoji stay as features (🎁 🎉 💰 ⚠️ are common in promotions and scams); other symbols become spaces
const EMOJI_RE = /[⌀-⏿☀-➿⬀-⯿\u{1F000}-\u{1FAFF}]/u;

// Khmer is often typed (and read by OCR) in several Unicode spellings that look the same on screen.
const KHMER_FIXES = [
  [/េី/g, "ើ"], // េ + ី → ើ
  [/េា/g, "ោ"], // េ + ា → ោ
  [/ំុ/g, "ុំ"], // ំ before ុ → ុំ
  [/ឱ្យ/g, "ឲ្យ"], // ឱ្យ and ឲ្យ are the same word
  [/ឣ/g, "អ"], // deprecated independent vowels
  [/ឤ/g, "អា"],
  [/្រ្([ក-ឣ])/g, "្$1្រ"], // subscript ro comes after the other subscript
];

export function fixKhmer(text) {
  for (const [pattern, repl] of KHMER_FIXES) text = text.replace(pattern, repl);
  return text;
}
const EMAIL_RE = /[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}/g;
const URL_RE = new RegExp(
  "(?:https?://|www\\.)[a-z0-9\\-._~:/?#\\[\\]@!$&'()*+,;=%]+" +
    "|[a-z0-9][a-z0-9\\-]*(?:\\.[a-z0-9\\-]+)*\\.(?:com|net|org|info|top|xyz|club|cn|kh|io|ly|me|co|vip|cc|site" +
    "|online|link|app|shop|do|gl|to|biz|asia|store|live)(?:/[a-z0-9\\-._~:/?#\\[\\]@!$&'()*+,;=%]*)?",
  "g",
);
const PHONE_RE = /\+?[0-9][0-9 \-()]{6,}[0-9]/g;
const DIGITS_RE = /[0-9]+/g;
const SPACE_RE = /[\s\x1c-\x1f\x85]+/g;
const CJK_GAP_RE = /([一-鿿]) (?=[一-鿿])/g;
const SYMBOLS_RE = /[\p{P}\p{Sm}\p{Sk}\p{So}]/gu; // punctuation and symbols, but not currency (Sc)
const KHMER_DIGIT_RE = /[០-៩]/g;

export const NGRAM_RANGE = [1, 4];

export function normalize(text) {
  let t = fixKhmer(text.normalize("NFKC"));
  t = t.toLowerCase();
  t = Array.from(t, (c) => T2S[c] ?? c).join("");
  t = t.replace(KHMER_DIGIT_RE, (c) => String(c.charCodeAt(0) - 0x17e0));
  t = t.replace(ZERO_WIDTH_RE, "");
  t = t.replace(EMAIL_RE, ` ${EMAIL} `);
  t = t.replace(URL_RE, ` ${URL} `);
  t = t.replace(PHONE_RE, ` ${PHONE} `);
  t = t.replace(SYMBOLS_RE, (c) => (EMOJI_RE.test(c) ? c : " "));
  t = t.replace(DIGITS_RE, "0");
  t = t.replace(SPACE_RE, " ").trim();
  t = t.replace(CJK_GAP_RE, "$1");
  return t;
}

// Every n-gram with its start position (in code points), so explanations can point at text.
export function charNgrams(normalized) {
  const chars = Array.from(normalized);
  const [lo, hi] = NGRAM_RANGE;
  const grams = [];
  for (let n = lo; n <= hi; n++) {
    for (let i = 0; i + n <= chars.length; i++) grams.push({ gram: chars.slice(i, i + n).join(""), start: i, n });
  }
  return { chars, grams };
}

export function detectLanguage(text) {
  let km = 0;
  let zh = 0;
  for (const c of text) {
    if (c >= "ក" && c <= "៿") km++;
    else if (c >= "一" && c <= "鿿") zh++;
  }
  if (km === 0 && zh === 0) return "en";
  return km >= zh ? "km" : "zh";
}

// Case-insensitive versions for finding links and phone numbers in the original message.
export const URL_FIND_RE = new RegExp(URL_RE.source, "gi");
export const PHONE_FIND_RE = new RegExp(PHONE_RE.source, "g");
