// Helpers for reading screenshots: clean up the picture before OCR, and the text after it.
// Pure functions (no DOM), so tests/check_ocr.mjs can run them in Node.

import { fixKhmer } from "./textprep.js";

// Grey scale, stretch the contrast, and turn dark-mode screenshots (light text on a dark
// background) into dark text on a light background, which Tesseract reads far better.
// `img` is an ImageData-like object {data, width, height}; it is changed in place.
export function enhanceForOcr(img) {
  const { data } = img;
  const n = data.length / 4;
  const grey = new Uint8ClampedArray(n);
  const hist = new Uint32Array(256);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const v = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    grey[p] = v;
    hist[v]++;
  }
  const percentile = (q) => {
    let seen = 0;
    for (let v = 0; v < 256; v++) {
      seen += hist[v];
      if (seen >= q * n) return v;
    }
    return 255;
  };
  const lo = percentile(0.01);
  const hi = percentile(0.99);
  const dark = percentile(0.5) < 100;
  const range = Math.max(hi - lo, 1);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    let v = hi - lo < 40 ? grey[p] : ((grey[p] - lo) * 255) / range;
    if (dark) v = 255 - v;
    data[i] = data[i + 1] = data[i + 2] = v;
  }
  return dark;
}

const LETTER_RE = /[\p{L}\p{M}]/gu;
const LEADING_NOISE_RE = /^[\p{P}\p{S}ៈ-៖\s]+/u;
const SYMBOL_TOKEN_RE = /^[\p{P}\p{Sm}\p{Sk}\p{So}]+$/u; // "@", "&", "|" left behind by emoji and icons
const STATUS_BAR_RE = /^\D{0,3}\d{1,2}:\d{2}\b/; // the phone's clock line: "‹ 12:47 Smart 📶"

// Screenshots also contain clock times, icons and emoji, which OCR turns into short junk
// such as "<< ឃ 2 ឃ J ng". Drop those lines and stray characters at the start of a line.
export function cleanOcrText(text) {
  const lines = text.split("\n").map((line) => {
    let tokens = line.trim().split(/[ \t]+/).filter((tok) => tok && !SYMBOL_TOKEN_RE.test(tok));
    while (tokens.length > 2 && Array.from(tokens[0]).length === 1) tokens = tokens.slice(1);
    return tokens.join(" ").replace(LEADING_NOISE_RE, "");
  });
  const kept = lines.filter((line, i) => {
    if (!line) return true; // keep paragraph breaks
    if (i === 0 && STATUS_BAR_RE.test(line)) return false;
    const letters = (line.match(LETTER_RE) || []).length;
    if (letters < 2) return false;
    const tokens = line.split(" ");
    const allShort = tokens.length >= 3 && tokens.every((tok) => Array.from(tok).length <= 2);
    return !allShort;
  });
  return fixKhmer(kept.join("\n"))
    .replace(/([一-鿿])[ \t]+(?=[一-鿿])/g, "$1") // OCR puts spaces between Chinese characters
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
