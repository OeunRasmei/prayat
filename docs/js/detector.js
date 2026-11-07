// Runs the exported Logistic Regression model: TF-IDF of character n-grams → softmax(W·x + b).
// Also explains the answer by spreading each n-gram's contribution back over its characters.

import { EMAIL, PHONE, PHONE_FIND_RE, URL, URL_FIND_RE, charNgrams, normalize, setT2S } from "./textprep.js";

let T2S_MAP = {};

export async function loadDetector(base) {
  const get = async (file, kind) => {
    const r = await fetch(`${base}/${file}`);
    if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
    return r[kind]();
  };
  const [meta, buffer, t2s] = await Promise.all([get("model.json", "json"), get("model.bin", "arrayBuffer"), get("t2s.json", "json")]);
  T2S_MAP = t2s;
  setT2S(t2s);
  return buildDetector(meta, buffer);
}

export function buildDetector(meta, buffer) {
  const f = new Float32Array(buffer);
  const n = meta.n_features;
  return {
    ...meta,
    idf: f.subarray(0, n),
    coef: meta.classes.map((_, k) => f.subarray(n * (k + 1), n * (k + 2))),
    index: new Map(meta.vocab.map((g, i) => [g, i])),
  };
}

export function classify(det, text) {
  const norm = normalize(text);
  const { chars, grams } = charNgrams(norm);

  const found = new Map(); // feature index → occurrences
  for (const g of grams) {
    const j = det.index.get(g.gram);
    if (j === undefined) continue;
    if (!found.has(j)) found.set(j, []);
    found.get(j).push(g);
  }

  // TF-IDF: (1 + ln count) × idf, then L2 normalisation (as scikit-learn with sublinear_tf=True)
  const x = new Map();
  let sumSq = 0;
  for (const [j, occ] of found) {
    const v = (1 + Math.log(occ.length)) * det.idf[j];
    x.set(j, v);
    sumSq += v * v;
  }
  const l2 = Math.sqrt(sumSq) || 1;
  for (const [j, v] of x) x.set(j, v / l2);

  const logits = det.classes.map((_, k) => {
    let s = det.intercept[k];
    for (const [j, v] of x) s += det.coef[k][j] * v;
    return s;
  });
  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));
  const total = exp.reduce((a, b) => a + b, 0);
  const probs = Object.fromEntries(det.classes.map((c, k) => [c, exp[k] / total]));
  const label = det.classes[logits.indexOf(max)];

  const risky = probs.scam >= probs.spam ? "scam" : "spam";
  const flags = x.size ? explain(det, chars, found, x, risky) : [];
  return { text, norm, probs, label, risky, flags };
}

// ---------- explanation ----------

const MARK_RE = /\p{M}/u;
const COENG = "្";

function explain(det, chars, found, x, risky) {
  const k = det.classes.indexOf(risky);
  const kNormal = det.classes.indexOf("normal");
  const score = new Float64Array(chars.length);
  for (const [j, occ] of found) {
    const contribution = x.get(j) * (det.coef[k][j] - det.coef[kNormal][j]);
    for (const g of occ) {
      for (let i = g.start; i < g.start + g.n; i++) score[i] += contribution / occ.length / g.n;
    }
  }
  const top = Math.max(...score);
  if (top <= 0) return [];
  const isLatin = (c) => /[a-z0-9]/.test(c);

  const wholeUnits = (a, b) => {
    // whole English words, whole Khmer clusters (never split a consonant from its vowel signs)
    while (a > 0 && isLatin(chars[a]) && isLatin(chars[a - 1])) a--;
    while (b < chars.length && isLatin(chars[b - 1]) && isLatin(chars[b])) b++;
    while (a > 0 && (MARK_RE.test(chars[a]) || chars[a - 1] === COENG)) a--;
    while (b < chars.length && (MARK_RE.test(chars[b]) || chars[b - 1] === COENG)) b++;
    return [a, b];
  };

  // runs of clearly suspicious characters, grown while neighbours also push towards the risky class
  const runs = [];
  let i = 0;
  while (i < chars.length) {
    if (score[i] < top * 0.3 || chars[i] === " ") { i++; continue; }
    let a = i;
    let b = i + 1;
    while (b < chars.length && chars[b] !== " " && score[b] > 0) b++;
    while (a > 0 && chars[a - 1] !== " " && score[a - 1] > 0) a--;
    const script = /[ក-៿]/.test(chars[i]) ? "km" : /[一-鿿]/.test(chars[i]) ? "zh" : "en";
    while (b - a > MAX_PHRASE[script]) {
      if (score[a] < score[b - 1]) a++;
      else b--;
    }
    [a, b] = wholeUnits(a, b);
    let s = 0;
    for (let t = a; t < b; t++) s += Math.max(0, score[t]);
    runs.push({ a, b, s });
    i = Math.max(b, i + 1);
  }

  const flags = [];
  for (const run of runs.sort((p, q) => q.s - p.s)) {
    const raw = chars.slice(run.a, run.b).join("");
    for (const [ph, kind] of [[URL, "link"], [PHONE, "phone"], [EMAIL, "email"]]) {
      if (raw.includes(ph) && !flags.some((f) => f.kind === kind)) flags.push({ kind, text: kind, weight: run.s });
    }
    const text = raw
      .replace(/[-]/g, " ")
      .replace(/[$£€¥៛]?0[$£€¥៛]?/g, " ") // numbers carry no meaning on their own
      .replace(/\s+/g, " ")
      .trim();
    const letters = Array.from(text.replace(/[\s0-9]/g, ""));
    const latinOnly = /^[a-z\s]+$/.test(text);
    const overlaps = flags.some((f) => f.kind === "text" && (f.text.includes(text) || text.includes(f.text)));
    if (letters.length >= (latinOnly ? 3 : 2) && !overlaps) {
      flags.push({ kind: "text", text, weight: run.s });
    }
    if (flags.length >= 6) break;
  }
  return flags;
}

// Longest phrase shown, in code points (Khmer syllables take several code points each)
const MAX_PHRASE = { km: 22, zh: 10, en: 24 };

// ---------- highlighting the original message ----------

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Returns [{start, end}] ranges in `original` to highlight for the given flags.
export function highlightRanges(original, flags) {
  const ranges = [];
  const lower = Array.from(original.toLowerCase(), (c) => T2S_MAP[c] ?? c).join("");
  const sameLength = lower.length === original.length;
  for (const f of flags) {
    let re;
    if (f.kind === "link") re = new RegExp(URL_FIND_RE.source, "gi");
    else if (f.kind === "phone") re = new RegExp(PHONE_FIND_RE.source, "g");
    else if (f.kind === "email") continue;
    else {
      // numbers became "0", punctuation became spaces, and spaces between Chinese characters were removed
      const cs = Array.from(f.text);
      const isHan = (c) => c >= "一" && c <= "鿿";
      let pattern = "";
      cs.forEach((c, idx) => {
        if (idx && isHan(c) && isHan(cs[idx - 1])) pattern += "[\\s\\p{P}]*";
        pattern += c === "0" ? "[0-9\\u17e0-\\u17e9][0-9\\u17e0-\\u17e9,.]*" : c === " " ? "[\\s\\p{P}\\p{S}]+" : escapeRe(c);
      });
      re = new RegExp(pattern, "gu");
    }
    const target = f.kind === "text" ? lower : original;
    if (f.kind === "text" && !sameLength) continue;
    for (const m of target.matchAll(re)) {
      if (m[0].length) ranges.push({ start: m.index, end: m.index + m[0].length });
    }
  }
  ranges.sort((p, q) => p.start - q.start);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r.start <= last.end) last.end = Math.max(last.end, r.end);
    else merged.push({ ...r });
  }
  return merged;
}
