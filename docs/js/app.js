import { classify, highlightRanges, loadDetector } from "./detector.js";
import { EXAMPLES, STRINGS } from "./i18n.js";
import { cleanOcrText, enhanceForOcr } from "./ocrprep.js";
import { detectLanguage, fixKhmer } from "./textprep.js";

const TESSERACT_URL = "https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js";
const UNSURE_BELOW = 0.6;
const SECOND_PASS = { zh: "chi_sim+eng", km: "khm+eng" };
const LANG_SELF_NAME = { en: "English", km: "ខ្មែរ", zh: "中文" };

const ICONS = {
  lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/>',
  free: '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M16 7l3 3"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1"/><path d="M14 10a4 4 0 0 0-5.66 0l-3 3a4 4 0 0 0 5.66 5.66l1-1"/>',
  phone: '<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2"/>',
  email: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/>',
  scam: '<path d="M12 3.5l9 16H3z"/><path d="M12 10v4M12 17v.01"/>',
  spam: '<path d="M3 11v2a1 1 0 0 0 1 1h3l6 4V6L7 10H4a1 1 0 0 0-1 1z"/><path d="M17 9a4 4 0 0 1 0 6"/>',
  normal: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  unsure: '<path d="M9 9a3 3 0 1 1 4.5 2.6c-.9.5-1.5 1.2-1.5 2.4"/><path d="M12 18v.01"/>',
  tick: '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

const $ = (id) => document.getElementById(id);
const state = {
  ui: "en",
  det: null,
  result: null,
  example: null,
  worker: null,
  workerLangs: null,
  thumbUrl: null,
};

const t = (key) => STRINGS[state.ui][key];
const fill = (text, vars) => text.replace(/\{(\w+)\}/g, (_, k) => vars[k]);
const pct = (p) => (p >= 0.995 ? ">99" : p < 0.005 ? "<1" : String(Math.round(p * 100)));

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "html") node.innerHTML = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k in node && typeof v !== "string") node[k] = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

// ---------- interface language ----------

function pickInitialLanguage() {
  try {
    const saved = localStorage.getItem("prayat-ui");
    if (saved && STRINGS[saved]) return saved;
  } catch { /* storage unavailable */ }
  const nav = (navigator.language || "en").toLowerCase();
  if (nav.startsWith("km")) return "km";
  if (nav.startsWith("zh")) return "zh";
  return "en";
}

function setLanguage(lang) {
  state.ui = lang;
  document.documentElement.lang = lang;
  try { localStorage.setItem("prayat-ui", lang); } catch { /* storage unavailable */ }
  document.querySelectorAll("[data-ui]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ui === lang)));
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => node.setAttribute("aria-label", t(node.dataset.i18nAria)));

  const trustIcons = ["lock", "globe", "image", "free"];
  $("trust").replaceChildren(...t("trust").map((s, i) => el("li", { html: icon(trustIcons[i]) }, s)));

  const signIcons = ["clock", "key", "link"];
  $("signs").replaceChildren(...t("signs").map(([title, body], i) =>
    el("li", {}, el("span", { class: "ico", html: icon(signIcons[i]) }), el("div", {}, el("h3", {}, title), el("p", {}, body)))));

  $("examples").replaceChildren(...EXAMPLES.map((ex, i) => {
    const b = el("button", { type: "button", class: "example", "aria-pressed": String(state.example === i) },
      ex.label[lang], el("small", { lang: ex.lang }, LANG_SELF_NAME[ex.lang]));
    b.addEventListener("click", () => useExample(i));
    return b;
  }));

  $("steps").replaceChildren(...t("steps").map(([title, body]) => el("li", {}, el("h3", {}, title), el("p", {}, body))));
  if (state.det) renderAccuracy();
  if (state.result) renderResult(state.result);
}

// ---------- checking ----------

function check() {
  const text = $("message").value.trim();
  if (!text) {
    $("hint").textContent = t("need_text");
    $("message").focus();
    return;
  }
  if (!state.det) {
    $("hint").textContent = t("loading_model");
    return;
  }
  if ($("hint").textContent === t("need_text")) $("hint").textContent = "";
  state.result = classify(state.det, fixKhmer(text));
  renderResult(state.result, true);
}

function useExample(i) {
  state.example = i;
  $("message").value = EXAMPLES[i].text;
  clearAttachment();
  $("hint").textContent = "";
  document.querySelectorAll(".example").forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));
  check();
}

function verdictOf(r) {
  if (r.label === "normal") return r.probs.normal >= UNSURE_BELOW ? "normal" : "unsure";
  return r.probs[r.label] >= UNSURE_BELOW || r.probs.normal < 0.25 ? r.label : "unsure";
}

function renderResult(r, fresh = false) {
  const verdict = verdictOf(r);
  const tone = verdict === "unsure" ? "spam" : verdict;
  $("empty").hidden = true;
  $("filled").hidden = false;
  $("result").dataset.tone = tone;

  // verdict header
  $("verdict-icon").innerHTML = icon(verdict);
  $("verdict").textContent = t("verdict")[verdict];
  const top = verdict === "unsure" ? Math.max(r.probs.spam, r.probs.scam) : r.probs[verdict];
  $("verdict-sub").replaceChildren(...[
    verdict === "unsure" ? null : fill(t("sure"), { pct: pct(top) }),
    fill(t("lang_detected"), { lang: t("lang_names")[detectLanguage(r.text)] }),
  ].filter(Boolean).map((s) => el("span", {}, s)));

  // message bubble: suspicious phrases underlined, each linked to its reason
  const flags = verdict === "normal" ? [] : r.flags;
  const perFlag = flags.map((f) => highlightRanges(r.text, [f]));
  const merged = highlightRanges(r.text, flags);
  const nodes = [el("span", { class: "stamp-space", "aria-hidden": "true" })];
  let pos = 0;
  for (const { start, end } of merged) {
    if (start > pos) nodes.push(document.createTextNode(r.text.slice(pos, start)));
    const owners = perFlag.flatMap((ranges, i) => (ranges.some((q) => q.start < end && q.end > start) ? [i] : []));
    nodes.push(el("mark", { dataset: { flags: owners.join(" ") } }, r.text.slice(start, end)));
    pos = end;
  }
  nodes.push(document.createTextNode(r.text.slice(pos)));
  $("bubble").replaceChildren(...nodes);
  $("bubble").lang = detectLanguage(r.text);

  const stamp = $("stamp");
  stamp.textContent = t("stamp")[verdict];
  stamp.classList.remove("thud");
  if (fresh) {
    void stamp.offsetWidth; // restart the animation
    stamp.classList.add("thud");
  }

  // reasons
  $("why").hidden = flags.length === 0;
  $("flags").replaceChildren(...flags.map((f, i) => {
    const isText = f.kind === "text";
    const b = el("button", { type: "button", dataset: { kind: f.kind, flag: String(i) }, html: isText ? "" : icon(f.kind) },
      isText ? `“${f.text}”` : t(`flag_${f.kind}`));
    if (isText) b.lang = detectLanguage(f.text);
    const hot = (on) => document.querySelectorAll("#bubble mark").forEach((m) => {
      if (m.dataset.flags.split(" ").includes(String(i))) m.classList.toggle("hot", on);
    });
    b.addEventListener("pointerenter", () => hot(true));
    b.addEventListener("pointerleave", () => hot(false));
    b.addEventListener("focus", () => hot(true));
    b.addEventListener("blur", () => hot(false));
    return el("li", {}, b);
  }));

  // scores
  const best = ["scam", "spam", "normal"].reduce((a, c) => (r.probs[c] > r.probs[a] ? c : a));
  $("scores").replaceChildren(...["scam", "spam", "normal"].map((c) =>
    el("li", { dataset: { c }, class: c === best ? "best" : "" },
      el("span", {}, t("labels")[c]),
      el("span", { class: "track" }, el("span", { class: "fill", style: `width:${(r.probs[c] * 100).toFixed(1)}%` })),
      el("span", { class: "val" }, `${pct(r.probs[c])}%`))));

  $("todo").replaceChildren(...t("todo")[tone].map((s) => el("li", { html: icon("tick") }, el("span", {}, s))));

  if (fresh && window.matchMedia("(max-width: 960px)").matches) {
    $("result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    $("verdict").focus({ preventScroll: true });
  }
}

function reset() {
  $("message").value = "";
  $("hint").textContent = "";
  state.result = null;
  state.example = null;
  document.querySelectorAll(".example").forEach((b) => b.setAttribute("aria-pressed", "false"));
  clearAttachment();
  $("filled").hidden = true;
  $("empty").hidden = false;
  delete $("result").dataset.tone;
}

function renderAccuracy() {
  const det = state.det;
  $("model-lede").textContent = fill(t("model_lede"), { n: det.n_test.toLocaleString("en-US") });
  const card = (name, m, n, total = false) => el("div", { class: `acc-card${total ? " total" : ""}` },
    el("h3", {}, name),
    el("p", { class: "big" }, `${(m["Macro F1"] * 100).toFixed(1)}%`),
    el("p", { class: "big-label" }, t("f1_label")),
    el("dl", {},
      el("dt", {}, t("caught_label")), el("dd", {}, `${(m["Scam recall"] * 100).toFixed(1)}%`),
      el("dt", {}, t("false_label")), el("dd", {}, `${(m["False alarms"] * 100).toFixed(1)}%`)),
    el("p", { class: "n" }, fill(t("n_label"), { n: n.toLocaleString("en-US") })));
  $("acc-cards").replaceChildren(
    ...["en", "km", "zh"].map((l) => card(t("lang_names")[l], det.per_language[l], det.per_language[l].n)),
    card(t("all_langs"), det.metrics, det.n_test, true),
  );
}

// ---------- screenshots (OCR, in the browser) ----------

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (window.Tesseract) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.append(s);
  });
}

function ocrStatus(text, progress = null) {
  $("ocr-status").textContent = text;
  $("progress").hidden = progress === null;
  if (progress !== null) $("progress").firstElementChild.style.width = `${Math.round(progress * 100)}%`;
}

function clearAttachment() {
  $("attachment").hidden = true;
  if (state.thumbUrl) URL.revokeObjectURL(state.thumbUrl);
  state.thumbUrl = null;
  $("file").value = "";
}

async function getWorker(langs) {
  await loadScript(TESSERACT_URL);
  const logger = (m) => {
    if (m.status === "recognizing text") ocrStatus(fill(t("ocr_reading"), { pct: Math.round(m.progress * 100) }), m.progress);
    else ocrStatus(t("ocr_loading"), m.progress ?? 0);
  };
  if (!state.worker) {
    state.worker = await Tesseract.createWorker(langs.split("+"), 1, { logger });
  } else if (state.workerLangs !== langs) {
    await state.worker.reinitialize(langs.split("+"));
  }
  state.workerLangs = langs;
  return state.worker;
}

// Upscale small screenshots: Tesseract reads text best at roughly 30 px per line.
async function prepareImage(file) {
  const bitmap = await createImageBitmap(file);
  const scale = bitmap.width < 1100 ? Math.min(2.5, 2200 / bitmap.width) : 1;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  enhanceForOcr(img);
  ctx.putImageData(img, 0, 0);
  return canvas;
}

async function readScreenshot(file) {
  if (!file || !file.type.startsWith("image/")) return;
  clearAttachment();
  state.example = null;
  document.querySelectorAll(".example").forEach((b) => b.setAttribute("aria-pressed", "false"));
  state.thumbUrl = URL.createObjectURL(file);
  $("thumb").src = state.thumbUrl;
  $("attachment").hidden = false;
  $("hint").textContent = "";
  ocrStatus(t("ocr_loading"), 0);
  try {
    const langs = $("ocr-lang").value;
    let [worker, image] = await Promise.all([getWorker(langs), prepareImage(file)]);
    let { data } = await worker.recognize(image);
    let text = cleanOcrText(data.text || "");
    // In automatic mode, read again with the detected script as the main language:
    // Tesseract reads mixed screenshots much better when the main language is right.
    const better = langs.includes("+") && text ? SECOND_PASS[detectLanguage(text)] : null;
    if (better) {
      worker = await getWorker(better);
      ({ data } = await worker.recognize(image));
      text = cleanOcrText(data.text || "") || text;
    }
    if (!text) {
      ocrStatus(t("ocr_none"));
      return;
    }
    ocrStatus(t("ocr_done"));
    $("message").value = text;
    check();
  } catch (err) {
    console.error(err);
    ocrStatus(t("ocr_error"));
  }
}

// ---------- wiring ----------

document.querySelectorAll("[data-ui]").forEach((b) => b.addEventListener("click", () => setLanguage(b.dataset.ui)));
$("check").addEventListener("click", check);
$("clear").addEventListener("click", () => { reset(); $("message").focus(); });
$("another").addEventListener("click", () => {
  reset();
  $("message").focus({ preventScroll: true });
  $("message").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
});
$("message").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) check();
});
$("message").addEventListener("input", () => {
  if (state.example !== null) {
    state.example = null;
    document.querySelectorAll(".example").forEach((b) => b.setAttribute("aria-pressed", "false"));
  }
});
$("file").addEventListener("change", (e) => readScreenshot(e.target.files[0]));
$("remove-image").addEventListener("click", clearAttachment);

// Drag a screenshot onto the message box (or anywhere on the page).
const composer = $("composer");
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  dragDepth++;
  composer.classList.add("over");
  $("drop-overlay").hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) { composer.classList.remove("over"); $("drop-overlay").hidden = true; }
});
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  composer.classList.remove("over");
  $("drop-overlay").hidden = true;
  readScreenshot(e.dataTransfer.files[0]);
});
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) {
    e.preventDefault();
    readScreenshot(item.getAsFile());
  }
});

setLanguage(pickInitialLanguage());
$("status").textContent = t("loading_model");
loadDetector("model")
  .then((det) => {
    state.det = det;
    $("status").textContent = "";
    renderAccuracy();
    if ($("message").value.trim()) check();
  })
  .catch((err) => {
    console.error(err);
    $("status").textContent = t("model_error");
    $("status").classList.add("is-error");
  });
