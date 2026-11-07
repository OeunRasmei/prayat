import { classify, highlightRanges, loadDetector } from "./detector.js";
import { EXAMPLES, STRINGS } from "./i18n.js";
import { detectLanguage } from "./textprep.js";

const TESSERACT_URL = "https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js";
const UNSURE_BELOW = 0.6;
const SECOND_PASS = { zh: "chi_sim+eng", km: "khm+eng" };

const $ = (id) => document.getElementById(id);
const state = {
  ui: "en",
  det: null,
  result: null,
  exampleIndex: { en: 0, km: 0, zh: 0 },
  worker: null,
  workerLangs: null,
};

const t = (key) => STRINGS[state.ui][key];
const fill = (text, vars) => text.replace(/\{(\w+)\}/g, (_, k) => vars[k]);
const pct = (p) => (p >= 0.995 ? ">99" : p < 0.005 ? "<1" : String(Math.round(p * 100)));

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
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  $("tips").replaceChildren(...t("tips").map((s) => li(s)));
  $("steps").replaceChildren(...t("steps").map(([title, body]) => {
    const item = document.createElement("li");
    const h = document.createElement("h3");
    h.textContent = title;
    const p = document.createElement("p");
    p.textContent = body;
    item.append(h, p);
    return item;
  }));
  if (state.det) renderAccuracy();
  if (state.result) renderResult(state.result);
}

function li(text) {
  const el = document.createElement("li");
  el.textContent = text;
  return el;
}

// ---------- tabs ----------

function selectTab(which) {
  const isText = which === "text";
  $("tab-text").setAttribute("aria-selected", String(isText));
  $("tab-image").setAttribute("aria-selected", String(!isText));
  $("tab-text").tabIndex = isText ? 0 : -1;
  $("tab-image").tabIndex = isText ? -1 : 0;
  $("pane-text").hidden = !isText;
  $("pane-image").hidden = isText;
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
  $("hint").textContent = "";
  state.result = classify(state.det, text);
  renderResult(state.result, true);
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

  // message bubble with suspicious parts underlined
  const bubble = $("bubble");
  const showFlags = verdict !== "normal";
  const ranges = showFlags ? highlightRanges(r.text, r.flags) : [];
  const nodes = [];
  let pos = 0;
  for (const { start, end } of ranges) {
    if (start > pos) nodes.push(document.createTextNode(r.text.slice(pos, start)));
    const mark = document.createElement("mark");
    mark.textContent = r.text.slice(start, end);
    nodes.push(mark);
    pos = end;
  }
  nodes.push(document.createTextNode(r.text.slice(pos)));
  bubble.replaceChildren(...nodes);
  bubble.lang = detectLanguage(r.text);

  const stamp = $("stamp");
  stamp.textContent = t("stamp")[verdict];
  stamp.classList.remove("thud");
  if (fresh) {
    void stamp.offsetWidth; // restart the animation
    stamp.classList.add("thud");
  }

  const top = verdict === "unsure" ? Math.max(r.probs.spam, r.probs.scam) : r.probs[verdict];
  $("verdict").textContent = t("verdict")[verdict];
  $("verdict-sub").replaceChildren(...[
    verdict === "unsure" ? null : fill(t("sure"), { pct: pct(top) }),
    fill(t("lang_detected"), { lang: t("lang_names")[detectLanguage(r.text)] }),
  ].filter(Boolean).map((s) => {
    const span = document.createElement("span");
    span.textContent = s;
    return span;
  }));

  const segs = $("bar").children;
  ["normal", "spam", "scam"].forEach((c, k) => { segs[k].style.width = `${(r.probs[c] * 100).toFixed(2)}%`; });
  $("legend").replaceChildren(...["scam", "spam", "normal"].map((c) => {
    const item = document.createElement("li");
    item.dataset.c = c;
    item.textContent = `${t("labels")[c]} ${pct(r.probs[c])}%`;
    return item;
  }));

  const flags = showFlags ? r.flags : [];
  $("why").hidden = flags.length === 0;
  $("flags").replaceChildren(...flags.map((f) => {
    const item = document.createElement("li");
    item.dataset.kind = f.kind;
    if (f.kind === "text") {
      item.textContent = `“${f.text}”`;
      item.lang = detectLanguage(f.text);
    } else {
      item.textContent = t(`flag_${f.kind}`);
    }
    return item;
  }));

  $("todo").replaceChildren(...t("todo")[verdict === "unsure" ? "spam" : verdict].map((s) => li(s)));
  if (fresh && window.matchMedia("(max-width: 900px)").matches) {
    $("verdict").focus({ preventScroll: true });
    $("result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }
}

function renderAccuracy() {
  const det = state.det;
  $("model-lede").textContent = fill(t("model_lede"), { n: det.n_test.toLocaleString("en-US") });
  const row = (name, m, n, isTotal = false) => {
    const tr = document.createElement("tr");
    if (isTotal) tr.className = "total";
    const cells = [name, n.toLocaleString("en-US"), `${(m["Macro F1"] * 100).toFixed(1)}%`,
      `${(m["Scam recall"] * 100).toFixed(1)}%`, `${(m["False alarms"] * 100).toFixed(1)}%`];
    cells.forEach((v, i) => {
      const td = document.createElement(i === 0 ? "th" : "td");
      if (i === 0) td.scope = "row";
      else td.className = "num";
      td.textContent = v;
      tr.append(td);
    });
    return tr;
  };
  $("acc-body").replaceChildren(
    ...["en", "km", "zh"].map((l) => row(t("lang_names")[l], det.per_language[l], det.per_language[l].n)),
    row(t("all_langs"), det.metrics, det.n_test, true),
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
  return canvas;
}

function cleanOcrText(text) {
  return text
    .replace(/([一-鿿])[ \t]+(?=[一-鿿])/g, "$1") // OCR puts spaces between Chinese characters
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function readScreenshot(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectTab("image");
  const thumb = $("thumb");
  thumb.src = URL.createObjectURL(file);
  thumb.hidden = false;
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
    selectTab("text");
    $("hint").textContent = t("ocr_done");
    check();
  } catch (err) {
    console.error(err);
    ocrStatus(t("ocr_error"));
  }
}

// ---------- wiring ----------

document.querySelectorAll("[data-ui]").forEach((b) => b.addEventListener("click", () => setLanguage(b.dataset.ui)));
$("tab-text").addEventListener("click", () => selectTab("text"));
$("tab-image").addEventListener("click", () => selectTab("image"));
document.querySelector(".tabs").addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const next = $("tab-text").getAttribute("aria-selected") === "true" ? "image" : "text";
    selectTab(next);
    $(`tab-${next}`).focus();
  }
});
$("check").addEventListener("click", check);
$("clear").addEventListener("click", () => {
  $("message").value = "";
  $("hint").textContent = "";
  state.result = null;
  $("filled").hidden = true;
  $("empty").hidden = false;
  delete $("result").dataset.tone;
  $("message").focus();
});
$("message").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) check();
});
document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
  const lang = b.dataset.example;
  const list = EXAMPLES[lang];
  $("message").value = list[state.exampleIndex[lang] % list.length];
  state.exampleIndex[lang]++;
  selectTab("text");
  check();
}));

$("file").addEventListener("change", (e) => readScreenshot(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
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
