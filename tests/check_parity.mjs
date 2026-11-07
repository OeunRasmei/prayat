// Checks that the browser code (docs/js) gives the same results as the Python training code.
// Run from the project root:  node tests/check_parity.mjs

import { readFileSync } from "node:fs";
import { buildDetector, classify } from "../docs/js/detector.js";
import { setT2S } from "../docs/js/textprep.js";

const dir = new URL("../docs/model/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("model.json", dir), "utf8"));
const bin = readFileSync(new URL("model.bin", dir));
setT2S(JSON.parse(readFileSync(new URL("t2s.json", dir), "utf8")));
const det = buildDetector(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const reference = JSON.parse(readFileSync(new URL("reference.json", dir), "utf8"));

let normMismatch = 0;
let maxDiff = 0;
for (const r of reference) {
  const out = classify(det, r.text);
  if (out.norm !== r.norm) {
    normMismatch++;
    if (normMismatch <= 3) console.log("normalise mismatch:\n  py:", JSON.stringify(r.norm), "\n  js:", JSON.stringify(out.norm));
  }
  meta.classes.forEach((c, k) => { maxDiff = Math.max(maxDiff, Math.abs(out.probs[c] - r.proba[k])); });
}
console.log(`${reference.length} messages | normalisation mismatches: ${normMismatch} | max probability difference: ${maxDiff.toExponential(2)}`);

const t0 = performance.now();
for (const r of reference) classify(det, r.text);
console.log(`speed: ${((performance.now() - t0) / reference.length).toFixed(2)} ms per message`);
process.exit(normMismatch === 0 && maxDiff < 1e-4 ? 0 : 1);
