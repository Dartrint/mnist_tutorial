// Headless smoke test of the web UI: loads the real page from a running server
// with jsdom, stubs the 2D canvas, then drives the UI like a user would.
//
//   npm install jsdom
//   python -m mnist serve --port 8123 --quiet &
//   node scripts/frontend_smoke.mjs http://127.0.0.1:8123 my_digit.png 7
//
// The PNG stands in for whatever the user draws on the canvas (jsdom has no
// real 2D canvas, so toDataURL() is stubbed to return it).
import { readFileSync } from "node:fs";
import { JSDOM, VirtualConsole } from "jsdom";

const base = process.argv[2] || "http://127.0.0.1:8123";
// Third argument: a PNG file OR a file containing a data URL (what the canvas would send).
const rawStub = readFileSync(process.argv[3] || "digit.png");
const digitDataUrl = rawStub.subarray(0, 5).toString() === "data:"
  ? rawStub.toString("utf8").trim()
  : "data:image/png;base64," + rawStub.toString("base64");
const expectedDigit = (process.argv[4] || "7").trim();

const failures = [];
const check = (name, ok, extra = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  — " + extra : ""}`);
  if (!ok) failures.push(name);
};

function makeContext() {
  const noop = () => {};
  return {
    strokeStyle: "", fillStyle: "", lineWidth: 1, lineCap: "round", lineJoin: "round",
    imageSmoothingEnabled: true, globalAlpha: 1,
    clearRect: noop, fillRect: noop, beginPath: noop, closePath: noop,
    moveTo: noop, lineTo: noop, stroke: noop, fill: noop, arc: noop,
    save: noop, restore: noop, translate: noop, scale: noop, drawImage: noop,
    getImageData: () => ({ data: new Uint8ClampedArray(280 * 280 * 4) }),
    putImageData: noop,
  };
}

const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (err) => console.log("  [jsdom error]", err.message));

const dom = await JSDOM.fromURL(base + "/", {
  runScripts: "dangerously",
  resources: "usable",
  pretendToBeVisual: true,
  virtualConsole,
  beforeParse(window) {
    // jsdom implements neither fetch nor canvas; supply both.
    // Node's fetch needs absolute URLs; browsers resolve them against the page.
    window.fetch = (input, init) =>
      globalThis.fetch(typeof input === "string" ? new URL(input, base) : input, init);
    window.HTMLCanvasElement.prototype.getContext = makeContext;
    window.HTMLCanvasElement.prototype.toDataURL = () => digitDataUrl;
    window.HTMLCanvasElement.prototype.getBoundingClientRect = () => ({
      left: 0, top: 0, width: 280, height: 280, right: 280, bottom: 280,
    });
    window.Element.prototype.scrollIntoView = () => {};
  },
});

const { window } = dom;
window.addEventListener("error", (e) => console.log("  [window error]", e.message));
const $ = (id) => window.document.getElementById(id);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const waitFor = async (fn, timeout = 15000, step = 100) => {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (fn()) return true;
    await wait(step);
  }
  return false;
};

// --- wait for DOMContentLoaded + the initial /api/models call -------------
// jsdom may fire DOMContentLoaded before a deferred external script has run;
// a real browser never does, so nudge it here if init() has not executed.
await waitFor(() => $("prob-chart") && $("prob-chart").children.length > 0, 5000, 50);
if ($("prob-chart").children.length === 0) {
  window.document.dispatchEvent(new window.Event("DOMContentLoaded", { bubbles: true }));
}
await waitFor(() => $("model-select") && $("model-select").options.length > 0);
check("model <select> populated from /api/models", $("model-select").options.length > 0,
  `${$("model-select").options.length} option(s)`);
check("selected model name is shown",
  $("model-select").options.length > 0 && $("model-select").value.length > 0,
  $("model-select").value);
check("no external resources referenced",
  !/https?:\/\//.test(window.document.documentElement.outerHTML.replace(base, "")),
  "index.html is self-contained");

// --- draw a stroke on the canvas and let the debounce fire ----------------
const canvas = $("draw-canvas");
const down = new window.MouseEvent("mousedown", { clientX: 40, clientY: 40, bubbles: true });
const move = new window.MouseEvent("mousemove", { clientX: 120, clientY: 160, bubbles: true });
const up = new window.MouseEvent("mouseup", { clientX: 120, clientY: 160, bubbles: true });
canvas.dispatchEvent(down);
canvas.dispatchEvent(move);
canvas.dispatchEvent(up);

const gotPrediction = await waitFor(() => $("digit-badge").textContent.trim() === expectedDigit);
check("auto-predict after drawing returns the right digit", gotPrediction,
  `badge = "${$("digit-badge").textContent}" (expected ${expectedDigit})`);
check("confidence rendered as a percentage", /%/.test($("confidence-value").textContent),
  $("confidence-value").textContent);
check("preview image of the 28x28 input is shown",
  ($("preview-img").getAttribute("src") || "").startsWith("data:image/png"),
  ($("preview-img").getAttribute("src") || "").slice(0, 30) + "…");

const bars = $("prob-chart").querySelectorAll("*");
const barCount = $("prob-chart").children.length;
check("probability chart has 10 class rows", barCount === 10, `${barCount} row(s)`);
check("probability chart rendered content", bars.length > 0, `${bars.length} node(s)`);

// --- explicit predict button + clear button ------------------------------
$("btn-predict").click();
await wait(1200);
check("manual 'Dự đoán' button keeps the prediction",
  $("digit-badge").textContent.trim() === expectedDigit, $("digit-badge").textContent);

$("btn-clear").click();
await wait(200);
check("'Xoá' resets the result panel", $("digit-badge").textContent.trim() === "–",
  `badge = "${$("digit-badge").textContent}"`);

// --- random test samples -------------------------------------------------
$("btn-samples").click();
const gotSamples = await waitFor(
  () => $("samples-grid").children.length >= 8 || /is-error/.test($("status-line").className),
  40000, 200);
check("'Lấy ảnh test' renders 8 sample thumbnails", gotSamples,
  `${$("samples-grid").children.length} thumbnail(s), status = "${$("status-line").textContent.trim()}"`);

// --- evaluation panel ----------------------------------------------------
$("btn-evaluate").click();
const gotEval = await waitFor(
  () => $("confusion").querySelectorAll("*").length >= 100 ||
        /is-error/.test($("status-line").className),
  90000, 250);
check("'Đánh giá model' renders summary stats", $("eval-summary").children.length >= 3,
  `${$("eval-summary").children.length} stat(s)`);
check("per-class accuracy bars rendered", $("per-class").children.length > 0,
  `${$("per-class").children.length} node(s)`);
const matrixCells = $("confusion").querySelectorAll("*").length;
check("confusion matrix rendered", gotEval && matrixCells >= 100, `${matrixCells} cell node(s)`);

check("no error status left on screen",
  !/error/i.test($("status-line").className || "") ||
    $("status-line").textContent.trim() === "",
  `status = "${$("status-line").textContent.trim()}"`);

console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nAll frontend checks passed");
window.close();
process.exit(failures.length ? 1 : 0);
