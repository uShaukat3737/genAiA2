"use strict";

const LIVE_INTERVAL_MS = 300;
const $ = (id) => document.getElementById(id);
const els = {
  file: $("file"), camera: $("camera"), snap: $("snap"), live: $("live"), mode: $("mode"),
  status: $("status"), video: $("video"), input: $("input"), output: $("output"),
  inputLabel: $("input-label"), outputLabel: $("output-label"), canvas: $("canvas"),
};
const LABELS = { photo2sketch: "Photo → Sketch", sketch2photo: "Sketch → Photo" };
let stream = null;
let busy = false;

function setStatus(text) { els.status.textContent = text; }

async function convert(blob) {
  const form = new FormData();
  form.append("image", blob, "input.png");
  form.append("mode", els.mode.value);
  const resp = await fetch("convert", { method: "POST", body: form });
  const body = await resp.json();
  if (!resp.ok) throw new Error(body.error || "conversion failed");
  return body;
}

function showInput(blob) {
  if (els.input.src.startsWith("blob:")) URL.revokeObjectURL(els.input.src);
  els.input.src = URL.createObjectURL(blob);
  els.input.hidden = false;
}

async function run(blob, showPreview) {
  if (busy || !blob) return; // toBlob gives null if the camera isn't ready yet
  busy = true;
  setStatus("Converting…");
  try {
    if (showPreview) showInput(blob);
    const t0 = performance.now();
    const res = await convert(blob);
    els.output.src = res.image; // server-generated data:image/png URL
    els.output.hidden = false;
    els.inputLabel.textContent = `Input (detected: ${res.detected})`;
    els.outputLabel.textContent = LABELS[res.direction] || "Output";
    setStatus(`Done in ${Math.round(performance.now() - t0)} ms`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    busy = false;
  }
}

function frameBlob() {
  const { video, canvas } = els;
  if (!video.videoWidth) return Promise.resolve(null);
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

async function toggleCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    els.video.hidden = true;
    els.live.checked = false;
    els.camera.textContent = "Start camera";
    els.snap.disabled = els.live.disabled = true;
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (err) {
    setStatus(`Camera unavailable: ${err.message}`);
    return;
  }
  els.video.srcObject = stream;
  els.video.hidden = false;
  els.input.hidden = true;
  els.camera.textContent = "Stop camera";
  els.snap.disabled = els.live.disabled = false;
}

async function liveLoop() {
  while (stream && els.live.checked) {
    await run(await frameBlob(), false);
    await new Promise((r) => setTimeout(r, LIVE_INTERVAL_MS));
  }
}

els.file.addEventListener("change", () => { if (els.file.files[0]) run(els.file.files[0], true); });
els.camera.addEventListener("click", toggleCamera);
els.snap.addEventListener("click", async () => run(await frameBlob(), true));
els.live.addEventListener("change", liveLoop);
