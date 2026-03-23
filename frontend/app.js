const API_URL = "http://127.0.0.1:5000/api/predict";

// input elements
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");

// buttons
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const downloadBtn = document.getElementById("downloadBtn");
const showBoxBtn = document.getElementById("showBoxBtn");
const showHeatmapBtn = document.getElementById("showHeatmapBtn");

// preview areas
const inputPreview = document.getElementById("inputPreview");
const inputPlaceholder = document.getElementById("inputPlaceholder");
const outputPreview = document.getElementById("outputPreview");
const outputPlaceholder = document.getElementById("outputPlaceholder");

// misc outputs
const fileMeta = document.getElementById("fileMeta");
const metricsBox = document.getElementById("metrics");
const statusEl = document.getElementById("status");
const latencyEl = document.getElementById("latency");
const thresholdInput = document.getElementById("threshold");

// state
let selectedFile = null;
let activeOutput = "box"; // "box" or "heatmap"

// store both returned images here
const outputImages = {
  box: null, // { blob, url }
  heatmap: null, // { blob, url }
};

// set the little element to running or whatever
function setStatus(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

// show the image that they just gave us
function showInputPreview(file) {
  const url = URL.createObjectURL(file);
  inputPreview.src = url;
  inputPreview.style.display = "block";
  inputPlaceholder.style.display = "none";
  fileMeta.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB • ${file.type || "unknown"}`;
}

// since we have two images now we need to only have one
function revokeOutputUrls() {
  if (outputImages.box?.url) URL.revokeObjectURL(outputImages.box.url);
  if (outputImages.heatmap?.url) URL.revokeObjectURL(outputImages.heatmap.url);

  outputImages.box = null;
  outputImages.heatmap = null;
}

// reset everything
function clearAll() {
  selectedFile = null;
  activeOutput = "box";

  revokeOutputUrls();

  inputPreview.src = "";
  inputPreview.style.display = "none";
  inputPlaceholder.style.display = "grid";

  outputPreview.src = "";
  outputPreview.style.display = "none";
  outputPlaceholder.style.display = "grid";

  fileMeta.textContent = "";
  metricsBox.textContent = "{}";
  latencyEl.textContent = "";
  setStatus("idle", "Idle");

  runBtn.disabled = true;
  downloadBtn.disabled = true;
  showBoxBtn.disabled = true;
  showHeatmapBtn.disabled = true;

  fileInput.value = "";
}

// to have a person be able to pick a file
function pickFile(file) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    alert("Please choose an image file.");
    return;
  }

  selectedFile = file;
  showInputPreview(file);
  runBtn.disabled = false;
}

// take the buffer of base64 image and make it to a png
function base64ToBlob(base64, mime = "image/png") {
  const byteChars = atob(base64);
  const byteNums = new Array(byteChars.length);

  for (let i = 0; i < byteChars.length; i++) {
    byteNums[i] = byteChars.charCodeAt(i);
  }

  return new Blob([new Uint8Array(byteNums)], { type: mime });
}

// show which image needs to be shown right now
function showActiveOutput() {
  const current = outputImages[activeOutput];
  console.log(current);

  if (!current?.url) {
    outputPreview.src = "";
    outputPreview.style.display = "none";
    outputPlaceholder.style.display = "grid";
    downloadBtn.disabled = true;
    showBoxBtn.disabled = true;
    showHeatmapBtn.disabled = true;
    return;
  }

  outputPreview.src = current.url;
  outputPreview.style.display = "block";
  outputPlaceholder.style.display = "none";
  downloadBtn.disabled = false;
  if (activeOutput == "box") {
    showBoxBtn.disabled = true;
    showHeatmapBtn.disabled = false;
  } else {
    showBoxBtn.disabled = false;
    showHeatmapBtn.disabled = true;
  }
}

// event listners to pick up the files in the drop box
fileInput.addEventListener("change", (e) => {
  pickFile(e.target.files?.[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files?.[0];
  pickFile(file);
});

clearBtn.addEventListener("click", clearAll);

// make a temp url so they can download it
downloadBtn.addEventListener("click", () => {
  const current = outputImages[activeOutput];
  if (!current?.blob) return;

  const a = document.createElement("a");
  a.href = current.url;
  a.download = activeOutput === "box" ? "output_box.png" : "output_heatmap.png";
  document.body.appendChild(a);
  a.click();
  a.remove();
});

// button listeners to toggle which image to look at
showBoxBtn.addEventListener("click", () => {
  activeOutput = "box";
  showActiveOutput();
});

showHeatmapBtn.addEventListener("click", () => {
  activeOutput = "heatmap";
  showActiveOutput();
});

// main thing to send what the user gives to the backend
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setStatus("loading", "Running...");
  latencyEl.textContent = "";

  outputPreview.style.display = "none";
  outputPlaceholder.style.display = "grid";
  downloadBtn.disabled = true;
  showBoxBtn.disabled = true;
  showHeatmapBtn.disabled = true;

  revokeOutputUrls();

  const t0 = performance.now();

  try {
    const form = new FormData();
    form.append("image", selectedFile);
    form.append("threshold", thresholdInput.value);

    const res = await fetch(API_URL, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Server error (${res.status}): ${text}`);
    }

    const data = await res.json();

    updateVisualMetrics(data.metrics);

    const mime = data.result_mime || "image/png";

    // store heatmap image
    if (data.result_image_base64_heatmap) {
      const heatmapBlob = base64ToBlob(data.result_image_base64_heatmap, mime);
      const heatmapUrl = URL.createObjectURL(heatmapBlob);

      outputImages.heatmap = {
        blob: heatmapBlob,
        url: heatmapUrl,
      };
    }

    // store box image
    if (data.result_image_base64_box) {
      const boxBlob = base64ToBlob(data.result_image_base64_box, mime);
      const boxUrl = URL.createObjectURL(boxBlob);

      outputImages.box = {
        blob: boxBlob,
        url: boxUrl,
      };
    }

    // default view
    if (outputImages.box) {
      activeOutput = "box";
    } else if (outputImages.heatmap) {
      activeOutput = "heatmap";
    }
    console.log(activeOutput);

    showActiveOutput();

    const t1 = performance.now();
    latencyEl.textContent = `Latency: ${(t1 - t0).toFixed(0)} ms`;
    setStatus("ok", "Done");
  } catch (err) {
    setStatus("err", "Error");
    metricsBox.textContent = JSON.stringify(
      { error: String(err.message || err) },
      null,
      2,
    );
  }
});

/* ... Keep existing state variables and file logic ... */

// NEW: Function to update visual bars
function updateVisualMetrics(metrics) {
  const probFill = document.getElementById("prob-fill");
  const probValue = document.getElementById("prob-value");
  const badge = document.getElementById("detection-badge");
  const metaModel = document.getElementById("meta-model");
  const metaLayer = document.getElementById("meta-layer");

  if (!metrics || Object.keys(metrics).length === 0) return;

  // 1. Update Probability Bar
  const prob = (metrics.probability * 100).toFixed(1);
  probFill.style.width = `${prob}%`;
  probValue.textContent = `${prob}%`;

  // Color change based on confidence
  if (prob > 70) probFill.style.background = "var(--accent-red)";
  else if (prob > 40) probFill.style.background = "var(--accent-orange)";
  else probFill.style.background = "var(--accent-green)";

  // 2. Update Badge
  badge.textContent = metrics.anomaly_detected
    ? "ANOMALY DETECTED"
    : "CLEAR / NORMAL";
  badge.className = `badge ${metrics.anomaly_detected ? "danger" : "success"}`;

  // 3. Update Meta
  metaModel.textContent = metrics.model || "Unknown";
  metaLayer.textContent = metrics.cam_layer || "Default";
}

clearAll();
