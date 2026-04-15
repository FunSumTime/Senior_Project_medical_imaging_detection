const API_URL = "https://prenatal-twice-deskbound.ngrok-free.dev/api/predict";

// input elements
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");

// buttons
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const downloadBtn = document.getElementById("downloadBtn");
const showBoxBtn = document.getElementById("showBoxBtn");
const showHeatmapBtn = document.getElementById("showHeatmapBtn");

// previews
const inputPreview = document.getElementById("inputPreview");
const inputPlaceholder = document.getElementById("inputPlaceholder");
const outputPreview = document.getElementById("outputPreview");
const outputPlaceholder = document.getElementById("outputPlaceholder");

// misc
const fileMeta = document.getElementById("fileMeta");
const metricsBox = document.getElementById("metrics");
const statusEl = document.getElementById("status");
const latencyEl = document.getElementById("latency");
const thresholdInput = document.getElementById("threshold");
const stage1ModelSelect = document.getElementById("stage1Model");
const stage2ModelSelect = document.getElementById("stage2Model");

// state
let selectedFile = null;
let activeOutput = "box";

const outputImages = {
  box: null,
  heatmap: null,
};

function setStatus(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

function showInputPreview(file) {
  const url = URL.createObjectURL(file);
  inputPreview.src = url;
  inputPreview.style.display = "block";
  inputPlaceholder.style.display = "none";
  fileMeta.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB • ${file.type || "unknown"}`;
}

function revokeOutputUrls() {
  if (outputImages.box?.url) URL.revokeObjectURL(outputImages.box.url);
  if (outputImages.heatmap?.url) URL.revokeObjectURL(outputImages.heatmap.url);
  outputImages.box = null;
  outputImages.heatmap = null;
}

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
  renderServerMetrics({});
  latencyEl.textContent = "";
  setStatus("idle", "System Idle");

  runBtn.disabled = true;
  downloadBtn.disabled = true;
  showBoxBtn.disabled = true;
  showHeatmapBtn.disabled = true;

  fileInput.value = "";
  updateVisualMetrics({});
}

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

function base64ToBlob(base64, mime = "image/png") {
  const byteChars = atob(base64);
  const byteNums = new Array(byteChars.length);

  for (let i = 0; i < byteChars.length; i++) {
    byteNums[i] = byteChars.charCodeAt(i);
  }

  return new Blob([new Uint8Array(byteNums)], { type: mime });
}

function showActiveOutput() {
  const current = outputImages[activeOutput];

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

  if (activeOutput === "box") {
    showBoxBtn.disabled = true;
    showHeatmapBtn.disabled = false;
  } else {
    showBoxBtn.disabled = false;
    showHeatmapBtn.disabled = true;
  }
}

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

showBoxBtn.addEventListener("click", () => {
  activeOutput = "box";
  showActiveOutput();
});

showHeatmapBtn.addEventListener("click", () => {
  activeOutput = "heatmap";
  showActiveOutput();
});

function updateVisualMetrics(metrics) {
  const probFill = document.getElementById("prob-fill");
  const probValue = document.getElementById("prob-value");
  const badge = document.getElementById("detection-badge");
  const metaDetected = document.getElementById("meta-detected");
  const metaModel = document.getElementById("meta-model");
  const metaLayer = document.getElementById("meta-layer");
  const metaStage2 = document.getElementById("meta-stage2");

  if (!metrics || Object.keys(metrics).length === 0) {
    probFill.style.width = "0%";
    probValue.textContent = "0%";
    badge.textContent = "N/A";
    badge.className = "badge";
    metaDetected.textContent = "-";
    metaModel.textContent = "-";
    metaLayer.textContent = "-";
    metaStage2.textContent = "-";
    return;
  }

  const prob = ((metrics.probability || 0) * 100).toFixed(1);
  probFill.style.width = `${prob}%`;
  probValue.textContent = `${prob}%`;

  if (prob > 70) probFill.style.background = "var(--accent-red)";
  else if (prob > 40) probFill.style.background = "var(--accent-orange)";
  else probFill.style.background = "var(--accent-green)";

  const detectedLabel =
    metrics.detected_label || metrics.predicted_class || "unknown";

  if (metrics.anomaly_detected) {
    const detectedLabel =
      metrics.detected_label ||
      metrics.stage2_predicted_class ||
      metrics.predicted_class ||
      "unknown";

    badge.textContent = metrics.anomaly_detected
      ? `ANOMALY DETECTED: ${detectedLabel}`
      : "CLEAR / NORMAL";
    badge.className = "badge danger";
    metaDetected.textContent = detectedLabel;
  } else {
    badge.textContent = "CLEAR / NORMAL";
    badge.className = "badge success";
    metaDetected.textContent = "normal";
  }

  metaModel.textContent = metrics.model || "Unknown";
  metaLayer.textContent = metrics.cam_layer || "-";
  metaStage2.textContent = metrics.stage2_ran ? "Ran" : "Skipped";
}

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
    form.append("stage1_model", stage1ModelSelect.value);
    form.append("stage2_model", stage2ModelSelect.value);

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
    renderServerMetrics(data.metrics);

    const mime = data.result_mime || "image/png";

    if (data.result_image_base64_heatmap) {
      const heatmapBlob = base64ToBlob(data.result_image_base64_heatmap, mime);
      const heatmapUrl = URL.createObjectURL(heatmapBlob);
      outputImages.heatmap = { blob: heatmapBlob, url: heatmapUrl };
    }

    if (data.result_image_base64_box) {
      const boxBlob = base64ToBlob(data.result_image_base64_box, mime);
      const boxUrl = URL.createObjectURL(boxBlob);
      outputImages.box = { blob: boxBlob, url: boxUrl };
    }

    if (outputImages.box) activeOutput = "box";
    else if (outputImages.heatmap) activeOutput = "heatmap";

    showActiveOutput();

    const t1 = performance.now();
    latencyEl.textContent = `Latency: ${(t1 - t0).toFixed(0)} ms`;
    setStatus("ok", "Done");
  } catch (err) {
    setStatus("err", "Error");
    renderServerMetrics({ error: String(err.message || err) });
  }
});

function formatMetricValue(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (Array.isArray(value)) {
    return value.map((v) => formatMetricValue(v)).join(", ");
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function prettifyKey(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function renderServerMetrics(metrics) {
  if (!metrics || Object.keys(metrics).length === 0) {
    metricsBox.innerHTML = `<div class="server-row">
      <div class="server-key">Status</div>
      <div class="server-value">No server response yet.</div>
    </div>`;
    return;
  }

  const preferredOrder = [
    "anomaly_detected",
    "detected_label",
    "predicted_class",
    "probability",
    "stage2_ran",
    "stage2_predicted_class",
    "stage2_probability",
    "model",
    "stage2_model",
    "cam_layer",
    "threshold",
    "inference_ms",
  ];

  const keys = [
    ...preferredOrder.filter((k) => k in metrics),
    ...Object.keys(metrics).filter((k) => !preferredOrder.includes(k)),
  ];

  metricsBox.innerHTML = keys
    .map((key) => {
      const value = formatMetricValue(metrics[key]);
      const isCodeLike = [
        "model",
        "stage2_model",
        "cam_layer",
        "predicted_class",
        "stage2_predicted_class",
        "detected_label",
      ].includes(key);

      return `
        <div class="server-row">
          <div class="server-key">${prettifyKey(key)}</div>
          <div class="server-value ${isCodeLike ? "code" : ""}">${value}</div>
        </div>
      `;
    })
    .join("");
}

clearAll();
