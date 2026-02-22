const API_URL = "http://127.0.0.1:5000/api/predict"; // adjust as needed

// might need to  add ffilter  to say weither the image is a  gray scale or  what so the model wont  freak out

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");

const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const downloadBtn = document.getElementById("downloadBtn");

const inputPreview = document.getElementById("inputPreview");
const inputPlaceholder = document.getElementById("inputPlaceholder");
const outputPreview = document.getElementById("outputPreview");
const outputPlaceholder = document.getElementById("outputPlaceholder");

const fileMeta = document.getElementById("fileMeta");
const metricsBox = document.getElementById("metrics");
const statusEl = document.getElementById("status");
const latencyEl = document.getElementById("latency");

let selectedFile = null; //current  image by user selected
let lastOutputBlob = null; //output from the server

// updates little badge for whats happening
function setStatus(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

function showInputPreview(file) {
  // creates a  temp  url  for the image  so we  can  view it
  const url = URL.createObjectURL(file);
  inputPreview.src = url;
  inputPreview.style.display = "block";
  inputPlaceholder.style.display = "none";
  fileMeta.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB • ${file.type || "unknown"}`;
}

// reset  the page
function clearAll() {
  selectedFile = null;
  lastOutputBlob = null;

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

  fileInput.value = "";
}

function pickFile(file) {
  // check if file and if its a image
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    alert("Please choose an image file.");
    return;
  }
  selectedFile = file;
  //   go  to show it
  showInputPreview(file);
  runBtn.disabled = false;
}

fileInput.addEventListener("change", (e) => {
  pickFile(e.target.files?.[0]);
});

// so  they can  drop  images in

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("dragover"),
);
// grabs  it and  shows it
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files?.[0];
  pickFile(file);
});

clearBtn.addEventListener("click", clearAll);

// create fake link so  it can download it
downloadBtn.addEventListener("click", () => {
  if (!lastOutputBlob) return;
  const url = URL.createObjectURL(lastOutputBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "output.png"; // or server-provided name
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

// event to  send it to the  backennd
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setStatus("loading", "Running...");
  latencyEl.textContent = "";
  outputPreview.style.display = "none";
  outputPlaceholder.style.display = "grid";
  downloadBtn.disabled = true;

  const t0 = performance.now();

  try {
    const form = new FormData();
    // form data so its  easy for backend
    form.append("image", selectedFile);

    // Future options (when enabled):
    // form.append("target", document.getElementById("targetSelect").value);
    // form.append("threshold", document.getElementById("threshold").value);

    const res = await fetch(API_URL, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Server error (${res.status}): ${text}`);
    }

    // Expect JSON: { result_image_base64, result_mime, metrics }
    const data = await res.json();

    // Show metrics
    metricsBox.textContent = JSON.stringify(data.metrics ?? {}, null, 2);

    // Show output image (base64 -> blob)
    if (data.result_image_base64) {
      const mime = data.result_mime || "image/png";
      const byteChars = atob(data.result_image_base64);
      const byteNums = new Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++)
        byteNums[i] = byteChars.charCodeAt(i);
      const blob = new Blob([new Uint8Array(byteNums)], { type: mime });

      lastOutputBlob = blob;

      const outUrl = URL.createObjectURL(blob);
      outputPreview.src = outUrl;
      outputPreview.style.display = "block";
      outputPlaceholder.style.display = "none";
      downloadBtn.disabled = false;
    } else {
      // If you decide server returns a URL instead:
      // outputPreview.src = data.result_image_url;
    }

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

clearAll();
