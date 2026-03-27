const state = {
  history: [],
};

const urlInput = document.getElementById("urlInput");
const predictForm = document.getElementById("predictForm");
const analyzeBtn = document.getElementById("analyzeBtn");

const resultCard = document.getElementById("resultCard");
const resultLabel = document.getElementById("resultLabel");
const resultUrl = document.getElementById("resultUrl");
const resultConfidence = document.getElementById("resultConfidence");
const riskLevel = document.getElementById("riskLevel");
const meterFill = document.getElementById("meterFill");
const signalList = document.getElementById("signalList");

const metricAccuracy = document.getElementById("metricAccuracy");
const metricF1 = document.getElementById("metricF1");
const metricPrecision = document.getElementById("metricPrecision");
const metricRecall = document.getElementById("metricRecall");
const datasetRows = document.getElementById("datasetRows");
const trainTestRows = document.getElementById("trainTestRows");
const trainedAt = document.getElementById("trainedAt");
const datasetSource = document.getElementById("datasetSource");
const historyBody = document.getElementById("historyBody");

function setBusy(isBusy) {
  analyzeBtn.disabled = isBusy;
  analyzeBtn.textContent = isBusy ? "Analyzing..." : "Analyze URL";
}

function prettifySignal(signal) {
  return signal
    .replace("brand-impersonation:", "brand impersonation: ")
    .replaceAll("-", " ");
}

function renderResult(result) {
  resultCard.classList.remove("hidden");
  resultLabel.classList.remove("bad", "good");

  const isBad = result.is_phishing;
  resultLabel.textContent = isBad ? "Phishing" : "Legitimate";
  resultLabel.classList.add(isBad ? "bad" : "good");

  resultUrl.textContent = result.url;
  riskLevel.textContent = `risk: ${result.risk_level}`;
  resultConfidence.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
  meterFill.style.width = `${(result.phishing_probability * 100).toFixed(2)}%`;

  const signals = Array.isArray(result.signals) ? result.signals : [];
  if (signals.length === 0) {
    signalList.innerHTML = `<li class="muted">No major phishing signals were triggered.</li>`;
  } else {
    signalList.innerHTML = signals.map((signal) => `<li>${prettifySignal(signal)}</li>`).join("");
  }
}

function renderHistory() {
  if (state.history.length === 0) {
    historyBody.innerHTML = `<tr><td colspan="3" class="muted">No checks yet.</td></tr>`;
    return;
  }

  historyBody.innerHTML = state.history
    .map(
      (item) => `
      <tr>
        <td title="${item.url}">${item.url}</td>
        <td>${item.prediction}</td>
        <td>${(item.confidence * 100).toFixed(2)}%</td>
      </tr>
    `
    )
    .join("");
}

function setMetrics(status) {
  const metrics = status.metrics || {};
  metricAccuracy.textContent = metrics.accuracy ? metrics.accuracy.toFixed(4) : "-";
  metricF1.textContent = metrics.f1_score ? metrics.f1_score.toFixed(4) : "-";
  metricPrecision.textContent = metrics.precision ? metrics.precision.toFixed(4) : "-";
  metricRecall.textContent = metrics.recall ? metrics.recall.toFixed(4) : "-";
  datasetRows.textContent = status.dataset_rows ?? "-";
  trainTestRows.textContent =
    status.train_rows && status.test_rows ? `${status.train_rows} / ${status.test_rows}` : "-";
  trainedAt.textContent = status.trained_at_utc ?? "-";
  datasetSource.textContent = status.generated_dataset ? "synthetic_urls.csv (auto)" : status.dataset_path ?? "-";
}

async function loadStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();
  if (payload.ok) {
    setMetrics(payload);
  }
}

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = urlInput.value.trim();
  if (!url) {
    return;
  }

  setBusy(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url }),
    });
    const payload = await response.json();
    if (!payload.ok) {
      throw new Error(payload.error || "Prediction failed");
    }

    renderResult(payload.result);
    state.history.unshift(payload.result);
    state.history = state.history.slice(0, 8);
    renderHistory();
  } catch (error) {
    alert(error.message || "Unable to analyze this URL");
  } finally {
    setBusy(false);
  }
});

loadStatus().catch(() => {
  datasetSource.textContent = "Unable to load model status";
});
