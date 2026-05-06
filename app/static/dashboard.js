const state = {
  currentUrl: "",
  history: [],
};

const urlInput = document.getElementById("urlInput");
const predictForm = document.getElementById("predictForm");
const analyzeBtn = document.getElementById("analyzeBtn");

const resultCard = document.getElementById("resultCard");
const resultIcon = document.getElementById("resultIcon");
const resultLabel = document.getElementById("resultLabel");
const resultAdvice = document.getElementById("resultAdvice");
const resultUrl = document.getElementById("resultUrl");
const riskLevel = document.getElementById("riskLevel");
const reasonBox = document.getElementById("reasonBox");
const reasonList = document.getElementById("reasonList");
const copyBtn = document.getElementById("copyBtn");
const clearBtn = document.getElementById("clearBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const historyBody = document.getElementById("historyBody");

function setBusy(isBusy) {
  analyzeBtn.disabled = isBusy;
  analyzeBtn.textContent = isBusy ? "Checking..." : "Check";
}

function sentenceJoin(values) {
  if (!values.length) {
    return "";
  }
  if (values.length === 1) {
    return values[0];
  }
  return `${values.slice(0, -1).join(", ")} and ${values[values.length - 1]}`;
}

function plainSignal(signal) {
  const [name, rawDetail = ""] = String(signal).split(":");
  const details = rawDetail.split(",").filter(Boolean);

  switch (name) {
    case "brand-impersonation":
      return `It mentions ${sentenceJoin(details)} but does not use the official website address.`;
    case "embedded-official-domain":
      return `It hides a familiar brand name inside a different website address.`;
    case "ip-address-host":
      return "It uses a number-based website address, which is uncommon for trusted login pages.";
    case "risky-tld":
      return "It uses a web ending often seen in suspicious links.";
    case "at-symbol-obfuscation":
      return "It contains an @ symbol, which can disguise where the link really goes.";
    case "url-shortener":
      return "It is a shortened link, so the destination is not obvious.";
    case "phishing-keywords":
      return "It uses words often found in fake account, payment, or password pages.";
    case "http-on-sensitive-brand-domain":
      return "It is not using a secure connection for a sensitive brand link.";
    case "http-with-phishing-keywords":
      return "It asks for sensitive action without using a secure connection.";
    case "deep-subdomain-chain":
      return "It has too many website sections before the main domain.";
    case "excessive-hyphenation":
      return "It uses repeated dashes, which can make fake domains look convincing.";
    case "credential-harvest-pattern":
      return "It looks like a page designed to collect login or payment details.";
    default:
      return "Something about this link looks unusual.";
  }
}

function riskCopy(result) {
  if (!result.is_phishing) {
    return {
      icon: "OK",
      label: "Looks safe",
      note: "No obvious warning",
      advice: "No obvious phishing signs were found. Still make sure you trust who sent it before entering private information.",
    };
  }

  if (result.risk_level === "high") {
    return {
      icon: "!",
      label: "Avoid this link",
      note: "High warning",
      advice: "Do not open this link or enter any passwords, card details, or one-time codes. Use the official app or type the website address yourself.",
    };
  }

  return {
    icon: "!",
    label: "Be careful",
    note: "Needs caution",
    advice: "This link has warning signs. Open it only if you are sure who sent it and why.",
  };
}

function replaceChildrenWithText(parent, tagName, text, className) {
  const element = document.createElement(tagName);
  element.textContent = text;
  if (className) {
    element.className = className;
  }
  parent.replaceChildren(element);
}

function renderReasons(signals) {
  reasonList.replaceChildren();

  if (!signals.length) {
    reasonBox.classList.add("hidden");
    return;
  }

  for (const signal of signals) {
    const item = document.createElement("li");
    item.textContent = plainSignal(signal);
    reasonList.appendChild(item);
  }

  reasonBox.classList.remove("hidden");
}

function renderResult(result) {
  const copy = riskCopy(result);
  const isBad = Boolean(result.is_phishing);
  const signals = Array.isArray(result.signals) ? result.signals : [];

  state.currentUrl = result.url || "";

  resultCard.classList.remove("hidden", "good", "bad");
  resultCard.classList.add(isBad ? "bad" : "good");
  resultIcon.textContent = copy.icon;
  resultLabel.textContent = copy.label;
  riskLevel.textContent = copy.note;
  resultAdvice.textContent = copy.advice;
  resultUrl.textContent = state.currentUrl;
  renderReasons(isBad ? signals : []);
}

function renderHistory() {
  historyBody.replaceChildren();

  if (state.history.length === 0) {
    replaceChildrenWithText(historyBody, "p", "No links checked yet.", "empty-state");
    return;
  }

  for (const item of state.history) {
    const row = document.createElement("div");
    row.className = "history-item";

    const url = document.createElement("p");
    url.className = "history-url";
    url.textContent = item.url;

    const status = document.createElement("span");
    status.className = `history-status ${item.is_phishing ? "bad" : "good"}`;
    status.textContent = item.is_phishing ? "Avoid" : "Looks safe";

    row.append(url, status);
    historyBody.appendChild(row);
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
      throw new Error(payload.error || "The link could not be checked.");
    }

    renderResult(payload.result);
    state.history.unshift(payload.result);
    state.history = state.history.slice(0, 6);
    renderHistory();
  } catch (error) {
    alert(error.message || "Unable to check this link right now.");
  } finally {
    setBusy(false);
  }
});

copyBtn.addEventListener("click", async () => {
  if (!state.currentUrl) {
    return;
  }

  try {
    await navigator.clipboard.writeText(state.currentUrl);
    copyBtn.textContent = "Copied";
    window.setTimeout(() => {
      copyBtn.textContent = "Copy link";
    }, 1400);
  } catch {
    copyBtn.textContent = "Copy failed";
    window.setTimeout(() => {
      copyBtn.textContent = "Copy link";
    }, 1400);
  }
});

clearBtn.addEventListener("click", () => {
  resultCard.classList.add("hidden");
  reasonBox.classList.add("hidden");
  state.currentUrl = "";
  urlInput.value = "";
  urlInput.focus();
});

clearHistoryBtn.addEventListener("click", () => {
  state.history = [];
  renderHistory();
});
