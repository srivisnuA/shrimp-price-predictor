/* Shrimp Price Predictor dashboard — advanced edition */
const $ = (id) => document.getElementById(id);
const state = { history: null, forecast: null, importance: null, debounce: null };

const showError = (s) => $("error-banner").classList.toggle("hidden", !s);

function toast(msg, ok) {
  const t = $("toast");
  t.textContent = msg;
  t.className = `toast ${ok ? "ok" : "err"}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 7000);
}

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${r.status} ${url}`);
  return data;
}

function fmtMoney(v) { return `$${Number(v).toFixed(2)}`; }

function opts(yTitles) {
  return {
    responsive: true,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { color: "#8ba0bd" } } },
    scales: {
      x: { ticks: { color: "#8ba0bd" }, grid: { color: "#1f2c47" } },
      y: { ticks: { color: "#8ba0bd", callback: (v) => "$" + v }, grid: { color: "#1f2c47" },
           title: { display: true, text: "USD / kg", color: "#8ba0bd" }, position: "left" },
      y1: { display: !!yTitles, title: { display: !!yTitles, text: yTitles || "", color: "#8ba0bd" },
            ticks: { color: "#8ba0bd" }, grid: { drawOnChartArea: false }, position: "right" },
    },
  };
}

function scenarioFromUI() {
  const sc = {};
  const t = +$("sc-temp").value, r = +$("sc-rain").value, d = +$("sc-disease").value;
  if ($("sc-temp").dataset.on) sc.avg_temp_c = t;
  if ($("sc-rain").dataset.on) sc.rainfall_mm = r;
  if ($("sc-disease").dataset.on) sc.disease_outbreak_severity = d;
  $("temp-value").textContent = sc.avg_temp_c != null ? `${t.toFixed(1)}°C` : "projected";
  $("rain-value").textContent = sc.rainfall_mm != null ? r : "projected";
  $("disease-value").textContent = sc.disease_outbreak_severity != null ? d : "projected";
  return sc;
}

async function loadHistory() {
  const rows = await fetchJSON("/api/prices/yearly");
  const labels = rows.map((r) => r.year);
  if (state.history) state.history.destroy();
  state.history = new Chart($("history-chart"), {
    data: {
      labels,
      datasets: [
        { type: "bar", label: "Disease severity (0–100)", data: rows.map((r) => r.disease_outbreak_severity ?? 0),
          backgroundColor: "rgba(255,107,107,.35)", yAxisID: "y1" },
        { type: "line", label: "Avg price", data: rows.map((r) => r.avg_price_usd_kg),
          borderColor: "#26c6a6", backgroundColor: "rgba(38,198,166,.12)", fill: true,
          tension: 0.3, pointRadius: 2, yAxisID: "y" },
      ],
    },
    options: opts("Disease severity"),
  });
}

async function loadForecast() {
  const h = +$("horizon").value;
  $("horizon-value").textContent = h;
  const sc = scenarioFromUI();
  const d = await fetchJSON(`/api/forecast/advanced?horizon=${h}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(sc),
  });
  // metric cards
  $("metric-model").textContent = d.model.replace(/_/g, " ");
  $("metric-mae").textContent = fmtMoney(d.metrics.mae);
  $("metric-rmse").textContent = fmtMoney(d.metrics.rmse);
  $("metric-base").textContent = fmtMoney(d.metrics.baseline_mae);

  const hist = d.history;
  const labels = [...hist.map((p) => p.year), ...d.predictions.map((p) => p.year)];
  const histData = [...hist.map((p) => p.price), ...d.predictions.map(() => null)];
  const predData = [...hist.map(() => null), ...d.predictions.map((p) => p.prediction)];
  predData[hist.length - 1] = hist[hist.length - 1].price;
  const lower = [...hist.map(() => null), ...d.predictions.map((p) => p.lower)];
  const upper = [...hist.map(() => null), ...d.predictions.map((p) => p.upper)];
  const disease = [...hist.map((p) => p.disease ?? 0), ...d.predictions.map(() => null)];

  if (state.forecast) state.forecast.destroy();
  state.forecast = new Chart($("forecast-chart"), {
    data: {
      labels,
      datasets: [
        { type: "bar", label: "Disease severity", data: disease,
          backgroundColor: "rgba(255,107,107,.3)", yAxisID: "y1" },
        { type: "line", label: "Confidence band", data: upper, fill: "+1",
          backgroundColor: "rgba(79,142,247,.15)", borderWidth: 0, pointRadius: 0, yAxisID: "y" },
        { type: "line", label: "Lower bound", data: lower, borderWidth: 0, pointRadius: 0, yAxisID: "y" },
        { type: "line", label: "History", data: histData, borderColor: "#26c6a6", tension: 0.3, pointRadius: 2, yAxisID: "y" },
        { type: "line", label: `Forecast (+${h}y)`, data: predData, borderColor: "#4f8ef7",
          borderDash: [6, 4], tension: 0.3, pointRadius: 2, yAxisID: "y" },
      ],
    },
    options: opts("Disease severity"),
  });

  // importances
  const imp = d.feature_importances || {};
  const labelsI = Object.keys(imp);
  const pretty = {
    trend: "Year trend", production: "Production volume", disease: "Disease severity",
    disease_lag1: "Disease (last year)", temp_anom: "Temperature anomaly",
    rain_anom: "Rainfall anomaly", temp_x_disease: "Temp × disease",
  };
  if (state.importance) state.importance.destroy();
  state.importance = new Chart($("importance-chart"), {
    type: "bar",
    data: {
      labels: labelsI.map((k) => pretty[k] || k),
      datasets: [{ label: "Importance", data: labelsI.map((k) => imp[k]),
        backgroundColor: labelsI.map((k) => (k.includes("disease") ? "#ff6b6b" : k.includes("temp") || k.includes("rain") ? "#4f8ef7" : "#26c6a6")) }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#8ba0bd" }, grid: { color: "#1f2c47" } },
                y: { ticks: { color: "#e6edf7" }, grid: { color: "#1f2c47" } } } },
  });
}

async function refreshAll() {
  showError(false);
  try { await Promise.all([loadHistory(), loadForecast()]); }
  catch (e) { console.error(e); showError(true); }
}

async function handleUpload() {
  const file = $("upload-file").files[0];
  if (!file) { toast("❌ Choose a CSV file first.", false); return; }
  const body = new FormData();
  body.append("file", file);
  const mode = $("upload-replace").checked ? "replace" : "merge";
  try {
    const d = await fetchJSON(`/api/data/upload?mode=${mode}`, { method: "POST", body });
    toast(`✅ Added ${d.rows_added} year(s), updated ${d.rows_updated}. Model ${d.retrained ? `retrained (${d.model})` : "not retrained — need ≥8 years of data"}.`, true);
    $("upload-file").value = "";
    await refreshAll();
  } catch (err) {
    toast(`❌ ${err.message}`, false);
  }
}

function wireEvents() {
  $("horizon").addEventListener("input", () => {
    clearTimeout(state.debounce);
    state.debounce = setTimeout(loadForecast, 150);
  });
  for (const id of ["sc-temp", "sc-rain", "sc-disease"]) {
    $(id).addEventListener("input", (e) => { e.target.dataset.on = "1"; });
    $(id).addEventListener("change", () => {
      clearTimeout(state.debounce);
      state.debounce = setTimeout(loadForecast, 200);
    });
  }
  $("upload-btn").addEventListener("click", handleUpload);
}

(async function init() {
  try {
    await fetchJSON("/api/prices/yearly");
  } catch (e) { showError(true); return; }
  wireEvents();
  await refreshAll();
})();
