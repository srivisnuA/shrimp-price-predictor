"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  series: [],
  quality: null,
  evaluation: null,
  sequence: 0,
  controller: null,
  loadedDataset: null,
};
const names = {
  latest_price: "Latest price",
  seasonal_naive: "Seasonal naïve",
  damped_drift: "Damped drift",
  ridge: "Ridge regression",
  hist_gradient_boosting: "Histogram boosting",
  lightgbm: "LightGBM",
};
const money = (value) => (Number.isFinite(value) ? "$" + value.toFixed(2) : "—");
const number = (value) => (Number.isFinite(value) ? value.toFixed(3) : "—");
const percent = (value) => (Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "Not estimable");
const base = () => "/api/v1/datasets/" + $("dataset").value;

async function get(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string" ? body.detail : "The service could not complete this request.",
    );
  }
  return response.json();
}
function setOptions(id, options, preferred) {
  const select = $(id),
    previous = select.value;
  select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
  select.value = options.some((x) => x[0] === previous)
    ? previous
    : options.some((x) => x[0] === preferred)
      ? preferred
      : options[0]?.[0] || "";
  select.disabled = options.length === 0;
}
function rebuild(level = "country") {
  if (level === "country")
    setOptions(
      "country",
      [...new Set(state.series.map((x) => x.country))].sort().map((x) => [x, x]),
      "India",
    );
  let rows = state.series.filter((x) => x.country === $("country").value);
  if (level !== "size")
    setOptions(
      "species",
      [...new Set(rows.map((x) => x.species))].sort().map((x) => [x, x.charAt(0).toUpperCase() + x.slice(1)]),
      "vannamei",
    );
  rows = rows
    .filter((x) => x.species === $("species").value)
    .sort((a, b) => a.size.localeCompare(b.size, undefined, { numeric: true }));
  setOptions(
    "size",
    rows.map((x) => [x.series_id, $("dataset").value === "farmgate" ? x.size + " count/kg" : x.size]),
    rows.find((x) => x.size === "60")?.series_id,
  );
}
function svgElement(tag, attributes = {}, text = null) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== null) element.textContent = text;
  return element;
}
function chart(history, predictions, origin) {
  const observed = history.slice(-32),
    all = [...observed.map((x) => ({ ...x, prediction: x.price })), ...predictions];
  const values = all.flatMap((x) => [x.prediction, x.lower, x.upper]).filter(Number.isFinite);
  const min = Math.max(0, Math.min(...values) * 0.96),
    max = Math.max(...values) * 1.04;
  const width = 1080,
    height = 310,
    left = 52,
    right = 25,
    top = 27,
    bottom = 39;
  const timestamps = all.map((x) => Date.parse(x.date));
  const start = Math.min(...timestamps),
    end = Math.max(...timestamps);
  const x = (date) => left + ((Date.parse(date) - start) / (end - start || 1)) * (width - left - right);
  const y = (value) => top + (1 - (value - min) / (max - min || 1)) * (height - top - bottom);
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label":
      "Observed shrimp prices followed by forecast estimates and exploratory uncertainty intervals",
  });
  for (let i = 0; i <= 4; i++) {
    const value = min + ((max - min) * i) / 4;
    svg.append(
      svgElement("line", { x1: left, x2: width - right, y1: y(value), y2: y(value), class: "chart-grid" }),
      svgElement(
        "text",
        { x: left - 12, y: y(value) + 4, "text-anchor": "end", class: "chart-label" },
        value.toFixed(2),
      ),
    );
  }
  const finite = predictions.filter((p) => Number.isFinite(p.lower) && Number.isFinite(p.upper));
  if (finite.length)
    svg.append(
      svgElement("polygon", {
        points: [
          ...finite.map((p) => `${x(p.date)},${y(p.upper)}`),
          ...finite.toReversed().map((p) => `${x(p.date)},${y(p.lower)}`),
        ].join(" "),
        class: "chart-band",
      }),
    );
  svg.append(
    svgElement("polyline", {
      points: observed.map((p) => `${x(p.date)},${y(p.price)}`).join(" "),
      class: "chart-observed",
    }),
  );
  const tail = observed.at(-1),
    future = [{ date: tail.date, prediction: tail.price }, ...predictions];
  svg.append(
    svgElement("line", { x1: x(origin), x2: x(origin), y1: top, y2: height - bottom, class: "chart-origin" }),
    svgElement("polyline", {
      points: future.map((p) => `${x(p.date)},${y(p.prediction)}`).join(" "),
      class: "chart-forecast",
    }),
  );
  for (const p of observed) {
    const point = svgElement("circle", { cx: x(p.date), cy: y(p.price), r: 2.8, class: "chart-point" });
    point.append(svgElement("title", {}, `${p.date}: ${money(p.price)}/kg`));
    svg.append(point);
  }
  const labelCount = 5;
  for (let i = 0; i < labelCount; i++) {
    const item = all[Math.round((i / (labelCount - 1)) * (all.length - 1))];
    svg.append(
      svgElement(
        "text",
        { x: x(item.date), y: height - 12, "text-anchor": "middle", class: "chart-label" },
        item.date.slice(0, 10),
      ),
    );
  }
  $("price-chart").replaceChildren(svg);
}
function render(history, result) {
  const selected = state.series.find((x) => x.series_id === $("size").value),
    q = state.quality,
    evaluation = state.evaluation;
  const monthly = $("dataset").value === "benchmark";
  $("next-price").textContent = money(result.predictions[0].prediction);
  $("next-date").textContent = result.predictions[0].date + " · USD / kg";
  $("model-name").textContent = names[result.model] || result.model;
  $("test-mae").textContent = money(evaluation.test.mae);
  $("baseline-comparison").textContent =
    "Latest-price baseline: " + money(evaluation.test_baseline.mae) + " / kg";
  $("history-count").textContent = selected.observations;
  $("history-range").textContent = selected.start + " → " + selected.end;
  $("selection-note").textContent =
    `${selected.country} · ${selected.species} · ${selected.size}${monthly ? "" : " count/kg"} · ${selected.observations} observations`;
  $("origin-note").textContent = "Forecast origin: " + result.forecast_origin;
  $("price-basis").textContent = monthly ? "Illustrative monthly sample" : "Observed farm-gate source";
  $("gaps").textContent = selected.missing_periods + " missing periods";
  $("age").textContent = selected.staleness_days + " days";
  $("coverage").textContent = percent(evaluation.test.interval_coverage);
  $("dataset-coverage").textContent = `${q.series} series · ${q.dates} dates`;
  $("run-id").textContent = result.run_id;
  $("data-hash").textContent = q.sha256;
  $("trained-at").textContent = q.trained_at;
  $("warning").textContent =
    (monthly ? "This monthly dataset is illustrative and is not verified export pricing. " : "") +
    result.warning;
  $("interval-note").textContent =
    `${percent(result.interval.nominal_coverage)} nominal interval from a separate calibration block. ${result.predictions.some((x) => x.lower === null) ? "Insufficient calibration samples: finite interval bounds are unavailable. " : ""}Actual held-out coverage: ${percent(evaluation.test.interval_coverage)}. Forecasts begin at the stored origin, which may be in the past.`;
  const entries = Object.entries(evaluation.validation).sort((a, b) => a[1].mae - b[1].mae);
  $("model-table").replaceChildren(
    ...entries.map(([name, scores]) => {
      const row = document.createElement("tr");
      if (name === result.model) row.className = "selected-row";
      for (const text of [names[name] || name, number(scores.mae), number(scores.rmse)]) {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.append(cell);
      }
      if (name === result.model) {
        const badge = document.createElement("span");
        badge.className = "model-check";
        badge.textContent = "SELECTED";
        row.firstChild.append(badge);
      }
      return row;
    }),
  );
  $("test-evidence").textContent =
    `Final audit: ${evaluation.test.observations} matched observations from ${evaluation.test.origins} unseen origin. Results pool series and horizons; they are not a series-specific accuracy guarantee.`;
  $("download").href =
    base() +
    "/forecast.csv?" +
    new URLSearchParams({ series_id: selected.series_id, horizon: $("horizon").value });
  $("download").classList.remove("disabled");
  $("download").setAttribute("aria-disabled", "false");
  chart(history, result.predictions, result.forecast_origin);
}
async function refresh(datasetChanged = false) {
  datasetChanged = datasetChanged || state.loadedDataset !== $("dataset").value;
  const sequence = ++state.sequence;
  state.controller?.abort();
  state.controller = new AbortController();
  const signal = state.controller.signal;
  document.body.classList.add("loading");
  $("price-chart").setAttribute("aria-busy", "true");
  $("error").classList.add("hidden");
  $("download").classList.add("disabled");
  $("download").setAttribute("aria-disabled", "true");
  if (datasetChanged) for (const id of ["country", "species", "size", "horizon"]) $(id).disabled = true;
  try {
    if (datasetChanged) {
      const [series, quality, evaluation] = await Promise.all([
        get(base() + "/series", signal),
        get(base() + "/quality", signal),
        get(base() + "/evaluation", signal),
      ]);
      if (sequence !== state.sequence) return;
      Object.assign(state, { series, quality, evaluation });
      rebuild();
      state.loadedDataset = $("dataset").value;
      const unit = $("dataset").value === "benchmark" ? "month" : "week";
      const previous = $("horizon").value;
      setOptions(
        "horizon",
        Array.from({ length: quality.max_horizon }, (_, i) => [
          String(i + 1),
          `${i + 1} ${unit}${i ? "s" : ""}`,
        ]),
        previous,
      );
    }
    if (!$("size").value) throw new Error("No matching price series is available.");
    const params = new URLSearchParams({ series_id: $("size").value, horizon: $("horizon").value });
    const [history, result] = await Promise.all([
      get(base() + "/history?" + params, signal),
      get(base() + "/forecast?" + params, signal),
    ]);
    if (sequence !== state.sequence) return;
    render(history, result);
    $("service-state").textContent = "Model loaded";
  } catch (error) {
    if (error.name === "AbortError" || sequence !== state.sequence) return;
    $("error").textContent = error.message;
    $("error").classList.remove("hidden");
    $("service-state").textContent = "Data unavailable";
    $("price-chart").textContent = "Forecast unavailable for this selection.";
    for (const id of ["next-price", "model-name", "test-mae", "history-count"]) $(id).textContent = "—";
  } finally {
    if (sequence === state.sequence) {
      document.body.classList.remove("loading");
      $("price-chart").setAttribute("aria-busy", "false");
    }
  }
}
$("dataset").addEventListener("change", () => refresh(true));
$("country").addEventListener("change", () => {
  rebuild("species");
  refresh();
});
$("species").addEventListener("change", () => {
  rebuild("size");
  refresh();
});
$("size").addEventListener("change", () => refresh());
$("horizon").addEventListener("change", () => refresh());
refresh(true);
