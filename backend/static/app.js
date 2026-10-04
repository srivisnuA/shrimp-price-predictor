const $ = (id) => document.getElementById(id);
const state = { history: null, forecast: null, exportRegions: [], farmgateOptions: [] };

function priceType() {
  return document.querySelector('input[name="price-type"]:checked').value;
}

async function fetchJSON(url) {
  const r = await fetch(url);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

function toast(msg, ok=true) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast " + (ok ? "ok" : "err");
}

function selectedFarmgate() {
  return {
    country_code: $("fg-country").value,
    species: $("fg-species").value,
    size: +$("fg-size").value
  };
}

function scenario() {
  if (!$("use-scenario").checked) return {};
  return {
    temperature_c: +$("sc-temp").value,
    rainfall_mm: +$("sc-rain").value,
    disease_events: +$("sc-disease").value
  };
}

function updateHorizonUI() {
  const farm = priceType() === "farm_gate";
  const h = $("horizon");
  h.max = farm ? 52 : 36;
  if (farm && +h.value > 52) h.value = 26;
  if (!farm && +h.value > 36) h.value = 12;
  $("horizon-unit").textContent = farm ? "weeks" : "months";
  $("horizon-value").textContent = h.value;
}

function updateModeUI() {
  const farm = priceType() === "farm_gate";
  $("export-controls").classList.toggle("hidden", farm);
  $("farmgate-controls").classList.toggle("hidden", !farm);
  $("price-basis-note").classList.toggle("hidden", farm);
  $("farmgate-note").classList.toggle("hidden", !farm);
  $("history-title").textContent = farm ? "Historical farm-gate price, weather and disease" : "Historical export price, weather and disease";
  $("forecast-title").textContent = farm ? "Future farm-gate shrimp price" : "Future export shrimp price";
  updateHorizonUI();
}

async function loadExportRegions() {
  state.exportRegions = await fetchJSON("/api/regional/options");
  $("region").innerHTML = state.exportRegions
    .map(x => "<option value="" + x.region + "">" + x.region.replaceAll("_", " ") + " — " + x.country + "</option>")
    .join("");
}

function rebuildFarmgateSelectors() {
  const country = $("fg-country").value;
  const countryRows = state.farmgateOptions.filter(x => x.country_code === country);

  const species = [...new Set(countryRows.map(x => x.species))].sort();
  const currentSpecies = $("fg-species").value;
  $("fg-species").innerHTML = species
    .map(x => "<option value="" + x + "">" + x + "</option>")
    .join("");
  if (species.includes(currentSpecies)) $("fg-species").value = currentSpecies;

  const activeSpecies = $("fg-species").value;
  const sizes = countryRows
    .filter(x => x.species === activeSpecies)
    .sort((a,b) => a.size - b.size);

  const currentSize = $("fg-size").value;
  $("fg-size").innerHTML = sizes
    .map(x => "<option value="" + x.size + "">" + x.size + " count/kg — " + x.observations + " obs.</option>")
    .join("");
  if (sizes.some(x => String(x.size) === currentSize)) $("fg-size").value = currentSize;

  const selected = sizes.find(x => String(x.size) === $("fg-size").value);
  $("farmgate-selection-info").textContent = selected
    ? country + " · " + activeSpecies + " · " + selected.size + " count/kg · " + selected.observations + " weekly observations"
    : "No matching farm-gate series.";
}

async function loadFarmgateOptions() {
  state.farmgateOptions = await fetchJSON("/api/farmgate/options");
  const countries = [...new Map(state.farmgateOptions.map(x => [x.country_code, x.country]))]
    .sort((a,b) => a[1].localeCompare(b[1]));

  $("fg-country").innerHTML = countries
    .map(x => "<option value="" + x[0] + "">" + x[1] + "</option>")
    .join("");
  rebuildFarmgateSelectors();
}

async function loadHistory() {
  if (state.history) state.history.destroy();

  if (priceType() === "farm_gate") {
    const s = selectedFarmgate();
    const rows = await fetchJSON("/api/farmgate/history?" + new URLSearchParams(s).toString());
    state.history = new Chart($("history-chart"), {
      data: {
        labels: rows.map(x => x.date),
        datasets: [
          {type:"line", label:"Farm-gate USD/kg", data:rows.map(x=>x.price), borderColor:"#26c6a6", yAxisID:"price", tension:.25},
          {type:"bar", label:"Disease pressure", data:rows.map(x=>x.disease), backgroundColor:"rgba(255,107,107,.25)", yAxisID:"disease"}
        ]
      },
      options: {
        responsive:true,
        interaction:{mode:"index",intersect:false},
        scales:{
          price:{position:"left",title:{display:true,text:"USD/kg"}},
          disease:{position:"right",grid:{drawOnChartArea:false},title:{display:true,text:"Disease pressure"}}
        }
      }
    });
    return;
  }

  const region = $("region").value;
  const rows = await fetchJSON("/api/regional/history?region=" + encodeURIComponent(region));
  state.history = new Chart($("history-chart"), {
    data: {
      labels: rows.map(x=>x.date),
      datasets: [
        {type:"line",label:"Export price",data:rows.map(x=>x.export_price),borderColor:"#26c6a6",yAxisID:"price",tension:.25},
        {type:"bar",label:"Disease pressure",data:rows.map(x=>x.disease),backgroundColor:"rgba(255,107,107,.25)",yAxisID:"disease"}
      ]
    },
    options:{
      responsive:true,
      interaction:{mode:"index",intersect:false},
      scales:{
        price:{position:"left",title:{display:true,text:"USD/kg"}},
        disease:{position:"right",grid:{drawOnChartArea:false},title:{display:true,text:"Disease pressure"}}
      }
    }
  });
}

async function loadForecast() {
  updateModeUI();
  const farm = priceType() === "farm_gate";
  const h = +$("horizon").value;
  $("horizon-value").textContent = h;
  $("temp-value").textContent = $("use-scenario").checked ? $("sc-temp").value + "°C" : "seasonal";
  $("rain-value").textContent = $("use-scenario").checked ? $("sc-rain").value : "seasonal";
  $("disease-value").textContent = $("use-scenario").checked ? $("sc-disease").value : "seasonal";

  let data;
  if (farm) {
    const s = selectedFarmgate();
    data = await fetchJSON("/api/farmgate/forecast?" + new URLSearchParams({...s, horizon_weeks:h, ...scenario()}).toString());
  } else {
    data = await fetchJSON("/api/forecast/regional?" + new URLSearchParams({
      region:$("region").value, horizon:h, price_type:"export", ...scenario()
    }).toString());
  }

  $("metric-model").textContent = data.model.replaceAll("_"," ");
  $("metric-history").textContent = farm ? data.history_weeks : h;
  $("metric-history-sub").textContent = farm ? "weekly target observations" : "forecast months";
  $("metric-mae").textContent = Number(data.metrics.mae).toFixed(2);
  $("metric-next").textContent = "$" + Number(data.predictions[0].prediction).toFixed(2);

  if (data.warning) {
    $("warning-card").classList.remove("hidden");
    $("warning-text").textContent = data.warning;
  } else {
    $("warning-card").classList.add("hidden");
  }

  if (state.forecast) state.forecast.destroy();
  state.forecast = new Chart($("forecast-chart"), {
    data:{
      labels:data.predictions.map(x=>x.date),
      datasets:[
        {type:"line",label:"Upper",data:data.predictions.map(x=>x.upper),borderWidth:0,pointRadius:0,fill:"+1",backgroundColor:"rgba(79,142,247,.16)"},
        {type:"line",label:"Lower",data:data.predictions.map(x=>x.lower),borderWidth:0,pointRadius:0},
        {type:"line",label:farm ? "Farm-gate forecast" : "Export forecast",data:data.predictions.map(x=>x.prediction),borderColor:"#4f8ef7",borderDash:[6,4],tension:.25}
      ]
    },
    options:{responsive:true,scales:{y:{title:{display:true,text:"USD/kg"}}}}
  });
}

async function refresh() {
  try {
    $("error-banner").classList.add("hidden");
    updateModeUI();
    await loadHistory();
    await loadForecast();
  } catch(e) {
    console.error(e);
    $("error-banner").textContent = e.message;
    $("error-banner").classList.remove("hidden");
    toast(e.message,false);
  }
}

$("price-export").addEventListener("change", refresh);
$("price-farmgate").addEventListener("change", refresh);
$("region").addEventListener("change", refresh);
$("fg-country").addEventListener("change", async () => { rebuildFarmgateSelectors(); await refresh(); });
$("fg-species").addEventListener("change", async () => { rebuildFarmgateSelectors(); await refresh(); });
$("fg-size").addEventListener("change", refresh);
$("horizon").addEventListener("input", loadForecast);
$("use-scenario").addEventListener("change", loadForecast);
for (const id of ["sc-temp","sc-rain","sc-disease"]) $(id).addEventListener("change", loadForecast);

(async()=>{
  try {
    await Promise.all([loadExportRegions(), loadFarmgateOptions()]);
    await refresh();
  } catch(e) {
    $("error-banner").textContent = e.message;
    $("error-banner").classList.remove("hidden");
    toast(e.message,false);
  }
})();
