const $ = (id) => document.getElementById(id);
const state = { history: null, forecast: null };

function priceType() {
  return document.querySelector('input[name="price-type"]:checked').value;
}

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

function toast(msg, ok=true) {
  const t=$("toast");
  t.textContent=msg;
  t.className="toast "+(ok?"ok":"err");
}

function setPriceLabels() {
  const farm = priceType() === "farm_gate";
  $("price-basis-note").classList.toggle("hidden", farm);
  $("farmgate-note").classList.toggle("hidden", !farm);
  $("history-title").textContent = farm
    ? "Historical farm-gate price, weather and disease"
    : "Historical export price, weather and disease";
  $("forecast-title").textContent = farm
    ? "Future farm-gate shrimp price"
    : "Future export shrimp price";
}

function scenario() {
  if (!$("use-scenario").checked) return {};
  return {
    temperature_c: +$("sc-temp").value,
    rainfall_mm: +$("sc-rain").value,
    disease_events: +$("sc-disease").value
  };
}

async function loadRegions() {
  const rows=await fetchJSON("/api/regional/options");
  $("region").innerHTML=rows.map(x =>
    `<option value="${x.region}">${x.region.replaceAll("_"," ")} — ${x.country}</option>`
  ).join("");
}

async function loadHistory() {
  const region=$("region").value;
  const rows=await fetchJSON("/api/regional/history?region="+encodeURIComponent(region));
  const farm = priceType() === "farm_gate";
  const values = farm ? rows.map(x=>x.farm_gate_price) : rows.map(x=>x.export_price);

  if(state.history) state.history.destroy();
  state.history=new Chart($("history-chart"), {
    data:{
      labels:rows.map(x=>x.date),
      datasets:[
        {type:"line",label:farm?"Farm-gate price":"Export price",data:values,borderColor:"#26c6a6",yAxisID:"price",tension:.25},
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
  setPriceLabels();

  const h=+$("horizon").value;
  $("horizon-value").textContent=h;
  $("temp-value").textContent=$("use-scenario").checked ? $("sc-temp").value+"°C" : "seasonal";
  $("rain-value").textContent=$("use-scenario").checked ? $("sc-rain").value : "seasonal";
  $("disease-value").textContent=$("use-scenario").checked ? $("sc-disease").value : "seasonal";

  const region=$("region").value;
  const sc=scenario();
  const qs=new URLSearchParams({region,horizon:h,price_type:priceType()});
  for(const [k,v] of Object.entries(sc)) qs.set(k,v);

  const d=await fetchJSON("/api/forecast/regional?"+qs.toString());

  $("metric-model").textContent=d.model.replaceAll("_"," ");
  $("metric-history").textContent=d.history_months;
  $("metric-mae").textContent=Number(d.metrics.mae).toFixed(2);
  $("metric-next").textContent="$"+Number(d.predictions[0].prediction).toFixed(2);

  if(d.warning){
    $("warning-card").classList.remove("hidden");
    $("warning-text").textContent=d.warning;
  } else {
    $("warning-card").classList.add("hidden");
  }

  if(state.forecast) state.forecast.destroy();
  state.forecast=new Chart($("forecast-chart"),{
    data:{
      labels:d.predictions.map(x=>x.date),
      datasets:[
        {type:"line",label:"Upper",data:d.predictions.map(x=>x.upper),borderWidth:0,pointRadius:0,fill:"+1",backgroundColor:"rgba(79,142,247,.16)"},
        {type:"line",label:"Lower",data:d.predictions.map(x=>x.lower),borderWidth:0,pointRadius:0},
        {type:"line",label:priceType()==="farm_gate"?"Farm-gate forecast":"Export forecast",data:d.predictions.map(x=>x.prediction),borderColor:"#4f8ef7",borderDash:[6,4],tension:.25}
      ]
    },
    options:{responsive:true,scales:{y:{title:{display:true,text:"USD/kg"}}}}
  });
}

async function refresh(){
  try {
    $("error-banner").classList.add("hidden");
    await loadHistory();
    await loadForecast();
  } catch(e) {
    console.error(e);
    $("error-banner").textContent = e.message;
    $("error-banner").classList.remove("hidden");
    toast(e.message,false);
  }
}

$("region").addEventListener("change",refresh);
$("horizon").addEventListener("input",loadForecast);
$("use-scenario").addEventListener("change",loadForecast);
for(const id of ["sc-temp","sc-rain","sc-disease"]) $(id).addEventListener("change",loadForecast);
for(const id of ["price-export","price-farmgate"]) $(id).addEventListener("change",refresh);

(async()=>{
  try {
    await loadRegions();
    await refresh();
  } catch(e) {
    $("error-banner").textContent = e.message;
    $("error-banner").classList.remove("hidden");
    toast(e.message,false);
  }
})();
