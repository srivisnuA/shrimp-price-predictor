const $ = (id) => document.getElementById(id);
const state = { history: null, forecast: null };

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}
function toast(msg, ok=true) {
  const t=$("toast"); t.textContent=msg; t.className="toast "+(ok?"ok":"err");
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
  $("region").innerHTML=rows.map(x=>`<option value="${x.region}">${x.region.replaceAll("_"," ")} — ${x.country}</option>`).join("");
}
async function loadHistory() {
  const region=$("region").value;
  const rows=await fetchJSON("/api/regional/history?region="+encodeURIComponent(region));
  if(state.history) state.history.destroy();
  state.history=new Chart($("history-chart"), {
    data:{labels:rows.map(x=>x.date),datasets:[
      {type:"line",label:"Price",data:rows.map(x=>x.price),borderColor:"#26c6a6",yAxisID:"price",tension:.25},
      {type:"bar",label:"Disease events",data:rows.map(x=>x.disease),backgroundColor:"rgba(255,107,107,.25)",yAxisID:"disease"}
    ]},
    options:{responsive:true,interaction:{mode:"index",intersect:false},scales:{
      price:{position:"left",title:{display:true,text:"USD/kg"}},
      disease:{position:"right",grid:{drawOnChartArea:false},title:{display:true,text:"Disease pressure"}}
    }}
  });
}
async function loadForecast() {
  const h=+$("horizon").value;
  $("horizon-value").textContent=h;
  $("temp-value").textContent=$("use-scenario").checked ? $("sc-temp").value+"°C" : "seasonal";
  $("rain-value").textContent=$("use-scenario").checked ? $("sc-rain").value : "seasonal";
  $("disease-value").textContent=$("use-scenario").checked ? $("sc-disease").value : "seasonal";
  const region=$("region").value, sc=scenario();
  const qs=new URLSearchParams({region,horizon:h});
  for(const [k,v] of Object.entries(sc)) qs.set(k,v);
  const d=await fetchJSON("/api/forecast/regional?"+qs.toString());
  $("metric-model").textContent=d.model.replaceAll("_"," ");
  $("metric-mae").textContent=Number(d.metrics.mae).toFixed(2);
  $("metric-rmse").textContent=Number(d.metrics.rmse).toFixed(2);
  $("metric-next").textContent="$"+Number(d.predictions[0].prediction).toFixed(2);
  if(state.forecast) state.forecast.destroy();
  state.forecast=new Chart($("forecast-chart"),{
    data:{labels:d.predictions.map(x=>x.date),datasets:[
      {type:"line",label:"Upper",data:d.predictions.map(x=>x.upper),borderWidth:0,pointRadius:0,fill:"+1",backgroundColor:"rgba(79,142,247,.16)"},
      {type:"line",label:"Lower",data:d.predictions.map(x=>x.lower),borderWidth:0,pointRadius:0},
      {type:"line",label:"Forecast",data:d.predictions.map(x=>x.prediction),borderColor:"#4f8ef7",borderDash:[6,4],tension:.25}
    ]},
    options:{responsive:true,scales:{y:{title:{display:true,text:"USD/kg"}}}}
  });
}
async function refresh(){ try { $("error-banner").classList.add("hidden"); await loadHistory(); await loadForecast(); } catch(e) { console.error(e); $("error-banner").classList.remove("hidden"); toast(e.message,false); } }
$("region").addEventListener("change",refresh);
$("horizon").addEventListener("input",loadForecast);
$("use-scenario").addEventListener("change",loadForecast);
for(const id of ["sc-temp","sc-rain","sc-disease"]) $(id).addEventListener("change",loadForecast);
(async()=>{try{await loadRegions();await refresh();}catch(e){$("error-banner").classList.remove("hidden");toast(e.message,false);}})();