const $ = (id) => document.getElementById(id);
const COLORS = { Low: "#1a7f4b", Medium: "#b7791f", High: "#c0392b" };
let chart = null;

function parseReadings(text) {
  return text.split(/[\s,;]+/).filter(Boolean).map(Number);
}

function sample(theft) {
  const base = 8 + Math.random() * 12;
  const out = [];
  for (let d = 0; d < 30; d++) {
    let v = base * (1 + 0.15 * Math.sin((2 * Math.PI * d) / 7)) * (0.85 + Math.random() * 0.3);
    if (theft && d >= 14) v *= 0.15 + Math.random() * 0.25;
    out.push(v.toFixed(1));
  }
  return out.join(", ");
}

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function drawChart(values) {
  const ctx = $("chart").getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: values.map((_, i) => i + 1),
      datasets: [{ label: "kWh per day", data: values, borderColor: "#1d5fd0",
                   backgroundColor: "rgba(29,95,208,.12)", fill: true, tension: .25, pointRadius: 2 }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, title: { display: true, text: "Daily consumption (kWh)" } },
      scales: { x: { title: { display: true, text: "Day" } }, y: { beginAtZero: true } },
    },
  });
}

function showResult(r, values) {
  $("resultCard").hidden = false;
  const pct = (r.probability * 100).toFixed(1);
  $("verdict").textContent = r.is_theft ? "Likely electricity theft" : "Looks normal";
  $("verdict").style.color = COLORS[r.risk];
  $("meterFill").style.width = pct + "%";
  $("meterFill").style.background = COLORS[r.risk];
  $("probText").textContent = `Theft probability ${pct}%  |  ${r.risk} risk  |  ${r.days} days analysed`;
  $("reasons").innerHTML = r.reasons.map((t) => `<li>${t}</li>`).join("");
  $("featTable").innerHTML = Object.entries(r.features)
    .map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td>${v}</td></tr>`).join("");
  drawChart(values);
  $("resultCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("analyze").addEventListener("click", async () => {
  $("error").textContent = "";
  const values = parseReadings($("readings").value);
  if (!values.length) { $("error").textContent = "Enter some readings first."; return; }
  $("analyze").disabled = true;
  try {
    const r = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ readings: values }),
    });
    showResult(r, values);
  } catch (e) {
    $("error").textContent = e.message;
  } finally {
    $("analyze").disabled = false;
  }
});

$("sampleNormal").addEventListener("click", () => { $("readings").value = sample(false); });
$("sampleTheft").addEventListener("click", () => { $("readings").value = sample(true); });
$("clear").addEventListener("click", () => {
  $("readings").value = ""; $("error").textContent = ""; $("resultCard").hidden = true;
});

$("uploadBtn").addEventListener("click", async () => {
  $("batchError").textContent = "";
  const file = $("csvFile").files[0];
  if (!file) { $("batchError").textContent = "Choose a CSV file first."; return; }
  const form = new FormData();
  form.append("file", file);
  $("uploadBtn").disabled = true;
  try {
    const d = await api("/api/predict-batch", { method: "POST", body: form });
    $("batchSummary").textContent =
      `${d.total} consumers scored, ${d.flagged} flagged as likely theft` +
      (d.skipped ? `, ${d.skipped} rows skipped` : "") + ".";
    const body = $("batchTable").querySelector("tbody");
    body.innerHTML = "";
    d.results.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td></td><td>${(r.probability * 100).toFixed(1)}%</td>` +
                     `<td><span class="pill ${r.risk}">${r.risk}</span></td>`;
      tr.firstChild.textContent = r.consumer;
      body.appendChild(tr);
    });
    $("batchTable").hidden = false;
  } catch (e) {
    $("batchError").textContent = e.message;
    $("batchTable").hidden = true;
  } finally {
    $("uploadBtn").disabled = false;
  }
});

(async function loadMetrics() {
  try {
    const m = await api("/api/metrics");
    const items = [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1-score", m.f1]];
    $("metrics").innerHTML = items
      .map(([n, v]) => `<div class="metric"><b>${(v * 100).toFixed(1)}%</b>${n}</div>`).join("") +
      `<div class="metric"><b>${m.model}</b>${m.samples} ${m.data} samples</div>`;
  } catch {
    $("metrics").textContent = "Metrics not available.";
  }
})();
