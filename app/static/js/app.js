// RoomGuard frontend: validation first, then API calls. All UI text is English.
const $ = (s) => document.querySelector(s);

function showError(el, msg) { el.textContent = msg; el.hidden = !msg; }

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const light = parseFloat($("#light").value), sound = parseFloat($("#sound").value);
  if (isNaN(light) || isNaN(sound)) return showError($("#form-error"), "Both Light and Sound are required numbers.");
  if (light < 0 || light > 2000) return showError($("#form-error"), "Light must be between 0 and 2000 Lux.");
  if (sound < 20 || sound > 120) return showError($("#form-error"), "Sound must be between 20 and 120 dB.");
  showError($("#form-error"), "");
  const r = await fetch("/api/predict", { method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ light_lux: light, sound_db: sound }) });
  const d = await r.json();
  if (!r.ok) return showError($("#form-error"), d.error || "Prediction failed.");
  $("#result").hidden = false;
  $("#result").className = d.prediction === "Occupied" ? "occupied" : "";
  $("#result-label").textContent = d.prediction === "Occupied" ? "🟢 Occupied" : "⚪ Empty";
  $("#result-prob").textContent = `Confidence: ${(d.probability * 100).toFixed(1)}%  (Empty ${(d.probabilities.Empty * 100).toFixed(1)}% / Occupied ${(d.probabilities.Occupied * 100).toFixed(1)}%) — record #${d.id}`;
  $("#result-desc").textContent = d.description;
  $("#result-rec").textContent = d.recommendation;
  $("#result-path").innerHTML = d.decision_path.map((s) => `<li>${s}</li>`).join("");
  loadHistory();
});

document.querySelectorAll(".presets button").forEach((b) => b.addEventListener("click", () => {
  $("#light").value = b.dataset.light; $("#sound").value = b.dataset.sound;
}));

$("#sample-csv").addEventListener("click", (e) => {
  e.preventDefault();
  const csv = "Light_Lux,Sound_dB,Occupied\n5,36,0\n120,56,1\n90,37,0\n8,52,1\n0,35,0\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "roomguard_sample.csv"; a.click();
});

document.getElementById("batch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("#csv-file").files[0];
  if (!file) return showError($("#batch-error"), "Choose a CSV file first.");
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch("/api/batch", { method: "POST", body: fd });
  const d = await r.json();
  if (!r.ok) return showError($("#batch-error"), d.error || "Batch prediction failed.");
  showError($("#batch-error"), "");
  $("#batch-result").hidden = false;
  $("#batch-summary").textContent = `${d.n_predicted} rows classified` +
    (d.batch_accuracy !== null ? ` — batch accuracy on labelled rows: ${(d.batch_accuracy * 100).toFixed(1)}% (${d.n_labelled} labelled)` : "") +
    (d.errors.length ? ` — ${d.errors.length} rows rejected (see console).` : "");
  $("#batch-body").innerHTML = d.results.slice(0, 100).map((x) =>
    `<tr><td>${x.light_lux}</td><td>${x.sound_db}</td><td>${x.prediction}</td><td>${(x.probability * 100).toFixed(1)}%</td><td>${x.true_label || "-"}</td></tr>`).join("");
  loadHistory();
});

async function loadHistory() {
  const r = await fetch("/api/history"); const rows = await r.json();
  $("#history-body").innerHTML = rows.map((x) => `<tr>
    <td>${x.id}</td><td>${x.light_lux}</td><td>${x.sound_db}</td><td>${x.prediction}</td>
    <td>${(x.probability * 100).toFixed(1)}%</td>
    <td><select data-id="${x.id}" class="true-label">
      <option value="">—</option>
      <option ${x.true_label === "Empty" ? "selected" : ""}>Empty</option>
      <option ${x.true_label === "Occupied" ? "selected" : ""}>Occupied</option></select></td>
    <td><input data-id="${x.id}" class="note" value="${(x.note || "").replace(/"/g, "&quot;")}" placeholder="add note"></td>
    <td><button class="save" data-id="${x.id}">Save</button>
        <button class="del" data-id="${x.id}">Delete</button></td></tr>`).join("");
  document.querySelectorAll(".save").forEach((b) => b.addEventListener("click", async () => {
    const id = b.dataset.id;
    const note = document.querySelector(`.note[data-id="${id}"]`).value;
    const true_label = document.querySelector(`.true-label[data-id="${id}"]`).value;
    if (!confirm("Save this correction?")) return;
    await fetch(`/api/history/${id}`, { method: "PUT", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ note, true_label }) });
    loadHistory();
  }));
  document.querySelectorAll(".del").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Delete this prediction record? This cannot be undone.")) return;
    await fetch(`/api/history/${b.dataset.id}`, { method: "DELETE" });
    loadHistory();
  }));
}
$("#refresh-history").addEventListener("click", loadHistory);
loadHistory();
