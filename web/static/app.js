/* CGate-MRI demo client: upload -> analyze -> render. Plain fetch, no frameworks.
   Every value rendered here comes from the server's real pipeline output. */

const $ = (id) => document.getElementById(id);
let token = null;

function showError(id, msg) {
  const el = $(id);
  el.textContent = msg;
  el.hidden = false;
}
function clearError(id) { $(id).hidden = true; }

async function jsonOrThrow(url, options) {
  const r = await fetch(url, options);
  let data = null;
  try { data = await r.json(); } catch (_) { /* non-JSON body */ }
  if (!r.ok || (data && data.error)) {
    throw new Error((data && data.error) || `request failed (HTTP ${r.status})`);
  }
  return data;
}

/* --- step 1: upload ---------------------------------------------------- */

const uploadForm = $("upload-form");
if (uploadForm) {
  const input = $("file-input");
  input.addEventListener("change", () => {
    const n = input.files.length;
    $("file-count").textContent =
      n === 0 ? "" : n === 1
        ? "1 file selected"
        : `${n} files selected (one case: t1, t1ce, t2, flair)`;
  });

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError("upload-error");
    if (!input.files.length) {
      showError("upload-error",
        "Choose the files first: one .zip, or the four .nii/.nii.gz sequences together.");
      return;
    }

    const btn = $("upload-btn");
    btn.disabled = true;
    btn.textContent = "Uploading…";
    try {
      const fd = new FormData();
      for (const f of input.files) fd.append("file", f);  // one form field, all files
      const data = await jsonOrThrow("/api/upload", { method: "POST", body: fd });
      token = data.token;
      $("preview-img").src = data.preview_url;
      $("case-id").textContent = data.patient_id;
      $("preview-card").hidden = false;
      $("results").hidden = true;
      clearError("analyze-error");
      $("preview-card").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError("upload-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Upload";
    }
  });
}

/* --- step 2: analyze ---------------------------------------------------- */

const analyzeBtn = $("analyze-btn");
if (analyzeBtn) {
  analyzeBtn.addEventListener("click", async () => {
    if (!token) { showError("analyze-error", "Upload a case first."); return; }
    clearError("analyze-error");
    const useVlm = $("use-vlm").checked;

    analyzeBtn.disabled = true;
    $("loading").hidden = false;
    $("loading-text").textContent = useVlm
      ? "Running the three experts, then routing and gating — and loading the VLM (first time only)…"
      : "Running radiomics, CNN and ViT experts, then quality routing and the confidence gate…";
    try {
      const data = await jsonOrThrow("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, use_vlm: useVlm }),
      });
      render(data);
    } catch (err) {
      showError("analyze-error", err.message);
    } finally {
      analyzeBtn.disabled = false;
      $("loading").hidden = true;
    }
  });
}

/* --- render the results -------------------------------------------------- */

function render(d) {
  // gate banner + confidence bar
  const released = d.accepted;
  const banner = $("gate-banner");
  banner.className = "banner " + (released ? "banner-ok" : "banner-warn");
  const badge = $("gate-badge");
  badge.className = "badge " + (released ? "badge-ok" : "badge-warn");
  badge.textContent = released ? "RELEASED" : "DEFERRED";
  $("gate-text").textContent = released
    ? `Confidence ${d.confidence.toFixed(3)} ≥ threshold ${d.gate_threshold.toFixed(3)} — the gate releases this case.`
    : `Confidence ${d.confidence.toFixed(3)} < threshold ${d.gate_threshold.toFixed(3)} — the gate holds this case for review.`;

  $("conf-fill").style.width = (d.confidence * 100).toFixed(1) + "%";
  $("conf-threshold").style.left = (d.gate_threshold * 100).toFixed(1) + "%";
  $("conf-caption").textContent =
    "Case confidence is the weakest calibrated probability among critical findings " +
    "(a minimum, not an average — one shaky finding holds a case back). " +
    `The threshold was fitted to release 80% of cases; this scan scored ${d.confidence.toFixed(3)}.`;

  // findings table
  const tbody = $("findings-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const f of d.findings) {
    const tr = document.createElement("tr");
    const pillClass = f.class === "present" ? "pill-present"
      : (f.class === "severe" || f.class === "high") ? "pill-severe" : "pill-absent";
    tr.innerHTML =
      `<td>${esc(f.label)}${f.critical ? "" : ' <span class="muted">(not critical)</span>'}</td>` +
      `<td><span class="pill ${pillClass}">${esc(f.class)}</span></td>` +
      `<td><div class="pbar"><i style="width:${(f.p * 100).toFixed(1)}%"></i>` +
      `<b>${(f.p * 100).toFixed(1)}%</b></div></td>`;
    tbody.appendChild(tr);
  }

  // routing table
  if (d.routing) {
    const rt = $("routing-table").querySelector("tbody");
    rt.innerHTML = "";
    for (const row of d.routing) {
      const tr = document.createElement("tr");
      let cells = `<td>${esc(row.finding)}</td>`;
      for (const name of d.expert_names) {
        const w = row.weights[name];
        cells += `<td class="wcell">${(w * 100).toFixed(0)}%` +
          `<div class="wbar"><i class="w-${esc(name)}" style="width:${(w * 100).toFixed(1)}%"></i></div></td>`;
      }
      tr.innerHTML = cells;
      rt.appendChild(tr);
    }
    $("quality-note").textContent =
      `Router input: this scan's image-quality profile (mean |z| = ${d.q_mean_abs_z.toFixed(2)} ` +
      `across ${Object.keys(d.iqms).length} metrics). The router shifts weight toward whichever ` +
      `expert historically does best at this quality level.`;
  } else {
    $("routing-card").hidden = true;
  }

  // VLM phrases (advisory; vetoes are enforced server-side)
  const vlm = d.vlm || {};
  const vlmCard = $("vlm-card");
  if (vlm.status && vlm.status !== "not requested") {
    vlmCard.hidden = false;
    $("vlm-status").textContent = `Model status: ${vlm.status}`;
    const kept = $("vlm-phrases");
    kept.innerHTML = "";
    for (const p of vlm.phrases || []) {
      const li = document.createElement("li");
      li.textContent = p;
      kept.appendChild(li);
    }
    if (!(vlm.phrases || []).length) {
      kept.innerHTML = '<li class="muted">No VLM phrase survived the veto filter for this case.</li>';
    }
    const vetoed = $("vlm-vetoed");
    vetoed.innerHTML = "";
    for (const v of vlm.vetoed || []) {
      const li = document.createElement("li");
      li.innerHTML = `<s>${esc(v.phrase)}</s> <span class="veto">vetoed: ${esc(v.reason)}</span>`;
      vetoed.appendChild(li);
    }
  } else {
    vlmCard.hidden = true;
  }

  $("report-link").href = d.report_url;
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
