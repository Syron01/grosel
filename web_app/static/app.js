(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const num = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "—");
  const secs = v => (Number.isFinite(v) ? (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + " s" : "—");
  const gb = bytes => `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  const avg = arr => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  const esc = v => String(v ?? "").replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  // The engine reports tiers and statuses in Turkish; the shell is English.
  const EN = new Map([
    ["Oto", "Auto"],
    ["Sınır 16", "Preview 16"],
    ["Hızlı 20", "Floor 20"],
    ["Dengeli 28", "Balanced 28"],
    ["Ultra 40", "Ultra 40"],
    ["Hızlı 3–4", "Fast 3–4"],
    ["Orta 8", "Standard 8"],
    ["Pro 12", "Pro 12"],
    ["Ultra 20", "Ultra 20"],
    ["Kalite kademesine göre 8–40 adım; 1080p'de 40", "8–40 steps by quality tier; 40 at 1080p"],
    ["720p'de ~29 s; 20 adım kalite referansı, 16 hızlı önizleme", "~29 s at 720p; 20 is the quality reference, 16 a fast preview"],
    ["20 adım; Qwen-Image 2.1 tabanı için alt sınır", "20 steps; floor for the Qwen-Image 2.1 base"],
    ["28 adım; varsayılan kalite/süre dengesi", "28 steps; default quality and time balance"],
    ["40 adım; önerilen tam yörünge, en uzun yol", "40 steps; recommended full trajectory, longest run"],
    ["Hızlı: damıtılmış mod; 420p'de 3, diğer çözünürlüklerde 4 adım", "Fast: distilled mode; 3 steps at 420p, 4 elsewhere"],
    ["Orta: 8 adım; süre artar, kalite artışı garanti değil", "Standard: 8 steps; slower, quality gain not guaranteed"],
    ["Pro: 12 adım; süre artar, kalite artışı garanti değil", "Pro: 12 steps; slower, quality gain not guaranteed"],
    ["Ultra: 20 adım; en uzun yol, damıtılmış modelde kazanç garanti değil", "Ultra: 20 steps; longest path, gain not guaranteed on a distilled model"],
    ["Mat Chiaroscuro", "Matte chiaroscuro"],
    ["Sinematik 35mm Film", "35mm film"],
    ["Premium Anime", "Anime"],
    ["Ham Belgesel Fotoğraf", "Documentary"],
    ["Karanlık Barok / Rembrandt", "Baroque"],
    ["Siberpunk Neon Noir", "Neon noir"],
    ["Filtresiz (Ham Prompt)", "Raw prompt"],
    ["Motor hazırlanıyor...", "Preparing engine..."],
    ["Prompt kodlanıyor...", "Encoding prompt..."],
    ["Prompt katman akışıyla kodlanıyor...", "Encoding prompt through the layer stream..."],
    ["Geçersiz üretim isteği. Alanları denetleyip yeniden deneyin.", "Invalid generation request. Check the fields and try again."],
    ["Bilinmeyen stil.", "Unknown style."],
    ["Üretim tamamlanamadı. Motor günlüklerini denetleyin.", "Generation failed. Check the engine logs."],
    ["hiçbir kademe tavana sığmıyor", "every tier is above the ceiling"],
  ]);
  const en = v => {
    const s = String(v ?? "");
    if (EN.has(s)) return EN.get(s);
    return s.replace(/^Varyasyon (\d+)\/(\d+)/, "Variant $1/$2");
  };

  const state = {
    style: "matte_chiaroscuro",
    ratio: "16:9",
    quality: "1080p",
    count: 1,
    steps: 0,
    crpf: false,
    geo: true,
    detail: true,
    system: null,
    info: null,
    running: false,
    runCount: 1,
    tiles: [],
    ws: null,
    timer: null,
    reconnectTimer: null,
    startedAt: 0,
    samples: [
      "Loş karanlık odada masaya yaslanmış zarif Asyatik kız, dağınık dökülen saç telleri, kahverengi örgü hırka, arkada sakin yanan şömine alevi",
      "Karanlık antika saat atölyesinde pirinç mekanizmaları büyüteçle tamir eden yaşlı saat ustası, sarı gaz lambası ışığı, tozlu hava",
      "Yağmurlu Tokyo sokağında eski şemsiyesiyle yürüyen yalnız gezgin, ıslak asfalt yansımaları, hafif buğulu loş sokak feneri",
      "Eski ahşap yelkenli güvertesinde fırtınalı denizi izleyen sakallı yaşlı deniz kaptanı, fırtına lambası, koyu melankolik gökyüzü",
    ],
    sampleIdx: 0,
    last: null,
    autoSteps: null,
    dims: null,
    tierSig: "",
    styleAuto: true,
    ratioAuto: true,
    geoAuto: true,
    math: {
      data: null, result: null, loading: false,
      budget: 5, time: 8, floor: 0.45, steps: 4, warm: true, objective: "quality",
    },
  };

  const els = {};
  [
    "gpuName", "modeBadge", "connDot", "connText",
    "promptInput", "optimizePromptBtn", "quickSampleBtn", "styleContainer",
    "ratioMeta", "ratioRow", "qualityRow", "seedInput", "seedRandomBtn",
    "variantRow", "stepsRow", "stepsMeta", "crpfToggle", "geoToggle", "detailToggle", "generateBtn", "generateLabel",
    "stageMeta", "copyPromptBtn", "downloadBtn", "stage", "stageEmpty", "mainRender",
    "runLine", "variantGrid", "timelineTotal", "timelineBar", "timelineLegend",
    "historyStrip", "clearAllBtn", "liveTag", "mLatency", "mStep", "mVram", "mRes",
    "vramHead", "vramFill", "vramMark", "vramFoot", "phaseNote",
    "contractTier", "contractMark", "contractText", "contractBadges", "contractNote",
    "promiseHead", "promiseChart", "promiseNote",
    "phaseEnc", "phaseDen", "phaseDec", "phaseEncBar", "phaseDenBar", "phaseDecBar",
    "cacheHead", "moduleList", "eventLog", "lightbox", "lightboxImg", "lightboxClose",
    "benchStart", "benchStatus", "benchModeNote", "suiteChips", "benchBody", "benchSummaryMeta",
    "sAvg", "sCache", "sVram", "sTotal", "timeChart", "vramChart", "vramScale",
    "pipeBf16", "pipeFp8", "pipePinned", "pipeDual", "pipeStepsV",
    "compareBf16", "compareFast", "compareClassic",
    "archBf16", "archFp8", "archPinned", "archDual", "archModules",
    "fBudget", "fTf", "fTe", "dualNote", "archGpu", "archBudget", "archWeights",
    "archCache", "limitBf16", "docStamp", "toClassic", "toFast", "toNote",
    "costModeNote", "archLimitNote",
    "mathN", "mathClipN", "mathFitLat", "mathFitVram", "mathFitNote",
    "mathBudget", "mathBudgetOut", "mathTime", "mathTimeOut", "mathFloor", "mathFloorOut",
    "mathStepsRow", "mathWarmRow", "mathObjRow", "mathSolveBtn", "mathSolveStatus",
    "mathChosen", "mathShadow", "mathAlts", "mathPareto", "mathParetoNote", "mathPlansBody",
    "mathDataN",
  ].forEach(id => { els[id] = document.getElementById(id); });

  /* ---------------------------------------------------------------- utils */

  async function getJSON(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  }

  function setConn(ok) {
    els.connDot.classList.toggle("on", ok);
    els.connDot.classList.toggle("off", !ok);
    els.connText.textContent = ok ? "Connected" : "Offline";
  }

  function logEvent(text, kind = "") {
    const empty = els.eventLog.querySelector(".event.empty");
    if (empty) empty.remove();
    const row = document.createElement("div");
    row.className = `event ${kind}`;
    const time = document.createElement("time");
    time.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
    const message = document.createElement("span");
    message.textContent = String(text);
    row.append(time, message);
    els.eventLog.prepend(row);
    while (els.eventLog.children.length > 8) els.eventLog.lastChild.remove();
  }

  function setLive(tag, label) {
    els.liveTag.className = `live-tag ${tag}`;
    els.liveTag.textContent = label;
  }

  /* ------------------------------------------------------------- navigation */

  function setView(name) {
    $$(".tab").forEach(t => {
      const on = t.dataset.view === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
    });
    $$(".view").forEach(v => {
      const on = v.id === `view-${name}`;
      v.classList.toggle("active", on);
      v.hidden = !on;
    });
    if (name === "arch") loadSystem();
    if (name === "bench") loadSystem();
    if (name === "math") loadMath();
  }

  const tabs = $$(".tab");
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setView(tab.dataset.view));
    tab.addEventListener("keydown", event => {
      let next = null;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === null) return;
      event.preventDefault();
      setView(tabs[next].dataset.view);
      tabs[next].focus();
    });
  });

  /* --------------------------------------------------------------- system */

  const moduleName = id => ({ transformer: "Transformer", text_encoder: "Text encoder", vae: "VAE" }[id] || id);

  function applySystem(s) {
    state.system = s;
    const isFast = s.engine === "fast";
    const isQwen = s.engine === "qwen";
    els.gpuName.textContent = `${s.gpu.replace("NVIDIA GeForce ", "")} · ${num(s.vram_total_gb, 2)} GB`;
    els.modeBadge.textContent = isQwen
      ? (s.resident_mode ? "Qwen · 4-bit resident" : "Qwen · NF4/FP8")
      : (isFast ? (s.native_fp8 ? "FP8 · native" : "FP8") : "BF16 · classic");

    els.cacheHead.textContent = `cache ${s.cache_entries}`;

    els.moduleList.innerHTML = s.modules.map(m => `
      <div class="module-row">
        <span class="module-name">${moduleName(m.id)}</span>
        <span class="module-bytes">${gb(m.bytes)}</span>
        <span class="module-state ${m.resident ? "on" : ""}">${m.resident ? "GPU" : (isQwen && s.resident_mode ? "RAM · per phase" : "RAM")}</span>
      </div>
    `).join("");

    els.vramHead.textContent = `${num(s.vram_total_gb, 2)} GB card`;
    paintCeilingMark(s.vram_total_gb);

    const fp8 = s.weights_fp8_bytes;
    els.pipeBf16.textContent = gb(s.weights_bf16_bytes);
    els.pipeFp8.textContent = isFast
      ? gb(fp8)
      : (isQwen ? (s.resident_mode ? `${gb(s.weights_gpu_bytes)} VRAM` : `${gb(fp8)} streamed`) : "BF16");
    els.pipePinned.textContent = isFast
      ? `${gb(fp8)} pinned RAM`
      : (isQwen ? (s.resident_mode ? "4-bit pack · no block stream" : `${gb(fp8)} packed on CPU`) : "—");
    els.pipeDual.textContent = isQwen
      ? (s.resident_mode ? "phase order · 4-bit resident" : "phase order · encode → denoise → decode")
      : (s.dual_resident ? "dual · both models on GPU" : "single module · swap at encode boundary");
    els.pipeStepsV.textContent = state.last ? `${state.last.steps} steps` : "—";

    els.archBf16.textContent = gb(s.weights_bf16_bytes);
    els.archFp8.textContent = isFast ? gb(fp8) : (isQwen ? (s.resident_mode ? `${gb(s.weights_gpu_bytes)} 4-bit` : `${gb(fp8)} streamed`) : "BF16 mode");
    els.archPinned.textContent = isFast ? `${gb(fp8)} pinned on CPU` : (isQwen ? (s.resident_mode ? "4-bit · no blocks" : `${s.group_size || 2} blocks/group`) : "sequential CPU flow");
    els.archDual.textContent = isQwen ? (s.resident_mode ? "4-bit resident" : "phase order") : (s.dual_resident ? "dual resident" : "single resident");
    els.fBudget.textContent = s.budget_gb ? num(s.budget_gb, 2) : "—";
    const tf = s.modules.find(m => m.id === "transformer");
    const te = s.modules.find(m => m.id === "text_encoder");
    const vae = s.modules.find(m => m.id === "vae");
    els.fTf.textContent = tf ? num(tf.bytes / 1024 ** 3, 2) : "—";
    els.fTe.textContent = te ? num(te.bytes / 1024 ** 3, 2) : "—";
    els.dualNote.textContent = isQwen
      ? (s.resident_mode
        ? "the 4-bit transformer stays in VRAM; the text encoder streams only during encode"
        : "one phase at a time in VRAM; weights stream from CPU at each phase boundary")
      : (s.dual_resident
        ? "both models stay on the GPU at once; no swap"
        : "the two models do not fit together; the swap happens at the encode boundary");
    els.archGpu.textContent = s.gpu;
    els.archBudget.textContent = s.budget_gb ? `${num(s.budget_gb, 2)} GB` : "—";
    els.archWeights.textContent = `${gb(s.weights_bf16_bytes)} → ${isFast || isQwen ? gb(fp8) : "BF16"}`;
    els.archCache.textContent = `${s.cache_entries} embedding`;
    els.limitBf16.textContent = gb(s.weights_bf16_bytes);

    els.archModules.innerHTML = s.modules.map(m => `
      <tr>
        <td>${moduleName(m.id)}</td>
        <td class="num">${m.layers || "—"}</td>
        <td class="num">${gb(m.bytes)}</td>
        <td>${m.resident ? "GPU" : (isQwen && s.resident_mode ? "RAM · per phase" : "RAM · streamed")}</td>
      </tr>
    `).join("");

    els.compareBf16.textContent = s.weights_bf16_bytes / 1024 ** 3 > s.vram_total_gb
      ? `BF16 full residency — does not fit · ${gb(s.weights_bf16_bytes)} weights / ${num(s.vram_total_gb, 2)} GB VRAM`
      : `BF16 full residency — ${gb(s.weights_bf16_bytes)} weights`;

    if (s.weights_bf16_bytes) {
      const ratio = s.weights_bf16_bytes / Math.max(s.weights_fp8_bytes, 1);
      if (s.engine === "fast" && ratio > 1) {
        els.archWeights.textContent = `${gb(s.weights_bf16_bytes)} → ${gb(fp8)} · ${num(ratio, 2)}×`;
      }
    }

    if (s.engine === "classic") {
      els.compareClassic.textContent = "Classic flow — BF16 + CPU stream";
      els.detailToggle.style.display = "none";
    }

    if (isQwen) {
      els.compareFast.textContent = "Qwen-Image 2.1 · FP8 + NF4 stream";
      els.crpfToggle.style.display = "none";
      els.detailToggle.style.display = "none";
    }
    // The benchmark protocol is calibrated on the FLUX fast path; a live Qwen
    // suite would take ~20 minutes and overwrite that table.
    els.benchStart.disabled = isQwen;
    els.benchStart.querySelector("span").textContent = isQwen ? "FLUX-only protocol" : "Run benchmark";
    if (els.costModeNote) {
      els.costModeNote.textContent = isQwen
        ? "Qwen runs phases in order: max(encode, denoise, decode). The dual/swap formula below describes the FLUX fast path."
        : "";
    }
    if (els.archLimitNote) {
      els.archLimitNote.textContent = isQwen
        ? "The items below describe the FLUX fast path. The Qwen quality path draws natively at 832×1248 (2:3) in 40 steps, measured 1.37 GB peak."
        : "";
    }
    if (els.benchModeNote) {
      els.benchModeNote.textContent = isQwen
        ? "The stored table comes from the FLUX fast path. In Qwen mode the five runs take 20/28 steps; live phase telemetry is in the Studio output."
        : "";
    }

    configureTiers(s.tiers, s.auto_steps, s.dims);
    if (isQwen) {
      if (state.styleAuto) {
        state.style = "anime_concept";
        $$("button", els.styleContainer).forEach(b => b.classList.toggle("active", b.dataset.style === state.style));
        syncPressed(els.styleContainer);
      }
      if (state.ratioAuto) {
        state.ratio = "2:3";
        $$("button", els.ratioRow).forEach(b => b.classList.toggle("active", b.dataset.ratio === state.ratio));
        syncPressed(els.ratioRow);
        updateDimMeta();
      }
      if (state.geoAuto && state.geo) {
        state.geo = false;
        els.geoToggle.classList.remove("active");
        els.geoToggle.setAttribute("aria-pressed", "false");
      }
    }
    els.docStamp.textContent = `last read ${new Date().toLocaleTimeString("en-GB", { hour12: false })} · ${s.engine}`;
  }

  function configureTiers(tiers, auto, dims) {
    if (!Array.isArray(tiers) || !tiers.length) return;
    const sig = JSON.stringify([tiers.map(t => t.value), auto, dims]);
    if (sig === state.tierSig) return;
    state.tierSig = sig;
    state.autoSteps = auto || null;
    state.dims = dims || null;
    els.stepsRow.className = `seg seg-${tiers.length}`;
    els.stepsRow.innerHTML = tiers.map((t, i) => {
      const name = EN.get(t.label) || t.label || "";
      const note = EN.get(t.title) || t.title || "";
      return `<button class="${i === 0 ? "active" : ""}" data-steps="${t.value}" title="${esc([name, note].filter(Boolean).join(" · "))}">${t.value === 0 ? "Auto" : esc(String(t.value))}</button>`;
    }).join("");
    state.steps = 0;
    syncPressed(els.stepsRow);
    updateStepsMeta();
    updateDimMeta();
  }

  async function loadSystem() {
    try {
      applySystem(await getJSON("/api/system"));
    } catch {
      /* stay quiet while the engine is not ready */
    }
  }

  /* ------------------------------------------------------------- promise */

  const ceilingGb = () => (state.info && state.info.vram_ceiling_gb) || 0;

  function paintCeilingMark(cardGb) {
    const ceiling = ceilingGb();
    if (!ceiling || !cardGb) {
      els.vramMark.style.display = "none";
      return;
    }
    els.vramMark.style.display = "block";
    els.vramMark.style.left = `${clamp(ceiling / cardGb * 100, 0, 100)}%`;
    els.vramMark.title = `product ceiling ${num(ceiling, 1)} GB`;
  }

  async function loadPromise() {
    try {
      state.info = await getJSON("/api/info");
    } catch {
      state.info = null;
    }
    renderPromise();
    if (state.system) paintCeilingMark(state.system.vram_total_gb);
    if (state.last) renderContract(state.last);
  }

  function renderPromise() {
    const info = state.info;
    if (!info || !els.promiseChart) return;
    const ceiling = info.vram_ceiling_gb || 0;
    const measured = info.measured || {};
    const ladder = (info.tier_ladder && info.tier_ladder.length ? info.tier_ladder : Object.keys(measured));
    const worst = Math.max(ceiling, ...ladder.map(t => (measured[t] || {}).worst_gb || 0), 1);
    els.promiseHead.textContent = `${info.backend} · ceiling ${num(ceiling, 1)} GB`;

    els.promiseChart.innerHTML = ladder.map(tier => {
      const m = measured[tier];
      const line = `<span class="ceil-line" style="left:${clamp(ceiling / worst * 100, 0, 100)}%"></span>`;
      if (!m) {
        return `<div class="ceil-row"><span class="ceil-tier off">${esc(tier)}</span>`
          + `<div class="ceil-cell">${line}</div>`
          + `<span class="val dim">—</span><span class="dim">not measured</span><span></span></div>`;
      }
      const over = ceiling > 0 && m.worst_gb > ceiling;
      const shape = [m.native, m.target].filter(Boolean).join(" → ");
      return `<div class="ceil-row">
        <span class="ceil-tier ${m.provisional ? "off" : ""}">${esc(tier)}</span>
        <div class="ceil-cell">
          <div class="ceil-track"><i class="ceil-bar ${m.peak_gb > ceiling ? "over" : ""}" style="width:${clamp(m.peak_gb / worst * 100, 0, 100)}%"></i></div>
          <span class="ceil-worst" style="left:${clamp(m.worst_gb / worst * 100, 0, 99.5)}%" title="worst run ${num(m.worst_gb, 2)} GB"></span>
          ${line}
        </div>
        <span class="val ${over ? "over" : ""}">${num(m.peak_gb, 2)} GB</span>
        <span class="dim" ${m.provisional ? `title="${m.runs} runs: fewer than three, provisional"` : ""}>${m.runs} runs</span>
        <span class="ceil-shape">${esc(shape)}</span>
      </div>`;
    }).join("");

    const tiers = ladder.filter(t => measured[t]);
    if (!tiers.length) {
      els.promiseNote.textContent = "Nothing measured on this backend yet; the ceiling is enforced from the first run that proves it.";
      return;
    }
    const breaches = tiers.filter(t => ceiling > 0 && measured[t].worst_gb > ceiling);
    const thin = tiers.filter(t => measured[t].provisional);
    const cap = (info.contract || {}).detail_mp_cap;
    els.promiseNote.textContent = (breaches.length
      ? `${breaches.join(", ")} has runs above the ceiling on this card, so those numbers are history, not the promise: `
        + "the guard shrinks the drawing grid before it steps the tier down. "
      : "Every measured tier sits under the ceiling on this card. ")
      + (thin.length ? `${thin.length} tier(s) rest on fewer than three runs and read dimmed. ` : "")
      + (cap ? `Detail grid capped at ${num(cap, 2)} MP by the peak curve fitted on this card.` : "");
  }

  async function loadTradeoff() {
    try {
      const runs = await getJSON("/api/runs?limit=300");
      const fast = runs.find(r => r.mode === "fast" && r.tag === "bench:720c")
        || runs.find(r => r.mode === "fast" && r.quality === "720p");
      const classicExact = fast ? runs.find(r => r.mode === "classic" && r.prompt === fast.prompt) : null;
      const classic = classicExact
        || runs.find(r => r.mode === "classic" && r.quality === "720p")
        || runs.find(r => r.mode === "classic");
      if (classic) {
        els.toClassic.textContent = `${secs(classic.total_time)} · ${num(classic.peak_vram_mb / 1024, 2)} GB`;
      }
      if (fast) {
        els.toFast.textContent = `${secs(fast.total_time)} · ${num(fast.peak_vram_mb / 1024, 2)} GB`;
      }
      if (classic && fast) {
        const vramRatio = fast.peak_vram_mb / classic.peak_vram_mb;
        const speedRatio = classic.total_time / fast.total_time;
        const samePrompt = classic.prompt === fast.prompt;
        els.toNote.textContent = `Same GPU${samePrompt ? ", same prompt" : ""}, cold encode: fast gives ${num(speedRatio, 1)}× speed for ${num(vramRatio, 1)}× VRAM.`;
      }
    } catch {
      /* ignore */
    }
  }

  /* ---------------------------------------------------------------- studio */

  function syncPressed(row) {
    $$("button", row).forEach(button => {
      button.setAttribute("aria-pressed", String(button.classList.contains("active")));
    });
  }

  function bindSegmented(row, attr, onPick) {
    syncPressed(row);
    row.addEventListener("click", e => {
      const btn = e.target.closest("button");
      if (!btn) return;
      $$("button", row).forEach(b => b.classList.toggle("active", b === btn));
      syncPressed(row);
      onPick(btn.dataset[attr]);
    });
  }

  const baseDims = {
    "420p": {
      "16:9": "640×368", "1:1": "512×512", "9:16": "368×640",
      "4:3": "640×480", "3:4": "480×640", "2:3": "512×768", "21:9": "768×336",
    },
    "720p": {
      "16:9": "896×512", "1:1": "704×704", "9:16": "512×896",
      "4:3": "800×608", "3:4": "608×800", "2:3": "640×960", "21:9": "1024×448",
    },
    "hd": {
      "16:9": "1280×720", "1:1": "1024×1024", "9:16": "720×1280",
      "4:3": "1088×816", "3:4": "816×1088", "2:3": "832×1248", "21:9": "1456×624",
    },
  };

  const legacyAutoSteps = { "420p": 3, "720p": 4, "1080p": 4, "2K": 4, "4K": 4 };

  function updateDimMeta() {
    const dynamic = state.dims && state.dims[state.quality];
    if (dynamic && dynamic[state.ratio]) {
      els.ratioMeta.textContent = dynamic[state.ratio];
      return;
    }
    const tier = baseDims[state.quality] ? state.quality : "hd";
    els.ratioMeta.textContent = baseDims[tier][state.ratio] || "";
  }

  function updateStepsMeta() {
    const auto = (state.autoSteps && state.autoSteps[state.quality])
      || legacyAutoSteps[state.quality] || 4;
    const effective = state.steps || auto;
    els.stepsMeta.textContent = state.steps ? `${effective} steps` : `${state.quality} · ${effective} steps`;
  }

  bindSegmented(els.styleContainer, "style", v => {
    state.style = v;
    state.styleAuto = false;
  });
  bindSegmented(els.ratioRow, "ratio", v => {
    state.ratio = v;
    state.ratioAuto = false;
    updateDimMeta();
  });
  bindSegmented(els.qualityRow, "quality", v => {
    state.quality = v;
    updateDimMeta();
    updateStepsMeta();
  });
  bindSegmented(els.variantRow, "count", v => {
    state.count = parseInt(v, 10) || 1;
  });
  bindSegmented(els.stepsRow, "steps", v => {
    state.steps = parseInt(v, 10) || 0;
    updateStepsMeta();
  });

  [els.crpfToggle, els.geoToggle, els.detailToggle].forEach(button => {
    button.setAttribute("aria-pressed", String(button.classList.contains("active")));
  });
  els.crpfToggle.addEventListener("click", () => {
    state.crpf = !state.crpf;
    els.crpfToggle.classList.toggle("active", state.crpf);
    els.crpfToggle.setAttribute("aria-pressed", String(state.crpf));
  });
  els.detailToggle.addEventListener("click", () => {
    state.detail = !state.detail;
    els.detailToggle.classList.toggle("active", state.detail);
    els.detailToggle.setAttribute("aria-pressed", String(state.detail));
  });
  els.geoToggle.addEventListener("click", () => {
    state.geo = !state.geo;
    state.geoAuto = false;
    els.geoToggle.classList.toggle("active", state.geo);
    els.geoToggle.setAttribute("aria-pressed", String(state.geo));
  });

  els.seedRandomBtn.addEventListener("click", () => { els.seedInput.value = "-1"; });

  els.quickSampleBtn.addEventListener("click", () => {
    els.promptInput.value = state.samples[state.sampleIdx % state.samples.length];
    state.sampleIdx += 1;
    els.promptInput.focus();
  });

  els.optimizePromptBtn.addEventListener("click", async () => {
    const prompt = els.promptInput.value.trim();
    if (!prompt) return els.promptInput.focus();
    const label = els.optimizePromptBtn.querySelector("span");
    const old = label.textContent;
    els.optimizePromptBtn.disabled = true;
    label.textContent = "Compiling…";
    try {
      const data = await getJSON("/api/optimize_prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, style: state.style }),
      });
      if (data.optimized) {
        els.promptInput.value = data.optimized;
        logEvent("prompt compiled", "ok");
      }
    } catch {
      logEvent("prompt compile failed", "warn");
    } finally {
      els.optimizePromptBtn.disabled = false;
      label.textContent = old;
    }
  });

  els.copyPromptBtn.addEventListener("click", async () => {
    const label = els.copyPromptBtn.querySelector("span");
    try {
      await navigator.clipboard.writeText(els.promptInput.value);
      label.textContent = "Copied";
      setTimeout(() => { label.textContent = "Copy prompt"; }, 1400);
    } catch {
      logEvent("clipboard write failed", "warn");
    }
  });

  /* -------------------------------------------------------------- gallery */

  async function loadGallery() {
    try {
      const items = await getJSON("/api/gallery");
      renderGallery(items);
    } catch {
      /* ignore */
    }
  }

  function renderGallery(items) {
    els.historyStrip.innerHTML = "";
    if (!items.length) {
      els.historyStrip.innerHTML = '<span class="history-empty">No generations yet.</span>';
      return;
    }
    items.forEach(item => {
      const wrap = document.createElement("div");
      wrap.className = "thumb-wrap";
      const dot = contractStateOf(item);
      wrap.innerHTML = `
        <button class="thumb">
          <img alt="">
          ${dot ? `<span class="thumb-q ${dot}"></span>` : ""}
          ${item.seed !== null && item.seed !== undefined ? `<span class="thumb-seed">#${String(item.seed).length > 6 ? String(item.seed).slice(-6) : item.seed}</span>` : ""}
        </button>
        <button class="thumb-del" title="Delete">×</button>
      `;
      const thumb = wrap.querySelector(".thumb");
      thumb.title = `${item.prompt || item.filename}${item.seed !== null && item.seed !== undefined ? ` · seed ${item.seed}` : ""}`;
      const img = wrap.querySelector("img");
      img.src = item.url;
      img.alt = item.filename;
      thumb.addEventListener("click", () => {
        showSingle(item.url, item.filename);
        if (item.seed !== null && item.seed !== undefined) els.seedInput.value = item.seed;
        if (item.quality_report) renderContract(item);
        if (item.style_name) logEvent(`history opened · ${en(item.style_name)}`, "");
      });
      const deleteButton = wrap.querySelector(".thumb-del");
      deleteButton.setAttribute("aria-label", `Delete ${item.filename}`);
      deleteButton.addEventListener("click", async e => {
        e.stopPropagation();
        try {
          await getJSON(`/api/gallery/${encodeURIComponent(item.filename)}`, { method: "DELETE" });
          loadGallery();
        } catch {
          logEvent("could not delete image", "warn");
        }
      });
      els.historyStrip.appendChild(wrap);
    });
  }

  els.clearAllBtn.addEventListener("click", async () => {
    if (!confirm("Delete the whole generation history?")) return;
    try {
      await getJSON("/api/gallery", { method: "DELETE" });
      loadGallery();
      logEvent("history cleared");
    } catch {
      logEvent("could not clear", "warn");
    }
  });

  /* ---------------------------------------------------------------- stage */

  function showSingle(url, filename) {
    els.variantGrid.classList.remove("active");
    els.variantGrid.innerHTML = "";
    state.tiles = [];
    els.stageEmpty.style.display = "none";
    els.mainRender.style.display = "block";
    els.mainRender.classList.remove("blurry");
    els.mainRender.src = url;
    els.mainRender.tabIndex = 0;
    els.downloadBtn.href = url;
    els.downloadBtn.download = filename || "grosel.png";
  }

  function resetGrid(count) {
    state.tiles = [];
    els.variantGrid.innerHTML = "";
    els.variantGrid.dataset.count = count;
    els.variantGrid.style.setProperty("--tile-ar", state.ratio.replace(":", " / "));
    if (count <= 1) {
      els.variantGrid.classList.remove("active");
      return;
    }
    els.stageEmpty.style.display = "none";
    els.variantGrid.classList.add("active");
    for (let i = 0; i < count; i++) {
      const tile = document.createElement("div");
      tile.className = "variant-tile pending";
      tile.innerHTML = `<img alt="Variant ${i + 1}"><span class="variant-idx">${String(i + 1).padStart(2, "0")}</span>`;
      tile.addEventListener("click", () => {
        const img = tile.querySelector("img");
        if (!img.getAttribute("src")) return;
        state.tiles.forEach(t => t.classList.remove("selected"));
        tile.classList.add("selected");
        els.downloadBtn.href = img.src;
        els.downloadBtn.download = img.dataset.filename || "grosel.png";
        openLightbox(img.src);
      });
      els.variantGrid.appendChild(tile);
      state.tiles.push(tile);
    }
  }

  let lightboxReturnFocus = null;

  function openLightbox(src) {
    lightboxReturnFocus = document.activeElement;
    els.lightboxImg.src = src;
    els.lightbox.classList.add("active");
    els.lightbox.setAttribute("aria-hidden", "false");
    els.lightboxClose.focus();
  }

  function closeLightbox() {
    if (!els.lightbox.classList.contains("active")) return;
    els.lightbox.classList.remove("active");
    els.lightbox.setAttribute("aria-hidden", "true");
    if (lightboxReturnFocus && lightboxReturnFocus.isConnected) lightboxReturnFocus.focus();
  }

  els.lightboxClose.addEventListener("click", closeLightbox);
  els.lightbox.addEventListener("click", e => {
    if (e.target === els.lightbox) closeLightbox();
  });
  document.addEventListener("keydown", e => {
    if (!els.lightbox.classList.contains("active")) return;
    if (e.key === "Escape") closeLightbox();
    if (e.key === "Tab") {
      e.preventDefault();
      els.lightboxClose.focus();
    }
  });
  els.mainRender.addEventListener("click", () => {
    if (els.mainRender.getAttribute("src")) openLightbox(els.mainRender.src);
  });
  els.mainRender.addEventListener("keydown", event => {
    if (!["Enter", " "].includes(event.key) || !els.mainRender.getAttribute("src")) return;
    event.preventDefault();
    openLightbox(els.mainRender.src);
  });

  /* ------------------------------------------------------------- telemetry */

  function mergeTelemetry(list) {
    const last = list[list.length - 1];
    // The panel quotes one frame's measurements, so quote the frame that cost
    // the most VRAM — the peak below is that same variant's, not a mix.
    const costliest = list.reduce((a, b) => ((b.peak_vram_mb || 0) > (a.peak_vram_mb || 0) ? b : a), last);
    const steps = [];
    list.forEach(t => (t.step_times_ms || []).forEach(ms => steps.push(ms)));
    return {
      ...last,
      images: list.length,
      quality_report: costliest.quality_report,
      encode_time: list.reduce((a, t) => a + (t.encode_time || 0), 0),
      decode_time: list.reduce((a, t) => a + (t.decode_time || 0), 0),
      swap_time: list.reduce((a, t) => a + (t.swap_time || 0), 0),
      total_time: list.reduce((a, t) => a + (t.total_time || 0), 0),
      peak_vram_mb: Math.max(...list.map(t => t.peak_vram_mb || 0)),
      step_times_ms: steps,
      prompt_cached: list.every(t => t.prompt_cached),
    };
  }

  function renderTelemetry(t) {
    state.last = t;
    const stepAvg = t.step_times_ms && t.step_times_ms.length
      ? avg(t.step_times_ms) / 1000
      : (t.step_time || 0);

    els.mLatency.textContent = secs(t.total_time);
    els.mStep.textContent = `${num(stepAvg * 1000, 0)} ms`;
    els.mVram.textContent = `${num(t.peak_vram_mb / 1024, 2)} GB`;
    els.mRes.textContent = t.detail_scale > 1
      ? `${t.resolution} · ${t.native_resolution} native`
      : t.resolution || "—";

    const totalVram = state.system ? state.system.vram_total_gb * 1024 : 7928;
    const ceiling = ceilingGb() * 1024;
    const peakPct = clamp((t.peak_vram_mb / totalVram) * 100, 0, 100);
    els.vramFill.classList.toggle("over", ceiling > 0 && t.peak_vram_mb > ceiling);
    els.vramFill.style.width = `${peakPct}%`;
    els.vramFoot.textContent = ceiling
      ? `peak ${num(t.peak_vram_mb / 1024, 2)} GB / ceiling ${num(ceiling / 1024, 1)} GB`
      : `peak ${num(t.peak_vram_mb / 1024, 2)} GB / ${num(totalVram / 1024, 2)} GB card`;

    const encMs = (t.encode_time || 0) * 1000;
    const denMs = (t.step_times_ms || []).reduce((a, b) => a + b, 0);
    const decMs = (t.decode_time || 0) * 1000;
    const totalMs = Math.max(encMs + denMs + decMs, 1);
    els.phaseEnc.textContent = encMs > 0 ? secs(encMs / 1000) : "0 (cache)";
    els.phaseDen.textContent = secs(denMs / 1000);
    els.phaseDec.textContent = secs(decMs / 1000);
    els.phaseEncBar.style.width = `${(encMs / totalMs) * 100}%`;
    els.phaseDenBar.style.width = `${(denMs / totalMs) * 100}%`;
    els.phaseDecBar.style.width = `${(decMs / totalMs) * 100}%`;
    els.phaseNote.textContent = t.swapped ? `swap ${num(t.swap_time, 2)} s` : (t.prompt_cached ? "cache hit" : "—");

    els.stageMeta.textContent = [
      t.seed !== undefined ? `seed ${t.seed}` : null,
      en(t.style_name) || null,
      t.resolution || null,
    ].filter(Boolean).join(" · ");

    renderTimeline(t);
    renderContract(t);

    els.downloadBtn.href = `/images/${t.filename}`;
    els.downloadBtn.download = t.filename;
    loadSystem();
  }

  /* --------------------------------------------------------- contract panel */

  const badge = (key, value, state, title) => `<span class="badge ${state || ""}"${title ? ` title="${esc(title)}"` : ""}>`
    + `<span class="k">${esc(key)}</span><span class="v">${esc(value)}</span></span>`;

  function contractBadges(t) {
    const c = (state.info && state.info.contract) || {};
    const m = (t.quality_report && t.quality_report.metrics) || {};
    const out = [];
    if (t.vae_blend_px) {
      out.push(badge("blend", `${t.vae_blend_px} px`,
        c.min_blend_px ? (t.vae_blend_px >= c.min_blend_px ? "ok" : "bad") : ""));
    }
    const zs = Object.values(m.seam_z || {}).filter(Number.isFinite);
    if (zs.length) {
      const z = Math.max(...zs);
      out.push(badge("seam z", num(z, 2),
        c.seam_z_limit ? (z <= c.seam_z_limit ? "ok" : "bad") : "",
        `tile periods ${Object.keys(m.seam_z).join(", ")}; limit ${num(c.seam_z_limit, 1)}`));
    } else {
      out.push(badge("seam z", "n/a", "",
        "fewer than three tile repeats in this frame: nothing to average. "
        + "Not measured is not the same as passed."));
    }
    if (Number.isFinite(m.detail_index)) {
      out.push(badge("detail", num(m.detail_index, 0), "",
        "median-filtered Laplacian variance on a fixed 1024 px grid; watchdog floor "
        + num(c.rel_detail_floor, 2) + "× the tier median"));
    }
    if (Number.isFinite(m.noise_index)) {
      out.push(badge("grain", num(m.noise_index, 2), "",
        "lag-ratio grain; an 8-bit PNG cannot read below ~0.39"));
    }
    if (t.native_resolution && t.resolution && t.native_resolution !== t.resolution) {
      out.push(badge("native", `${t.native_resolution} → ${t.resolution}`, "ok",
        "drawn natively, then resampled to the delivery size"));
    }
    return out;
  }

  function contractStateOf(t) {
    const rep = t.quality_report;
    if (!rep || rep.ok === null || rep.ok === undefined) return "";
    const ceiling = ceilingGb();
    if ((rep.failures || []).length
      || (ceiling > 0 && (t.peak_vram_mb || 0) / 1024 > ceiling)) return "bad";
    return (rep.warnings || []).length ? "warn" : "ok";
  }

  function renderContract(t) {
    if (!els.contractMark) return;
    const rep = t.quality_report || {};
    const failures = rep.failures || [];
    const warnings = rep.warnings || [];
    const ceiling = ceilingGb();
    const peak = (t.peak_vram_mb || 0) / 1024;

    els.contractTier.textContent = [t.quality, t.steps ? `${t.steps} steps` : null, t.aspect_ratio]
      .filter(Boolean).join(" · ") || "awaiting frame";
    const cls = contractStateOf(t);
    const mark = { ok: "MET", warn: "CHECK", bad: "BROKEN" }[cls] || "—";
    els.contractMark.className = `verdict-mark ${cls}`;
    els.contractMark.textContent = mark;
    els.contractText.textContent = ceiling
      ? `${num(peak, 2)} / ${num(ceiling, 1)} GB · ${t.resolution || "—"}`
      : `${num(peak, 2)} GB · ${t.resolution || "—"}`;

    els.contractBadges.innerHTML = contractBadges(t).join("");

    const guard = t.vram_guard;
    const note = failures[0]
      || warnings[0]
      || (guard ? guardNote(guard) : null)
      || rep.not
      || "";
    els.contractNote.textContent = note;
  }

  function guardNote(guard) {
    const ceiling = num(guard.ceiling_gb, 1);
    if (!guard.used) return `no tier fits the ${ceiling} GB ceiling (${en(guard.reason || "unclear")})`;
    const basis = Number.isFinite(guard.predicted_peak_gb)
      ? `predicted ${num(guard.predicted_peak_gb, 2)} GB`
      : `${guard.used} not yet measured on this card`;
    return `tier guard: ${guard.requested} → ${guard.used} · ${basis} · ceiling ${ceiling} GB`;
  }

  function renderTimeline(t) {
    const segs = [];
    const swap = (t.swap_time || 0) * 1000;
    const enc = Math.max((t.encode_time || 0) * 1000 - swap, 0);
    const steps = t.step_times_ms || [];
    const denTotal = steps.reduce((a, b) => a + b, 0);
    const dec = (t.decode_time || 0) * 1000;
    if (swap > 0) segs.push({ cls: "tl-swap", ms: swap, label: "swap" });
    if (enc > 0) segs.push({ cls: "tl-enc", ms: enc, label: "enc" });
    if (t.images > 1) {
      if (denTotal > 0) segs.push({ cls: "tl-step", ms: denTotal, label: `${t.images}×${Math.round(steps.length / t.images)} steps` });
    } else {
      steps.forEach((ms, i) => segs.push({ cls: "tl-step", ms, label: String(i + 1) }));
    }
    if (dec > 0) segs.push({ cls: "tl-dec", ms: dec, label: "dec" });

    const total = segs.reduce((a, s) => a + s.ms, 0);
    if (!total) {
      els.timelineBar.className = "timeline-bar empty";
      els.timelineBar.innerHTML = "";
      els.timelineLegend.innerHTML = "";
      els.timelineTotal.textContent = "no measurement";
      return;
    }

    els.timelineBar.className = "timeline-bar";
    els.timelineBar.innerHTML = segs.map(s => {
      const pct = (s.ms / total) * 100;
      return `<div class="tl-seg ${s.cls}" style="width:${pct}%" title="${s.label} · ${Math.round(s.ms)} ms">${pct > 7 ? s.label : ""}</div>`;
    }).join("");

    const encTotal = enc;
    const stepLabel = t.images > 1
      ? `${t.images}×${Math.round(steps.length / t.images)} steps`
      : `${steps.length} steps`;
    const legend = [];
    legend.push(`<span class="lg"><span class="sw" style="background:var(--primary)"></span>encode ${secs((encTotal + swap) / 1000)}</span>`);
    legend.push(`<span class="lg"><span class="sw" style="background:#48505E"></span>${stepLabel} ${secs(denTotal / 1000)}</span>`);
    legend.push(`<span class="lg"><span class="sw" style="background:#6A7382"></span>decode ${secs(dec / 1000)}</span>`);
    if (swap > 0) legend.push(`<span class="lg"><span class="sw" style="background:var(--warn)"></span>swap ${secs(swap / 1000)}</span>`);
    els.timelineLegend.innerHTML = legend.join("");
    els.timelineTotal.textContent = `${secs(t.total_time)} total`;
  }

  /* -------------------------------------------------------------- websocket */

  function resetGenerationUI() {
    clearInterval(state.timer);
    state.running = false;
    els.generateBtn.disabled = false;
    els.generateLabel.textContent = "Generate";
    els.generateBtn.querySelector("svg").classList.remove("spin");
    els.runLine.style.width = "0%";
  }

  function showGenerationError(message) {
    resetGenerationUI();
    els.liveTag.className = "live-tag error";
    els.liveTag.textContent = "error";
    logEvent(message, "warn");
  }

  function connectStudio() {
    if (state.ws && [WebSocket.OPEN, WebSocket.CONNECTING].includes(state.ws.readyState)) return;
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/generate`);
    state.ws = ws;
    ws.onopen = () => {
      if (state.ws === ws) setConn(true);
    };
    ws.onclose = () => {
      if (state.ws !== ws) return;
      setConn(false);
      state.ws = null;
      if (state.running) showGenerationError("Connection lost; generation state cannot be verified.");
      state.reconnectTimer = setTimeout(connectStudio, 1500);
    };
    ws.onmessage = event => {
      try {
        handleStudioMessage(JSON.parse(event.data));
      } catch {
        /* ignore malformed */
      }
    };
  }

  function handleStudioMessage(msg) {
    if (msg.type === "status") {
      logEvent(msg.guard ? guardNote(msg.guard) : en(msg.message));
      return;
    }
    if (msg.type === "step_progress") {
      const vidx = typeof msg.variant === "number" ? msg.variant : -1;
      if (msg.preview_b64) {
        if (vidx >= 0 && state.tiles[vidx]) {
          const img = state.tiles[vidx].querySelector("img");
          img.src = msg.preview_b64;
          img.classList.add("blurry");
          state.tiles[vidx].classList.remove("pending");
        } else {
          els.stageEmpty.style.display = "none";
          els.mainRender.style.display = "block";
          els.mainRender.src = msg.preview_b64;
          els.mainRender.classList.add("blurry");
        }
      }
      const count = msg.variant_count || 1;
      const overall = vidx >= 0 && count > 1
        ? ((vidx + msg.percent / 100) / count) * 100
        : msg.percent;
      els.runLine.style.width = `${overall}%`;
      els.mVram.textContent = `${num(msg.vram_peak_mb / 1024, 2)} GB`;
      setLive("running", `step ${msg.step}/${msg.total_steps}`);
      return;
    }
    if (msg.type === "variant") {
      const t = msg.telemetry;
      const tile = state.tiles[msg.index];
      if (tile) {
        const img = tile.querySelector("img");
        img.src = t.image_url;
        img.dataset.filename = t.filename;
        img.classList.remove("blurry");
        tile.classList.remove("pending");
        if (msg.index === 0) tile.classList.add("selected");
      }
      return;
    }
    if (msg.type === "complete") {
      const list = msg.variants && msg.variants.length ? msg.variants : [msg.telemetry];
      const merged = mergeTelemetry(list);
      resetGenerationUI();
      setLive("done", "done");

      if (list.length > 1) {
        els.mainRender.style.display = "none";
        els.stageEmpty.style.display = "none";
        const first = list[0];
        renderTelemetry(merged);
        els.downloadBtn.href = `/images/${first.filename}`;
        els.downloadBtn.download = first.filename;
        els.stageMeta.textContent = `${list.length}× · ${first.resolution}`;
      } else {
        showSingle(msg.telemetry.image_url, msg.telemetry.filename);
        renderTelemetry(merged);
      }

      logEvent(
        `completed · ${secs(merged.total_time)}${merged.prompt_cached ? " · cache" : ""}${merged.swapped ? ` · swap ${num(merged.swap_time, 2)} s` : ""}`,
        "ok"
      );
      loadGallery();
      loadPromise();
      return;
    }
    if (msg.type === "error") {
      showGenerationError(msg.guard ? `${guardNote(msg.guard)} · pick a lower tier or raise HYPERFLUX_VRAM_CEILING_GB` : en(msg.message));
    }
  }

  function startGeneration() {
    const prompt = els.promptInput.value.trim();
    if (!prompt) return els.promptInput.focus();
    if (state.running) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      logEvent("not connected; retrying", "warn");
      connectStudio();
      return;
    }

    state.running = true;
    state.runCount = state.count;
    state.startedAt = performance.now();
    els.generateBtn.disabled = true;
    els.generateLabel.textContent = "Working…";
    els.generateBtn.querySelector("svg").classList.add("spin");
    setLive("running", "running");
    els.runLine.style.width = "2%";
    els.mainRender.classList.remove("blurry");

    resetGrid(state.runCount);
    els.stageEmpty.style.display = "none";
    els.mainRender.style.display = "none";

    clearInterval(state.timer);
    state.timer = setInterval(() => {
      els.mLatency.textContent = `${((performance.now() - state.startedAt) / 1000).toFixed(2)} s`;
    }, 50);

    logEvent(`generation started · ${state.runCount}× · ${state.quality}`);
    state.ws.send(JSON.stringify({
      prompt,
      style: state.style,
      aspect_ratio: state.ratio,
      quality: state.quality,
      enable_crpf: state.crpf,
      enable_geo: state.geo,
      detail_pass: state.detail,
      steps: state.steps,
      seed: els.seedInput.value,
      count: state.runCount,
      tag: "studio",
    }));
  }

  els.generateBtn.addEventListener("click", startGeneration);
  els.promptInput.addEventListener("keydown", e => {
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      startGeneration();
    }
  });

  /* --------------------------------------------------------------- bench */

  const BENCH = [
    { key: "1080c", label: "1080p · cold", quality: "1080p", seed: 7101, prompt: "A" },
    { key: "1080w", label: "1080p · warm", quality: "1080p", seed: 7102, prompt: "A" },
    { key: "720c", label: "720p · cold", quality: "720p", seed: 7105, prompt: "C" },
    { key: "720w", label: "720p · warm", quality: "720p", seed: 7106, prompt: "C" },
    { key: "1080c2", label: "1080p · second prompt", quality: "1080p", seed: 7104, prompt: "B" },
  ];
  const BENCH_PROMPTS = {
    A: "Karanlık antik kütüphanede mum ışığında el yazması inceleyen yaşlı bilgin, deri ciltli kitaplar, tozlu hava",
    B: "Sisli bir göl kıyısında ahşap iskelede duran balıkçı, paslı teneke kova, soğuk şafak ışığı",
    C: "Terk edilmiş tren istasyonunda bankta oturan genç kadın, sis, sarı istasyon lambası, ıslak beton",
  };
  const bench = { running: false, rows: [], ws: null, results: [] };

  function initBench() {
    bench.results = BENCH.map(() => null);
    els.suiteChips.innerHTML = BENCH.map((b, i) => `<span class="suite-chip" data-i="${i}">${b.label}</span>`).join("");
    els.benchBody.innerHTML = BENCH.map((b, i) => `
      <tr data-i="${i}">
        <td class="mono">${String(i + 1).padStart(2, "0")}</td>
        <td>${b.label}</td>
        <td class="mono">—</td>
        <td class="num">—</td>
        <td class="num">—</td>
        <td class="num">—</td>
        <td class="num">—</td>
        <td class="num">—</td>
      </tr>
    `).join("");
  }

  function benchRow(i) { return $(`#benchBody tr[data-i="${i}"]`); }

  function setBenchStatus(text) {
    els.benchStatus.textContent = text;
  }

  function fillBenchRow(i, t) {
    const row = benchRow(i);
    const cells = row.querySelectorAll("td");
    const den = (t.step_times_ms || []).reduce((a, b) => a + b, 0) / 1000;
    cells[1].textContent = BENCH[i].label;
    cells[2].textContent = t.resolution || "—";
    cells[3].textContent = t.prompt_cached ? "cache" : secs(t.encode_time);
    cells[4].textContent = secs(den || t.step_time * t.steps);
    cells[5].textContent = secs(t.decode_time);
    cells[6].textContent = secs(t.total_time);
    cells[7].textContent = `${num(t.peak_vram_mb / 1024, 2)} GB`;
    row.classList.remove("running");
    row.classList.add("done");
  }

  function benchSummary() {
    const done = bench.results.filter(Boolean);
    if (!done.length) return;
    const total = done.reduce((a, t) => a + t.total_time, 0);
    const peak = Math.max(...done.map(t => t.peak_vram_mb));
    els.sAvg.textContent = secs(avg(done.map(t => t.total_time)));
    els.sTotal.textContent = secs(total);
    els.sVram.textContent = `${num(peak / 1024, 2)} GB`;
    els.benchSummaryMeta.textContent = `${done.length}/${BENCH.length} runs`;

    const cold = bench.results[0];
    const warm = bench.results[1];
    els.sCache.textContent = cold && warm && !warm.prompt_cached
      ? "no cache gain"
      : cold && warm
        ? `−${num(Math.max(cold.encode_time - warm.encode_time, 0), 2)} s`
        : "—";

    renderBenchCharts(done);
    updateCompare();
  }

  function renderBenchCharts(done) {
    const items = bench.results.map((t, i) => ({ t, b: BENCH[i] })).filter(x => x.t);
    const maxTotal = Math.max(...items.map(x => x.t.total_time), 0.001);

    els.timeChart.innerHTML = items.map(({ t, b }) => {
      const enc = (t.encode_time || 0) * 1000;
      const den = ((t.step_times_ms || []).reduce((a, c) => a + c, 0));
      const dec = (t.decode_time || 0) * 1000;
      const total = Math.max(enc + den + dec, 1);
      const w = (total / (maxTotal * 1000)) * 100;
      return `
        <div class="chart-row">
          <span class="label">${b.label}</span>
          <span class="chart-track" style="width:${w}%">
            <i class="c-enc" style="width:${(enc / total) * 100}%"></i>
            <i class="c-step" style="width:${(den / total) * 100}%"></i>
            <i class="c-dec" style="width:${(dec / total) * 100}%"></i>
          </span>
          <span class="val">${secs(t.total_time)}</span>
        </div>`;
    }).join("");

    const vramTotal = state.system ? state.system.vram_total_gb : 8.15;
    els.vramScale.textContent = `${num(vramTotal, 2)} GB scale`;
    els.vramChart.innerHTML = items.map(({ t, b }, idx) => `
      <div class="chart-row">
        <span class="label">${b.label}</span>
        <span class="chart-track">
          <i class="c-vram ${idx === 0 ? "" : "alt"}" style="width:${clamp((t.peak_vram_mb / 1024 / vramTotal) * 100, 0, 100)}%"></i>
        </span>
        <span class="val">${num(t.peak_vram_mb / 1024, 2)}</span>
      </div>`).join("");
  }

  async function updateCompare() {
    try {
      const runs = await getJSON("/api/runs?limit=300");
      const classic = runs.find(r => r.mode === "classic");
      if (classic) {
        els.compareClassic.textContent = `Classic flow — ${classic.quality} · ${secs(classic.total_time)} · ${num(classic.peak_vram_mb / 1024, 2)} GB peak`;
      } else {
        els.compareClassic.textContent = "Classic flow — no record";
      }
      const lastFast = runs.find(r => r.mode === "fast" && r.tag === "studio");
      if (lastFast && !bench.results.some(Boolean)) {
        els.compareFast.textContent = `FP8 fast — ${lastFast.quality} · ${secs(lastFast.total_time)} · ${num(lastFast.peak_vram_mb / 1024, 2)} GB peak`;
      }
      if (bench.results.some(Boolean)) {
        const done = bench.results.filter(Boolean);
        const avg = done.reduce((a, t) => a + t.total_time, 0) / done.length;
        const peak = Math.max(...done.map(t => t.peak_vram_mb));
        els.compareFast.textContent = `FP8 fast — ${done.length} conditions · mean ${secs(avg)} · ${num(peak / 1024, 2)} GB peak`;
      }
    } catch {
      /* ignore */
    }
  }

  function startBench() {
    if (bench.running) return;
    if (state.system && state.system.engine === "qwen") {
      logEvent("benchmark protocol is FLUX-only", "warn");
      return;
    }
    bench.running = true;
    bench.results = BENCH.map(() => null);
    els.benchStart.disabled = true;
    setBenchStatus("preparing…");
    initBench();
    els.timeChart.innerHTML = '<div class="chart-empty">measuring</div>';
    els.vramChart.innerHTML = '<div class="chart-empty">measuring</div>';
    ["sAvg", "sCache", "sVram", "sTotal"].forEach(id => { els[id].textContent = "—"; });

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/generate`);
    bench.ws = ws;

    ws.onopen = async () => {
      await fetch("/api/cache/clear", { method: "POST" }).catch(() => {});
      for (let i = 0; i < BENCH.length; i++) {
        const item = BENCH[i];
        const chip = $(`.suite-chip[data-i="${i}"]`);
        chip.classList.add("running");
        const row = benchRow(i);
        row.classList.add("running");
        row.querySelectorAll("td")[1].textContent = `${item.label} · running`;
        setBenchStatus(`${i + 1}/${BENCH.length} · ${item.label}`);

        try {
          const result = await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error("timeout")), 600000);
            const onMsg = e => {
              const msg = JSON.parse(e.data);
              if (msg.type === "step_progress") {
                row.querySelectorAll("td")[1].textContent = `${item.label} · step ${msg.step}/${msg.total_steps}`;
              }
              if (msg.type === "complete") {
                clearTimeout(timer);
                ws.removeEventListener("message", onMsg);
                resolve(msg.telemetry);
              }
              if (msg.type === "error") {
                clearTimeout(timer);
                ws.removeEventListener("message", onMsg);
                reject(new Error(msg.message));
              }
            };
            ws.addEventListener("message", onMsg);
            ws.send(JSON.stringify({
              prompt: BENCH_PROMPTS[item.prompt],
              style: "matte_chiaroscuro",
              aspect_ratio: "16:9",
              quality: item.quality,
              enable_crpf: false,
              enable_geo: true,
              seed: item.seed,
              count: 1,
              tag: `bench:${item.key}`,
            }));
          });
          bench.results[i] = result;
          fillBenchRow(i, result);
          chip.classList.remove("running");
          chip.classList.add("done");
          benchSummary();
        } catch (err) {
          row.classList.remove("running");
          row.classList.add("fail");
          row.querySelectorAll("td")[1].textContent = `${item.label} · error`;
          setBenchStatus(`error: ${err.message}`);
          logEvent(`measurement error · ${err.message}`, "warn");
          bench.running = false;
          els.benchStart.disabled = false;
          ws.close();
          return;
        }
      }
      bench.running = false;
      els.benchStart.disabled = false;
      setBenchStatus("completed");
      logEvent("benchmark completed", "ok");
      loadSystem();
      loadGallery();
      loadTradeoff();
      ws.close();
    };

    ws.onerror = () => {
      bench.running = false;
      els.benchStart.disabled = false;
      setBenchStatus("connection error");
    };
  }

  els.benchStart.addEventListener("click", startBench);

  async function restoreBench() {
    try {
      const runs = await getJSON("/api/runs?limit=100");
      const byKey = {};
      runs.forEach(r => {
        const tag = String(r.tag || "");
        if (!tag.startsWith("bench:")) return;
        const key = tag.slice(6);
        if (!byKey[key]) byKey[key] = r;
      });
      let found = 0;
      BENCH.forEach((b, i) => {
        const r = byKey[b.key];
        if (r && r.mode === "fast") {
          bench.results[i] = r;
          fillBenchRow(i, r);
          const chip = $(`.suite-chip[data-i="${i}"]`);
          if (chip) chip.classList.add("done");
          found += 1;
        }
      });
      if (found) {
        els.benchStatus.textContent = "stored measurement";
        benchSummary();
      }
    } catch {
      /* ignore */
    }
  }

  /* ---------------------------------------------------------------- math */

  const mathKey = p => p.filename || `${p.quality}-${p.steps}-${p.cold ? "c" : "w"}-${p.native_mp}`;
  const mp = v => `${num(v, 2)} MP`;

  function renderMathMeta(data) {
    const fit = data.fit || {};
    const lat = fit.latency || {};
    const cv = lat.cv || {};
    const vram = fit.vram || {};
    const plans = n => `${n} ${n === 1 ? "plan" : "plans"}`;
    els.mathN.textContent = `${plans(data.n_aggregated)} · ${data.n_plans} measured`;
    els.mathClipN.textContent = `${data.n_clip}/${data.n_plans}`;
    if (els.mathDataN) els.mathDataN.textContent = data.n_plans;
    els.mathFitLat.textContent = `${num(cv.r2, 3)} · ${num(cv.mae, 2)} s`;
    els.mathFitVram.textContent = `${vram.n || 0} runs · tier median`;
    els.mathFitNote.textContent = `Latency: leave-one-out (LOOCV) over ${plans(cv.n || 0)}. VRAM is not a regression; it is the measured warm median of the same tier. Cold encode floor ${num(fit.encode_cold_gb, 2)} GB.`;
  }

  function renderMathTable(plans) {
    const rows = [...(plans || [])].sort((a, b) => a.total_time - b.total_time).slice(0, 12);
    els.mathPlansBody.innerHTML = rows.map(p => `
      <tr>
        <td>${p.image_url ? `<img class="math-thumb" src="${esc(p.image_url)}" alt="">` : "—"}</td>
        <td><strong>${esc(p.quality)}</strong> <span class="mono dim">${mp(p.native_mp)}</span></td>
        <td class="num">${p.steps}</td>
        <td>${p.mode === "classic" ? "classic" : p.cold ? "cold" : "warm"}</td>
        <td class="num">${secs(p.total_time)}</td>
        <td class="num">${num(p.peak_gb, 2)} GB</td>
        <td class="num">${p.clip != null ? num(p.clip, 3) : "—"}</td>
        <td>${esc(p.mode)}${p.n_measured > 1 ? ` · n=${p.n_measured}` : ""}</td>
      </tr>`).join("");
  }

  function renderMathPareto(points, selected) {
    const list = [...(points || [])];
    if (selected && !list.some(p => mathKey(p) === mathKey(selected))) list.push(selected);
    if (!list.length) {
      els.mathPareto.innerHTML = '<div class="chart-empty">no measurement</div>';
      return;
    }
    const w = 680, h = 300, l = 48, r = 18, t = 16, b = 44;
    const xs = list.map(p => p.total_time);
    const ys = list.map(p => p.peak_gb);
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const xPad = Math.max((x1 - x0) * 0.08, 0.35);
    const yPad = Math.max((y1 - y0) * 0.12, 0.12);
    x0 -= xPad; x1 += xPad; y0 -= yPad; y1 += yPad;
    const sx = v => l + ((v - x0) / (x1 - x0)) * (w - l - r);
    const sy = v => t + (1 - (v - y0) / (y1 - y0)) * (h - t - b);
    const xTicks = [0, 0.25, 0.5, 0.75, 1].map(f => x0 + (x1 - x0) * f);
    const yTicks = [0, 0.5, 1].map(f => y0 + (y1 - y0) * f);
    const grid = [
      ...xTicks.map(v => `<line class="grid" x1="${sx(v)}" y1="${t}" x2="${sx(v)}" y2="${h - b}"/>`),
      ...yTicks.map(v => `<line class="grid" x1="${l}" y1="${sy(v)}" x2="${w - r}" y2="${sy(v)}"/>`),
    ].join("");
    const xLabels = xTicks.map(v => `<text x="${sx(v)}" y="${h - 22}" text-anchor="middle">${num(v, 1)}</text>`).join("");
    const yLabels = yTicks.map(v => `<text x="${l - 7}" y="${sy(v) + 3}" text-anchor="end">${num(v, 2)}</text>`).join("");
    const line = [...list].sort((a, b) => a.total_time - b.total_time)
      .map(p => `${sx(p.total_time)},${sy(p.peak_gb)}`).join(" ");
    const native = list.map(p => p.native_mp);
    const nMin = Math.min(...native), nMax = Math.max(...native);
    const dots = list.map(p => {
      const isSel = selected && mathKey(p) === mathKey(selected);
      const isClassic = p.mode === "classic";
      const radius = 4 + 4 * ((p.native_mp - nMin) / Math.max(nMax - nMin, 1e-6));
      const cls = `pt${isClassic ? " classic" : ""}${isSel ? " selected" : ""}`;
      const label = `${p.mode === "classic" ? "C·" : ""}${p.quality}·${p.steps}`;
      return `<g class="${cls}">
        <title>${esc(p.quality)} · ${esc(p.steps)} steps · ${num(p.total_time, 2)} s · ${num(p.peak_gb, 2)} GB · ${mp(p.native_mp)}${p.predicted ? " · predicted" : " · measured"}</title>
        <circle cx="${sx(p.total_time)}" cy="${sy(p.peak_gb)}" r="${radius}"/>
        <text x="${sx(p.total_time) + radius + 3}" y="${sy(p.peak_gb) + 3}">${esc(label)}</text>
      </g>`;
    }).join("");
    els.mathPareto.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Pareto frontier of time against VRAM">
        <text x="${l}" y="11">VRAM (GB)</text>
        ${grid}${xLabels}${yLabels}
        <line class="axis" x1="${l}" y1="${h - b}" x2="${w - r}" y2="${h - b}"/>
        <line class="axis" x1="${l}" y1="${t}" x2="${l}" y2="${h - b}"/>
        <polyline points="${line}" fill="none" stroke="#5A626E" stroke-width="1" stroke-dasharray="3 3"/>
        ${dots}
        <text x="${(l + w - r) / 2}" y="${h - 4}" text-anchor="middle">time (s)</text>
      </svg>
      <div class="math-legend">
        <span><i></i>fast · warm</span>
        <span><i class="classic"></i>classic</span>
        <span><i class="selected"></i>selected</span>
        <span>circle size = native resolution</span>
      </div>`;
    els.mathParetoNote.textContent = `${list.length} frontier points. Left is faster, up costs more VRAM. Points come from measured plans; the curve is never invented.`;
  }

  function renderMathChosen(result) {
    const c = result.chosen;
    if (!c) {
      els.mathSolveStatus.className = "math-status bad";
      els.mathSolveStatus.textContent = "no data · produce plans first";
      els.mathChosen.innerHTML = '<div class="chart-empty">no measured plan</div>';
      els.mathShadow.innerHTML = "";
      els.mathAlts.innerHTML = "";
      return;
    }
    const ok = result.status === "optimal";
    els.mathSolveStatus.className = `math-status ${ok ? "ok" : "bad"}`;
    els.mathSolveStatus.textContent = ok ? "optimal · constraints satisfied" : "infeasible · nearest plan";
    const source = c.predicted ? "predicted" : `measured · n=${c.n_measured}`;
    els.mathChosen.innerHTML = `
      ${c.image_url ? `<img src="${esc(c.image_url)}" alt="">` : `<div class="chart-empty">${c.predicted ? "predicted · no image" : "no image · measurement stored"}</div>`}
      <div class="chosen-title"><strong>${esc(c.quality)} · ${c.steps} steps</strong><span class="mono dim">${c.mode === "classic" ? "classic" : c.cold ? "cold" : "warm"}</span></div>
      <div class="math-kv">
        <div><span>Time</span><span class="mono">${secs(c.total_time)}</span></div>
        <div><span>VRAM</span><span class="mono">${num(c.peak_gb, 2)} GB</span></div>
        <div><span>Native</span><span class="mono">${mp(c.native_mp)}</span></div>
        <div><span>CLIP</span><span class="mono">${c.clip != null ? num(c.clip, 3) : "—"}</span></div>
        <div><span>Source</span><span class="mono">${source}</span></div>
        <div><span>Output</span><span class="mono">${esc(c.resolution)}</span></div>
      </div>`;

    const shadow = result.shadow || {};
    const rows = [];
    const gainText = (v, unit) => Math.abs(v) < 1e-9
      ? "inactive · constraint not binding"
      : `${num(v, 3)} ${esc(unit)}`;
    if (shadow.budget_gain !== undefined) {
      rows.push(`Budget +0.5 GB → <strong>${gainText(shadow.budget_gain, shadow.budget_unit)}</strong>`);
    }
    if (shadow.time_gain !== undefined) {
      rows.push(`Time +1 s → <strong>${gainText(shadow.time_gain, shadow.time_unit)}</strong>`);
    }
    if (!rows.length) rows.push("no shadow price");
    if (result.binding && result.binding.length) {
      rows.push(`Binding: <strong>${result.binding.map(esc).join(", ")}</strong>`);
    }
    if (result.violations && Object.keys(result.violations).length) {
      const v = result.violations;
      const bits = [];
      if (v.vram_gb) bits.push(`VRAM +${num(v.vram_gb, 2)} GB`);
      if (v.time_s) bits.push(`time +${num(v.time_s, 2)} s`);
      if (v.native_mp) bits.push(`quality −${num(v.native_mp, 3)} MP`);
      rows.push(`<strong>Violation:</strong> ${bits.join(" · ") || "—"}`);
    }
    els.mathShadow.innerHTML = rows.map(r => `<div>${r}</div>`).join("");

    const alts = result.alternatives || [];
    els.mathAlts.innerHTML = alts.length
      ? alts.map(p => `<div class="math-alt"><span><b>${esc(p.quality)}</b> · ${p.steps} steps · ${p.mode === "classic" ? "classic" : p.cold ? "cold" : "warm"}</span><span class="mono">${secs(p.total_time)} · ${num(p.peak_gb, 2)} GB</span></div>`).join("")
      : '<div class="math-alt">no alternative · constraints too tight</div>';
  }

  async function solveMath() {
    const m = state.math;
    m.budget = parseFloat(els.mathBudget.value);
    m.time = parseFloat(els.mathTime.value);
    m.floor = parseFloat(els.mathFloor.value);
    els.mathSolveStatus.className = "math-status";
    els.mathSolveStatus.textContent = "solving…";
    try {
      const result = await getJSON("/api/math/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          budget_gb: m.budget, max_time_s: m.time, min_native_mp: m.floor,
          steps: m.steps, warm: m.warm, objective: m.objective,
        }),
      });
      m.result = result;
      renderMathChosen(result);
      renderMathPareto(result.frontier, result.chosen);
    } catch (err) {
      els.mathSolveStatus.className = "math-status bad";
      els.mathSolveStatus.textContent = `solver error · ${err.message}`;
    }
  }

  async function loadMath() {
    const m = state.math;
    if (m.loading) return;
    m.loading = true;
    try {
      const data = await getJSON("/api/math");
      m.data = data;
      renderMathMeta(data);
      renderMathTable(data.aggregated);
      renderMathPareto(data.frontier, null);
      await solveMath();
    } catch (err) {
      els.mathSolveStatus.className = "math-status bad";
      els.mathSolveStatus.textContent = `math layer not ready · ${err.message}`;
    } finally {
      m.loading = false;
    }
  }

  function initMath() {
    const m = state.math;
    const sync = () => {
      els.mathBudgetOut.textContent = `${num(parseFloat(els.mathBudget.value), 1)} GB`;
      els.mathTimeOut.textContent = `${num(parseFloat(els.mathTime.value), 1)} s`;
      els.mathFloorOut.textContent = `${num(parseFloat(els.mathFloor.value), 2)} MP`;
    };
    ["mathBudget", "mathTime", "mathFloor"].forEach(id => {
      els[id].addEventListener("input", sync);
    });
    bindSegmented(els.mathStepsRow, "steps", v => { m.steps = parseInt(v, 10) || 4; });
    bindSegmented(els.mathWarmRow, "warm", v => { m.warm = v === "1"; });
    bindSegmented(els.mathObjRow, "objective", v => { m.objective = v; });
    els.mathSolveBtn.addEventListener("click", solveMath);
    sync();
  }

  /* ---------------------------------------------------------------- init */

  updateStepsMeta();
  initBench();
  initMath();
  connectStudio();
  loadPromise();
  loadSystem().then(() => {
    if (state.system) {
      const mode = state.system.engine === "fast"
        ? "FP8"
        : (state.system.engine === "qwen" ? "Qwen stream" : "BF16");
      logEvent(`engine ready · ${mode} · ${state.system.gpu}`, "ok");
    }
  });
  loadGallery().then(updateCompare);
  loadTradeoff();
  restoreBench();
})();
