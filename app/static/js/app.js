/* SpamShield Mail – front-end controller (vanilla JS, no build step). */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const state = {
    folder: "inbox",
    query: "",
    selected: null,
    threshold: 0.5,
    autoTimer: null,
    knownUids: new Set(),
    model: null,
  };

  const FOLDER_TITLES = { inbox: "Bandeja de entrada", spam: "Spam" };
  const AVATAR_COLORS = ["#2563eb", "#7c3aed", "#db2777", "#ea580c", "#0d9488", "#4f46e5", "#0891b2", "#65a30d", "#c026d3", "#b45309"];

  const SAMPLES = {
    spam: { sender_name: "Rewards Center", sender_email: "winner@claim-prize-now.biz",
            subject: "Congratulations! You have won a £1000 prize",
            body: "URGENT! You have been selected to receive a £1000 cash prize. To claim call 09061701461 now. Claim code KL341. Valid 12 hours only. T&Cs apply, 150p/min." },
    ham:  { sender_name: "Laura Gómez", sender_email: "laura.gomez@unal.edu.co",
            subject: "Reunión del proyecto mañana",
            body: "Hi! Are we still meeting tomorrow at 3pm to review the notebook? I'll bring the EDA charts. Let me know if you need anything else." },
    phish:{ sender_name: "Account Security", sender_email: "security@verify-account-login.info",
            subject: "Your account has been suspended",
            body: "Dear customer, your account will be closed. Verify your details now at www.secure-verify-login.com to avoid losing access. Reply STOP to opt out." },
  };

  // ------------------------------------------------------------------ helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`);
  const initials = (name) => (name || "?").trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
  const colorFor = (s) => { let h = 0; for (const c of s || "") h = (h * 31 + c.charCodeAt(0)) >>> 0; return AVATAR_COLORS[h % AVATAR_COLORS.length]; };

  function fmtTime(iso) {
    const d = new Date(iso), now = new Date();
    return d.toDateString() === now.toDateString()
      ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      : d.toLocaleDateString([], { day: "numeric", month: "short" });
  }

  async function api(path, opts = {}) {
    const res = await fetch(`/api${path}`, {
      headers: { "Content-Type": "application/json" },
      ...opts,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    if (res.status === 204) return null;
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function toast(msg, isErr = false) {
    const el = document.createElement("div");
    el.className = `toast${isErr ? " err" : ""}`;
    el.innerHTML = msg;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), 3800);
  }

  // ------------------------------------------------------------------ stats
  function renderStats(s) {
    const inbox = s.folders.inbox, spam = s.folders.spam;
    $("#cntInbox").textContent = inbox.unread ? `${inbox.unread}/${inbox.total}` : inbox.total;
    $("#cntSpam").textContent = spam.unread ? `${spam.unread}/${spam.total}` : spam.total;
    $("#cntInbox").title = `${inbox.unread} sin leer de ${inbox.total}`;
    $("#cntSpam").title = `${spam.unread} sin leer de ${spam.total}`;
    $("#statTotal").textContent = s.total;
    $("#statInbox").textContent = inbox.total;
    $("#statSpam").textContent = spam.total;
    const t = s.total || 1;
    $("#ratioInbox").style.width = `${(inbox.total / t) * 100}%`;
    $("#ratioSpam").style.width = `${(spam.total / t) * 100}%`;
    const e = s.evaluation;
    $("#mAcc").textContent = pct(e.accuracy);
    $("#mPrec").textContent = pct(e.precision);
    $("#mRec").textContent = pct(e.recall);
    $("#cmTN").textContent = e.tn; $("#cmFP").textContent = e.fp;
    $("#cmFN").textContent = e.fn; $("#cmTP").textContent = e.tp;
    document.title = inbox.unread ? `(${inbox.unread}) SpamShield Mail` : "SpamShield Mail";
  }

  const refreshStats = async () => renderStats(await api("/stats"));

  // ------------------------------------------------------------------ list
  async function loadList({ highlightNew = false } = {}) {
    const q = encodeURIComponent(state.query);
    const emails = await api(`/emails?folder=${state.folder}&q=${q}`);
    const list = $("#mailList");
    list.innerHTML = emails.map((e) => {
      const isNew = highlightNew && !state.knownUids.has(e.uid);
      const p = e.prediction.spam_probability;
      const pill = e.user_moved
        ? `<span class="pill moved">Movido por ti</span>`
        : `<span class="pill ${e.prediction.label}" title="Probabilidad de spam">${pct(p, 0)} spam</span>`;
      return `
        <li class="mail${e.read ? "" : " unread"}${e.uid === state.selected ? " selected" : ""}${isNew ? " new" : ""}" data-uid="${e.uid}" tabindex="0">
          <div class="avatar" style="background:${colorFor(e.sender_email)}">${esc(initials(e.sender_name))}</div>
          <div class="from">${esc(e.sender_name)}</div>
          <div class="time">${fmtTime(e.received_at)}</div>
          <div class="line"><span class="text"><span class="subj">${esc(e.subject)}</span> — ${esc(e.body)}</span>${pill}</div>
        </li>`;
    }).join("");
    emails.forEach((e) => state.knownUids.add(e.uid));

    const empty = emails.length === 0;
    $("#emptyState").hidden = !empty;
    $("#emptyText").textContent = state.query
      ? "Ningún correo coincide con la búsqueda."
      : state.folder === "spam" ? "¡Genial! No hay spam por ahora." : "No hay correos aquí todavía.";
    $("#emptyGen").hidden = !!state.query;
    $("#spamBanner").hidden = state.folder !== "spam" || empty;
    $("#listInfo").textContent = empty ? "" : `${emails.length} mensaje${emails.length === 1 ? "" : "s"}`;
  }

  // ------------------------------------------------------------------ reader
  function verdictHtml(e) {
    const pred = e.prediction, p = pred.spam_probability, isSpam = pred.label === "spam";
    let truth = `<span class="pill neutral">Sin etiqueta (redactado)</span>`;
    if (e.true_label) {
      const correct = e.true_label === pred.label;
      const kind = correct ? "Correcto" : isSpam ? "Falso positivo" : "Falso negativo";
      truth = `<span class="pill ${correct ? "ham" : "spam"}">${kind}</span>`;
    }
    return `
      <div class="verdict">
        <div class="verdict-head">
          <div class="icon ${pred.label}"><svg><use href="#i-${isSpam ? "spam" : "shield"}"/></svg></div>
          <div style="flex:1">
            <h4>${isSpam ? "Clasificado como SPAM" : "Clasificado como válido"}</h4>
            <p>Confianza del modelo: <b>${pct(pred.confidence)}</b></p>
          </div>
          ${truth}
        </div>
        <div class="prob">
          <div class="prob-row"><span>Probabilidad de spam</span><b>${pct(p, 2)}</b></div>
          <div class="prob-track">
            <div class="prob-fill" style="width:${Math.max(p * 100, 1)}%"></div>
            <div class="prob-thr" style="left:${state.threshold * 100}%" title="Umbral ${state.threshold}"></div>
          </div>
          <div class="prob-row muted small"><span>0 · válido</span><span>umbral ${state.threshold}</span><span>1 · spam</span></div>
        </div>
        <div class="facts">
          <div class="fact"><div class="k">Etiqueta real</div><div class="v">${e.true_label ? (e.true_label === "spam" ? "spam" : "ham (válido)") : "—"}</div></div>
          <div class="fact"><div class="k">Carpeta asignada</div><div class="v">${FOLDER_TITLES[e.classified_folder]}</div></div>
          <div class="fact"><div class="k">Modelo</div><div class="v">${esc(state.model?.short || "spam-mail-classifier")}</div></div>
          ${e.source_id ? `<div class="fact"><div class="k">ID dataset</div><div class="v">${esc(e.source_id)}</div></div>` : ""}
        </div>
      </div>`;
  }

  async function openEmail(uid) {
    const e = await api(`/emails/${uid}`);
    state.selected = uid;
    history.replaceState(null, "", `#${uid}`);
    if (e.folder !== state.folder) {
      state.folder = e.folder;
      $$(".folder").forEach((b) => b.classList.toggle("active", b.dataset.folder === e.folder));
      $("#folderTitle").textContent = FOLDER_TITLES[e.folder];
    }
    const moveBtn = e.folder === "spam"
      ? `<button class="btn success" data-act="move" data-to="inbox"><svg><use href="#i-inbox"/></svg>No es spam</button>`
      : `<button class="btn danger" data-act="move" data-to="spam"><svg><use href="#i-spam"/></svg>Marcar como spam</button>`;
    $("#readerBody").innerHTML = `
      <div class="reader-toolbar">
        <button class="icon-btn back-btn" data-act="back" aria-label="Volver"><svg><use href="#i-back"/></svg></button>
        ${moveBtn}
        <button class="btn ghost" data-act="unread"><svg><use href="#i-mail"/></svg>Marcar no leído</button>
        <span class="spacer"></span>
        <button class="icon-btn" data-act="star" title="Destacar" style="color:${e.starred ? "#eab308" : ""}"><svg><use href="#i-star"/></svg></button>
        <button class="icon-btn" data-act="delete" title="Eliminar"><svg><use href="#i-trash"/></svg></button>
      </div>
      <h1 class="r-subject">${esc(e.subject)} <span class="pill ${e.folder === "spam" ? "spam" : "neutral"}">${FOLDER_TITLES[e.folder]}</span></h1>
      <div class="r-meta">
        <div class="avatar" style="background:${colorFor(e.sender_email)}">${esc(initials(e.sender_name))}</div>
        <div class="who">
          <div><b>${esc(e.sender_name)}</b> <span class="addr">&lt;${esc(e.sender_email)}&gt;</span></div>
          <div class="to">para mí</div>
        </div>
        <div class="date">${new Date(e.received_at).toLocaleString()}</div>
      </div>
      <div class="r-body">${esc(e.body)}</div>
      ${verdictHtml(e)}`;
    $("#readerBody").dataset.uid = uid;
    $("#readerBody").hidden = false;
    $("#readerEmpty").hidden = true;
    $(".content").classList.add("reading");
    await Promise.all([loadList(), refreshStats()]);
  }

  function closeReader() {
    state.selected = null;
    if (location.hash) history.replaceState(null, "", location.pathname);
    $("#readerBody").hidden = true;
    $("#readerEmpty").hidden = false;
    $(".content").classList.remove("reading");
  }

  async function readerAction(act, btn) {
    const uid = $("#readerBody").dataset.uid;
    if (act === "back") { closeReader(); return loadList(); }
    if (act === "move") {
      const to = btn.dataset.to;
      await api(`/emails/${uid}`, { method: "PATCH", body: { folder: to } });
      toast(to === "spam" ? "Conversación movida a Spam" : "Movido a la Bandeja de entrada");
      closeReader();
    } else if (act === "unread") {
      await api(`/emails/${uid}`, { method: "PATCH", body: { read: false } });
      closeReader();
    } else if (act === "star") {
      const e = await api(`/emails/${uid}`);
      await api(`/emails/${uid}`, { method: "PATCH", body: { starred: !e.starred } });
      return openEmail(uid);
    } else if (act === "delete") {
      await api(`/emails/${uid}`, { method: "DELETE" });
      toast("Correo eliminado");
      closeReader();
    }
    await Promise.all([loadList(), refreshStats()]);
  }

  // ------------------------------------------------------------------ generation
  async function generate(count) {
    const btn = $("#genBtn");
    btn.disabled = true;
    try {
      const ratio = $("#genRatio").value;
      const res = await api("/generate", { method: "POST", body: { count, spam_ratio: ratio === "" ? null : Number(ratio) } });
      const nSpam = res.delivered.filter((e) => e.folder === "spam").length;
      const nInbox = res.delivered.length - nSpam;
      renderStats(res.stats);
      await loadList({ highlightNew: true });
      toast(`📬 ${res.delivered.length} nuevo${res.delivered.length === 1 ? "" : "s"}: <b>${nInbox}</b> a bandeja de entrada · <b>${nSpam}</b> a spam`);
    } catch (err) {
      toast(esc(err.message), true);
      stopAuto();
    } finally {
      btn.disabled = false;
    }
  }

  function startAuto() {
    stopAuto();
    generate(1);
    state.autoTimer = setInterval(() => generate(1), 4000);
  }
  function stopAuto() {
    clearInterval(state.autoTimer);
    state.autoTimer = null;
    $("#autoToggle").checked = false;
  }

  // ------------------------------------------------------------------ compose
  let previewTimer = null;
  function schedulePreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      const f = $("#composeForm");
      const subject = f.subject.value.trim(), body = f.body.value.trim();
      const box = $("#composePreview");
      if (!subject && !body) { box.innerHTML = `<span class="muted small">Escribe para ver la predicción en tiempo real.</span>`; return; }
      try {
        const p = await api("/classify", { method: "POST", body: { subject, body } });
        box.innerHTML = `<span class="pill ${p.label}">${p.label === "spam" ? "SPAM" : "VÁLIDO"}</span>
          <span class="small">Probabilidad de spam: <b>${pct(p.spam_probability, 2)}</b> → irá a <b>${p.label === "spam" ? "Spam" : "Bandeja de entrada"}</b></span>`;
      } catch (err) { box.innerHTML = `<span class="small" style="color:var(--bad)">${esc(err.message)}</span>`; }
    }, 350);
  }

  function openCompose() {
    $("#composeForm").reset();
    schedulePreview();
    $("#composeModal").showModal();
  }

  async function submitCompose(ev) {
    ev.preventDefault();
    const f = ev.target;
    const payload = Object.fromEntries(new FormData(f).entries());
    try {
      const res = await api("/compose", { method: "POST", body: payload });
      const e = res.delivered[0];
      $("#composeModal").close();
      renderStats(res.stats);
      toast(`Mensaje entregado en <b>${FOLDER_TITLES[e.folder]}</b> (${pct(e.prediction.spam_probability)} spam)`);
      if (state.folder !== e.folder) switchFolder(e.folder);
      await loadList({ highlightNew: true });
    } catch (err) { toast(esc(err.message), true); }
  }

  // ------------------------------------------------------------------ navigation
  function switchFolder(folder) {
    state.folder = folder;
    $$(".folder").forEach((b) => b.classList.toggle("active", b.dataset.folder === folder));
    $("#folderTitle").textContent = FOLDER_TITLES[folder];
    closeReader();
    $("#sidebar").classList.remove("open");
    loadList();
  }

  async function pollModel() {
    try {
      const h = await api("/health");
      state.threshold = h.threshold;
      const short = h.model.split("/").pop();
      state.model = { ...h, short };
      $("#modelText").textContent = `${h.fine_tuned ? "MiniLM fine-tuned (SMS)" : "Goodmotion/spam-mail-classifier"} · ${h.model_loaded ? "listo" : "cargando…"}`;
      $("#modelChip").title = `Modelo: ${h.model}\nUmbral: ${h.threshold}\nCorreos en el pool: ${h.pool_size}`;
      $("#modelChip").classList.toggle("ready", h.model_loaded);
      if (!h.model_loaded) setTimeout(pollModel, 1500);
    } catch { setTimeout(pollModel, 3000); }
  }

  // ------------------------------------------------------------------ events
  function bind() {
    $$(".folder").forEach((b) => b.addEventListener("click", () => switchFolder(b.dataset.folder)));
    $("#genBtn").addEventListener("click", () => generate(Number($("#genCount").value)));
    $("#emptyGen").addEventListener("click", () => generate(Number($("#genCount").value)));
    $("#autoToggle").addEventListener("change", (e) => (e.target.checked ? startAuto() : stopAuto()));
    $("#resetBtn").addEventListener("click", async () => {
      if (!confirm("¿Vaciar ambas carpetas del buzón?")) return;
      stopAuto();
      renderStats(await api("/reset", { method: "POST" }));
      state.knownUids.clear();
      closeReader();
      loadList();
      toast("Buzón vaciado");
    });
    $("#mailList").addEventListener("click", (e) => { const li = e.target.closest(".mail"); if (li) openEmail(li.dataset.uid); });
    $("#mailList").addEventListener("keydown", (e) => { if (e.key === "Enter") { const li = e.target.closest(".mail"); if (li) openEmail(li.dataset.uid); } });
    $("#readerBody").addEventListener("click", (e) => { const b = e.target.closest("[data-act]"); if (b) readerAction(b.dataset.act, b).catch((err) => toast(esc(err.message), true)); });

    let searchTimer;
    $("#search").addEventListener("input", (e) => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.query = e.target.value; loadList(); }, 200); });

    $("#composeBtn").addEventListener("click", openCompose);
    $("#composeClose").addEventListener("click", () => $("#composeModal").close());
    $("#composeCancel").addEventListener("click", () => $("#composeModal").close());
    $("#composeForm").addEventListener("submit", submitCompose);
    $("#composeForm").addEventListener("input", (e) => { if (["subject", "body"].includes(e.target.name)) schedulePreview(); });
    $$("[data-sample]").forEach((b) => b.addEventListener("click", () => {
      const s = SAMPLES[b.dataset.sample], f = $("#composeForm");
      Object.entries(s).forEach(([k, v]) => (f[k].value = v));
      schedulePreview();
    }));

    $("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, textarea, select")) return;
      if (e.key === "c") { e.preventDefault(); openCompose(); }
      if (e.key === "g") generate(Number($("#genCount").value));
      if (e.key === "Escape") closeReader();
    });
  }

  bind();
  pollModel();
  refreshStats();
  loadList().then(() => {
    const uid = location.hash.slice(1);
    if (uid) openEmail(uid).catch(() => closeReader());
  });
})();
