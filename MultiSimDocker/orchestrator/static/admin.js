import { api, el, fmtTime, keyModal, now, openModal, syncClock } from "./common.js";

const $ = (id) => document.getElementById(id);
const LIMIT_FIELDS = ["max_sims", "idle_timeout_minutes", "max_hold_hours", "max_held_sims", "max_claims_per_client"];

let state = null;
let stats = {};
let refreshTimer = null;
let statsTimer = null;
let limitsDirty = false;

function showLogin(message = "") {
	clearInterval(refreshTimer);
	clearInterval(statsTimer);
	$("dashboard").classList.add("hidden");
	$("login").classList.remove("hidden");
	$("login-error").textContent = message;
	$("password").focus();
}

function showDashboard() {
	$("login").classList.add("hidden");
	$("dashboard").classList.remove("hidden");
}

async function guarded(fn) {
	try {
		return await fn();
	} catch (err) {
		if (err.status === 401 || err.status === 503) showLogin(err.status === 503 ? err.message : "");
		else alert(err.message);
		throw err;
	}
}

function fmtBytes(bytes) {
	if (bytes == null) return "--";
	const mb = bytes / 1024 / 1024;
	return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

function toLocalInputValue(date) {
	const pad = (n) => String(n).padStart(2, "0");
	return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

// datetime-local value (browser clock) -> server epoch seconds
function inputToServerEpoch(value) {
	const date = new Date(value);
	if (isNaN(date)) return null;
	return date.getTime() / 1000 + (now() - Date.now() / 1000);
}

// -- simulations -----------------------------------------------------------

function simAction(label, handler, cls = "secondary small") {
	return el("button", { class: cls, onclick: handler }, label);
}

async function releaseSim(sim) {
	if (!confirm(`Release ${sim.id} (${sim.sceneTitle}, ${sim.ownerName || "anonymous"})? Its container is stopped immediately.`)) return;
	await guarded(() => api(`/admin/api/sims/${sim.id}/release`, { method: "POST" }));
	refresh();
}

async function restartSim(sim) {
	if (!confirm(`Restart the container of ${sim.id}? Its simulation state is lost; the owner keeps their claim.`)) return;
	await guarded(() => api(`/admin/api/sims/${sim.id}/restart`, { method: "POST" }));
	refresh();
}

async function reissueKey(sim) {
	if (!confirm(`Issue a new key for ${sim.id}? The current key stops working immediately.`)) return;
	const { key } = await guarded(() => api(`/admin/api/sims/${sim.id}/reissue-key`, { method: "POST" }));
	keyModal(key, {
		title: `New key for ${sim.id}`,
		intro: "Give this key to the person who should control the simulation. They can enter it in the lobby under \"Have a key?\" or use the reclaim link.",
	});
}

async function showLogs(sim) {
	const pre = el("pre", { class: "log" }, "Loading...");
	openModal([el("h3", {}, `Container logs: ${sim.containerName}`), pre], { wide: true });
	try {
		const response = await fetch(`/admin/api/sims/${sim.id}/logs`, { credentials: "same-origin" });
		pre.textContent = (await response.text()) || "(no output)";
		pre.scrollTop = pre.scrollHeight;
	} catch (err) {
		pre.textContent = String(err);
	}
}

function editHold(sim) {
	const untilInput = el("input", {
		type: "datetime-local",
		value: toLocalInputValue(sim.held ? new Date((sim.holdUntil - (now() - Date.now() / 1000)) * 1000) : new Date(Date.now() + 4 * 3600e3)),
	});
	const reasonInput = el("input", { maxlength: 300 });
	reasonInput.value = sim.holdReason || "";
	const errorEl = el("p", { class: "error small" });
	const save = async (clear) => {
		const body = clear ? { until: null } : { until: inputToServerEpoch(untilInput.value), reason: reasonInput.value.trim() };
		try {
			await api(`/admin/api/sims/${sim.id}/hold`, { method: "PUT", body });
			close();
			refresh();
		} catch (err) {
			errorEl.textContent = err.message;
		}
	};
	const { close } = openModal([
		el("h3", {}, `Hold for ${sim.id}`),
		el("p", { class: "small" }, "Admin holds ignore the visitor limits on duration and number of holds."),
		el("div", { class: "field" }, el("label", {}, "Hold until"), untilInput),
		el("div", { class: "field" }, el("label", {}, "Reason"), reasonInput),
		errorEl,
		el(
			"div",
			{ class: "buttons" },
			sim.held ? el("button", { class: "secondary", onclick: () => save(true) }, "Clear hold") : null,
			el("button", { class: "secondary", onclick: () => close() }, "Cancel"),
			el("button", { onclick: () => save(false) }, "Save hold")
		),
	]);
}

function renderSims() {
	const body = $("sims-body");
	if (!state.sims.length) {
		body.replaceChildren(el("tr", {}, el("td", { colspan: 11, class: "muted" }, "No simulations running.")));
		return;
	}
	body.replaceChildren(
		...state.sims.map((sim) => {
			const st = stats[sim.id];
			const containerNote = sim.containerState && sim.containerState !== "running" ? ` (container: ${sim.containerState})` : "";
			return el(
				"tr",
				{},
				el("td", {}, el("a", { href: `/view/${sim.id}`, target: "_blank", class: "mono" }, sim.id)),
				el("td", {}, sim.sceneTitle),
				el("td", {}, sim.ownerName || el("span", { class: "muted" }, "anonymous"), el("div", { class: "muted small" }, sim.clientAddr)),
				el("td", {}, el("span", { class: `badge ${sim.status}` }, sim.status), el("span", { class: "muted small" }, containerNote), sim.ownerConnected ? el("div", { class: "small muted" }, "owner connected") : null),
				el("td", {}, fmtTime(sim.createdAt)),
				el("td", {}, fmtTime(sim.lastActive)),
				el("td", {}, fmtTime(sim.expiresAt)),
				el("td", { class: "reason" }, sim.held ? [el("div", {}, `until ${fmtTime(sim.holdUntil)}`), el("div", { class: "muted small" }, sim.holdReason)] : el("span", { class: "muted" }, "--")),
				el("td", {}, String(sim.watchers)),
				el("td", { class: "small" }, st ? `${st.cpuPercent != null ? st.cpuPercent.toFixed(0) : "--"}% / ${fmtBytes(st.memoryBytes)}` : "--"),
				el(
					"td",
					{},
					el(
						"div",
						{ class: "actions" },
						simAction("Logs", () => showLogs(sim)),
						simAction("Hold...", () => editHold(sim)),
						simAction("New key", () => reissueKey(sim)),
						simAction("Restart", () => restartSim(sim)),
						simAction("Release", () => releaseSim(sim), "danger small")
					)
				)
			);
		})
	);
}

// -- limits ----------------------------------------------------------------

function renderLimits() {
	if (limitsDirty) return; // don't clobber an edit in progress
	for (const field of LIMIT_FIELDS) $(field).value = state.limits[field];
}

for (const field of LIMIT_FIELDS) $(field).addEventListener("input", () => (limitsDirty = true));

$("limits-form").addEventListener("submit", async (event) => {
	event.preventDefault();
	const body = {};
	for (const field of LIMIT_FIELDS) {
		const raw = $(field).value;
		if (raw !== "") body[field] = Number(raw);
	}
	try {
		await guarded(() => api("/admin/api/limits", { method: "PUT", body }));
		limitsDirty = false;
		$("limits-status").textContent = "Saved.";
		refresh();
	} catch {}
});

$("reset-limits-btn").addEventListener("click", async () => {
	if (!confirm("Discard all limit changes made here and go back to the values from the server's environment?")) return;
	await guarded(() => api("/admin/api/limits", { method: "DELETE" }));
	limitsDirty = false;
	$("limits-status").textContent = "Reset to defaults.";
	refresh();
});

// -- scenes ----------------------------------------------------------------

function renderScenes() {
	const select = $("create-scene");
	const selected = select.value;
	select.replaceChildren(...state.scenes.map((s) => el("option", { value: s.id }, s.title)));
	if (selected) select.value = selected;

	$("scenes-body").replaceChildren(
		...state.scenes.map((scene) => {
			const build = scene.build;
			let status = scene.imageAvailable ? el("span", { class: "badge running" }, "image ready") : el("span", { class: "badge starting" }, "image missing");
			let detail = null;
			if (build?.state === "building") status = el("span", { class: "badge starting" }, "building...");
			if (build?.state === "failed") detail = el("details", {}, el("summary", { class: "error small" }, "last build failed"), el("pre", { class: "log" }, build.detail));
			if (build?.state === "done") detail = el("div", { class: "muted small" }, `built at ${fmtTime(build.ts)}`);
			return el(
				"tr",
				{},
				el("td", {}, el("strong", {}, scene.title), el("div", { class: "muted small mono" }, scene.id)),
				el("td", { class: "mono small" }, scene.image),
				el("td", {}, status, detail),
				el(
					"td",
					{},
					scene.buildable
						? el(
								"button",
								{
									class: "secondary small",
									disabled: build?.state === "building",
									onclick: async () => {
										await guarded(() => api(`/admin/api/scenes/${scene.id}/build`, { method: "POST" }));
										refresh();
									},
								},
								scene.imageAvailable ? "Rebuild image" : "Build image"
						  )
						: el("span", { class: "muted small" }, "no Dockerfile")
				)
			);
		})
	);
}

$("create-form").addEventListener("submit", async (event) => {
	event.preventDefault();
	const statusEl = $("create-status");
	const holdValue = $("create-hold").value;
	const body = {
		sceneId: $("create-scene").value,
		ownerName: $("create-owner").value.trim(),
		holdUntil: holdValue ? inputToServerEpoch(holdValue) : null,
		holdReason: $("create-reason").value.trim(),
		ignoreCapacity: $("create-ignore-cap").checked,
	};
	statusEl.className = "small muted";
	statusEl.textContent = "Starting...";
	try {
		const result = await api("/admin/api/sims", { method: "POST", body });
		statusEl.textContent = "";
		keyModal(result.key, {
			title: `Started ${result.sim.id}`,
			intro: "Use this key (lobby -> \"Have a key?\") to control the simulation, or hand it to the person who needs it.",
		});
		refresh();
	} catch (err) {
		statusEl.className = "small error";
		statusEl.textContent = err.message;
		if (err.status === 401) showLogin();
	}
});

// -- events ----------------------------------------------------------------

function renderEvents() {
	$("events-body").replaceChildren(
		...state.events.map((event) =>
			el(
				"tr",
				{},
				el("td", { class: "small", style: { whiteSpace: "nowrap" } }, fmtTime(event.ts)),
				el("td", { class: "mono small" }, event.sim_id || ""),
				el("td", { class: "event-kind small" }, event.kind),
				el("td", { class: "small reason" }, event.detail)
			)
		)
	);
}

// -- refresh loop ----------------------------------------------------------

async function refresh() {
	try {
		state = await api("/admin/api/state");
	} catch (err) {
		if (err.status === 401 || err.status === 503) showLogin(err.status === 503 ? err.message : "");
		return;
	}
	syncClock(state.serverTime);
	showDashboard();
	renderSims();
	renderLimits();
	renderScenes();
	renderEvents();
}

async function refreshStats() {
	try {
		stats = await api("/admin/api/stats");
		if (state) renderSims();
	} catch {}
}

function startLoops() {
	clearInterval(refreshTimer);
	clearInterval(statsTimer);
	refreshTimer = setInterval(refresh, 5000);
	statsTimer = setInterval(refreshStats, 15000);
	refreshStats();
}

$("login-form").addEventListener("submit", async (event) => {
	event.preventDefault();
	try {
		await api("/admin/api/login", { method: "POST", body: { password: $("password").value } });
		$("password").value = "";
		await refresh();
		startLoops();
	} catch (err) {
		$("login-error").textContent = err.message;
	}
});

$("logout-btn").addEventListener("click", async () => {
	await api("/admin/api/logout", { method: "POST" }).catch(() => {});
	showLogin();
});

(async () => {
	await refresh();
	if (state) startLoops();
})();
