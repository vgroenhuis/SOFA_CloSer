import { api, el, fmtCountdown, fmtTime, keyModal, now, openModal, syncClock } from "./common.js";

const simId = location.pathname.split("/").filter(Boolean)[1] || "";

const $ = (id) => document.getElementById(id);
const frame = $("frame");
const startingOverlay = $("starting-overlay");
const endedOverlay = $("ended-overlay");
const idleBanner = $("idle-banner");

// An owner counts as "active" if they interacted with the page (including
// the simulation inside the iframe) within this window while the tab was
// visible. Only active heartbeats keep the simulation from being reclaimed.
const ACTIVE_WINDOW_MS = 2 * 60 * 1000;
const IDLE_WARNING_SECONDS = 3 * 60;
const ACTIVITY_EVENTS = ["pointerdown", "pointermove", "keydown", "wheel", "touchstart"];

let sim = null;
let frameSrc = null;
let ended = false;
let lastInteraction = Date.now();
let pollTimer = null;

function markActive() {
	lastInteraction = Date.now();
}
for (const type of ACTIVITY_EVENTS) window.addEventListener(type, markActive, { passive: true });
frame.addEventListener("load", () => {
	// Same origin, so interaction inside the simulation's own UI (orbiting
	// the camera, dragging sliders) counts as activity too.
	try {
		for (const type of ACTIVITY_EVENTS) frame.contentWindow.addEventListener(type, markActive, { passive: true });
	} catch {}
});

function isActive() {
	return !document.hidden && Date.now() - lastInteraction < ACTIVE_WINDOW_MS;
}

function showEnded(reason) {
	ended = true;
	clearTimeout(pollTimer);
	frame.src = "about:blank";
	frameSrc = null;
	idleBanner.classList.add("hidden");
	startingOverlay.classList.add("hidden");
	endedOverlay.classList.remove("hidden");
	$("ended-reason").textContent = reason || "";
	$("owner-controls").hidden = true;
	$("viewer-controls").hidden = true;
	$("status-badge").textContent = "ended";
	$("status-badge").className = "badge";
}

function render() {
	if (!sim || ended) return;
	syncClock(sim.serverTime);
	document.title = `${sim.sceneTitle} (${sim.id})`;
	$("title").textContent = sim.sceneTitle;
	$("sim-id").textContent = sim.id;
	$("status-badge").textContent = sim.status;
	$("status-badge").className = `badge ${sim.status}`;
	$("held-badge").classList.toggle("hidden", !sim.held);
	$("held-badge").textContent = sim.held ? `held until ${fmtTime(sim.holdUntil)}` : "";

	const role = $("role");
	role.textContent = sim.mine ? "You control this simulation" : `Watching${sim.ownerName ? ` ${sim.ownerName}'s simulation` : ""} (read-only)`;
	role.className = sim.mine ? "owner" : "muted";
	$("watchers").textContent = `${sim.watchers} watching`;

	$("owner-controls").hidden = !sim.mine;
	$("viewer-controls").hidden = sim.mine;
	$("viewer-expiry").textContent = sim.held ? `Reserved until ${fmtTime(sim.holdUntil)}` : "";
	renderExpiry();

	if (sim.status === "running") {
		startingOverlay.classList.add("hidden");
		const desired = `/sim/${sim.id}/?role=${sim.mine ? "owner" : "viewer"}`;
		if (frameSrc !== desired) {
			frameSrc = desired;
			frame.src = desired;
		}
	} else {
		// Blank the frame so it reloads once the (re)started container is up.
		if (frameSrc !== null) {
			frame.src = "about:blank";
			frameSrc = null;
		}
		startingOverlay.classList.remove("hidden");
		$("starting-text").textContent = "Starting the simulation container... this usually takes a few seconds.";
	}
}

function renderExpiry() {
	if (!sim || ended) return;
	const expiry = $("expiry");
	const remaining = sim.expiresAt - now();
	const warn = sim.mine && !sim.held && remaining < IDLE_WARNING_SECONDS;
	if (sim.held) {
		expiry.textContent = `Reserved until ${fmtTime(sim.holdUntil)}`;
	} else {
		expiry.textContent = `Returns to the pool at ${fmtTime(sim.expiresAt)} if idle`;
	}
	expiry.classList.toggle("soon", warn);
	idleBanner.classList.toggle("hidden", !warn);
	if (warn) $("idle-text").textContent = `No activity detected -- this simulation returns to the pool in ${fmtCountdown(remaining)}.`;
}
setInterval(renderExpiry, 1000);

async function poll() {
	clearTimeout(pollTimer);
	if (ended) return;
	try {
		if (sim?.mine) {
			try {
				sim = await api(`/api/sims/${simId}/heartbeat`, { method: "POST", body: { active: isActive() } });
			} catch (err) {
				if (err.status !== 403) throw err;
				sim = await api(`/api/sims/${simId}`); // key no longer valid (e.g. reissued by admin)
			}
		} else {
			sim = await api(`/api/sims/${simId}`);
		}
		render();
	} catch (err) {
		if (err.status === 410) {
			const reason = err.detail?.reason;
			showEnded(reason ? `Reason: ${reason} (at ${fmtTime(err.detail.endedAt)}).` : "");
			return;
		}
		if (err.status === 404) {
			showEnded("There is no simulation with this id.");
			return;
		}
		$("starting-text").textContent = "Lost contact with the server, retrying...";
	}
	pollTimer = setTimeout(poll, sim?.status === "starting" ? 1500 : 5000);
}

// -- owner actions ---------------------------------------------------------

$("key-btn").addEventListener("click", async () => {
	try {
		const { key } = await api(`/api/sims/${simId}/key`);
		keyModal(key, { title: "Your simulation key" });
	} catch (err) {
		alert(err.message);
	}
});

$("still-here-btn").addEventListener("click", () => {
	markActive();
	poll();
});

$("restart-btn").addEventListener("click", () => {
	const { close } = openModal([
		el("h3", {}, "Restart simulation?"),
		el("p", {}, "The simulation container is restarted from scratch; its current state is lost. You keep your claim and key."),
		el(
			"div",
			{ class: "buttons" },
			el("button", { class: "secondary", onclick: () => close() }, "Cancel"),
			el(
				"button",
				{
					onclick: async () => {
						close();
						try {
							sim = await api(`/api/sims/${simId}/restart`, { method: "POST" });
							render();
							poll();
						} catch (err) {
							alert(err.message);
						}
					},
				},
				"Restart"
			)
		),
	]);
});

$("release-btn").addEventListener("click", () => {
	const { close } = openModal([
		el("h3", {}, "Release this simulation?"),
		el("p", {}, "The simulation is stopped and its slot is returned to the pool for someone else. This can't be undone."),
		el(
			"div",
			{ class: "buttons" },
			el("button", { class: "secondary", onclick: () => close() }, "Cancel"),
			el(
				"button",
				{
					class: "danger",
					onclick: async () => {
						close();
						try {
							await api(`/api/sims/${simId}/release`, { method: "POST" });
							showEnded("You released it. Thanks for freeing up the slot!");
						} catch (err) {
							alert(err.message);
						}
					},
				},
				"Release"
			)
		),
	]);
});

function toLocalInputValue(date) {
	const pad = (n) => String(n).padStart(2, "0");
	return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

$("hold-btn").addEventListener("click", async () => {
	let limits;
	try {
		limits = await api("/api/status");
	} catch (err) {
		alert(err.message);
		return;
	}
	const maxHours = limits.maxHoldHours;
	const durations = [1, 2, 3, 4, 6, 8, 12, 24, 48].filter((h) => h <= maxHours);
	if (!durations.includes(maxHours)) durations.push(maxHours);

	let mode = "duration";
	const durationSelect = el(
		"select",
		{},
		durations.map((h) => el("option", { value: h }, h === 1 ? "1 hour" : `${h} hours`))
	);
	durationSelect.value = String(Math.min(4, maxHours));
	const nowDate = new Date();
	const defaultUntil = new Date(Math.ceil((nowDate.getTime() + 2 * 3600e3) / 900e3) * 900e3);
	const untilInput = el("input", {
		type: "datetime-local",
		value: toLocalInputValue(defaultUntil),
		min: toLocalInputValue(nowDate),
		max: toLocalInputValue(new Date(nowDate.getTime() + maxHours * 3600e3)),
	});
	const reasonInput = el("textarea", { rows: 3, maxlength: 300, placeholder: "e.g. Live demo for the visiting delegation at 15:00 in room Z-203" });
	reasonInput.value = sim?.holdReason || "";
	const errorEl = el("p", { class: "error small hidden" });

	const durationField = el("div", { class: "field" }, el("label", {}, "Keep it for"), durationSelect);
	const untilField = el("div", { class: "field hidden" }, el("label", {}, "Keep it until"), untilInput);
	const durationTab = el("button", { class: "secondary small active", type: "button" }, "For a duration");
	const untilTab = el("button", { class: "secondary small", type: "button" }, "Until a time");
	const setMode = (next) => {
		mode = next;
		durationTab.classList.toggle("active", mode === "duration");
		untilTab.classList.toggle("active", mode === "until");
		durationField.classList.toggle("hidden", mode !== "duration");
		untilField.classList.toggle("hidden", mode !== "until");
	};
	durationTab.addEventListener("click", () => setMode("duration"));
	untilTab.addEventListener("click", () => setMode("until"));

	const submit = async () => {
		errorEl.classList.add("hidden");
		let untilBrowser;
		if (mode === "duration") {
			untilBrowser = Date.now() / 1000 + parseFloat(durationSelect.value) * 3600;
		} else {
			const parsed = new Date(untilInput.value);
			if (isNaN(parsed)) {
				errorEl.textContent = "Pick a date and time.";
				errorEl.classList.remove("hidden");
				return;
			}
			untilBrowser = parsed.getTime() / 1000;
		}
		// Convert from this browser's clock to the server's.
		const until = untilBrowser + (now() - Date.now() / 1000);
		try {
			sim = await api(`/api/sims/${simId}/hold`, { method: "PUT", body: { until, reason: reasonInput.value.trim() } });
			render();
			close();
		} catch (err) {
			errorEl.textContent = err.message;
			errorEl.classList.remove("hidden");
		}
	};

	const clearHold = async () => {
		try {
			sim = await api(`/api/sims/${simId}/hold`, { method: "DELETE" });
			render();
			close();
		} catch (err) {
			errorEl.textContent = err.message;
			errorEl.classList.remove("hidden");
		}
	};

	const { close } = openModal([
		el("h3", {}, "Hold this simulation"),
		el(
			"p",
			{},
			`Normally a simulation returns to the pool after ${limits.idleTimeoutMinutes} minutes without activity. ` +
				`A hold keeps it reserved for you -- for example for a live demo later today -- for up to ${maxHours} hours. ` +
				`Slots are scarce, so please release it when you're done.`
		),
		sim?.held ? el("p", { class: "small" }, el("strong", {}, `Currently held until ${fmtTime(sim.holdUntil)}.`), " Setting a new hold replaces it.") : null,
		el("div", { class: "segmented" }, durationTab, untilTab),
		durationField,
		untilField,
		el("div", { class: "field" }, el("label", {}, "Reason (visible to the administrator)"), reasonInput),
		errorEl,
		el(
			"div",
			{ class: "buttons" },
			sim?.held ? el("button", { class: "secondary", onclick: clearHold }, "Remove hold") : null,
			el("button", { class: "secondary", onclick: () => close() }, "Cancel"),
			el("button", { onclick: submit }, "Set hold")
		),
	]);
});

// -- viewer actions --------------------------------------------------------

$("have-key-btn").addEventListener("click", () => {
	const input = el("input", { class: "mono", placeholder: "XXXX-XXXX-XXXX-XXXX", autocomplete: "off", spellcheck: "false" });
	const errorEl = el("p", { class: "error small hidden" });
	const submit = async () => {
		try {
			const result = await api("/api/reclaim", { method: "POST", body: { key: input.value.trim() } });
			close();
			if (result.simId !== simId) {
				location.href = `/view/${result.simId}`;
				return;
			}
			sim = await api(`/api/sims/${simId}`);
			render();
			poll();
		} catch (err) {
			errorEl.textContent = err.message;
			errorEl.classList.remove("hidden");
		}
	};
	input.addEventListener("keydown", (event) => {
		if (event.key === "Enter") submit();
	});
	const { close } = openModal([
		el("h3", {}, "Take control with your key"),
		el("p", {}, "Enter the key you got when you started this simulation."),
		el("div", { class: "field" }, input),
		errorEl,
		el("div", { class: "buttons" }, el("button", { class: "secondary", onclick: () => close() }, "Cancel"), el("button", { onclick: submit }, "Take control")),
	]);
});

// -- start -----------------------------------------------------------------

let newKey = null;
try {
	newKey = sessionStorage.getItem(`msd_newkey_${simId}`);
	sessionStorage.removeItem(`msd_newkey_${simId}`);
} catch {}
if (newKey) keyModal(newKey);

poll();
