import { api, el, fmtTime, now, syncClock } from "./common.js";

const sceneGrid = document.getElementById("scene-grid");
const runningList = document.getElementById("running-list");
const mineSection = document.getElementById("mine-section");
const mineList = document.getElementById("mine-list");
const startNotice = document.getElementById("start-notice");
const capacitySlots = document.getElementById("capacity-slots");
const capacityText = document.getElementById("capacity-text");
const ownerNameInput = document.getElementById("owner-name");
const footer = document.getElementById("footer");

let scenes = [];
let status = null;
let starting = false;

try {
	ownerNameInput.value = localStorage.getItem("msd_owner_name") || "";
} catch {}
ownerNameInput.addEventListener("change", () => {
	try {
		localStorage.setItem("msd_owner_name", ownerNameInput.value.trim());
	} catch {}
});

function showNotice(content) {
	startNotice.replaceChildren(...[content].flat());
	startNotice.classList.toggle("hidden", !content);
}

function simBadges(sim) {
	return [
		el("span", { class: `badge ${sim.status}` }, sim.status),
		sim.held ? el("span", { class: "badge held" }, `held until ${fmtTime(sim.holdUntil)}`) : null,
		sim.mine ? el("span", { class: "badge mine" }, "yours") : null,
	];
}

function simRow(sim) {
	const meta = [
		el("span", {}, sim.ownerName ? `by ${sim.ownerName}` : "anonymous"),
		el("span", {}, `started ${fmtTime(sim.createdAt)}`),
		el("span", {}, `${sim.watchers} watching`),
		el("span", {}, sim.held ? `reserved until ${fmtTime(sim.holdUntil)}` : `free by ${fmtTime(sim.expiresAt)} if idle`),
	];
	return el(
		"div",
		{ class: "card sim-row" },
		el(
			"div",
			{},
			el("div", { class: "title" }, el("h3", {}, sim.sceneTitle), el("span", { class: "muted mono small" }, sim.id), simBadges(sim)),
			el("div", { class: "meta" }, meta)
		),
		el(
			"div",
			{ class: "actions" },
			el("a", { class: sim.mine ? "button" : "button secondary", href: `/view/${sim.id}` }, sim.mine ? "Open" : "Watch")
		)
	);
}

function renderStatus() {
	if (!status) return;
	const used = status.sims.length;
	const full = used >= status.capacity;
	capacitySlots.replaceChildren(
		...Array.from({ length: Math.max(status.capacity, used) }, (_, i) => el("div", { class: `capacity-slot${i < used ? " used" : ""}` }))
	);
	capacityText.textContent = `${used} of ${status.capacity} slots in use`;

	const mine = status.sims.filter((s) => s.mine);
	mineSection.classList.toggle("hidden", mine.length === 0);
	mineList.replaceChildren(...mine.map(simRow));

	const others = status.sims.filter((s) => !s.mine);
	runningList.replaceChildren(
		...(others.length ? others.map(simRow) : [el("p", { class: "muted" }, mine.length ? "Nobody else is running a simulation." : "No simulations are running right now.")])
	);

	for (const button of sceneGrid.querySelectorAll("button[data-scene]")) {
		button.disabled = full || starting;
	}
	if (full && !starting) {
		const nextFree = Math.min(...status.sims.map((s) => s.expiresAt));
		showNotice([
			el("strong", {}, "All slots are in use. "),
			`The earliest a slot can free up is around ${fmtTime(nextFree)} (sooner if someone releases theirs). You can watch any running simulation below in the meantime.`,
		]);
	} else if (!starting && startNotice.dataset.kind !== "error") {
		showNotice(null);
	}

	footer.textContent =
		`Simulations return to the pool after ${status.idleTimeoutMinutes} minutes without activity. ` +
		`Owners can reserve theirs for up to ${status.maxHoldHours} hours (e.g. for a demo later today).`;
}

function renderScenes() {
	if (!scenes.length) {
		sceneGrid.replaceChildren(el("p", { class: "muted" }, "No scenes are installed."));
		return;
	}
	sceneGrid.replaceChildren(
		...scenes.map((scene) =>
			el(
				"div",
				{ class: "card scene-card" },
				el("div", { class: "scene-thumb", style: scene.thumbnail ? { backgroundImage: `url("${scene.thumbnail}")` } : {} }),
				el(
					"div",
					{ class: "scene-body" },
					el("h3", {}, scene.title),
					el("p", {}, scene.description),
					el("div", { class: "actions" }, el("button", { "data-scene": scene.id, onclick: () => startScene(scene) }, "Start simulation"))
				)
			)
		)
	);
	renderStatus();
}

async function startScene(scene) {
	starting = true;
	startNotice.dataset.kind = "";
	showNotice([el("strong", {}, `Starting "${scene.title}"...`)]);
	renderStatus();
	try {
		const result = await api("/api/sims", { method: "POST", body: { sceneId: scene.id, ownerName: ownerNameInput.value.trim() } });
		// The claim cookie set by this response is what identifies the owner;
		// the key itself stays available from the viewer's Key button.
		location.href = `/view/${result.sim.id}`;
	} catch (err) {
		starting = false;
		startNotice.dataset.kind = "error";
		const links = (err.detail?.simIds || []).map((id) => el("a", { href: `/view/${id}` }, `open ${id}`));
		showNotice([el("strong", { class: "error" }, "Could not start: "), err.message, " ", ...links]);
		refresh();
	}
}

async function reclaim(key) {
	const statusEl = document.getElementById("reclaim-status");
	statusEl.className = "small muted";
	statusEl.textContent = "Checking...";
	try {
		const result = await api("/api/reclaim", { method: "POST", body: { key } });
		location.href = `/view/${result.simId}`;
	} catch (err) {
		statusEl.className = "small error";
		statusEl.textContent = err.message;
	}
}

document.getElementById("reclaim-form").addEventListener("submit", (event) => {
	event.preventDefault();
	const key = document.getElementById("reclaim-key").value.trim();
	if (key) reclaim(key);
});

// Reclaim links look like https://host/#reclaim=KEY -- the key sits in the
// fragment so it never reaches server logs or Referer headers.
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.get("reclaim")) {
	const key = fragment.get("reclaim");
	history.replaceState(null, "", location.pathname);
	document.getElementById("reclaim-key").value = key;
	reclaim(key);
}

async function refresh() {
	try {
		status = await api("/api/status");
		syncClock(status.serverTime);
		renderStatus();
	} catch (err) {
		capacityText.textContent = "server unreachable";
	}
}

async function init() {
	try {
		scenes = await api("/api/scenes");
	} catch {
		scenes = [];
	}
	await refresh();
	renderScenes();
	setInterval(refresh, 5000);
}
init();
