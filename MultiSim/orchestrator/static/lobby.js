import { api, el, fmtTime, now, openModal, syncClock } from "./common.js";

// One collapsible group per scene.json "group" (CloSer project, Playground, ...).
const sceneGroups = document.getElementById("scene-groups");
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

// Only private / admin-only scenes get a badge; visitors never see those
// unless they're allowed to.
const VISIBILITY_BADGES = { private: ["private", "private"], admin: ["admin-only", "admin only"] };

function visibilityBadge(level) {
	const badge = VISIBILITY_BADGES[level];
	return badge ? el("span", { class: `badge ${badge[0]}` }, badge[1]) : null;
}

function sceneLevel(sceneId) {
	return scenes.find((s) => s.id === sceneId)?.visibility;
}

function groupBadge(scene) {
	return scene ? el("span", { class: `badge group-${scene.group}` }, scene.groupTitle) : null;
}

function simBadges(sim) {
	return [
		el("span", { class: `badge ${sim.status}` }, sim.status),
		groupBadge(scenes.find((s) => s.id === sim.sceneId)),
		visibilityBadge(sceneLevel(sim.sceneId)),
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
			el("a", { class: sim.mine ? "button" : "button secondary", href: `view/${sim.id}` }, sim.mine ? "Open" : "Watch"),
			sim.mine ? el("button", { class: "danger", onclick: () => confirmRelease(sim) }, "Release") : null
		)
	);
}

function confirmRelease(sim) {
	const errorEl = el("p", { class: "error small hidden" });
	const { close } = openModal([
		el("h3", {}, `Release ${sim.sceneTitle} (${sim.id})?`),
		el("p", {}, "The simulation is stopped and its slot is returned to the pool for someone else. This can't be undone."),
		errorEl,
		el(
			"div",
			{ class: "buttons" },
			el("button", { class: "secondary", onclick: () => close() }, "Cancel"),
			el(
				"button",
				{
					class: "danger",
					onclick: async () => {
						try {
							await api(`api/sims/${sim.id}/release`, { method: "POST" });
							close();
							refresh();
						} catch (err) {
							errorEl.textContent = err.message;
							errorEl.classList.remove("hidden");
						}
					},
				},
				"Release"
			)
		),
	]);
}

function renderStatus() {
	if (!status) return;
	// Private simulations this browser can't see still take slots.
	const used = status.used ?? status.sims.length;
	const full = used >= status.capacity;
	capacitySlots.replaceChildren(
		...Array.from({ length: Math.max(status.capacity, used) }, (_, i) => el("div", { class: `capacity-slot${i < used ? " used" : ""}` }))
	);
	capacityText.textContent = `${used} of ${status.capacity} slots in use`;

	const mine = status.sims.filter((s) => s.mine);
	mineSection.classList.toggle("hidden", mine.length === 0);
	mineList.replaceChildren(...mine.map(simRow));
	// Starting a new simulation releases the visitor's un-held ones -- only
	// once half or more of the slots are in use (see main.plenty_of_room).
	const roomy = used < status.capacity / 2;
	document.getElementById("mine-note").classList.toggle("hidden", roomy || !mine.some((s) => !s.held));

	const others = status.sims.filter((s) => !s.mine);
	runningList.replaceChildren(
		...(others.length ? others.map(simRow) : [el("p", { class: "muted" }, mine.length ? "Nobody else is running a simulation." : "No simulations are running right now.")])
	);

	for (const button of document.querySelectorAll("#scene-grid button[data-scene], #admin-scene-grid button[data-scene]")) {
		button.disabled = full || starting;
	}
	if (full && !starting) {
		const nextFree = Math.min(...status.sims.map((s) => s.expiresAt));
		showNotice([
			el("strong", {}, "All slots are in use. "),
			Number.isFinite(nextFree)
				? `The earliest a slot can free up is around ${fmtTime(nextFree)} (sooner if someone releases theirs). You can watch any running simulation below in the meantime.`
				: "Please try again later.",
		]);
	} else if (!starting && startNotice.dataset.kind !== "error") {
		showNotice(null);
	}

	renderAccess();

	footer.textContent =
		`Simulations return to the pool after ${status.idleTimeoutMinutes} minutes without activity. ` +
		`Owners can reserve theirs for up to ${status.maxHoldHours} hours (e.g. for a demo later today).`;
}

function sceneCard(scene) {
	// The picture starts the simulation too, like the button (and is as disabled as it).
	const startFromPicture = () => {
		const button = document.querySelector(`button[data-scene="${scene.id}"]`);
		if (button && !button.disabled) startScene(scene);
	};
	const thumb = el("div", {
		class: "scene-thumb",
		role: "button",
		tabindex: "0",
		title: `Start "${scene.title}"`,
		style: scene.thumbnail ? { backgroundImage: `url("${scene.thumbnail}")` } : {},
		onclick: startFromPicture,
	});
	thumb.addEventListener("keydown", (event) => {
		if (event.key === "Enter" || event.key === " ") {
			event.preventDefault();
			startFromPicture();
		}
	});
	return el(
		"div",
		{ class: "card scene-card" },
		thumb,
		el(
			"div",
			{ class: "scene-body" },
			el("h3", {}, scene.title, visibilityBadge(scene.visibility)),
			el("p", {}, scene.description),
			el("div", { class: "actions" }, el("button", { "data-scene": scene.id, onclick: () => startScene(scene) }, "Start simulation"))
		)
	);
}

// Groups open or closed as this browser last left them; Playground and
// Archive start closed (not part of the current CloSer work).
const GROUP_ORDER = ["closer", "playground", "archive", "other"];
const GROUPS_CLOSED_BY_DEFAULT = new Set(["playground", "archive"]);
const GROUPS_KEY = "msd_lobby_groups_open";

function groupOpenState() {
	try {
		return JSON.parse(localStorage.getItem(GROUPS_KEY) || "{}");
	} catch {
		return {};
	}
}

function renderScenes() {
	if (!scenes.length) {
		sceneGroups.replaceChildren(el("p", { class: "muted" }, "No scenes are installed."));
		renderStatus();
		return;
	}
	const open = groupOpenState();
	const groups = [...new Set(scenes.map((s) => s.group))].sort((a, b) => GROUP_ORDER.indexOf(a) - GROUP_ORDER.indexOf(b));
	sceneGroups.replaceChildren(
		...groups.map((group) => {
			// Public scenes first, then private, then admin-only ones (badged);
			// by title within each (Finger 1, Finger 2, ..., Finger 10).
			const rank = { public: 0, private: 1, admin: 2 };
			const members = scenes
				.filter((s) => s.group === group)
				.sort((a, b) => (rank[a.visibility] ?? 0) - (rank[b.visibility] ?? 0) || a.title.localeCompare(b.title, undefined, { numeric: true }));
			const details = el(
				"details",
				{ class: "scene-group", "data-group": group },
				el("summary", {}, el("span", { class: "group-title" }, members[0].groupTitle), el("span", { class: "muted small" }, `${members.length} simulation${members.length === 1 ? "" : "s"}`)),
				el("div", { class: "grid" }, members.map((scene) => sceneCard(scene)))
			);
			details.open = group in open ? !!open[group] : !GROUPS_CLOSED_BY_DEFAULT.has(group);
			// On the user's click only ("toggle" also fires for the line above).
			details.firstChild.addEventListener("click", () => {
				const state = groupOpenState();
				state[group] = !details.open;
				try {
					localStorage.setItem(GROUPS_KEY, JSON.stringify(state));
				} catch {}
			});
			return details;
		})
	);
	renderStatus();
}

async function startScene(scene) {
	starting = true;
	startNotice.dataset.kind = "";
	showNotice([el("strong", {}, `Starting "${scene.title}"...`)]);
	renderStatus();
	try {
		const result = await api("api/sims", { method: "POST", body: { sceneId: scene.id, ownerName: ownerNameInput.value.trim() } });
		// The claim cookie set by this response is what identifies the owner;
		// the key itself stays available from the viewer's Key button.
		location.href = `view/${result.sim.id}`;
	} catch (err) {
		starting = false;
		startNotice.dataset.kind = "error";
		const links = (err.detail?.simIds || []).map((id) => el("a", { href: `view/${id}` }, `open ${id}`));
		showNotice([el("strong", { class: "error" }, "Could not start: "), err.message, " ", ...links]);
		refresh();
	}
}

async function reclaim(key) {
	const statusEl = document.getElementById("reclaim-status");
	statusEl.className = "small muted";
	statusEl.textContent = "Checking...";
	try {
		const result = await api("api/reclaim", { method: "POST", body: { key } });
		location.href = `view/${result.simId}`;
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

// -- private scenes: unlocked per browser with the access password --------

const accessSection = document.getElementById("access-section");

function renderAccess() {
	accessSection.classList.toggle("hidden", !status.accessLogin);
	document.getElementById("access-form").classList.toggle("hidden", status.access);
	document.getElementById("access-unlocked").classList.toggle("hidden", !status.access);
	const hint = document.getElementById("access-hint");
	hint.textContent = status.accessHint || "";
	hint.classList.toggle("hidden", status.access || !status.accessHint);
}

async function reloadAll() {
	try {
		scenes = await api("api/scenes");
	} catch {
		scenes = [];
	}
	await refresh();
	renderScenes();
}

document.getElementById("access-form").addEventListener("submit", async (event) => {
	event.preventDefault();
	const input = document.getElementById("access-password");
	const statusEl = document.getElementById("access-status");
	statusEl.textContent = "";
	try {
		await api("api/access", { method: "POST", body: { password: input.value } });
		input.value = "";
		await reloadAll();
	} catch (err) {
		statusEl.textContent = err.message;
	}
});

document.getElementById("access-lock").addEventListener("click", async () => {
	await api("api/access", { method: "DELETE" }).catch(() => {});
	await reloadAll();
});

async function refresh() {
	try {
		status = await api("api/status");
		syncClock(status.serverTime);
		renderStatus();
	} catch (err) {
		capacityText.textContent = "server unreachable";
	}
}

// Going back to the lobby after starting a simulation restores this page
// from the browser's back/forward cache exactly as it was left: mid-start,
// with the start buttons disabled. Reset that and fetch fresh state.
window.addEventListener("pageshow", (event) => {
	if (!event.persisted) return;
	starting = false;
	startNotice.dataset.kind = "";
	showNotice(null);
	refresh();
});

async function init() {
	try {
		scenes = await api("api/scenes");
	} catch {
		scenes = [];
	}
	await refresh();
	renderScenes();
	setInterval(refresh, 5000);
}

// A link back to the site's home page, when MultiSim is part of a bigger site.
async function showHomeLink() {
	try {
		const { home } = await api("api/site");
		if (!home) return;
		const link = document.getElementById("home-link");
		link.href = home.url;
		link.textContent = `\u2190 ${home.title}`;
		link.classList.remove("hidden");
	} catch {
		// no link
	}
}

init();
showHomeLink();
