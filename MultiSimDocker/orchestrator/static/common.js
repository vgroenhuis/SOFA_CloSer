// Shared helpers for the lobby, viewer and admin pages.

export class ApiError extends Error {
	constructor(message, status, detail) {
		super(message);
		this.status = status;
		this.detail = detail;
	}
}

export async function api(path, { method = "GET", body } = {}) {
	const init = { method, credentials: "same-origin", headers: {} };
	if (body !== undefined) {
		init.headers["Content-Type"] = "application/json";
		init.body = JSON.stringify(body);
	}
	const response = await fetch(path, init);
	let data = null;
	try {
		data = await response.json();
	} catch {
		// not JSON (or empty)
	}
	if (!response.ok) {
		const detail = data?.detail;
		let message = `Request failed (HTTP ${response.status})`;
		if (typeof detail === "string") message = detail;
		else if (Array.isArray(detail)) message = detail.map((d) => d.msg).join("; ");
		else if (detail?.message) message = detail.message;
		throw new ApiError(message, response.status, detail);
	}
	return data;
}

// Server epoch seconds minus browser epoch seconds, so countdowns stay right
// on machines with a skewed clock.
let clockOffset = 0;
export function syncClock(serverTime) {
	if (typeof serverTime === "number") clockOffset = serverTime - Date.now() / 1000;
}
export function now() {
	return Date.now() / 1000 + clockOffset;
}

function sameDay(a, b) {
	return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

export function fmtTime(epoch) {
	if (epoch == null) return "--";
	const date = new Date((epoch - clockOffset) * 1000);
	const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
	if (sameDay(date, new Date())) return time;
	return `${date.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })} ${time}`;
}

export function fmtDuration(seconds) {
	seconds = Math.max(0, Math.round(seconds));
	if (seconds < 60) return `${seconds}s`;
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) return `${minutes} min`;
	const hours = Math.floor(minutes / 60);
	const rest = minutes % 60;
	return rest ? `${hours} h ${rest} min` : `${hours} h`;
}

export function fmtCountdown(seconds) {
	seconds = Math.max(0, Math.floor(seconds));
	const m = Math.floor(seconds / 60);
	const s = String(seconds % 60).padStart(2, "0");
	return `${m}:${s}`;
}

// Tiny DOM builder: el("div", { class: "x", onclick: fn }, "text", childNode).
// Text is always inserted as text nodes, never as HTML, since owner names and
// hold reasons are user-supplied.
export function el(tag, attrs = {}, ...children) {
	const node = document.createElement(tag);
	for (const [key, value] of Object.entries(attrs)) {
		if (value == null || value === false) continue;
		if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
		else if (key === "class") node.className = value;
		else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
		else node.setAttribute(key, value === true ? "" : value);
	}
	for (const child of children.flat()) {
		if (child == null || child === false) continue;
		node.append(child instanceof Node ? child : document.createTextNode(String(child)));
	}
	return node;
}

export async function copyText(text) {
	try {
		await navigator.clipboard.writeText(text);
		return true;
	} catch {
		return false;
	}
}

export function reclaimLink(key) {
	return `${location.origin}/#reclaim=${encodeURIComponent(key)}`;
}

// Minimal modal: returns { overlay, close }. Closes on Escape / backdrop click.
export function openModal(content, { wide = false, onClose } = {}) {
	const modal = el("div", { class: `modal${wide ? " wide" : ""}`, role: "dialog", "aria-modal": "true" }, content);
	const overlay = el("div", { class: "overlay" }, modal);
	const close = () => {
		overlay.remove();
		window.removeEventListener("keydown", onKey);
		onClose?.();
	};
	const onKey = (event) => {
		if (event.key === "Escape") close();
	};
	overlay.addEventListener("mousedown", (event) => {
		if (event.target === overlay) close();
	});
	window.addEventListener("keydown", onKey);
	document.body.append(overlay);
	modal.querySelector("input, textarea, select, button")?.focus();
	return { overlay, modal, close };
}

export function keyModal(key, { title = "Save your simulation key", intro } = {}) {
	const status = el("span", { class: "muted small" });
	const { close } = openModal([
		el("h3", {}, title),
		el(
			"p",
			{},
			intro ??
				"This key lets you take control of this simulation again from any browser -- for example after closing this tab or switching computers. Anyone with the key can control the simulation, so keep it to yourself."
		),
		el("div", { class: "key-box" }, key),
		el(
			"div",
			{ class: "inline-form" },
			el("button", { class: "secondary small", onclick: async () => (status.textContent = (await copyText(key)) ? "Key copied." : "Copy failed -- select it manually.") }, "Copy key"),
			el(
				"button",
				{ class: "secondary small", onclick: async () => (status.textContent = (await copyText(reclaimLink(key))) ? "Reclaim link copied." : "Copy failed.") },
				"Copy reclaim link"
			),
			status
		),
		el("div", { class: "buttons" }, el("button", { onclick: () => close() }, "I've saved it")),
	]);
}
