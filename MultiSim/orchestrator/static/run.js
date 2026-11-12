// A saved run (run/<token>): the scene's viewer, read-only, playing back the
// run from the orchestrator (no simulation behind it); a link to share it.
import { api, copyText } from "./common.js";

const token = (location.pathname.match(/\/run\/([A-Za-z0-9_-]+)/) || [])[1] || "";
const $ = (id) => document.getElementById(id);
const LEVELS = { private: "private (unlocked visitors)", admin: "admin only" };

function shareUrl() {
	return new URL(`run/${token}`, document.baseURI).href;
}

function fmtDate(epoch) {
	return new Date(epoch * 1000).toLocaleString([], { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

async function load() {
	let run;
	try {
		run = await api(`api/runs/${encodeURIComponent(token)}`);
	} catch {
		$("missing-overlay").classList.remove("hidden");
		$("share-btn").disabled = true;
		return;
	}
	const title = run.name || run.sceneTitle;
	document.title = `${title} (saved run)`;
	$("title").textContent = title;
	const parts = [];
	if (run.name) parts.push(run.sceneTitle);
	parts.push(`${run.simTime.toFixed(1)} s simulated`);
	parts.push(`saved ${fmtDate(run.createdAt)}${run.ownerName ? ` by ${run.ownerName}` : ""}`);
	$("details").textContent = parts.join(" · ");
	if (LEVELS[run.visibility]) {
		$("level-badge").textContent = LEVELS[run.visibility];
		$("level-badge").title = "Listed in the lobby only for these visitors; anyone with the link can watch it";
		$("level-badge").classList.remove("hidden");
	}
	$("frame").src = `run/${token}/v/?role=viewer&saved=1`;
}

$("share-btn").addEventListener("click", async () => {
	const url = shareUrl();
	const ok = await copyText(url);
	$("share-status").textContent = ok ? "Link copied." : url;
	if (ok) setTimeout(() => ($("share-status").textContent = ""), 4000);
});

load();
