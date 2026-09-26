// Generic viewer for a sofaweb scene: rebuilds the SOFA scene's visual
// models in three.js from the server's "setup" message and moves them with
// every binary frame. Everything scene-specific (parameters, charts,
// readouts) comes from api/scene.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import Chart from "chart.js/auto";

const isViewer = new URLSearchParams(location.search).get("role") === "viewer";
if (isViewer) document.body.classList.add("viewer");

const $ = (id) => document.getElementById(id);

// SOFA's renderer is classic fixed-function OpenGL working directly in
// display color space; turning three.js' color management off (and
// scaling light intensities by PI, see buildLights) reproduces its colors.
THREE.ColorManagement.enabled = false;

// ---------------------------------------------------------------------------
// three.js setup
// ---------------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.sortObjects = true;
$("viewer").appendChild(renderer.domElement);

const scene = new THREE.Scene();
let camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 1000);
let controls = null;
let setup = null;
let generation = null;
const models = new Map(); // id -> { object, geometry, info }
let ambientLight = null;
let directionalLights = [];
let orthoHalfHeight = 1;

function sofaColor(rgba) {
	return new THREE.Color(rgba[0], rgba[1], rgba[2]);
}

function disposeScene() {
	for (const { object, geometry } of models.values()) {
		scene.remove(object);
		geometry.dispose();
		object.material.dispose();
	}
	models.clear();
	for (const light of [ambientLight, ...directionalLights]) if (light) scene.remove(light);
	directionalLights = [];
	ambientLight = null;
}

function buildLights(ambient, lights) {
	// Fixed-function GL: material ambient is 0.2 x the model color (SOFA's
	// Material::setColor), times the LightManager's ambient; each light adds
	// its color x max(0, n.l). three.js' Lambert BRDF divides by PI, so the
	// intensities are scaled back up by PI.
	ambientLight = new THREE.AmbientLight(sofaColor(ambient), 0.2 * Math.PI);
	scene.add(ambientLight);
	for (const light of lights) {
		const dir = new THREE.DirectionalLight(sofaColor(light.color), Math.PI);
		scene.add(dir);
		scene.add(dir.target);
		directionalLights.push(dir);
	}
	updateLights(lights);
}

function updateLights(lights) {
	lights.forEach((light, i) => {
		const dir = directionalLights[i];
		if (!dir) return;
		dir.color.copy(sofaColor(light.color));
		// SOFA's DirectionalLight.direction points towards the light.
		const d = light.direction || [0, 0, 1];
		dir.position.set(d[0], d[1], d[2]).normalize().multiplyScalar(100);
		dir.target.position.set(0, 0, 0);
	});
}

function buildModel(info, order) {
	const geometry = new THREE.BufferGeometry();
	const positions = new Float32Array(info.vertexCount * 3);
	geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
	geometry.setIndex(info.index);
	const [r, g, b, a] = info.color;
	const color = new THREE.Color(r, g, b);
	const transparent = a < 0.999;
	const is2d = setup.view === "2d";
	let object;
	if (info.kind === "lines") {
		const material = new THREE.LineBasicMaterial({ color, transparent, opacity: a });
		object = new THREE.LineSegments(geometry, material);
	} else {
		const material = new THREE.MeshLambertMaterial({
			color,
			transparent,
			opacity: a,
			side: THREE.DoubleSide,
			depthWrite: !transparent,
		});
		object = new THREE.Mesh(geometry, material);
	}
	object.frustumCulled = false;
	if (is2d) {
		// Flat 2D scenes put everything in one plane: draw in creation
		// order (painter's algorithm) instead of fighting over depth.
		object.material.depthTest = false;
		object.material.depthWrite = false;
		object.renderOrder = order;
	} else if (transparent) {
		object.renderOrder = 1000 + order;
	}
	scene.add(object);
	models.set(info.id, { object, geometry, info });
}

function applyPositions(entry, packed) {
	const target = entry.geometry.attributes.position.array;
	const inverse = entry.info.inverse;
	if (inverse) {
		for (let i = 0; i < inverse.length; i++) {
			const j = inverse[i] * 3;
			target[i * 3] = packed[j];
			target[i * 3 + 1] = packed[j + 1];
			target[i * 3 + 2] = packed[j + 2];
		}
	} else {
		target.set(packed.subarray(0, target.length));
	}
	entry.geometry.attributes.position.needsUpdate = true;
	// Explicit normals rather than three.js' derivative-based flatShading
	// (which lit some faces from the wrong side). Models the scene
	// flat-shades itself share no vertices between triangles, so this
	// still yields one normal per face, as in SOFA.
	if (entry.info.kind === "mesh") entry.geometry.computeVertexNormals();
	entry.geometry.computeBoundingSphere();
}

// -- camera ------------------------------------------------------------------

function makeCamera() {
	const cam = setup.camera;
	const aspect = window.innerWidth / window.innerHeight;
	const distance = cam ? cam.distance : 5;
	const near = Math.max(distance / 2000, 1e-5);
	const far = distance * 200;
	if (cam && cam.projection === "Orthographic") {
		orthoHalfHeight = distance * Math.tan(((cam.fov || 45) * Math.PI) / 360);
		camera = new THREE.OrthographicCamera(-orthoHalfHeight * aspect, orthoHalfHeight * aspect, orthoHalfHeight, -orthoHalfHeight, -far, far);
	} else {
		camera = new THREE.PerspectiveCamera(cam ? cam.fov : 45, aspect, near, far);
	}
	camera.up.set(...(setup.up || [0, 0, 1]));
	if (controls) controls.dispose();
	controls = new OrbitControls(camera, renderer.domElement);
	controls.enableDamping = true;
	if (setup.view === "2d") {
		controls.enableRotate = false;
		controls.screenSpacePanning = true;
		controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
		controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
	}
	resetView();
}

function resetView() {
	const cam = setup && setup.camera;
	if (!cam) return;
	// SOFA's camera orientation is the same convention as three.js': local
	// -Z is the view direction, +Y is up.
	const q = new THREE.Quaternion(...cam.orientation);
	camera.position.set(...cam.position);
	camera.quaternion.copy(q);
	const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
	controls.target.copy(camera.position).addScaledVector(forward, cam.distance);
	camera.zoom = 1;
	camera.updateProjectionMatrix();
	controls.update();
}

function onResize() {
	const aspect = window.innerWidth / window.innerHeight;
	renderer.setSize(window.innerWidth, window.innerHeight);
	if (camera.isOrthographicCamera) {
		camera.left = -orthoHalfHeight * aspect;
		camera.right = orthoHalfHeight * aspect;
	} else {
		camera.aspect = aspect;
	}
	camera.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);
$("view-btn").addEventListener("click", resetView);

function applySetup(message) {
	const firstSetup = setup === null;
	const cameraChanged = !setup || JSON.stringify(setup.camera) !== JSON.stringify(message.camera) || setup.view !== message.view;
	const newRun = message.generation !== generation;
	setup = message;
	generation = message.generation;
	disposeScene();
	scene.background = sofaColor(message.background);
	document.body.style.background = `#${scene.background.getHexString()}`;
	buildLights(message.ambient, message.lights);
	message.models.forEach((info, order) => buildModel(info, order));
	// Keep the visitor's own view across rebuilds unless the scene's
	// camera itself changed (e.g. a geometry parameter moved it).
	if (firstSetup || cameraChanged) makeCamera();
	if (newRun) clearCharts();
}

// ---------------------------------------------------------------------------
// Charts, readouts, status
// ---------------------------------------------------------------------------

let sceneInfo = null;
const charts = [];
const readoutEls = new Map();
const MAX_CHART_POINTS = 4000;
const chartFont = { size: 10 };

function buildCharts(config) {
	const container = $("charts");
	for (const def of config.charts) {
		const panel = document.createElement("div");
		panel.className = "chart-panel panel";
		panel.innerHTML = `<div class="resize-handle-tr" title="Drag to resize (grows upward)"></div><h2></h2><div class="chart-canvas-wrap"><canvas></canvas></div>`;
		panel.querySelector("h2").textContent = def.unit ? `${def.title} (${def.unit})` : def.title;
		container.appendChild(panel);
		const chart = new Chart(panel.querySelector("canvas"), {
			type: "line",
			data: {
				datasets: def.series.map((s) => ({
					label: s.label,
					data: [],
					borderColor: s.color,
					backgroundColor: "transparent",
					borderWidth: 1.5,
					pointRadius: 0,
					tension: 0,
				})),
			},
			options: {
				animation: false,
				responsive: true,
				maintainAspectRatio: false,
				parsing: false,
				plugins: { legend: { display: def.series.length > 1, labels: { color: "#cfd3da", boxWidth: 10, font: chartFont } } },
				scales: {
					x: {
						type: "linear",
						title: { display: true, text: "time (s)", color: "#8f97a3", font: chartFont },
						ticks: { color: "#8f97a3", font: chartFont },
						grid: { color: "#2f323b" },
					},
					y: { ticks: { color: "#8f97a3", font: chartFont }, grid: { color: "#2f323b" } },
				},
			},
		});
		charts.push({ def, chart });
		makeTopRightResizable(panel, panel.querySelector(".resize-handle-tr"));
	}
	if (config.readouts.length) {
		const panel = document.createElement("div");
		panel.className = "readout-panel panel";
		panel.innerHTML = `<h2>Readouts</h2><div class="readout-grid"></div>`;
		const grid = panel.querySelector(".readout-grid");
		for (const r of config.readouts) {
			const label = document.createElement("span");
			label.className = "readout-label";
			label.textContent = r.label;
			const value = document.createElement("span");
			value.className = "readout-value";
			value.textContent = "--";
			grid.append(label, value);
			readoutEls.set(r.key, { el: value, def: r });
		}
		container.appendChild(panel);
	}
}

function clearCharts() {
	for (const { chart } of charts) for (const ds of chart.data.datasets) ds.data = [];
	chartsDirty = true;
}

let chartsDirty = false;
let lastChartT = -Infinity;
function updateTelemetry(t, scalars, full) {
	if (!full && t > lastChartT) {
		for (const { def, chart } of charts) {
			def.series.forEach((s, i) => {
				const v = scalars[s.key];
				if (v === undefined || !Number.isFinite(v)) return;
				const data = chart.data.datasets[i].data;
				data.push({ x: t, y: v });
				if (data.length > MAX_CHART_POINTS) data.splice(0, data.length - MAX_CHART_POINTS);
			});
		}
		chartsDirty = true;
	}
	if (t < lastChartT) clearCharts();
	lastChartT = t;
	for (const [key, { el, def }] of readoutEls) {
		const v = scalars[key];
		if (v === undefined || !Number.isFinite(v)) continue;
		el.textContent = `${v.toFixed(def.digits ?? 3)}${def.unit ? " " + def.unit : ""}`;
	}
}

// Chart history for a browser that connects mid-run (e.g. a watcher).
function applyHistory(message) {
	if (message.generation !== generation) return;
	clearCharts();
	for (const [t, scalars] of message.rows) {
		for (const { def, chart } of charts) {
			def.series.forEach((s, i) => {
				const v = scalars[s.key];
				if (v !== undefined && Number.isFinite(v)) chart.data.datasets[i].data.push({ x: t, y: v });
			});
		}
		lastChartT = t;
	}
	chartsDirty = true;
}

let lastStatus = null;
function renderStatus() {
	const statusEl = $("status");
	const banner = $("banner");
	const s = lastStatus;
	if (!connected) {
		statusEl.textContent = "disconnected -- retrying...";
		statusEl.className = "disconnected";
		return;
	}
	if (!s) return;
	let text = "running";
	let cls = "connected";
	if (s.status === "building") [text, cls] = ["building...", "building"];
	else if (s.status === "error") [text, cls] = ["error", "error"];
	else if (s.finished) [text, cls] = ["finished", "finished"];
	else if (s.paused) [text, cls] = ["paused", "paused"];
	statusEl.textContent = text;
	statusEl.className = cls;
	$("building").classList.toggle("hidden", s.status !== "building");
	$("pause-btn").textContent = s.paused || s.finished ? "Resume" : "Pause";
	// While paused this only resets to the start (and stays paused).
	$("reset-btn").textContent = s.paused ? "Reset" : "Restart";
	$("auto-restart-checkbox").checked = !!s.autoRestart;

	let message = "";
	banner.className = "";
	if (s.error) {
		message = s.error;
		banner.className = "error";
	} else if (s.finished && !s.autoRestart) {
		message = `Run finished (${s.endReason}).` + (isViewer ? "" : " Resume to continue, or Restart to run it again.");
	} else if (s.finished && s.autoRestart) {
		message = `Run finished (${s.endReason}) -- restarting...`;
	}
	banner.textContent = message;
	banner.classList.toggle("hidden", !message);
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

let connected = false;
const decoder = new TextDecoder();

function handleFrame(buffer) {
	const view = new DataView(buffer);
	const headerLength = view.getUint32(0, true);
	const header = JSON.parse(decoder.decode(new Uint8Array(buffer, 4, headerLength)));
	if (header.generation !== generation) return; // from a previous build
	let offset = 4 + headerLength;
	offset += (4 - (offset % 4)) % 4;
	for (const [id, count] of header.models) {
		const packed = new Float32Array(buffer, offset, count * 3);
		offset += count * 12;
		const entry = models.get(id);
		if (entry) applyPositions(entry, packed);
	}
	if (header.lights) updateLights(header.lights);
	$("time").textContent = `t = ${header.t.toFixed(2)} s`;
	const rtfEl = $("rtf");
	if (header.rtf > 0 && !header.paused && !header.finished) {
		rtfEl.textContent = `${header.rtf.toFixed(2)}× real time`;
		rtfEl.className = header.rtf < 0.95 ? "slow" : "";
	} else {
		rtfEl.textContent = "";
	}
	updateTelemetry(header.t, header.scalars || {}, header.full);
}

function connect() {
	const wsUrl = new URL("ws/sim", location.href);
	wsUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
	wsUrl.search = "";
	const ws = new WebSocket(wsUrl);
	ws.binaryType = "arraybuffer";
	ws.addEventListener("open", () => {
		connected = true;
		renderStatus();
	});
	ws.addEventListener("close", () => {
		connected = false;
		renderStatus();
		setTimeout(connect, 1000);
	});
	ws.addEventListener("error", () => ws.close());
	ws.addEventListener("message", (event) => {
		if (typeof event.data === "string") {
			const message = JSON.parse(event.data);
			if (message.type === "setup") applySetup(message);
			else if (message.type === "history") applyHistory(message);
			else if (message.type === "status") {
				const rebuilt = lastStatus && lastStatus.status === "building" && message.status === "running";
				lastStatus = message;
				renderStatus();
				if (rebuilt) loadParams();
			}
		} else {
			handleFrame(event.data);
		}
	});
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const fieldInputs = new Map(); // key -> { field, input }
let currentParams = {};

function fmtNumber(value, field) {
	const shown = value / (field.scale || 1);
	return field.integer ? String(Math.round(shown)) : String(+shown.toFixed(Math.max(field.decimals ?? 3, 0) + 2));
}

function buildParamsPanel(spec) {
	const form = $("params-form");
	form.replaceChildren();
	for (const section of spec) {
		const wrap = document.createElement("div");
		wrap.className = "param-section";
		if (spec.length > 1) {
			const h = document.createElement("h3");
			h.textContent = section.title;
			wrap.appendChild(h);
		}
		for (const field of section.fields) {
			const row = document.createElement("div");
			row.className = `param-row${field.type === "bool" ? " bool" : ""}`;
			const label = document.createElement("label");
			label.htmlFor = `p-${field.key}`;
			label.textContent = field.label;
			if (field.unit) {
				const unit = document.createElement("span");
				unit.className = "unit";
				unit.textContent = ` ${field.unit}`;
				label.appendChild(unit);
			}
			if (field.live) {
				const badge = document.createElement("span");
				badge.className = "live-badge";
				badge.textContent = "live";
				label.appendChild(badge);
			}
			const input = document.createElement("input");
			input.id = `p-${field.key}`;
			if (field.type === "bool") input.type = "checkbox";
			else if (field.type === "vector") input.type = "text";
			else {
				input.type = "number";
				input.step = field.integer ? "1" : String(10 ** -Math.max(field.decimals ?? 3, 0));
				if (field.min !== undefined) input.min = field.min / (field.scale || 1);
				if (field.max !== undefined) input.max = field.max / (field.scale || 1);
			}
			if (isViewer) input.disabled = true;
			input.addEventListener(field.type === "bool" ? "change" : "input", () => {
				if (field.live) return;
				input.classList.add("dirty");
				$("apply-params-btn").disabled = false;
			});
			if (field.live) input.addEventListener("change", () => submitParams([field.key]));
			row.append(label, input);
			wrap.appendChild(row);
			fieldInputs.set(field.key, { field, input });
		}
		form.appendChild(wrap);
	}
}

function showParams(params) {
	currentParams = params;
	for (const [key, { field, input }] of fieldInputs) {
		if (!(key in params)) continue;
		if (document.activeElement === input && !isViewer) continue;
		const value = params[key];
		if (field.type === "bool") input.checked = !!value;
		else if (field.type === "vector") input.value = value.map((v) => +v.toFixed(4)).join(", ");
		else input.value = fmtNumber(value, field);
		input.classList.remove("dirty");
	}
	$("apply-params-btn").disabled = true;
}

function readField({ field, input }) {
	if (field.type === "bool") return input.checked;
	if (field.type === "vector") return input.value.split(/[,\s]+/).filter(Boolean).map(Number);
	const shown = parseFloat(input.value);
	return Number.isFinite(shown) ? shown * (field.scale || 1) : undefined;
}

async function submitParams(keys) {
	if (isViewer) return;
	const changes = {};
	for (const key of keys) {
		const value = readField(fieldInputs.get(key));
		if (value !== undefined) changes[key] = value;
	}
	try {
		const response = await fetch("api/params", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ changes }),
		});
		showParams(await response.json());
	} catch (err) {
		console.error("Failed to apply parameters", err);
	}
}

$("params-form").addEventListener("submit", (event) => {
	event.preventDefault();
	const dirty = [...fieldInputs].filter(([, { input }]) => input.classList.contains("dirty")).map(([key]) => key);
	document.activeElement?.blur();
	if (dirty.length) submitParams(dirty);
});

$("default-params-btn").addEventListener("click", async () => {
	try {
		showParams(await (await fetch("api/params/default", { method: "POST" })).json());
	} catch (err) {
		console.error("Failed to restore defaults", err);
	}
});

async function loadParams() {
	try {
		showParams(await (await fetch("api/params")).json());
	} catch (err) {
		console.error("Failed to load parameters", err);
	}
}

// ---------------------------------------------------------------------------
// Buttons, console, info
// ---------------------------------------------------------------------------

$("reset-btn").addEventListener("click", () => fetch("api/reset", { method: "POST" }).catch(() => {}));
$("pause-btn").addEventListener("click", () => {
	const resume = lastStatus && (lastStatus.paused || lastStatus.finished);
	fetch(resume ? "api/resume" : "api/pause", { method: "POST" }).catch(() => {});
});
$("auto-restart-checkbox").addEventListener("change", (event) => {
	fetch(`api/auto-restart?enabled=${event.target.checked}`, { method: "POST" }).catch(() => {});
});

$("download-btn").addEventListener("click", async () => {
	const btn = $("download-btn");
	const label = btn.textContent;
	btn.disabled = true;
	btn.textContent = "Downloading...";
	try {
		const response = await fetch("api/data.csv");
		const name = (response.headers.get("Content-Disposition") || "").match(/filename=([^;]+)/)?.[1] || "sofa-data.csv";
		const url = URL.createObjectURL(await response.blob());
		const a = document.createElement("a");
		a.href = url;
		a.download = name.replace(".csv", `-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`);
		document.body.appendChild(a);
		a.click();
		a.remove();
		URL.revokeObjectURL(url);
	} finally {
		btn.disabled = false;
		btn.textContent = label;
	}
});

let consoleAfter = 0;
let consoleTimer = null;
const consoleLog = $("console-log");
async function pollConsole() {
	try {
		const lines = await (await fetch(`api/console?after=${consoleAfter}`)).json();
		if (lines.length) {
			const atBottom = consoleLog.scrollTop + consoleLog.clientHeight >= consoleLog.scrollHeight - 4;
			consoleAfter = lines[lines.length - 1].seq;
			consoleLog.textContent += lines.map((l) => l.text).join("\n") + "\n";
			const all = consoleLog.textContent.split("\n");
			if (all.length > 400) consoleLog.textContent = all.slice(-400).join("\n");
			if (atBottom) consoleLog.scrollTop = consoleLog.scrollHeight;
		}
	} catch {}
}
function toggleConsole(show) {
	$("console-panel").classList.toggle("hidden", !show);
	clearInterval(consoleTimer);
	if (show) {
		pollConsole();
		consoleTimer = setInterval(pollConsole, 1000);
	}
}
$("console-btn").addEventListener("click", () => toggleConsole($("console-panel").classList.contains("hidden")));
$("console-close").addEventListener("click", () => toggleConsole(false));

async function openInfo() {
	$("info-overlay").classList.remove("hidden");
	const list = $("info-sim-list");
	list.replaceChildren();
	const add = (label, value) => {
		const dt = document.createElement("dt");
		dt.textContent = label;
		const dd = document.createElement("dd");
		dd.textContent = value;
		list.append(dt, dd);
	};
	try {
		const info = await (await fetch("api/info")).json();
		if (info.dt) add("Timestep", `${(info.dt * 1000).toFixed(2)} ms`);
		add("Simulated time", `${info.simTime.toFixed(3)} s`);
		if (info.stopAt != null) add("Run ends at", `${info.stopAt.toFixed(2)} s simulated time`);
		add("Speed", info.rtf > 0 ? `${info.rtf.toFixed(2)}× real time` : "not running");
		add("Visual models", info.models.map((m) => `${m.name} (${m.vertices} vertices)`).join(", "));
		add("Build", String(info.generation));
	} catch {
		add("Error", "Could not load simulation info.");
	}
}
$("info-btn").addEventListener("click", openInfo);
$("info-close-btn").addEventListener("click", () => $("info-overlay").classList.add("hidden"));
$("info-overlay").addEventListener("click", (event) => {
	if (event.target === $("info-overlay")) $("info-overlay").classList.add("hidden");
});
window.addEventListener("keydown", (event) => {
	if (event.key === "Escape") $("info-overlay").classList.add("hidden");
});

function makeTopRightResizable(panel, handle) {
	let startX = 0, startY = 0, startWidth = 0, startHeight = 0;
	handle.addEventListener("pointerdown", (event) => {
		startX = event.clientX;
		startY = event.clientY;
		const rect = panel.getBoundingClientRect();
		startWidth = rect.width;
		startHeight = rect.height;
		handle.setPointerCapture(event.pointerId);
		event.preventDefault();
	});
	handle.addEventListener("pointermove", (event) => {
		if (!handle.hasPointerCapture(event.pointerId)) return;
		panel.style.width = `${startWidth + event.clientX - startX}px`;
		panel.style.height = `${startHeight + startY - event.clientY}px`;
	});
	const stop = (event) => {
		if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
	};
	handle.addEventListener("pointerup", stop);
	handle.addEventListener("pointercancel", stop);
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function init() {
	try {
		sceneInfo = await (await fetch("api/scene")).json();
	} catch {
		sceneInfo = { title: "SOFA Simulation", about: [], charts: [], readouts: [], spec: [] };
	}
	document.title = sceneInfo.title;
	$("title").textContent = sceneInfo.title;
	$("info-title").textContent = sceneInfo.title;
	const about = $("info-about");
	for (const text of sceneInfo.about) {
		const p = document.createElement("p");
		p.textContent = text;
		about.appendChild(p);
	}
	if (sceneInfo.origin) {
		const p = document.createElement("p");
		p.className = "origin";
		p.textContent = sceneInfo.origin;
		about.appendChild(p);
	}
	buildParamsPanel(sceneInfo.spec);
	if (!sceneInfo.spec.length) $("params-panel").classList.add("hidden");
	buildCharts(sceneInfo);
	await loadParams();
	if (isViewer) setInterval(loadParams, 3000);
	connect();
}
init();

// For debugging from the browser console.
window.sofaweb = { scene, models, renderer, get camera() { return camera; }, get setup() { return setup; } };

function animate() {
	requestAnimationFrame(animate);
	if (chartsDirty) {
		for (const { chart } of charts) chart.update("none");
		chartsDirty = false;
	}
	if (controls) controls.update();
	renderer.render(scene, camera);
}
animate();
