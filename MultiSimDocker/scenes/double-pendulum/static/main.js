import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import Chart from "chart.js/auto";

// MultiSimDocker loads this page with ?role=viewer for watchers; the proxy
// rejects their changes regardless, this just makes the UI read-only too.
const isViewer = new URLSearchParams(location.search).get("role") === "viewer";
if (isViewer) document.body.classList.add("viewer");

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const pauseBtn = $("pause-btn");

// ---------------------------------------------------------------------------
// Scene: the original's flat cartoon look -- everything lies in the X-Z
// plane (Z up), seen by an orthographic camera looking along +Y. Colors and
// dimensions are the original DoublePendulum.py constants.
// ---------------------------------------------------------------------------

const SKY_COLOR = [0.78, 0.87, 0.96];
const FLOOR = { length: 4.0, z: 0.0, halfThickness: 0.02, color: [0.1, 0.55, 0.15] };
const SUN = { cx: 1.3, cz: 3.0, radius: 0.25, color: [1.0, 0.85, 0.0], rayCount: 20, rayColor: [1.0, 0.75, 0.0] };
const CLOUDS = [
	[-1.4, 2.5, 0.6],
	[1.1, 2.9, 0.5],
];
const MASS1 = { radius: 0.06, color: [0.55, 0.15, 0.75] };
const MASS2 = { radius: 0.05, color: [0.85, 0.4, 0.15] };
const ROD = { halfThickness: 0.012, color: [0.35, 0.35, 0.35] };
const PIVOT = { radius: 0.035, color: [0.3, 0.3, 0.3] };

// Centered a little below the middle of the scenery, so the pendulum
// (z 0..1.9 by default) clears the chart panels along the bottom edge.
const VIEW_CENTER = new THREE.Vector3(0, 0, 0.9);
const VIEW_HALF_HEIGHT = 2.5;

// Depth layers along Y (camera sits at -Y, so smaller Y is in front).
const LAYER = { sky: 0.4, floor: 0.2, trail: 0.1, rod: 0.0, pivot: -0.02, mass: -0.04 };

const rgb = ([r, g, b]) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
const flat = (color, extra = {}) => new THREE.MeshBasicMaterial({ color: rgb(color), side: THREE.DoubleSide, ...extra });

const scene = new THREE.Scene();
scene.background = rgb(SKY_COLOR);
document.body.style.background = `#${rgb(SKY_COLOR).getHexString()}`;

const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 50);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
$("viewer").appendChild(renderer.domElement);

// A 2D scene: navigation is pan (drag) and zoom (wheel/pinch). Rotating
// would only show the flat shapes edge-on.
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableRotate = false;
controls.screenSpacePanning = true;
controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
controls.minZoom = 0.3;
controls.maxZoom = 12;

function fitFrustum() {
	const aspect = window.innerWidth / window.innerHeight;
	camera.left = -VIEW_HALF_HEIGHT * aspect;
	camera.right = VIEW_HALF_HEIGHT * aspect;
	camera.top = VIEW_HALF_HEIGHT;
	camera.bottom = -VIEW_HALF_HEIGHT;
	camera.updateProjectionMatrix();
}

function resetView() {
	camera.position.set(VIEW_CENTER.x, -10, VIEW_CENTER.z);
	camera.zoom = 1;
	controls.target.copy(VIEW_CENTER);
	fitFrustum();
	controls.update();
}
resetView();
$("view-btn").addEventListener("click", resetView);

// Shapes are built in the X-Y plane (three.js' default for 2D geometry) and
// rotated into X-Z, so "y" in these helpers is world Z.
function inXZ(geometry, layerY) {
	geometry.rotateX(Math.PI / 2);
	geometry.translate(0, layerY, 0);
	return geometry;
}

function disc(cx, cz, radius, color, layerY) {
	const mesh = new THREE.Mesh(inXZ(new THREE.CircleGeometry(radius, 40), 0), flat(color));
	mesh.position.set(cx, layerY, cz);
	scene.add(mesh);
	return mesh;
}

function bar(x0, z0, x1, z1, halfThickness, color, layerY) {
	const mesh = new THREE.Mesh(inXZ(new THREE.PlaneGeometry(1, 2 * halfThickness), 0), flat(color));
	placeBar(mesh, x0, z0, x1, z1, layerY);
	scene.add(mesh);
	return mesh;
}

function placeBar(mesh, x0, z0, x1, z1, layerY = mesh.position.y) {
	mesh.position.set((x0 + x1) / 2, layerY, (z0 + z1) / 2);
	// Rotation about +Y by phi maps +X to (cos phi, 0, -sin phi).
	mesh.rotation.set(0, -Math.atan2(z1 - z0, x1 - x0), 0);
	mesh.scale.set(Math.hypot(x1 - x0, z1 - z0), 1, 1);
}

// Floor (a thin green bar, as in the original).
bar(-FLOOR.length / 2, FLOOR.z, FLOOR.length / 2, FLOOR.z, FLOOR.halfThickness, FLOOR.color, LAYER.floor);

// Sun with rays.
disc(SUN.cx, SUN.cz, SUN.radius, SUN.color, LAYER.sky);
for (let i = 0; i < SUN.rayCount; i++) {
	const a = (2 * Math.PI * i) / SUN.rayCount;
	const [ux, uz] = [Math.cos(a), Math.sin(a)];
	const inner = SUN.radius * 1.1;
	const outer = SUN.radius * 1.9;
	bar(SUN.cx + inner * ux, SUN.cz + inner * uz, SUN.cx + outer * ux, SUN.cz + outer * uz, 0.015, SUN.rayColor, LAYER.sky);
}

// Clouds: four overlapping puffs each.
for (const [cx, cz, s] of CLOUDS) {
	for (const [dx, dz, r] of [
		[0, 0, 0.35],
		[-0.34, 0.02, 0.25],
		[0.36, 0, 0.27],
		[0.05, 0.22, 0.27],
	]) {
		disc(cx + dx * s, cz + dz * s, r * s, [1, 1, 1], LAYER.sky);
	}
}

// Pendulum parts, positioned by incoming frames.
const pivot = disc(0, 1, PIVOT.radius, PIVOT.color, LAYER.pivot);
const rod1 = bar(0, 1, 0, 0.5, ROD.halfThickness, ROD.color, LAYER.rod);
const rod2 = bar(0, 0.5, 0, 0.1, ROD.halfThickness, ROD.color, LAYER.rod);
const mass1 = disc(0, 0.5, MASS1.radius, MASS1.color, LAYER.mass);
const mass2 = disc(0, 0.1, MASS2.radius, MASS2.color, LAYER.mass);

// Trail of the outer mass -- the chaotic part is easiest to see here.
const TRAIL_POINTS = 400; // 8 s at 50 frames/s
const trailPositions = new Float32Array(TRAIL_POINTS * 3);
const trailColors = new Float32Array(TRAIL_POINTS * 3);
const trailGeometry = new THREE.BufferGeometry();
trailGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
trailGeometry.setAttribute("color", new THREE.BufferAttribute(trailColors, 3));
const trail = new THREE.Line(trailGeometry, new THREE.LineBasicMaterial({ vertexColors: true }));
trail.frustumCulled = false;
scene.add(trail);
let trailPoints = [];

const trailHead = rgb(MASS2.color);
const trailTail = rgb(SKY_COLOR);
function updateTrail() {
	const n = trailPoints.length;
	const tmp = new THREE.Color();
	for (let i = 0; i < n; i++) {
		const [x, z] = trailPoints[i];
		trailPositions.set([x, LAYER.trail, z], i * 3);
		// Fade from sky color (oldest) to the mass color (newest).
		tmp.copy(trailTail).lerp(trailHead, n > 1 ? i / (n - 1) : 1);
		trailColors.set([tmp.r, tmp.g, tmp.b], i * 3);
	}
	trailGeometry.setDrawRange(0, n);
	trailGeometry.attributes.position.needsUpdate = true;
	trailGeometry.attributes.color.needsUpdate = true;
}

const trailCheckbox = $("trail-checkbox");
trailCheckbox.addEventListener("change", () => (trail.visible = trailCheckbox.checked));

function applyFrameToScene(frame) {
	const [ax, az] = frame.anchor;
	const [x1, z1] = frame.mass1;
	const [x2, z2] = frame.mass2;
	pivot.position.set(ax, LAYER.pivot, az);
	placeBar(rod1, ax, az, x1, z1);
	placeBar(rod2, x1, z1, x2, z2);
	mass1.position.set(x1, LAYER.mass, z1);
	mass2.position.set(x2, LAYER.mass, z2);

	if (frame.reset) trailPoints = [];
	trailPoints.push([x2, z2]);
	if (trailPoints.length > TRAIL_POINTS) trailPoints.shift();
	updateTrail();
}

// ---------------------------------------------------------------------------
// Charts (replace the original's separate matplotlib live-plot window; same
// 10 s scrolling window and series colors as its live_plot.py).
// ---------------------------------------------------------------------------

const CHART_WINDOW_SECONDS = 10;
const chartFont = { size: 10 };
const axisColor = "#8f97a3";
const gridColor = "#2f323b";

function lineDataset(label, color, extra = {}) {
	return { label, data: [], borderColor: color, backgroundColor: "transparent", borderWidth: 1.5, pointRadius: 0, tension: 0, ...extra };
}

function chartOptions(yOptions) {
	return {
		animation: false,
		responsive: true,
		maintainAspectRatio: false,
		parsing: false,
		spanGaps: false,
		plugins: { legend: { display: true, labels: { color: "#cfd3da", boxWidth: 10, font: chartFont } } },
		scales: {
			x: {
				type: "linear",
				title: { display: true, text: "time (s)", color: axisColor, font: chartFont },
				ticks: { color: axisColor, font: chartFont },
				grid: { color: gridColor },
			},
			y: { ticks: { color: axisColor, font: chartFont }, grid: { color: gridColor }, ...yOptions },
		},
	};
}

const angleChart = new Chart($("angle-chart"), {
	type: "line",
	data: { datasets: [lineDataset("theta1", "#2a78d6"), lineDataset("theta2", "#eb6834")] },
	options: chartOptions({ min: -180, max: 180, ticks: { color: axisColor, font: chartFont, stepSize: 90 } }),
});

const energyChart = new Chart($("energy-chart"), {
	type: "line",
	data: {
		datasets: [
			lineDataset("potential", "#4a9be8"),
			lineDataset("kinetic", "#e8933c"),
			lineDataset("total", "#6bcf7a", { borderWidth: 2 }),
		],
	},
	options: chartOptions({}),
});

function pushPoint(dataset, t, value) {
	dataset.data.push({ x: t, y: value });
	while (dataset.data.length && dataset.data[0].x < t - CHART_WINDOW_SECONDS) dataset.data.shift();
}

// Angles are wrapped to (-180, 180]; a jump across the wrap would otherwise
// draw a vertical line through the whole chart, so insert a gap there.
const lastAngle = [null, null];
function pushAngle(index, t, value) {
	const dataset = angleChart.data.datasets[index];
	if (lastAngle[index] !== null && Math.abs(value - lastAngle[index]) > 180) {
		dataset.data.push({ x: t, y: null });
	}
	lastAngle[index] = value;
	pushPoint(dataset, t, value);
}

function clearCharts() {
	for (const chart of [angleChart, energyChart]) for (const dataset of chart.data.datasets) dataset.data = [];
	lastAngle[0] = lastAngle[1] = null;
}

const readouts = {
	pe1: $("readout-pe1"),
	ke1: $("readout-ke1"),
	pe2: $("readout-pe2"),
	ke2: $("readout-ke2"),
	e: $("readout-e"),
	drift: $("readout-drift"),
};

function updateCharts(frame) {
	if (frame.reset) clearCharts();
	pushAngle(0, frame.t, frame.theta1);
	pushAngle(1, frame.t, frame.theta2);
	const [pot, kin] = energyChart.data.datasets;
	pushPoint(pot, frame.t, frame.PE1 + frame.PE2);
	pushPoint(kin, frame.t, frame.KE1 + frame.KE2);
	pushPoint(energyChart.data.datasets[2], frame.t, frame.E);

	readouts.pe1.textContent = frame.PE1.toFixed(3);
	readouts.ke1.textContent = frame.KE1.toFixed(3);
	readouts.pe2.textContent = frame.PE2.toFixed(3);
	readouts.ke2.textContent = frame.KE2.toFixed(3);
	readouts.e.textContent = frame.E.toFixed(4);
	readouts.drift.textContent = `${(frame.energyDrift * 100).toFixed(4)}%`;
}

// Charts redraw at most once per animation frame, not once per message.
let chartsDirty = false;

// ---------------------------------------------------------------------------
// Connection and controls
// ---------------------------------------------------------------------------

let paused = false;
let connected = false;

function renderStatus() {
	if (!connected) return;
	statusEl.textContent = paused ? "paused" : "connected";
	statusEl.className = paused ? "paused" : "connected";
}

function syncPaused(serverPaused) {
	if (serverPaused === undefined || serverPaused === paused) return;
	paused = serverPaused;
	pauseBtn.textContent = paused ? "Resume" : "Pause";
	// While paused this only resets to the start (and stays paused).
	$("reset-btn").textContent = paused ? "Reset" : "Restart";
	renderStatus();
}

function connect() {
	// Relative, so this works both standalone and under /sim/<id>/.
	const wsUrl = new URL("ws/sim", location.href);
	wsUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
	wsUrl.search = "";
	const ws = new WebSocket(wsUrl);

	ws.addEventListener("open", () => {
		connected = true;
		renderStatus();
	});
	ws.addEventListener("close", () => {
		connected = false;
		statusEl.textContent = "disconnected -- retrying...";
		statusEl.className = "disconnected";
		setTimeout(connect, 1000);
	});
	ws.addEventListener("error", () => ws.close());
	ws.addEventListener("message", (event) => {
		const frame = JSON.parse(event.data);
		syncPaused(frame.paused);
		applyFrameToScene(frame);
		updateCharts(frame);
		chartsDirty = true;
		// A restart can come from the owner in another tab, or from applying
		// parameters; refresh the panel so everyone sees what's running.
		if (frame.reset) loadParams();
	});
}
connect();

$("reset-btn").addEventListener("click", () => {
	fetch("api/reset", { method: "POST" }).catch(() => {});
});

pauseBtn.addEventListener("click", () => {
	const next = !paused;
	fetch(next ? "api/pause" : "api/resume", { method: "POST" }).catch(() => {});
	syncPaused(next);
});

$("download-btn").addEventListener("click", async () => {
	const btn = $("download-btn");
	const label = btn.textContent;
	btn.disabled = true;
	btn.textContent = "Downloading...";
	try {
		const blob = await (await fetch("api/data.csv")).blob();
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = `sofa-double-pendulum-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
		document.body.appendChild(a);
		a.click();
		a.remove();
		URL.revokeObjectURL(url);
	} catch (err) {
		console.error("Failed to download simulation data", err);
	} finally {
		btn.disabled = false;
		btn.textContent = label;
	}
});

// -- parameters --------------------------------------------------------------

const PARAM_IDS = ["l1", "m1_kg", "theta1_initial_deg", "omega1_initial", "l2", "m2_kg", "theta2_initial_deg", "omega2_initial", "anchor_z", "gravity"];
const paramInputs = Object.fromEntries(PARAM_IDS.map((id) => [id, $(id)]));
if (isViewer) for (const input of Object.values(paramInputs)) input.readOnly = true;

function showParams(params) {
	for (const id of PARAM_IDS) {
		const input = paramInputs[id];
		// Don't overwrite what the owner is in the middle of typing.
		if (params[id] === undefined || (document.activeElement === input && !isViewer)) continue;
		input.value = +params[id].toFixed(4);
		input.classList.remove("dirty");
	}
}

async function loadParams() {
	try {
		showParams(await (await fetch("api/params")).json());
	} catch (err) {
		console.error("Failed to load parameters", err);
	}
}

async function loadLimits() {
	try {
		const info = await (await fetch("api/info")).json();
		for (const [id, [min, max]] of Object.entries(info.paramLimits || {})) {
			if (paramInputs[id]) Object.assign(paramInputs[id], { min, max });
		}
	} catch {}
}
loadLimits();
loadParams();
if (isViewer) setInterval(loadParams, 3000);

for (const input of Object.values(paramInputs)) {
	input.addEventListener("input", () => input.classList.add("dirty"));
}

$("params-form").addEventListener("submit", async (event) => {
	event.preventDefault();
	if (isViewer) return;
	const body = {};
	for (const id of PARAM_IDS) {
		const value = parseFloat(paramInputs[id].value);
		if (Number.isFinite(value)) body[id] = value;
	}
	try {
		const response = await fetch("api/params", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
		});
		document.activeElement?.blur();
		showParams(await response.json()); // shows any clamping the server applied
	} catch (err) {
		console.error("Failed to apply parameters", err);
	}
});

$("default-params-btn").addEventListener("click", async () => {
	try {
		showParams(await (await fetch("api/params/default", { method: "POST" })).json());
	} catch (err) {
		console.error("Failed to restore default parameters", err);
	}
});

// -- info modal --------------------------------------------------------------

function addInfoRow(label, value) {
	const list = $("info-sim-list");
	const dt = document.createElement("dt");
	dt.textContent = label;
	const dd = document.createElement("dd");
	dd.textContent = value;
	list.append(dt, dd);
}

async function openInfo() {
	$("info-overlay").classList.remove("hidden");
	$("info-sim-list").replaceChildren();
	try {
		const info = await (await fetch("api/info")).json();
		const p = info.params;
		addInfoRow("Integrator", info.integrator);
		addInfoRow("Physics timestep", `${(info.physicsDt * 1000).toFixed(1)} ms (${Math.round(1 / info.physicsDt)} Hz)`);
		addInfoRow("Broadcast interval", `${(info.broadcastDt * 1000).toFixed(0)} ms -- ${info.stepsPerBroadcast} physics steps per frame`);
		addInfoRow("Rod 1", `${p.l1} m, mass ${p.m1_kg} kg, starts at ${p.theta1_initial_deg} deg / ${p.omega1_initial} rad/s`);
		addInfoRow("Rod 2", `${p.l2} m, mass ${p.m2_kg} kg, starts at ${p.theta2_initial_deg} deg / ${p.omega2_initial} rad/s`);
		addInfoRow("Anchor height", `${p.anchor_z} m`);
		addInfoRow("Gravity", `${p.gravity} m/s²`);
		addInfoRow("CSV download", `every physics step of the last ${info.logSeconds} s`);
	} catch (err) {
		addInfoRow("Error", "Could not load simulation info.");
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

// -- resizable chart panels (grow upward from a top-right handle) ------------

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
for (const panel of document.querySelectorAll(".chart-panel")) {
	makeTopRightResizable(panel, panel.querySelector(".resize-handle-tr"));
}

// -- render loop ---------------------------------------------------------------

window.addEventListener("resize", () => {
	renderer.setSize(window.innerWidth, window.innerHeight);
	fitFrustum();
});

function animate() {
	requestAnimationFrame(animate);
	if (chartsDirty) {
		angleChart.update("none");
		energyChart.update("none");
		chartsDirty = false;
	}
	controls.update();
	renderer.render(scene, camera);
}
animate();
