import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import Chart from "chart.js/auto";

const viewerEl = document.getElementById("viewer");
const statusEl = document.getElementById("status");
const resetBtn = document.getElementById("reset-btn");
const pauseBtn = document.getElementById("pause-btn");
const downloadBtn = document.getElementById("download-btn");
const autoResetCheckbox = document.getElementById("auto-reset-checkbox");
const infoBtn = document.getElementById("info-btn");
const infoOverlay = document.getElementById("info-overlay");
const infoCloseBtn = document.getElementById("info-close-btn");
const infoSimList = document.getElementById("info-sim-list");

// MultiSimDocker loads this page with ?role=viewer for people watching
// someone else's simulation. The orchestrator's proxy already rejects their
// control requests; this just hides/disables the controls so the UI doesn't
// pretend otherwise.
const isViewer = new URLSearchParams(location.search).get("role") === "viewer";
if (isViewer) document.body.classList.add("viewer");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1d22);
scene.fog = new THREE.Fog(0x1b1d22, 15, 40);

const camera = new THREE.PerspectiveCamera(
	50,
	window.innerWidth / window.innerHeight,
	0.1,
	100
);
camera.position.set(6, 5, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
viewerEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);
controls.enableDamping = true;

const hemiLight = new THREE.HemisphereLight(0xbfd4ff, 0x30302a, 0.9);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
dirLight.position.set(6, 10, 4);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(1024, 1024);
dirLight.shadow.camera.left = -8;
dirLight.shadow.camera.right = 8;
dirLight.shadow.camera.top = 8;
dirLight.shadow.camera.bottom = -8;
scene.add(dirLight);

// Floor. This matches the SOFA scene's PlaneForceField (normal 0 1 0, d=0)
// but is drawn directly -- it isn't simulated data from the backend.
const floorGeometry = new THREE.PlaneGeometry(30, 30);
const floorMaterial = new THREE.MeshStandardMaterial({ color: 0x3a3d46, roughness: 0.9 });
const floor = new THREE.Mesh(floorGeometry, floorMaterial);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(30, 30, 0x555a66, 0x2a2d34);
scene.add(grid);

const cubeSize = 1; // matches CUBE_HALF_SIZE = 0.5 in backend/simulation.py
const cubeGeometry = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
const cubeMaterial = new THREE.MeshStandardMaterial({ color: 0xe8933c, roughness: 0.5, metalness: 0.05 });
const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
cube.castShadow = true;
cube.position.set(0, 5, 0);
scene.add(cube);

const chartFont = { size: 10 };
const chartAxisOptions = {
	x: {
		type: "linear",
		title: { display: true, text: "time (s)", color: "#8f97a3", font: chartFont },
		ticks: { color: "#8f97a3", font: chartFont },
		grid: { color: "#2f323b" },
	},
	y: {
		beginAtZero: true,
		ticks: { color: "#8f97a3", font: chartFont },
		grid: { color: "#2f323b" },
	},
};

const heightChart = new Chart(document.getElementById("height-chart"), {
	type: "line",
	data: {
		datasets: [
			{
				label: "height",
				data: [],
				borderColor: "#e8933c",
				backgroundColor: "transparent",
				borderWidth: 1.5,
				pointRadius: 0,
				tension: 0,
			},
		],
	},
	options: {
		animation: false,
		responsive: true,
		maintainAspectRatio: false,
		parsing: false,
		plugins: { legend: { display: false } },
		scales: chartAxisOptions,
	},
});

function energyDataset(label, color) {
	return {
		label,
		data: [],
		borderColor: color,
		backgroundColor: "transparent",
		borderWidth: 1.5,
		pointRadius: 0,
		tension: 0,
	};
}

const energyChart = new Chart(document.getElementById("energy-chart"), {
	type: "line",
	data: {
		datasets: [
			energyDataset("kinetic", "#e8933c"),
			energyDataset("potential", "#4a9be8"),
			energyDataset("elastic", "#c77dff"),
			energyDataset("dissipated", "#8f97a3"),
			energyDataset("Cube (KE+PE+elastic)", "#6bcf7a"),
			energyDataset("sum (cube+dissipated)", "#e85f8b"),
		],
	},
	options: {
		animation: false,
		responsive: true,
		maintainAspectRatio: false,
		parsing: false,
		plugins: {
			legend: {
				display: true,
				labels: { color: "#cfd3da", boxWidth: 10, font: chartFont },
			},
		},
		scales: chartAxisOptions,
	},
});

const MAX_CHART_POINTS = 600; // ~12s of data at 50Hz -- comfortably covers one drop-to-settle cycle

function pushPoint(dataset, t, value) {
	dataset.data.push({ x: t, y: value });
	if (dataset.data.length > MAX_CHART_POINTS) {
		dataset.data.shift();
	}
}

function clearCharts() {
	heightChart.data.datasets[0].data = [];
	for (const dataset of energyChart.data.datasets) {
		dataset.data = [];
	}
}

const energyReadouts = {
	kinetic: document.getElementById("readout-kinetic"),
	potential: document.getElementById("readout-potential"),
	elastic: document.getElementById("readout-elastic"),
	dissipated: document.getElementById("readout-dissipated"),
	cube: document.getElementById("readout-cube"),
	sum: document.getElementById("readout-sum"),
};

function updateEnergyReadouts(kinetic, potential, elastic, dissipated, cubeEnergy, sumEnergy) {
	energyReadouts.kinetic.textContent = kinetic.toFixed(2);
	energyReadouts.potential.textContent = potential.toFixed(2);
	energyReadouts.elastic.textContent = elastic.toFixed(2);
	energyReadouts.dissipated.textContent = dissipated.toFixed(2);
	energyReadouts.cube.textContent = cubeEnergy.toFixed(2);
	energyReadouts.sum.textContent = sumEnergy.toFixed(2);
}

function updateCharts(frame) {
	if (frame.reset) {
		clearCharts();
	}
	pushPoint(heightChart.data.datasets[0], frame.t, frame.height);
	pushPoint(energyChart.data.datasets[0], frame.t, frame.kineticEnergy);
	pushPoint(energyChart.data.datasets[1], frame.t, frame.potentialEnergy);
	pushPoint(energyChart.data.datasets[2], frame.t, frame.elasticEnergy);
	pushPoint(energyChart.data.datasets[3], frame.t, frame.dissipatedEnergy);
	const cubeEnergy = frame.kineticEnergy + frame.potentialEnergy + frame.elasticEnergy;
	const sumEnergy = cubeEnergy + frame.dissipatedEnergy;
	pushPoint(energyChart.data.datasets[4], frame.t, cubeEnergy);
	pushPoint(energyChart.data.datasets[5], frame.t, sumEnergy);
	updateEnergyReadouts(
		frame.kineticEnergy,
		frame.potentialEnergy,
		frame.elasticEnergy,
		frame.dissipatedEnergy,
		cubeEnergy,
		sumEnergy
	);
	heightChart.update("none");
	energyChart.update("none");
}

function setStatus(text, className) {
	statusEl.textContent = text;
	statusEl.className = className;
}

function connect() {
	// Relative to the page, so this works both standalone and behind the
	// MultiSimDocker proxy (served under /sim/<id>/).
	const wsUrl = new URL("ws/sim", location.href);
	wsUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
	wsUrl.search = "";
	const ws = new WebSocket(wsUrl);

	ws.addEventListener("open", () => setStatus("connected", "connected"));
	ws.addEventListener("close", () => {
		setStatus("disconnected -- retrying...", "disconnected");
		setTimeout(connect, 1000);
	});
	ws.addEventListener("error", () => ws.close());
	ws.addEventListener("message", (event) => {
		const frame = JSON.parse(event.data);
		syncPaused(frame.paused);
		const [x, y, z] = frame.position;
		const [qx, qy, qz, qw] = frame.quaternion;
		cube.position.set(x, y, z);
		cube.quaternion.set(qx, qy, qz, qw);
		updateCharts(frame);
	});
}
connect();

resetBtn.addEventListener("click", () => {
	fetch("api/reset", { method: "POST" }).catch(() => {});
});

let paused = false;
// The owner may pause from another tab, and watchers need to see it too, so
// the pause state follows what the simulation broadcasts.
function syncPaused(serverPaused) {
	if (serverPaused === undefined || serverPaused === paused) return;
	paused = serverPaused;
	updatePauseLabels();
	setStatus(paused ? "paused" : "connected", paused ? "" : "connected");
}
// While paused, the reset button only resets to the start (and stays paused).
function updatePauseLabels() {
	pauseBtn.textContent = paused ? "Resume" : "Pause";
	resetBtn.textContent = paused ? "Reset" : "Restart";
}
pauseBtn.addEventListener("click", () => {
	paused = !paused;
	updatePauseLabels();
	fetch(paused ? "api/pause" : "api/resume", { method: "POST" }).catch(() => {});
});

autoResetCheckbox.addEventListener("change", () => {
	const enabled = autoResetCheckbox.checked;
	fetch(`api/auto-reset?enabled=${enabled}`, { method: "POST" }).catch(() => {});
});

downloadBtn.addEventListener("click", async () => {
	// Full PHYSICS_DT-resolution data lives server-side (the websocket only
	// ever streams the throttled BROADCAST_DT frames driving the charts
	// above), so this fetches the raw log from the backend rather than
	// exporting what's been received in the browser.
	const originalLabel = downloadBtn.textContent;
	downloadBtn.disabled = true;
	downloadBtn.textContent = "Downloading...";
	try {
		const response = await fetch("api/data.csv");
		const blob = await response.blob();
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = `sofa-cube-sim-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
		document.body.appendChild(a);
		a.click();
		a.remove();
		URL.revokeObjectURL(url);
	} catch (err) {
		console.error("Failed to download simulation data", err);
	} finally {
		downloadBtn.disabled = false;
		downloadBtn.textContent = originalLabel;
	}
});

// Physics parameter sliders -- these edit the live SOFA scene's Data fields
// directly (see SimulationRunner.set_params in backend/simulation.py), so
// changes take effect on the very next physics step, mid-drop or at rest.
const PARAM_IDS = ["floorStiffness", "floorDamping", "rayleighMass", "rayleighStiffness", "lateralFrictionDamping"];
const paramInputs = Object.fromEntries(PARAM_IDS.map((id) => [id, document.getElementById(id)]));
const paramValueLabels = Object.fromEntries(PARAM_IDS.map((id) => [id, document.getElementById(`${id}-value`)]));

function formatParam(id, value) {
	if (id === "floorStiffness") return Math.round(value).toString();
	if (id === "rayleighStiffness") return value.toFixed(4);
	return value.toFixed(2);
}

function setParamDisplay(id, value) {
	paramInputs[id].value = value;
	paramValueLabels[id].textContent = formatParam(id, value);
}

async function loadParams() {
	try {
		const response = await fetch("api/params");
		const params = await response.json();
		for (const id of PARAM_IDS) {
			if (params[id] !== undefined) setParamDisplay(id, params[id]);
		}
	} catch (err) {
		console.error("Failed to load simulation parameters", err);
	}
}
loadParams();
if (isViewer) setInterval(loadParams, 2000);

for (const id of PARAM_IDS) {
	paramInputs[id].addEventListener("input", () => {
		const value = parseFloat(paramInputs[id].value);
		paramValueLabels[id].textContent = formatParam(id, value);
		fetch("api/params", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ [id]: value }),
		}).catch(() => {});
	});
}

document.getElementById("default-params-btn").addEventListener("click", async () => {
	try {
		const response = await fetch("api/params/default", { method: "POST" });
		const params = await response.json();
		for (const id of PARAM_IDS) {
			if (params[id] !== undefined) setParamDisplay(id, params[id]);
		}
	} catch (err) {
		console.error("Failed to reset simulation parameters", err);
	}
});

function addInfoRow(label, value) {
	const dt = document.createElement("dt");
	dt.textContent = label;
	const dd = document.createElement("dd");
	dd.textContent = value;
	infoSimList.appendChild(dt);
	infoSimList.appendChild(dd);
}

async function openInfo() {
	infoOverlay.classList.remove("hidden");
	infoSimList.innerHTML = "";
	try {
		const info = await (await fetch("api/info")).json();
		addInfoRow("Physics timestep", `${(info.physicsDt * 1000).toFixed(1)} ms (${Math.round(1 / info.physicsDt)} Hz)`);
		addInfoRow(
			"Broadcast interval",
			`${(info.broadcastDt * 1000).toFixed(0)} ms (${Math.round(1 / info.broadcastDt)} Hz) -- ${info.stepsPerBroadcast} physics steps per broadcast`
		);
		addInfoRow("Gravity", `${info.gravity.toFixed(2)} m/s²`);
		addInfoRow("Cube", `${info.cubeSide.toFixed(2)} m side, ${info.cubeMass.toFixed(2)} kg, inertia ${info.cubeInertia.toFixed(4)} kg·m²`);
		addInfoRow("Drop height", `${info.startHeight.toFixed(2)} m`);
		addInfoRow("Max initial spin", `±${info.maxAngularVelocity.toFixed(1)} rad/s per axis`);
		addInfoRow("Contact threshold", `${info.contactHeight.toFixed(2)} m`);
		addInfoRow(
			"Settle threshold",
			`linear < ${info.settleLinearVelocity} m/s and angular < ${info.settleAngularVelocity} rad/s, sustained for ${info.settleStepsRequired} ticks`
		);
		addInfoRow("Auto-reset pause", `${info.settlePauseSeconds.toFixed(1)} s after settling`);
		addInfoRow("Floor stiffness", info.params.floorStiffness.toFixed(0));
		addInfoRow("Floor damping", info.params.floorDamping.toFixed(2));
		addInfoRow("Rayleigh mass damping", info.params.rayleighMass.toFixed(2));
		addInfoRow("Rayleigh stiffness damping", info.params.rayleighStiffness.toFixed(4));
		addInfoRow("Lateral friction damping", info.params.lateralFrictionDamping.toFixed(2));
	} catch (err) {
		console.error("Failed to load simulation info", err);
		addInfoRow("Error", "Could not load live simulation info.");
	}
}

infoBtn.addEventListener("click", openInfo);
infoCloseBtn.addEventListener("click", () => infoOverlay.classList.add("hidden"));
infoOverlay.addEventListener("click", (event) => {
	if (event.target === infoOverlay) infoOverlay.classList.add("hidden");
});
window.addEventListener("keydown", (event) => {
	if (event.key === "Escape") infoOverlay.classList.add("hidden");
});

// The chart panels sit near the bottom of the viewport, so the browser's
// native bottom-right resize handle runs out of room to drag into almost
// immediately. This adds a second handle at the top-right corner that grows
// the panel upward instead (width still grows rightward, same as native).
function makeTopRightResizable(panel, handle) {
	let startX = 0;
	let startY = 0;
	let startWidth = 0;
	let startHeight = 0;

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
		const dx = event.clientX - startX;
		const dy = startY - event.clientY; // moving the mouse up grows the panel
		panel.style.width = `${startWidth + dx}px`;
		panel.style.height = `${startHeight + dy}px`;
	});

	const stopDragging = (event) => {
		if (handle.hasPointerCapture(event.pointerId)) {
			handle.releasePointerCapture(event.pointerId);
		}
	};
	handle.addEventListener("pointerup", stopDragging);
	handle.addEventListener("pointercancel", stopDragging);
}

for (const panel of document.querySelectorAll(".chart-panel")) {
	const handle = panel.querySelector(".resize-handle-tr");
	if (handle) makeTopRightResizable(panel, handle);
}

window.addEventListener("resize", () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
	requestAnimationFrame(animate);
	controls.update();
	renderer.render(scene, camera);
}
animate();
