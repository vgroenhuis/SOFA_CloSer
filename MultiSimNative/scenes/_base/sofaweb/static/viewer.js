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
	for (const { object, geometry, mirrors, wire, edges, points } of models.values()) {
		scene.remove(object);
		if (wire) {
			scene.remove(wire);
			wire.material.dispose();
		}
		if (edges) {
			scene.remove(edges); // shares the element geometry
			edges.material.dispose();
		}
		if (points) {
			scene.remove(points);
			points.geometry.dispose();
			points.material.dispose();
		}
		for (const copy of mirrors || []) {
			scene.remove(copy.mesh); // shares geometry/material
			if (copy.wire) scene.remove(copy.wire);
		}
		geometry.dispose();
		object.material.dispose();
	}
	models.clear();
	for (const light of [ambientLight, ...directionalLights]) if (light) scene.remove(light);
	directionalLights = [];
	ambientLight = null;
}

// Fixed-function GL: material ambient is 0.2 x the model color (SOFA's
// Material::setColor), times the LightManager's ambient; each light adds
// its color x max(0, n.l). three.js' Lambert BRDF divides by PI, so the
// intensities are scaled back up by PI.
const AMBIENT_INTENSITY = 0.2 * Math.PI;
const DIRECTIONAL_INTENSITY = Math.PI;

// The scene's own lights, as last streamed (a controller may move them,
// e.g. a rotating key light). The Lighting panel can override, per viewer,
// each light's brightness and a directional light's yaw/pitch; whatever
// isn't overridden follows the scene.
let sceneAmbient = [1, 1, 1];
let sceneLights = [];

function buildLights(ambient, lights) {
	sceneAmbient = ambient || [1, 1, 1];
	ambientLight = new THREE.AmbientLight(0xffffff, AMBIENT_INTENSITY);
	scene.add(ambientLight);
	for (const _ of lights) {
		const dir = new THREE.DirectionalLight(0xffffff, DIRECTIONAL_INTENSITY);
		scene.add(dir);
		scene.add(dir.target);
		directionalLights.push(dir);
	}
	updateLights(lights);
}

function updateLights(lights) {
	sceneLights = lights;
	applyLights();
	syncLightSliders();
}

// A SOFA light's brightness is its color's largest component (1 for
// white); the color divided by it is its hue, which stays the scene's.
function splitColor(rgb) {
	const m = Math.max(rgb[0], rgb[1], rgb[2]);
	return m > 0 ? { hue: [rgb[0] / m, rgb[1] / m, rgb[2] / m], brightness: m } : { hue: [1, 1, 1], brightness: 0 };
}

function lightColor(id) {
	return id === "ambient" ? sceneAmbient : sceneLights[id]?.color || [1, 1, 1];
}

function lightBrightness(id) {
	const value = displayState[`light:${id}:brightness`];
	return Number.isFinite(value) ? value : splitColor(lightColor(id)).brightness;
}

// Yaw/pitch (degrees) of a direction pointing towards the light: pitch is
// the elevation above the plane perpendicular to the scene's up axis, yaw
// the angle within that plane from its first axis (world X, or Y when up
// is X).
function upBasis() {
	const u = new THREE.Vector3(...((setup && setup.up) || [0, 0, 1])).normalize();
	const f = Math.abs(u.x) > 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
	f.addScaledVector(u, -f.dot(u)).normalize();
	return { u, f, g: new THREE.Vector3().crossVectors(u, f) };
}

function anglesFromDirection(d) {
	const { u, f, g } = upBasis();
	const v = new THREE.Vector3(d[0], d[1], d[2]).normalize();
	const deg = 180 / Math.PI;
	return [Math.atan2(v.dot(g), v.dot(f)) * deg, Math.asin(Math.min(1, Math.max(-1, v.dot(u)))) * deg];
}

function directionFromAngles(yaw, pitch) {
	const { u, f, g } = upBasis();
	const y = (yaw * Math.PI) / 180, p = (pitch * Math.PI) / 180;
	return f.multiplyScalar(Math.cos(p) * Math.cos(y)).addScaledVector(g, Math.cos(p) * Math.sin(y)).addScaledVector(u, Math.sin(p));
}

function lightAngles(i) {
	const override = displayState[`light:${i}:dir`];
	if (Array.isArray(override) && override.length === 2) return override.map(Number);
	return anglesFromDirection(sceneLights[i]?.direction || [0, 0, 1]);
}

function applyLights() {
	if (ambientLight) {
		ambientLight.color.setRGB(...splitColor(sceneAmbient).hue);
		ambientLight.intensity = AMBIENT_INTENSITY * lightBrightness("ambient");
	}
	sceneLights.forEach((light, i) => {
		const dir = directionalLights[i];
		if (!dir) return;
		dir.color.setRGB(...splitColor(light.color || [1, 1, 1]).hue);
		dir.intensity = DIRECTIONAL_INTENSITY * lightBrightness(i);
		// SOFA's DirectionalLight.direction points towards the light.
		const override = displayState[`light:${i}:dir`];
		if (Array.isArray(override)) dir.position.copy(directionFromAngles(...lightAngles(i)));
		else dir.position.set(...(light.direction || [0, 0, 1]));
		dir.position.normalize().multiplyScalar(100);
		dir.target.position.set(0, 0, 0);
	});
}

// FEM elements, drawn like SOFA's showForceFields: each tetrahedron's 4
// faces in 4 shades of blue, shrunk towards its centre so the elements stay
// apart. Not lit, like SOFA's debug drawing.
const TETRA_SHRINK = 0.8;
const TETRA_FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
const TETRA_COLORS = [[0, 0, 1], [0, 0.5, 1], [0, 1, 1], [0.5, 1, 1]];

// Strain coloring: each element in one color from a blue-cyan-green-yellow-red
// scale (0 .. the current maximum), its 4 faces in slightly different shades
// so the shape stays readable.
const STRAIN_SCALE = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]];
const STRAIN_SHADES = [1, 0.88, 0.76, 0.64];
const STRAIN_SCALE_MIN = 1e-3; // the color scale's smallest maximum

function strainColor(s) {
	const x = Math.min(1, Math.max(0, s)) * (STRAIN_SCALE.length - 1);
	const i = Math.min(STRAIN_SCALE.length - 2, Math.floor(x));
	const f = x - i;
	const a = STRAIN_SCALE[i], b = STRAIN_SCALE[i + 1];
	return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}

const TETRA_EDGE_COLOR = [0.05, 0.1, 0.3];
const TETRA_NODE_COLOR = [1.0, 0.85, 0.1];
const TETRA_NODE_SIZE = 4; // px

function buildTetraModel(info, order) {
	const tetraCount = info.index.length / 4;
	const faceCount = tetraCount * 4;
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(faceCount * 9), 3).setUsage(THREE.DynamicDrawUsage));
	// Colors per face slot in the canonical (unsorted) order; when the faces
	// are depth-sorted they're copied along with the positions.
	const faceColors = new Float32Array(faceCount * 9);
	for (let face = 0; face < faceCount; face++) {
		for (let v = 0; v < 3; v++) faceColors.set(TETRA_COLORS[face % 4], (face * 3 + v) * 3);
	}
	geometry.setAttribute("color", new THREE.BufferAttribute(faceColors.slice(), 3).setUsage(THREE.DynamicDrawUsage));
	geometry.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(faceCount * 9), 3).setUsage(THREE.DynamicDrawUsage));
	// forceSinglePass: three.js otherwise draws a transparent double-sided
	// mesh in two passes (all back-facing triangles, then all front-facing),
	// which would undo the explicit back-to-front order of writeTetraFaces.
	// Unlit (SOFA style, strain colors) or lit (the "lit" color mode).
	const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, forceSinglePass: true });
	const litMaterial = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, forceSinglePass: true });
	const object = new THREE.Mesh(geometry, material);
	object.frustumCulled = false;
	object.visible = false;
	scene.add(object);

	// Element edges: the same (shrunk, sorted) faces drawn as lines.
	const edges = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ wireframe: true, color: new THREE.Color(...TETRA_EDGE_COLOR), transparent: true }));
	// Nodes: the FEM mesh's own node positions, as fixed-size points.
	const nodes = new Float32Array(info.vertexCount * 3);
	const nodeGeometry = new THREE.BufferGeometry();
	nodeGeometry.setAttribute("position", new THREE.BufferAttribute(nodes, 3).setUsage(THREE.DynamicDrawUsage));
	const points = new THREE.Points(nodeGeometry, new THREE.PointsMaterial({
		color: new THREE.Color(...TETRA_NODE_COLOR), size: TETRA_NODE_SIZE, sizeAttenuation: false, transparent: true,
	}));
	for (const extra of [edges, points]) {
		extra.frustumCulled = false;
		extra.visible = false;
		scene.add(extra);
	}

	// Strain needs each element's rest shape: the inverse of its rest edge
	// matrix Dm = [r1-r0, r2-r0, r3-r0] (row-major, 9 values per element).
	let restInverse = null;
	let restCentre = null;
	if (Array.isArray(info.rest) && info.rest.length >= info.vertexCount * 3) {
		// Rest centres, for hiding elements by region (femRegion controls).
		restCentre = new Float64Array(tetraCount * 3);
		for (let t = 0; t < tetraCount; t++) {
			for (let k = 0; k < 4; k++) {
				const n = info.index[t * 4 + k] * 3;
				for (let c = 0; c < 3; c++) restCentre[t * 3 + c] += info.rest[n + c] / 4;
			}
		}
		restInverse = new Float64Array(tetraCount * 9);
		const r = info.rest;
		const m = new THREE.Matrix3();
		for (let t = 0; t < tetraCount; t++) {
			const n = [0, 1, 2, 3].map((k) => info.index[t * 4 + k] * 3);
			const col = (k, c) => r[n[k] + c] - r[n[0] + c];
			m.set(
				col(1, 0), col(2, 0), col(3, 0),
				col(1, 1), col(2, 1), col(3, 1),
				col(1, 2), col(2, 2), col(3, 2),
			);
			if (Math.abs(m.determinant()) < 1e-30) continue; // degenerate: stays zero strain
			m.invert();
			// THREE.Matrix3.elements is column-major; store row-major.
			const e = m.elements;
			restInverse.set([e[0], e[3], e[6], e[1], e[4], e[7], e[2], e[5], e[8]], t * 9);
		}
	}

	models.set(info.id, {
		object, geometry, info, order,
		tets: Uint32Array.from(info.index),
		nodes,
		edges,
		points,
		faceVerts: new Float32Array(faceCount * 9), // shrunk faces, canonical order, wound outward
		faceNormals: new Float32Array(faceCount * 9), // outward unit normal, per vertex
		faceColors,
		materials: { unlit: material, lit: litMaterial },
		baseColors: faceColors.slice(), // the blue shades
		restInverse,
		restCentre,
		hiddenTets: null, // Uint8Array, 1 = hidden by a femRegion control; null = none hidden
		hiddenKey: "",
		strain: new Float64Array(tetraCount),
		maxStrain: 0,
		colorMode: "sofa", // what faceColors currently hold: see femColorMode
		tetCentre: new Float64Array(tetraCount * 3),
		tetDepth: new Float64Array(tetraCount),
		tetOrder: new Uint32Array(tetraCount).map((_, i) => i),
		sortedFor: null, // camera matrix the current face order was sorted for
	});
}

function applyTetraPositions(entry, packed) {
	const nodes = entry.nodes;
	nodes.set(packed.subarray(0, nodes.length));
	const out = entry.faceVerts;
	const normals = entry.faceNormals;
	const tets = entry.tets;
	const corner = [0, 0, 0, 0];
	for (let t = 0, o = 0; t < tets.length / 4; t++) {
		let cx = 0, cy = 0, cz = 0;
		for (let k = 0; k < 4; k++) {
			const n = tets[t * 4 + k] * 3;
			corner[k] = n;
			cx += nodes[n];
			cy += nodes[n + 1];
			cz += nodes[n + 2];
		}
		cx /= 4;
		cy /= 4;
		cz /= 4;
		entry.tetCentre[t * 3] = cx;
		entry.tetCentre[t * 3 + 1] = cy;
		entry.tetCentre[t * 3 + 2] = cz;
		for (const face of TETRA_FACES) {
			const f = o;
			for (const k of face) {
				const n = corner[k];
				out[o++] = cx + TETRA_SHRINK * (nodes[n] - cx);
				out[o++] = cy + TETRA_SHRINK * (nodes[n + 1] - cy);
				out[o++] = cz + TETRA_SHRINK * (nodes[n + 2] - cz);
			}
			// Outward (away from the element's centre) unit normal; flip the
			// winding to match, so front faces are the outside ones.
			const ax = out[f + 3] - out[f], ay = out[f + 4] - out[f + 1], az = out[f + 5] - out[f + 2];
			const bx = out[f + 6] - out[f], by = out[f + 7] - out[f + 1], bz = out[f + 8] - out[f + 2];
			let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
			const fx = (out[f] + out[f + 3] + out[f + 6]) / 3 - cx;
			const fy = (out[f + 1] + out[f + 4] + out[f + 7]) / 3 - cy;
			const fz = (out[f + 2] + out[f + 5] + out[f + 8]) / 3 - cz;
			if (nx * fx + ny * fy + nz * fz < 0) {
				nx = -nx;
				ny = -ny;
				nz = -nz;
				for (let c = 0; c < 3; c++) {
					const tmp = out[f + 3 + c];
					out[f + 3 + c] = out[f + 6 + c];
					out[f + 6 + c] = tmp;
				}
			}
			const len = Math.hypot(nx, ny, nz) || 1;
			for (let v = 0; v < 3; v++) {
				normals[f + v * 3] = nx / len;
				normals[f + v * 3 + 1] = ny / len;
				normals[f + v * 3 + 2] = nz / len;
			}
		}
	}
	entry.points.geometry.attributes.position.needsUpdate = true;
	entry.points.geometry.computeBoundingSphere();
	entry.strainFresh = false;
	entry.hasPositions = true;
	updateTetraColors(entry, true);
	entry.sortedFor = null; // new positions: (re)write, sorted if needed
	writeTetraFaces(entry);
	entry.geometry.computeBoundingSphere();
}

// Green-Lagrange strain per element: F = Ds Dm^-1, E = (F^T F - I) / 2,
// reduced to its Frobenius norm. Zero for any rigid motion (rotation
// included), so only actual deformation shows.
function computeTetraStrain(entry) {
	const { nodes, tets, restInverse: inv, strain } = entry;
	let max = 0;
	for (let t = 0; t < strain.length; t++) {
		const a = tets[t * 4] * 3, b = tets[t * 4 + 1] * 3, c = tets[t * 4 + 2] * 3, d = tets[t * 4 + 3] * 3;
		const o = t * 9;
		const F = new Array(9);
		for (let row = 0; row < 3; row++) {
			const e1 = nodes[b + row] - nodes[a + row];
			const e2 = nodes[c + row] - nodes[a + row];
			const e3 = nodes[d + row] - nodes[a + row];
			for (let colIdx = 0; colIdx < 3; colIdx++) {
				F[row * 3 + colIdx] = e1 * inv[o + colIdx] + e2 * inv[o + 3 + colIdx] + e3 * inv[o + 6 + colIdx];
			}
		}
		let sum = 0;
		for (let i = 0; i < 3; i++) {
			for (let j = i; j < 3; j++) {
				const cij = F[i] * F[j] + F[3 + i] * F[3 + j] + F[6 + i] * F[6 + j];
				const e = 0.5 * (cij - (i === j ? 1 : 0));
				sum += i === j ? e * e : 2 * e * e;
			}
		}
		const degenerate = inv.subarray(o, o + 9).every((x) => x === 0); // see buildTetraModel
		const s = degenerate ? 0 : Math.sqrt(sum);
		strain[t] = s;
		if (s > max) max = s;
	}
	entry.maxStrain = max;
	entry.strainFresh = true;
}

// How FEM elements are colored (the Overlays panel's "Element colors"):
//   "sofa":   SOFA's showForceFields look, 4 unlit shades of blue per element;
//   "lit":    one blue, lit by the scene's lights like the visual models;
//   "strain": strain colors (unlit, 4 shades per element), see STRAIN_SCALE;
//   "strainLit": strain colors, lit -- the default.
const FEM_LIT_COLOR = [0.3, 0.5, 1.0];
const isStrainMode = (mode) => mode === "strain" || mode === "strainLit";
const isLitMode = (mode) => mode === "lit" || mode === "strainLit";

function femColorMode(entry) {
	const mode = displayState["overlay:femColor"];
	if (mode === "sofa" || mode === "lit") return mode;
	// No strain before the first positions arrive (the nodes are all zero),
	// nor without the mesh's rest shape.
	if (!entry.restInverse || !entry.hasPositions) return "sofa";
	return mode === "strain" ? "strain" : "strainLit";
}

// Fills faceColors for the current color mode -- for strain, scaled to the
// color scale's maximum (see strainPeak) -- and updates the max-strain
// readout. Returns whether the colors changed.
function updateTetraColors(entry, positionsChanged = false) {
	const wanted = femColorMode(entry);
	// The material goes with the mode -- also when the mode itself changes
	// on its own, e.g. from "sofa" to strain once the first positions arrive.
	const material = isLitMode(wanted) ? entry.materials.lit : entry.materials.unlit;
	if (entry.object.material !== material) {
		entry.object.material = material;
		entry.sortedFor = null;
	}
	if (!isStrainMode(wanted)) {
		if (entry.colorMode === wanted) return false;
		if (wanted === "sofa") {
			entry.faceColors.set(entry.baseColors);
		} else {
			for (let i = 0; i < entry.faceColors.length; i += 3) entry.faceColors.set(FEM_LIT_COLOR, i);
		}
		entry.colorMode = wanted;
		return true;
	}
	if (entry.colorMode === wanted && !positionsChanged) return false;
	if (!entry.strainFresh) computeTetraStrain(entry);
	if (entry.maxStrain > strainPeak) setStrainPeak(entry.maxStrain);
	// At least 0.1 %: an undeformed mesh's float rounding (~1e-7) would
	// otherwise fill the whole color range.
	const scale = 1 / Math.max(strainPeak, STRAIN_SCALE_MIN);
	const colors = entry.faceColors;
	const lit = isLitMode(wanted); // the lighting shades the faces instead
	for (let t = 0; t < entry.strain.length; t++) {
		const [r, g, b] = strainColor(entry.strain[t] * scale);
		for (let k = 0; k < 4; k++) {
			const shade = lit ? 1 : STRAIN_SHADES[k];
			for (let v = 0; v < 3; v++) colors.set([r * shade, g * shade, b * shade], ((t * 4 + k) * 3 + v) * 3);
		}
	}
	entry.colorMode = wanted;
	updateStrainReadout();
	return true;
}

// The strain color scale runs up to the largest strain seen so far, kept
// across restarts, rebuilds and reloads (it's part of the saved display
// settings), so runs compare on the same scale. The reset button next to
// the readout (or the Display panel's Reset) starts it over.
let strainPeak = 0;
let strainPeakSaved = 0;

function setStrainPeak(value) {
	strainPeak = value;
	displayState["overlay:femStrainPeak"] = value;
	// Save at most every 1 s while it's growing.
	const now = performance.now();
	if (value === 0 || now - strainPeakSaved > 1000) {
		strainPeakSaved = now;
		saveDisplay();
	}
}

function resetStrainPeak() {
	setStrainPeak(0);
	for (const entry of models.values()) {
		if (entry.info.kind === "tetra" && isStrainMode(entry.colorMode)) {
			updateTetraColors(entry, true);
			entry.sortedFor = null;
			writeTetraFaces(entry);
		}
	}
	saveDisplay();
	updateStrainReadout();
}

function updateStrainReadout() {
	const shown = $("d-strain-max");
	if (!shown) return;
	let max = null;
	for (const entry of models.values()) {
		if (entry.info.kind === "tetra" && isStrainMode(entry.colorMode)) max =Math.max(max ?? 0, entry.maxStrain);
	}
	const percent = (v) => `${(v * 100).toFixed(1)} %`;
	shown.textContent = max === null ? "" : `max now ${percent(max)} · scale ${percent(Math.max(strainPeak, STRAIN_SCALE_MIN))}`;
	$("d-strain-info")?.classList.toggle("hidden", max === null);
}

// Writes the element faces into the drawn geometry -- back to front when
// the elements are transparent, canonical otherwise:
//   * elements sorted by the view depth of their centre, farthest first
//     (the shrunk elements don't overlap, so this orders them correctly);
//   * within an element, the faces turned away from the camera before the
//     ones turned towards it -- exact for a convex shape, unlike sorting
//     faces by their own centres (a far-side face of a thin element can have
//     a nearer centre than a near-side face). "Away" uses the outward normal
//     (pointing away from the element centre, see applyTetraPositions).
// A few thousand elements take well under a millisecond, and it only runs
// after the camera moved or new positions arrived.
function writeTetraFaces(entry) {
	const transparent = entry.object.visible && entry.object.material.transparent;
	const key = transparent ? camera.matrixWorldInverse.elements.join(",") : "canonical";
	if (entry.sortedFor === key) return;
	entry.sortedFor = key;
	const pos = entry.geometry.attributes.position.array;
	const col = entry.geometry.attributes.color.array;
	const nor = entry.geometry.attributes.normal.array;
	const hidden = entry.hiddenTets;
	let w = 0; // faces written
	if (!transparent && !hidden) {
		pos.set(entry.faceVerts);
		col.set(entry.faceColors);
		nor.set(entry.faceNormals);
		w = entry.faceVerts.length / 9;
	} else if (!transparent) {
		// Canonical order, minus the hidden elements.
		for (let t = 0; t < hidden.length; t++) {
			if (hidden[t]) continue;
			const src = t * 36;
			pos.set(entry.faceVerts.subarray(src, src + 36), w * 9);
			col.set(entry.faceColors.subarray(src, src + 36), w * 9);
			nor.set(entry.faceNormals.subarray(src, src + 36), w * 9);
			w += 4;
		}
	} else {
		// The matrix's z row gives view-space depth; it's also the world
		// vector pointing from the scene towards an orthographic camera.
		const m = camera.matrixWorldInverse.elements;
		const centre = entry.tetCentre;
		const depth = entry.tetDepth;
		for (let t = 0; t < depth.length; t++) {
			depth[t] = m[2] * centre[t * 3] + m[6] * centre[t * 3 + 1] + m[10] * centre[t * 3 + 2] + m[14];
		}
		const order = entry.tetOrder.sort((a, b) => depth[a] - depth[b]); // most negative (farthest) first
		const v = entry.faceVerts;
		const colors = entry.faceColors;
		const normals = entry.faceNormals;
		const ortho = camera.isOrthographicCamera;
		const eye = camera.position;
		const front = [0, 0, 0, 0];
		const write = (face) => {
			const src = face * 9;
			pos.set(v.subarray(src, src + 9), w * 9);
			col.set(colors.subarray(src, src + 9), w * 9);
			nor.set(normals.subarray(src, src + 9), w * 9);
			w++;
		};
		for (const t of order) {
			if (hidden && hidden[t]) continue;
			for (let k = 0; k < 4; k++) {
				const o = (t * 4 + k) * 9;
				const nx = normals[o], ny = normals[o + 1], nz = normals[o + 2]; // outward
				const fx = (v[o] + v[o + 3] + v[o + 6]) / 3;
				const fy = (v[o + 1] + v[o + 4] + v[o + 7]) / 3;
				const fz = (v[o + 2] + v[o + 5] + v[o + 8]) / 3;
				const toCamera = ortho
					? nx * m[2] + ny * m[6] + nz * m[10]
					: nx * (eye.x - fx) + ny * (eye.y - fy) + nz * (eye.z - fz);
				front[k] = toCamera > 0;
			}
			for (let k = 0; k < 4; k++) if (!front[k]) write(t * 4 + k); // far side first
			for (let k = 0; k < 4; k++) if (front[k]) write(t * 4 + k);
		}
	}
	entry.geometry.setDrawRange(0, w * 3); // fills and edges share this geometry
	entry.geometry.attributes.position.needsUpdate = true;
	entry.geometry.attributes.color.needsUpdate = true;
	entry.geometry.attributes.normal.needsUpdate = true;
}

// femRegion display controls: while one is switched off, the elements whose
// rest centre lies in any of its regions (half-spaces such as "X below 0",
// each counted only while its `requires` parameter is on) are hidden, with
// the nodes that only they use. Returns whether anything changed.
function updateTetraHidden(entry) {
	const regions = [];
	for (const d of displaySpec) {
		if (d.type !== "femRegion" || displayState[d.key] !== false) continue;
		for (const r of d.regions || []) {
			if (!r.requires || currentParams[r.requires]) regions.push(r);
		}
	}
	const key = entry.restCentre ? JSON.stringify(regions) : "";
	if (key === entry.hiddenKey) return false;
	entry.hiddenKey = key;
	const tetraCount = entry.tets.length / 4;
	let hidden = null;
	if (regions.length && entry.restCentre) {
		const axisIndex = { X: 0, Y: 1, Z: 2 };
		const eps = 1e-9;
		hidden = new Uint8Array(tetraCount);
		for (let t = 0; t < tetraCount; t++) {
			for (const r of regions) {
				const c = entry.restCentre[t * 3 + axisIndex[r.axis]];
				if (("below" in r && c < r.below - eps) || ("above" in r && c > r.above + eps)) {
					hidden[t] = 1;
					break;
				}
			}
		}
	}
	entry.hiddenTets = hidden;
	// Nodes: only those used by a shown element.
	const nodeGeometry = entry.points.geometry;
	if (!hidden) {
		nodeGeometry.setIndex(null);
	} else {
		const used = new Uint8Array(entry.nodes.length / 3);
		for (let t = 0; t < tetraCount; t++) {
			if (!hidden[t]) for (let k = 0; k < 4; k++) used[entry.tets[t * 4 + k]] = 1;
		}
		const index = [];
		used.forEach((u, i) => u && index.push(i));
		nodeGeometry.setIndex(index);
	}
	entry.sortedFor = null;
	return true;
}

// Wireframe overlay: the same geometry drawn as lines, in a darkened model
// color so it shows on top of the fill. Shown independently of the fill's
// opacity, so a model can be seen as wireframe only.
function makeWire(geometry, info) {
	const [r, g, b] = info.color;
	const material = new THREE.MeshBasicMaterial({ wireframe: true, color: new THREE.Color(r * 0.45, g * 0.45, b * 0.45), transparent: true });
	const wire = new THREE.Mesh(geometry, material);
	wire.frustumCulled = false;
	wire.visible = false;
	scene.add(wire);
	return wire;
}

function buildModel(info, order) {
	if (info.kind === "tetra") return buildTetraModel(info, order);
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
	const wire = info.kind === "mesh" && !is2d ? makeWire(geometry, info) : null;
	models.set(info.id, { object, geometry, info, order, wire });
}

function applyPositions(entry, packed) {
	if (entry.info.kind === "tetra") return applyTetraPositions(entry, packed);
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
	// Reports the pose to the Camera fields while moving, and remembers it
	// once the user stops (after the damping has settled).
	controls.addEventListener("change", showCameraPose);
	controls.addEventListener("end", () => {
		clearTimeout(cameraSaveTimer);
		cameraSaveTimer = setTimeout(() => {
			cameraSaveTimer = null;
			displayState.camera = currentCameraPose();
			saveDisplay();
		}, 600);
	});
	sceneView();
	applyCameraPose(displayState.camera); // this viewer's (or the default) saved view, if any
}

// The camera pose, as part of the display settings (displayState.camera):
// {position, target, zoom}. Absent means the scene's own camera.
let cameraSaveTimer = null;

function currentCameraPose() {
	const round = (v) => +v.toPrecision(6);
	return {
		position: camera.position.toArray().map(round),
		target: controls.target.toArray().map(round),
		zoom: round(camera.zoom),
	};
}

function validPose(pose) {
	const vec = (v) => Array.isArray(v) && v.length === 3 && v.every((c) => Number.isFinite(+c));
	return pose && vec(pose.position) && vec(pose.target);
}

function applyCameraPose(pose) {
	if (!controls || !validPose(pose)) return false;
	camera.position.set(...pose.position.map(Number));
	controls.target.set(...pose.target.map(Number));
	camera.zoom = Number.isFinite(+pose.zoom) && +pose.zoom > 0 ? +pose.zoom : 1;
	camera.updateProjectionMatrix();
	controls.update();
	showCameraPose();
	return true;
}

// "Reset view": the scene's default view -- an admin's saved default
// camera, else the scene's own -- and forget this viewer's own.
function resetView() {
	clearTimeout(cameraSaveTimer);
	cameraSaveTimer = null;
	const fallback = (sceneInfo.displayDefaults || {}).camera;
	if (validPose(fallback)) displayState.camera = fallback;
	else delete displayState.camera;
	saveDisplay();
	sceneView();
	applyCameraPose(displayState.camera);
}

// Camera fields in the Display panel: "x, y, z" text, applied on Enter /
// leaving the field.
const cameraFields = {};

function buildCameraControls() {
	const box = $("display-camera");
	box.replaceChildren(heading("Camera"));
	for (const [key, label] of [["position", "Position"], ["target", "Look at"], ["zoom", "Zoom"]]) {
		const row = document.createElement("div");
		row.className = "display-row camera";
		const labelEl = document.createElement("label");
		labelEl.textContent = label;
		const input = document.createElement("input");
		input.type = "text";
		input.spellcheck = false;
		labelEl.htmlFor = input.id = `d-camera-${key}`;
		input.title = key === "zoom" ? "Zoom factor (orthographic view)" : "x, y, z in scene units";
		input.addEventListener("change", () => {
			const pose = currentCameraPose();
			const numbers = input.value.split(/[\s,;]+/).filter(Boolean).map(Number);
			if (key === "zoom" ? numbers.length === 1 && numbers[0] > 0 : numbers.length === 3 && numbers.every(Number.isFinite)) {
				pose[key] = key === "zoom" ? numbers[0] : numbers;
				applyCameraPose(pose);
				displayState.camera = currentCameraPose();
				saveDisplay();
			}
			showCameraPose(); // also restores the field after invalid input
		});
		row.append(labelEl, input);
		box.appendChild(row);
		cameraFields[key] = { row, input };
	}
	showCameraPose();
}

function showCameraPose() {
	if (!controls || !cameraFields.position) return;
	const pose = currentCameraPose();
	const fmt = (v) => (+v.toPrecision(4)).toString();
	for (const [key, { row, input }] of Object.entries(cameraFields)) {
		if (key === "zoom") row.classList.toggle("hidden", !camera.isOrthographicCamera);
		if (document.activeElement === input) continue; // being edited
		input.value = key === "zoom" ? fmt(pose.zoom) : pose[key].map(fmt).join(", ");
	}
}

// The scene's own camera (SOFA's InteractiveCamera).
function sceneView() {
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
	showCameraPose();
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
	buildMirrorCopies();
	buildLightControls(message);
	applyDisplay(); // this viewer's display settings, onto the fresh models and lights
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
		makeCollapsible(panel, `chart:${def.title}`);
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
		makeCollapsible(panel, "readouts");
	}
}

// Clicking a chart's or the readouts' title collapses the panel to just
// that title; remembered per scene in this browser.
const panelStoreKey = () => `sofaweb-panels:${sceneInfo.title}`;

function collapsedPanels() {
	try {
		return JSON.parse(localStorage.getItem(panelStoreKey()) || "{}");
	} catch {
		return {};
	}
}

function makeCollapsible(panel, id) {
	const title = panel.querySelector("h2");
	title.classList.add("collapsible-title");
	title.title = "Click to collapse or expand";
	panel.classList.toggle("collapsed", !!collapsedPanels()[id]);
	title.addEventListener("click", () => {
		const collapsed = panel.classList.toggle("collapsed");
		const state = collapsedPanels();
		if (collapsed) state[id] = true;
		else delete state[id];
		try {
			localStorage.setItem(panelStoreKey(), JSON.stringify(state));
		} catch {}
	});
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
	showPlayState($("play-btn"), !(s.paused || s.finished), s.finished ? "Continue the finished run" : "Play");
	$("auto-restart-checkbox").checked = !!s.autoRestart;

	let message = "";
	banner.className = "";
	if (s.error) {
		message = s.error;
		banner.className = "error";
	} else if (s.finished && !s.autoRestart) {
		message = `Run finished (${s.endReason}).` + (isViewer ? "" : " Press ▶ to continue it, or ■ to go back to the start.");
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
	updateMirrorPlanes();
	if (header.lights) updateLights(header.lights);
	$("time").textContent = `t = ${header.t.toFixed(2)} s · step ${header.step ?? "--"}`;
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

// Display rows that depend on a scene parameter (a mirror needs its
// symmetry to be in use) are grayed out while it's off in the running scene.
function updateDisplayAvailability() {
	for (const d of displaySpec) {
		if (!d.requires) continue;
		const input = $(`d-${d.key}`);
		if (!input) continue;
		const available = !!currentParams[d.requires];
		input.disabled = !available;
		const row = input.closest(".display-row");
		row.classList.toggle("disabled", !available);
		const label = fieldInputs.get(d.requires)?.field.label || d.requires;
		row.title = available ? "" : `No effect: "${label}" is off`;
	}
}

function showParams(params) {
	currentParams = params;
	updateDisplayAvailability();
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
	applyDisplay(); // mirrored copies depend on the symmetry parameters
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
// Display settings: applied to this browser's three.js models only -- no
// rebuild, the simulation isn't touched, and every viewer (watchers too)
// has their own, remembered in localStorage.
// ---------------------------------------------------------------------------

let displaySpec = [];
const displayState = {};
const displayStoreKey = () => `sofaweb-display:${sceneInfo.title}`;

function hexFromRgb(c) {
	return "#" + c.slice(0, 3).map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0")).join("");
}
function rgbFromHex(hex) {
	return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}

function displayDefault(d, value) {
	// A key that isn't in params.json has no value: fully opaque.
	if (d.type === "opacity") return value !== null && value !== undefined && Number.isFinite(+value) ? Math.min(1, Math.max(0, +value)) : 1;
	if (d.type === "color") return Array.isArray(value) && value.length >= 3 ? value.slice(0, 3).map(Number) : [1, 1, 1];
	return value === undefined || value === null ? true : !!value;
}

function displayNumber(key, fallback) {
	const value = displayState[key];
	return Number.isFinite(value) ? value : fallback;
}

function setOpacity(material, alpha, is2d) {
	const transparent = alpha < 0.999;
	if (material.transparent !== transparent) {
		material.transparent = transparent;
		material.needsUpdate = true;
	}
	material.opacity = alpha;
	if (!is2d) material.depthWrite = !transparent;
}

function applyDisplay() {
	if (!setup) return;
	applyLights();
	const is2d = setup.view === "2d";
	const wireOn = !!displayState["overlay:wireframe"];
	const wireAlpha = displayNumber("overlay:wireframeAlpha", 1);
	const femOn = !!displayState["overlay:fem"];
	const femAlpha = displayNumber("overlay:femAlpha", 1);
	const femWireOn = !!displayState["overlay:femWire"];
	const femWireAlpha = displayNumber("overlay:femWireAlpha", 1);
	const femNodesOn = !!displayState["overlay:femNodes"];
	const femNodesAlpha = displayNumber("overlay:femNodesAlpha", 1);
	for (const entry of models.values()) {
		if (entry.info.kind === "tetra") {
			// Both, so the one updateTetraColors switches to is ready.
			setOpacity(entry.materials.unlit, femAlpha, is2d);
			setOpacity(entry.materials.lit, femAlpha, is2d);
			entry.object.visible = femOn && femAlpha > 0.001;
			setOpacity(entry.edges.material, femWireAlpha, is2d);
			entry.edges.material.transparent = true; // sorted with the transparent fills
			entry.edges.visible = femWireOn && femWireAlpha > 0.001;
			setOpacity(entry.points.material, femNodesAlpha, is2d);
			entry.points.material.transparent = true;
			entry.points.visible = femNodesOn && femNodesAlpha > 0.001;
			updateTetraHidden(entry);
			if (updateTetraColors(entry)) entry.sortedFor = null;
			writeTetraFaces(entry); // sorted only while the elements are transparent
			continue;
		}
		const name = entry.info.name;
		let [r, g, b, alpha] = entry.info.color;
		let color = [r, g, b];
		let visible = true;
		for (const d of displaySpec) {
			const value = displayState[d.key];
			if (d.type === "mirror" || !d.re || !d.re.test(name)) continue;
			if (d.type === "opacity") {
				alpha = value;
				if (displayState[`${d.key}:on`] === false) visible = false; // its quick show/hide switch
			}
			else if (d.type === "color") color = value;
			else if (d.type === "visible") visible = visible && value;
		}
		const material = entry.object.material;
		material.color.setRGB(color[0], color[1], color[2]);
		const transparent = alpha < 0.999;
		if (material.transparent !== transparent) {
			material.transparent = transparent;
			material.needsUpdate = true;
		}
		material.opacity = alpha;
		if (setup.view !== "2d") {
			material.depthWrite = !transparent;
			entry.object.renderOrder = transparent ? 1000 + entry.order : 0;
		}
		entry.object.visible = visible && alpha > 0.001;
		// The wireframe follows show/hide, but not the fill's opacity: a
		// fill at opacity 0 leaves a wireframe-only view.
		const showWire = wireOn && wireAlpha > 0.001;
		if (entry.wire) {
			setOpacity(entry.wire.material, wireAlpha, is2d);
			entry.wire.material.transparent = true; // sorted with the transparent fills (see sortTransparent)
			entry.wire.visible = showWire && visible;
		}
		for (const copy of entry.mirrors || []) {
			const shown = visible && !copy.planeMissing && (!copy.chain || chainCopyShown(copy.chain)) &&
				copy.specs.every((d) => displayState[d.key] && (!d.requires || currentParams[d.requires]));
			copy.mesh.renderOrder = entry.object.renderOrder;
			copy.mesh.visible = shown && alpha > 0.001;
			if (copy.wire) copy.wire.visible = showWire && shown;
		}
	}
	updateStrainReadout();
}

// Mirrored copies across symmetry planes, drawn here rather than simulated
// or copied on the server: the same geometry and material, reflected by the
// copy's matrix. A plane is either fixed (`axis`, at `plane`, default 0)
// or follows a visual model of the scene that marks it (`planeModel`, a
// regex for its SOFA path -- e.g. a moving, tilting far-end plane), read
// from that model's current vertices every frame. Every combination of
// mirrors gets a copy (X, Y and XY for two axes), shown when all of its
// mirrors are switched on and the scene actually uses that symmetry (the
// `requires` parameter). In a combination the fixed planes reflect first,
// then the moving ones.
//
// A moving mirror with a `chain` repeats, for a finger of many chambers:
// reflecting alternately across the moving plane (b) and the chamber's own
// midplane (a: chain.axis at chain.plane) walks along the finger. With the
// half chamber H the scene simulates (chain.half, e.g. symmetry_x, on),
// the copies are the half chambers bH, baH, babH, ... -- two per extra
// chamber; with a whole chamber C, they're bC, babC, ... -- one per
// chamber. The number of extra chambers is the display setting
// chain.countKey (1 .. chain.max). Each chain copy also comes in every
// combination with the other fixed mirrors (e.g. Y).
const axisIndex = { X: 0, Y: 1, Z: 2 };

function axisReflection(axis, plane = 0) {
	const i = axisIndex[axis];
	const scale = new THREE.Vector3(1, 1, 1).setComponent(i, -1);
	const reflect = new THREE.Matrix4().makeScale(scale.x, scale.y, scale.z);
	reflect.elements[12 + i] = 2 * plane;
	return reflect;
}

function buildMirrorCopies() {
	const mirrors = displaySpec.filter((d) => d.type === "mirror");
	if (!mirrors.length) return;
	for (const entry of models.values()) {
		const specs = mirrors.filter((d) => d.re && d.re.test(entry.info.name));
		if (!specs.length || entry.info.kind !== "mesh") continue;
		entry.mirrors = [];
		const addCopy = (props) => {
			const mesh = new THREE.Mesh(entry.geometry, entry.object.material);
			const wire = entry.wire ? new THREE.Mesh(entry.geometry, entry.wire.material) : null;
			for (const object of wire ? [mesh, wire] : [mesh]) {
				object.frustumCulled = false;
				object.matrixAutoUpdate = false;
				object.matrix.copy(props.fixed);
				scene.add(object);
			}
			entry.mirrors.push({ mesh, wire, moving: [], chain: null, ...props });
		};
		const fixedOf = (subset) => {
			const fixed = new THREE.Matrix4();
			for (const d of subset.filter((d) => !d.planeModel)) fixed.premultiply(axisReflection(d.axis, d.plane || 0));
			return fixed;
		};
		// Plain combinations (chains have their own copies below).
		const plain = specs.filter((d) => !d.chain);
		for (let mask = 1; mask < 1 << plain.length; mask++) {
			const subset = plain.filter((_, i) => mask & (1 << i));
			const moving = subset.filter((d) => d.planeModel).map((d) => new RegExp(d.planeModel));
			addCopy({ specs: subset, fixed: fixedOf(subset), moving, planeMissing: moving.length > 0 });
		}
		for (const chainSpec of specs.filter((d) => d.chain && d.planeModel)) {
			// With every combination of the fixed mirrors other than the
			// chain's own axis (that one is part of the chain).
			const others = plain.filter((d) => !d.planeModel && d.axis !== chainSpec.chain.axis);
			const max = Math.max(1, Math.min(32, chainSpec.chain.max || 10));
			for (let mask = 0; mask < 1 << others.length; mask++) {
				const subset = others.filter((_, i) => mask & (1 << i));
				for (let k = 1; k <= 2 * max; k++) {
					addCopy({
						specs: [chainSpec, ...subset],
						fixed: fixedOf(subset),
						moving: [new RegExp(chainSpec.planeModel)],
						chain: { spec: chainSpec, k },
						planeMissing: true,
					});
				}
			}
		}
	}
	updateMirrorPlanes();
}

// Whether chain copy k is within the chosen number of extra chambers.
function chainCopyShown(chain) {
	const c = chain.spec.chain;
	const count = Math.max(1, Math.min(c.max || 10, Math.round(+displayState[c.countKey] || 1)));
	const half = c.half ? !!currentParams[c.half] : true;
	return half ? chain.k <= 2 * count : chain.k % 2 === 1 && chain.k <= 2 * count - 1;
}

// The reflection across the plane a (flat) visual model currently lies in:
// the normal of its largest triangle (the sign doesn't matter for a
// reflection) through the centroid of its vertices. False until it has
// positions.
const _pa = new THREE.Vector3(), _pb = new THREE.Vector3(), _pc = new THREE.Vector3();
const _cross = new THREE.Vector3(), _normal = new THREE.Vector3(), _centroid = new THREE.Vector3();

function reflectionAcross(planeEntry, out) {
	const pos = planeEntry.geometry.attributes.position;
	const index = planeEntry.geometry.index;
	const count = index ? index.count : pos.count;
	let best = 0;
	for (let i = 0; i + 2 < count; i += 3) {
		_pa.fromBufferAttribute(pos, index ? index.getX(i) : i);
		_pb.fromBufferAttribute(pos, index ? index.getX(i + 1) : i + 1).sub(_pa);
		_pc.fromBufferAttribute(pos, index ? index.getX(i + 2) : i + 2).sub(_pa);
		_cross.crossVectors(_pb, _pc);
		const area = _cross.lengthSq();
		if (area > best) {
			best = area;
			_normal.copy(_cross);
		}
	}
	if (!(best > 1e-30)) return false;
	_normal.normalize();
	_centroid.set(0, 0, 0);
	for (let i = 0; i < pos.count; i++) _centroid.add(_pa.fromBufferAttribute(pos, i));
	_centroid.divideScalar(pos.count);
	const { x, y, z } = _normal;
	const d = 2 * _normal.dot(_centroid);
	out.set(
		1 - 2 * x * x, -2 * x * y, -2 * x * z, d * x,
		-2 * y * x, 1 - 2 * y * y, -2 * y * z, d * y,
		-2 * z * x, -2 * z * y, 1 - 2 * z * z, d * z,
		0, 0, 0, 1,
	);
	return true;
}

// Re-aims the copies across moving planes (after every frame).
const _reflection = new THREE.Matrix4();
function updateMirrorPlanes() {
	let visibilityChanged = false;
	const planes = new Map(); // planeModel regex source -> reflection, or null
	const planeReflection = (re) => {
		if (!planes.has(re.source)) {
			const planeEntry = [...models.values()].find((m) => re.test(m.info.name));
			const out = new THREE.Matrix4();
			planes.set(re.source, planeEntry && reflectionAcross(planeEntry, out) ? out : null);
		}
		return planes.get(re.source);
	};
	const chainWords = new Map(); // chain spec -> [W_1, W_2, ...]
	const words = (spec, b) => {
		if (!chainWords.has(spec)) {
			const a = axisReflection(spec.chain.axis, spec.chain.plane || 0);
			const list = [b.clone()];
			const max = Math.max(1, Math.min(32, spec.chain.max || 10));
			for (let k = 2; k <= 2 * max; k++) list.push(list[k - 2].clone().multiply(k % 2 === 0 ? a : b));
			chainWords.set(spec, list);
		}
		return chainWords.get(spec);
	};
	for (const entry of models.values()) {
		for (const copy of entry.mirrors || []) {
			if (!copy.moving.length) continue;
			const matrix = copy.fixed.clone();
			let missing = false;
			if (copy.chain) {
				const b = planeReflection(copy.moving[0]);
				if (b) matrix.premultiply(words(copy.chain.spec, b)[copy.chain.k - 1]);
				else missing = true;
			} else {
				for (const re of copy.moving) {
					const reflection = planeReflection(re);
					if (!reflection) {
						missing = true;
						break;
					}
					matrix.premultiply(reflection);
				}
			}
			if (missing !== copy.planeMissing) {
				copy.planeMissing = missing;
				visibilityChanged = true;
			}
			if (missing) continue;
			for (const object of copy.wire ? [copy.mesh, copy.wire] : [copy.mesh]) {
				object.matrix.copy(matrix);
				object.matrixWorldNeedsUpdate = true;
			}
		}
	}
	if (visibilityChanged) applyDisplay();
}

function saveDisplay() {
	try {
		localStorage.setItem(displayStoreKey(), JSON.stringify(displayState));
	} catch {}
}

let savedDisplay = {};
let overlaySignature = null;
let lightingSignature = null;

function lightLabel(name, index) {
	if (!name) return `Light ${index + 1}`;
	const words = name.replace(/([a-z0-9])([A-Z])/g, "$1 $2").toLowerCase();
	return words.charAt(0).toUpperCase() + words.slice(1);
}

function savedOr(key, fallback) {
	if (!(key in savedDisplay)) return fallback;
	const value = savedDisplay[key];
	if (typeof fallback === "string") return typeof value === "string" ? value : fallback;
	return typeof fallback === "boolean" ? !!value : Number.isFinite(+value) ? +value : fallback;
}

function displayChanged(key, value) {
	displayState[key] = value;
	saveDisplay();
	applyDisplay();
}

// A slider row: label, range input, shown value.
function sliderRow(key, labelText, max, step, format, toggleKey = null) {
	const row = document.createElement("div");
	row.className = "display-row opacity";
	const label = document.createElement("label");
	label.htmlFor = `d-${key}`;
	if (toggleKey) {
		// Switch and slider in one row: "[x] Wireframe  ----o----  0.80".
		const toggle = document.createElement("input");
		toggle.type = "checkbox";
		toggle.id = `d-${toggleKey}`;
		toggle.checked = displayState[toggleKey];
		toggle.addEventListener("change", () => displayChanged(toggleKey, toggle.checked));
		label.htmlFor = toggle.id;
		label.className = "toggle-label";
		label.append(toggle, ` ${labelText}`);
	} else {
		label.textContent = labelText;
	}
	const input = document.createElement("input");
	input.id = `d-${key}`;
	input.type = "range";
	Object.assign(input, { min: 0, max, step });
	input.value = displayState[key];
	const shown = document.createElement("span");
	shown.className = "display-value";
	shown.textContent = format(+input.value);
	input.addEventListener("input", () => {
		shown.textContent = format(+input.value);
		displayChanged(key, +input.value);
	});
	row.append(label, input, shown);
	return row;
}

// A checkbox row: "[x] Label", optionally with a value shown on the right.
function checkboxRow(key, labelText, valueId = null) {
	const row = document.createElement("div");
	row.className = "display-row visible";
	const toggle = document.createElement("input");
	toggle.type = "checkbox";
	toggle.id = `d-${key}`;
	toggle.checked = !!displayState[key];
	toggle.addEventListener("change", () => displayChanged(key, toggle.checked));
	const label = document.createElement("label");
	label.htmlFor = toggle.id;
	label.className = "toggle-label";
	label.append(toggle, ` ${labelText}`);
	row.appendChild(label);
	if (valueId) {
		const shown = document.createElement("span");
		shown.className = "display-value";
		shown.id = valueId;
		row.appendChild(shown);
	}
	return row;
}

// A drop-down row: "Label   [option v]", options as [value, text] pairs.
function selectRow(key, labelText, options) {
	const row = document.createElement("div");
	row.className = "display-row select";
	const label = document.createElement("label");
	label.htmlFor = `d-${key}`;
	label.textContent = labelText;
	const select = document.createElement("select");
	select.id = `d-${key}`;
	for (const [value, text] of options) select.add(new Option(text, value));
	select.value = displayState[key];
	select.addEventListener("change", () => displayChanged(key, select.value));
	row.append(label, select);
	return row;
}

function heading(text) {
	const h = document.createElement("h3");
	h.textContent = text;
	return h;
}

// Controls every scene gets, built once the setup message says which
// lights and FEM meshes it has: wireframe / FEM-element overlays (switch +
// opacity) in the Display panel, and the Lighting panel.
function buildLightControls(message) {
	buildLightingPanel(message);
	const hasFem = message.models.some((m) => m.kind === "tetra");
	const hasRest = message.models.some((m) => m.kind === "tetra" && m.rest);
	const signature = JSON.stringify({ hasFem, hasRest });
	if (signature === overlaySignature) return; // same as before this rebuild
	overlaySignature = signature;
	const box = $("display-overlays");
	box.replaceChildren();
	const alpha = (v) => v.toFixed(2);

	box.appendChild(heading("Overlays"));
	displayState["overlay:wireframe"] = savedOr("overlay:wireframe", false);
	displayState["overlay:wireframeAlpha"] = savedOr("overlay:wireframeAlpha", 1);
	box.appendChild(sliderRow("overlay:wireframeAlpha", "Surface wireframe", 1, 0.01, alpha, "overlay:wireframe"));
	if (hasFem) {
		for (const [key, label] of [["fem", "FEM elements"], ["femWire", "FEM element edges"], ["femNodes", "FEM nodes"]]) {
			displayState[`overlay:${key}`] = savedOr(`overlay:${key}`, false);
			displayState[`overlay:${key}Alpha`] = savedOr(`overlay:${key}Alpha`, 1);
			box.appendChild(sliderRow(`overlay:${key}Alpha`, label, 1, 0.01, alpha, `overlay:${key}`));
			if (key === "fem") {
				const modes = [["sofa", "SOFA style"], ["lit", "Blue, lit"]];
				if (hasRest) modes.push(["strain", "By strain"], ["strainLit", "By strain, lit"]);
				const saved = savedOr("overlay:femColor", "strainLit");
				displayState["overlay:femColor"] = modes.some(([v]) => v === saved) ? saved : "sofa";
				box.appendChild(selectRow("overlay:femColor", "Element colors", modes));
				if (!hasRest) continue;
				strainPeak = savedOr("overlay:femStrainPeak", 0);
				displayState["overlay:femStrainPeak"] = strainPeak;
				// Below it, while coloring by strain: the current maximum, the
				// color scale's maximum (the largest seen so far) and a button
				// to reset it.
				const info = document.createElement("div");
				info.id = "d-strain-info";
				info.className = "display-row strain hidden";
				const shown = document.createElement("span");
				shown.id = "d-strain-max";
				shown.className = "display-value";
				const reset = document.createElement("button");
				reset.className = "mini-btn secondary";
				reset.textContent = "Reset scale";
				reset.title = "Restart the color scale from the current strain";
				reset.addEventListener("click", resetStrainPeak);
				info.append(shown, reset);
				box.appendChild(info);
			}
		}
	}

	$("display-panel").classList.remove("hidden");
}

// Lighting panel: absolute brightness per light (the scene's own value until
// changed) and yaw/pitch per directional light (following the scene's
// direction, including a light a controller moves, until changed). Stored
// with the display settings as light:<id>:brightness and light:<i>:dir.
const lightSliders = [];

function rangeRow(labelText, min, max, step, format, onInput) {
	const row = document.createElement("div");
	row.className = "display-row opacity";
	const label = document.createElement("label");
	label.textContent = labelText;
	const input = document.createElement("input");
	input.type = "range";
	Object.assign(input, { min, max, step });
	label.htmlFor = input.id = `l-${Math.random().toString(36).slice(2)}`;
	const shown = document.createElement("span");
	shown.className = "display-value";
	const range = { row, input, shown, format };
	input.addEventListener("input", () => {
		shown.textContent = format(+input.value);
		onInput(+input.value);
	});
	row.append(label, input, shown);
	return range;
}

function setRange(range, value) {
	if (document.activeElement === range.input) return; // being dragged
	range.input.value = value;
	range.shown.textContent = range.format(+range.input.value);
}

// Shows the current values: the overrides, or the scene's own (which may
// be moving).
function syncLightSliders() {
	for (const entry of lightSliders) {
		setRange(entry.brightness, lightBrightness(entry.id));
		if (entry.yaw) {
			const [yaw, pitch] = lightAngles(entry.id);
			setRange(entry.yaw, yaw);
			setRange(entry.pitch, pitch);
		}
	}
}

function buildLightingPanel(message, force = false) {
	const lights = [
		{ id: "ambient", label: "Ambient light" },
		...message.lights.map((light, i) => ({ id: i, label: lightLabel(light.name, i) })),
	];
	const signature = JSON.stringify(lights);
	if (signature === lightingSignature && !force) return;
	lightingSignature = signature;
	for (const [key, value] of Object.entries(savedDisplay)) {
		if (/^light:.+:(brightness|dir)$/.test(key) && !(key in displayState)) displayState[key] = value;
	}
	const box = $("lighting-controls");
	box.replaceChildren();
	lightSliders.length = 0;
	const degrees = (v) => `${Math.round(v)}°`;
	for (const light of lights) {
		box.appendChild(heading(light.label));
		const entry = { id: light.id };
		entry.brightness = rangeRow("Brightness", 0, 2, 0.01, (v) => v.toFixed(2), (v) => {
			displayState[`light:${light.id}:brightness`] = v;
			saveDisplay();
			applyLights();
		});
		box.appendChild(entry.brightness.row);
		if (light.id !== "ambient") {
			const setDirection = () => {
				displayState[`light:${light.id}:dir`] = [+entry.yaw.input.value, +entry.pitch.input.value];
				saveDisplay();
				applyLights();
			};
			entry.yaw = rangeRow("Yaw", -180, 180, 1, degrees, setDirection);
			entry.pitch = rangeRow("Pitch", -90, 90, 1, degrees, setDirection);
			box.append(entry.yaw.row, entry.pitch.row);
		}
		lightSliders.push(entry);
	}
	syncLightSliders();
	$("lighting-panel").classList.remove("hidden");
}

$("lighting-reset-btn").addEventListener("click", () => {
	// Back to the scene's lights, or its saved lighting defaults.
	for (const key of Object.keys(displayState)) if (key.startsWith("light:")) delete displayState[key];
	for (const key of Object.keys(savedDisplay)) if (key.startsWith("light:")) delete savedDisplay[key];
	for (const [key, value] of Object.entries(sceneInfo.displayDefaults || {})) {
		if (key.startsWith("light:")) displayState[key] = savedDisplay[key] = value;
	}
	saveDisplay();
	applyLights();
	syncLightSliders();
});

// "Set as default" (admins only): stores a panel's current state as the
// scene's default in the orchestrator, for every viewer without settings of
// their own and every simulation of this scene started from now on.
const simIdMatch = location.pathname.match(/^\/sim\/([a-z0-9]+)\//);

async function checkAdmin() {
	if (!simIdMatch) return; // not behind the orchestrator
	try {
		const response = await fetch("/api/admin-session", { cache: "no-store" });
		if (response.ok && (await response.json()).admin) document.body.classList.add("admin");
	} catch {}
}

async function setAsDefault(section, values, button, what) {
	if (!confirm(`Make ${what} the default for this scene? It applies to everyone, and to every simulation of this scene started from now on.`)) return;
	const label = button.textContent;
	button.disabled = true;
	try {
		const response = await fetch(`/admin/api/sims/${simIdMatch[1]}/defaults/${section}`, {
			method: "PUT",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ values }),
		});
		if (!response.ok) {
			const body = await response.json().catch(() => ({}));
			throw new Error(body?.detail?.message || `${response.status} ${response.statusText}`);
		}
		if (section !== "params") {
			// The section is replaced as a whole, so this page's Reset
			// buttons go back to exactly these right away.
			const others = Object.entries(sceneInfo.displayDefaults || {}).filter(
				([key]) => key.startsWith("light:") !== (section === "lighting"),
			);
			sceneInfo.displayDefaults = { ...Object.fromEntries(others), ...values };
		}
		button.textContent = "Saved ✓";
		setTimeout(() => (button.textContent = label), 2000);
	} catch (err) {
		alert(`Couldn't save the defaults: ${err.message}`);
	} finally {
		button.disabled = false;
	}
}

$("display-default-btn").addEventListener("click", (event) => {
	// The view as it is now, if it's been moved from the scene's (the last
	// move may not be remembered yet).
	if (displayState.camera || cameraSaveTimer !== null) displayState.camera = currentCameraPose();
	clearTimeout(cameraSaveTimer);
	cameraSaveTimer = null;
	const values = Object.fromEntries(
		Object.entries(displayState).filter(([key]) => !key.startsWith("light:") && key !== "overlay:femStrainPeak"),
	);
	setAsDefault("display", values, event.currentTarget, "these display settings");
});

$("lighting-default-btn").addEventListener("click", (event) => {
	// Only what's been changed here: an untouched light keeps following the scene.
	const values = Object.fromEntries(Object.entries(displayState).filter(([key]) => key.startsWith("light:")));
	setAsDefault("lighting", values, event.currentTarget, "these lighting settings");
});

$("params-default-btn").addEventListener("click", (event) => {
	// The running simulation's parameters (not unapplied edits in the form).
	const values = Object.fromEntries(Object.entries(currentParams).filter(([key]) => fieldInputs.has(key)));
	setAsDefault("params", values, event.currentTarget, "the running parameters");
});

checkAdmin();

function buildDisplayPanel(spec, useSaved = true) {
	displaySpec = spec.map((d) => ({ ...d, re: d.models ? new RegExp(d.models) : null }));
	// The scene's defaults (an admin's "Set as default"), under this
	// browser's own saved settings; Reset goes back to just the defaults.
	let saved = { ...(sceneInfo.displayDefaults || {}) };
	if (useSaved) {
		try {
			Object.assign(saved, JSON.parse(localStorage.getItem(displayStoreKey()) || "{}"));
		} catch {}
	}
	savedDisplay = saved;
	if (validPose(saved.camera)) displayState.camera = saved.camera;
	else delete displayState.camera;
	const form = $("display-form");
	form.replaceChildren();
	for (const d of displaySpec) {
		// `default`: for a key that isn't in params.json.
		displayState[d.key] = displayDefault(d, d.key in saved ? saved[d.key] : d.value ?? d.default);
		if (d.type === "opacity") {
			// A show/hide switch in front of the slider, for quick toggling
			// without losing the opacity.
			const onKey = `${d.key}:on`;
			displayState[onKey] = onKey in saved ? !!saved[onKey] : true;
			form.appendChild(sliderRow(d.key, d.label, 1, 0.01, (v) => v.toFixed(2), onKey));
		} else if (d.type === "color") {
			const row = document.createElement("div");
			row.className = "display-row color";
			const label = document.createElement("label");
			label.htmlFor = `d-${d.key}`;
			label.textContent = d.label;
			const input = document.createElement("input");
			input.id = `d-${d.key}`;
			input.type = "color";
			input.value = hexFromRgb(displayState[d.key]);
			input.addEventListener("input", () => displayChanged(d.key, rgbFromHex(input.value)));
			row.append(label, input);
			form.appendChild(row);
		} else if (d.type === "mirror" && d.chain && d.chain.countKey) {
			// The mirror switch plus how many chambers to add.
			const row = checkboxRow(d.key, d.label);
			row.classList.add("count");
			const key = d.chain.countKey;
			const max = d.chain.max || 10;
			const value = Math.round(+(key in saved ? saved[key] : d.chain.default ?? 1));
			displayState[key] = Number.isFinite(value) ? Math.max(1, Math.min(max, value)) : 1;
			const input = document.createElement("input");
			input.type = "number";
			Object.assign(input, { min: 1, max, step: 1, value: displayState[key] });
			input.id = `d-${key}`;
			input.title = "Number of chambers to add";
			input.addEventListener("change", () => {
				const n = Math.max(1, Math.min(max, Math.round(+input.value) || 1));
				input.value = n;
				displayChanged(key, n);
			});
			row.appendChild(input);
			form.appendChild(row);
		} else {
			form.appendChild(checkboxRow(d.key, d.label));
		}
	}
	updateDisplayAvailability();
}

$("display-reset-btn").addEventListener("click", () => {
	try {
		localStorage.removeItem(displayStoreKey());
	} catch {}
	buildDisplayPanel(sceneInfo.display || [], false);
	overlaySignature = null; // the Lighting panel keeps its own settings (its own Reset)
	if (setup) buildLightControls(setup);
	if (controls) {
		sceneView();
		applyCameraPose(displayState.camera); // the default view
	}
	applyDisplay();
	resetStrainPeak();
});

// ---------------------------------------------------------------------------
// Buttons, console, info
// ---------------------------------------------------------------------------

// Media-style controls: one play/pause toggle, and stop = pause + back to
// the start (the simulation stays paused there until play is pressed).
const ICONS = {
	play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9.5-5.5z"/></svg>',
	pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 2.5h3.2v11H3.5zM9.3 2.5h3.2v11H9.3z"/></svg>',
	stop: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="3" width="10" height="10" rx="1"/></svg>',
};
function showPlayState(button, running, pausedLabel = "Play") {
	button.innerHTML = running ? ICONS.pause : ICONS.play;
	button.classList.toggle("running", running); // green play / blue pause
	button.title = running ? "Pause" : pausedLabel;
	button.setAttribute("aria-label", button.title);
}

// Simulations start paused; the status message corrects this if not.
showPlayState($("play-btn"), false);
$("stop-btn").innerHTML = ICONS.stop;
$("play-btn").addEventListener("click", () => {
	const resume = lastStatus && (lastStatus.paused || lastStatus.finished);
	fetch(resume ? "api/resume" : "api/pause", { method: "POST" }).catch(() => {});
});
$("stop-btn").addEventListener("click", async () => {
	try {
		await fetch("api/pause", { method: "POST" });
		await fetch("api/reset", { method: "POST" });
	} catch {}
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
	buildDisplayPanel(sceneInfo.display || []);
	buildCameraControls();
	if (!displaySpec.length) $("display-panel").classList.add("hidden");
	buildCharts(sceneInfo);
	await loadParams();
	if (isViewer) setInterval(loadParams, 3000);
	connect();
}
init();

// For debugging from the browser console.
window.sofaweb = { scene, models, renderer, get camera() { return camera; }, get setup() { return setup; } };

// Transparent models must be drawn back to front. three.js sorts by each
// object's origin, but every SOFA model sits at the world origin (its
// vertices are in world coordinates), so that sort can't tell them apart.
// Instead, order the visible transparent models -- and mirrored copies --
// by the view-space depth of their bounding-sphere centre each frame,
// farthest first. Faces within one model aren't sorted against each other.
const _centre = new THREE.Vector3();
function sortTransparent() {
	if (!setup || setup.view === "2d") return; // 2D scenes keep creation order
	camera.updateMatrixWorld();
	const items = [];
	// `bias` nudges a wireframe to just after (in front of) its own fill.
	const consider = (mesh, geometry, bias = 0) => {
		if (!mesh || !mesh.visible || !mesh.material.transparent || !geometry.boundingSphere) return;
		mesh.updateMatrixWorld();
		_centre.copy(geometry.boundingSphere.center).applyMatrix4(mesh.matrixWorld).applyMatrix4(camera.matrixWorldInverse);
		items.push({ mesh, depth: _centre.z + bias }); // view space: more negative = farther away
	};
	for (const entry of models.values()) {
		consider(entry.object, entry.geometry);
		consider(entry.wire, entry.geometry, 1e-9);
		if (entry.info.kind === "tetra") {
			// The elements' faces themselves are sorted in writeTetraFaces;
			// edges and nodes go right after them.
			consider(entry.edges, entry.geometry, 1e-9);
			consider(entry.points, entry.points.geometry, 2e-9);
			if (entry.object.visible) writeTetraFaces(entry); // re-sorts if the camera moved
		}
		for (const copy of entry.mirrors || []) {
			consider(copy.mesh, entry.geometry);
			consider(copy.wire, entry.geometry, 1e-9);
		}
	}
	items.sort((a, b) => a.depth - b.depth);
	items.forEach((item, i) => (item.mesh.renderOrder = 1000 + i));
}

function animate() {
	requestAnimationFrame(animate);
	if (chartsDirty) {
		for (const { chart } of charts) chart.update("none");
		chartsDirty = false;
	}
	if (controls) controls.update();
	sortTransparent();
	renderer.render(scene, camera);
}
animate();
