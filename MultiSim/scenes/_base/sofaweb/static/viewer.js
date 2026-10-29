// Generic viewer for a sofaweb scene: rebuilds the SOFA scene's visual
// models in three.js from the server's "setup" message and moves them with
// every binary frame. Everything scene-specific (parameters, charts,
// readouts) comes from api/scene.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import Chart from "chart.js/auto";
import { createManipulator } from "./manipulator.js";
import { texturedMaterial, setAnisotropy } from "./textures.js";

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
setAnisotropy(Math.min(8, renderer.capabilities.getMaxAnisotropy()));

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
	for (const { object, geometry, mirrors, wire, edges, points, plainMaterial, texMaterial } of models.values()) {
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
		for (const m of [plainMaterial, texMaterial]) if (m && m !== object.material) m.dispose();
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
	// A surface texture (see textures.js): the textured material and the
	// plain one, switched by the Display panel's "Textures".
	if (info.texture && info.texture.label && info.texture.label.overlay && !is2d) {
		// A label part sits a fraction of a millimetre on its package:
		// pulled forward in depth so it doesn't flicker through it.
		Object.assign(object.material, { polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8 });
	}
	const texMaterial = info.kind === "mesh" && !is2d ? texturedMaterial(object.material, geometry, info) : null;
	models.set(info.id, { object, geometry, info, order, wire, plainMaterial: object.material, texMaterial });
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
	// A portrait screen (a phone) shows less width for the same vertical field: back off to keep the scene's
	// width in view (the scene's camera is set for a landscape window).
	const aspect = window.innerWidth / window.innerHeight;
	if (aspect < 1) {
		if (camera.isOrthographicCamera) camera.zoom = Math.max(0.45, aspect);
		else camera.position.copy(controls.target).addScaledVector(forward, -cam.distance * Math.min(2.2, 1 / aspect));
	}
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

// The side panels may not reach below the window: their column gets the
// height left under its top, which depends on the HUD (wrapped, "..." open).
function fitSidePanels() {
	const column = $("side-panels");
	const room = window.innerHeight - column.getBoundingClientRect().top - 12;
	column.style.setProperty("--side-max", `${Math.max(120, Math.floor(room))}px`);
}
window.addEventListener("resize", fitSidePanels);
new ResizeObserver(fitSidePanels).observe($("hud"));
fitSidePanels();
$("view-btn").addEventListener("click", resetView);

function applySetup(message) {
	const firstSetup = setup === null;
	const cameraChanged = !setup || JSON.stringify(setup.camera) !== JSON.stringify(message.camera) || setup.view !== message.view;
	const newRun = message.generation !== generation;
	setup = message;
	generation = message.generation;
	// New models (or a model's vertices now sent separately): the recording
	// and the live positions so far no longer fit them.
	resetReplay();
	livePositions.clear();
	disposeScene();
	scene.background = sofaColor(message.background);
	rulerKey = null; // the ruler follows the new background
	document.body.style.background = `#${scene.background.getHexString()}`;
	buildLights(message.ambient, message.lights);
	message.models.forEach((info, order) => buildModel(info, order));
	buildGauge(message.gauge);
	buildMirrorCopies();
	buildLightControls(message);
	updateTextureRow();
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

// Where the playback is, as a vertical line on every chart.
const cursorPlugin = {
	id: "replayCursor",
	afterDatasetsDraw(chart) {
		if (!replay.active || replay.cursorT === null) return;
		const x = chart.scales.x.getPixelForValue(replay.cursorT);
		const { top, bottom, left, right } = chart.chartArea;
		if (x < left || x > right) return;
		const ctx = chart.ctx;
		ctx.save();
		ctx.strokeStyle = "rgba(232, 232, 232, 0.7)";
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(x, top);
		ctx.lineTo(x, bottom);
		ctx.stroke();
		ctx.restore();
	},
};

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
			plugins: [cursorPlugin],
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
// that title; the side panels (Controls, Display, Lighting, Parameters) open
// and close with theirs. Everything starts collapsed, and what a visitor
// opens stays open in this browser, per scene ({id: true (collapsed) |
// false (open)}; no entry: collapsed).
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
	panel.classList.toggle("collapsed", collapsedPanels()[id] !== false);
	title.addEventListener("click", () => {
		const collapsed = panel.classList.toggle("collapsed");
		const state = collapsedPanels();
		state[id] = collapsed;
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
	if (!replay.active) showReadouts(scalars);
}

let shownScalars = {}; // what the readouts show (live, or the recording's frame)

function showReadouts(scalars) {
	shownScalars = scalars;
	updateGauge(scalars);
	updateHeat(scalars);
	for (const [key, { el, def }] of readoutEls) {
		const v = scalars[key];
		if (v === undefined || !Number.isFinite(v)) continue;
		el.textContent = `${v.toFixed(def.digits ?? 3)}${def.unit ? " " + def.unit : ""}`;
	}
}

// Heat maps (the scene's `heat`): models coloured by a probe, e.g. a tactile
// skin by the force on each pad: their own colour at 0, then blue, yellow,
// red at `max`. Switched in the Display panel.
const HEAT_STOPS = [[0.1, 0.35, 1.0], [1.0, 0.85, 0.1], [1.0, 0.12, 0.05]];

function heatColor(v) {
	const x = Math.max(0, Math.min(1, v)) * (HEAT_STOPS.length - 1);
	const i = Math.min(Math.floor(x), HEAT_STOPS.length - 2);
	const f = x - i;
	return HEAT_STOPS[i].map((c, k) => c + (HEAT_STOPS[i + 1][k] - c) * f);
}

function updateHeat(scalars) {
	const specs = sceneInfo?.heat || [];
	if (!specs.length || !setup) return;
	const on = displayState.heat !== false;
	for (const entry of models.values()) {
		if (entry.heat === undefined) { // which probe colours it (once per model)
			entry.heat = null;
			for (const h of specs) {
				h.re ||= new RegExp(h.models);
				const m = h.re.exec(entry.info.name || "");
				if (m) {
					entry.heat = { key: h.key.replace("{}", m[1] ?? ""), max: h.max || 1 };
					break;
				}
			}
		}
		if (!entry.heat || !entry.object.material?.color) continue;
		const v = on ? Number(scalars[entry.heat.key]) / entry.heat.max : 0;
		const [r, g, b] = entry.info.color;
		if (!(v > 0.01)) {
			entry.object.material.color.setRGB(r, g, b);
			continue;
		}
		const hot = heatColor(v);
		const k = Math.min(1, v * 5); // a light touch already shows
		entry.object.material.color.setRGB(r + (hot[0] - r) * k, g + (hot[1] - g) * k, b + (hot[2] - b) * k);
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
// The "recorded run" hint waits a few seconds, so it doesn't cover the
// scene the moment it loads: when it first applies, and whether it's due.
const RECORDED_HINT_DELAY_MS = 4000;
let recordedHintSince = null;
let recordedHintTimer = null;

// A message in the banner can be closed with its x, and hides itself after a while unless it is an error.
// Either way it stays away until the message changes or the situation ends, so it doesn't pop back on every status update.
const BANNER_AUTO_HIDE_MS = 20000;
let shownMessage = "";
let shownMessageSince = 0;
let dismissedMessage = "";
let bannerTimer = null;

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
	else if (s.recording?.cached) [text, cls] = ["recorded run", "finished"];
	else if (s.finished) [text, cls] = ["finished", "finished"];
	else if (s.paused) [text, cls] = ["paused", "paused"];
	statusEl.textContent = text;
	statusEl.className = cls;
	$("building").classList.toggle("hidden", s.status !== "building");
	$("auto-restart-checkbox").checked = !!s.autoRestart;
	updateTimeline();
	updateControls(s);
	updateAutoApply(s);

	let message = "";
	banner.className = "";
	if (s.error) {
		message = s.error;
		banner.className = "error";
	} else if (s.recording?.cached) {
		if (recordedHintSince === null) {
			recordedHintSince = performance.now();
			clearTimeout(recordedHintTimer);
			recordedHintTimer = setTimeout(renderStatus, RECORDED_HINT_DELAY_MS);
		}
		if (performance.now() - recordedHintSince >= RECORDED_HINT_DELAY_MS - 50) {
			message = "These parameters' run was recorded before (they're a preset's): play it back with ▶ or the timeline."
				+ (isViewer ? "" : " Simulate live runs the simulation itself.");
			banner.classList.add("fade-in");
		}
	} else if (s.finished && !s.autoRestart) {
		message = `Run finished (${s.endReason}).` + (isViewer ? " Replay it with the timeline." : " Replay it with the timeline, press ▶ to continue it, or ■ to go back to the start.");
	} else if (s.finished && s.autoRestart) {
		message = `Run finished (${s.endReason}) -- restarting...`;
	}
	if (!s.recording?.cached) {
		recordedHintSince = null;
		clearTimeout(recordedHintTimer);
	}
	if (!message) {
		shownMessage = "";
		dismissedMessage = "";
		clearTimeout(bannerTimer);
	} else if (message !== shownMessage) {
		shownMessage = message;
		shownMessageSince = performance.now();
		clearTimeout(bannerTimer);
		if (!s.error) bannerTimer = setTimeout(renderStatus, BANNER_AUTO_HIDE_MS + 50);
	}
	if (message && !s.error && performance.now() - shownMessageSince >= BANNER_AUTO_HIDE_MS) dismissedMessage = message;
	if (message === dismissedMessage) message = "";
	$("banner-text").textContent = message;
	banner.classList.toggle("hidden", !message);
}

$("banner-close").addEventListener("click", () => {
	dismissedMessage = shownMessage;
	renderStatus();
});

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
		livePositions.set(id, packed); // to come back to after a playback
		const entry = models.get(id);
		if (entry && !replay.active) applyPositions(entry, packed);
	}
	if (header.lights) liveLights = header.lights;
	liveHeader = header;
	if (!replay.active) {
		updateMirrorPlanes();
		if (header.lights) updateLights(header.lights);
		$("time").textContent = `t = ${header.t.toFixed(2)} s · step ${header.step ?? "--"}`;
	}
	const rtfEl = $("rtf");
	if (header.rtf > 0 && !header.paused && !header.finished) {
		rtfEl.textContent = `${header.rtf.toFixed(2)}× real time${header.fps > 0 ? ` · ${formatFps(header.fps)}` : ""}`;
		rtfEl.className = header.rtf < 0.95 ? "slow" : "";
	} else {
		rtfEl.textContent = "";
	}
	updateTelemetry(header.t, header.scalars || {}, header.full);
}

// Frames per second the server sends (one per step when the scene runs slower than real time).
function formatFps(fps) {
	return `${fps < 10 ? fps.toFixed(1) : Math.round(fps)} fps`;
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
// Controls: sliders and buttons that act on the running simulation (the
// scene config's `controls`). The owner moves them; everyone sees them move
// with the status messages.
// ---------------------------------------------------------------------------

const controlRows = new Map(); // key -> { control, row, show(value) }
const controlSends = new Map(); // key -> { busy, next } (a slider sends its latest value only)

function postControl(key, value) {
	return fetch("api/control", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ key, value }),
	}).catch(() => {});
}

function sendControl(key, value) {
	let state = controlSends.get(key);
	if (!state) controlSends.set(key, (state = { busy: false, next: undefined }));
	if (state.busy) {
		state.next = value;
		return;
	}
	state.busy = true;
	postControl(key, value).finally(() => {
		state.busy = false;
		if (state.next !== undefined) {
			const next = state.next;
			state.next = undefined;
			sendControl(key, next);
		}
	});
}

const fmtControl = (c, v) => `${Number(v).toFixed(c.digits ?? 2)}${c.unit ? " " + c.unit : ""}`;

function controlLabel(c, forId) {
	const label = document.createElement("label");
	label.textContent = c.label;
	if (forId) label.htmlFor = forId;
	return label;
}

function buildControlsPanel(controls) {
	const form = $("controls-form");
	form.replaceChildren();
	controlRows.clear();
	$("controls-panel").classList.toggle("hidden", !controls.length);
	for (const c of controls) {
		const row = document.createElement("div");
		row.className = `display-row control ${c.type}`;
		row.hidden = !!c.hidden; // set by the manipulator only (e.g. a clicked point)
		if (c.description) row.title = c.description;
		const id = `c-${c.key}`;
		let show = () => {};
		if (c.type === "slider") {
			const input = document.createElement("input");
			Object.assign(input, { type: "range", id, min: c.min, max: c.max, step: c.step ?? "any", value: c.default ?? c.min, disabled: isViewer });
			const value = document.createElement("span");
			value.className = "display-value";
			value.textContent = fmtControl(c, input.value);
			const reset = document.createElement("button");
			Object.assign(reset, { type: "button", className: "secondary mini-btn owner-only", textContent: "↺", title: `Back to ${fmtControl(c, c.default ?? c.min)}` });
			let grabbed = false;
			let keepAlive = null;
			input.addEventListener("pointerdown", () => {
				grabbed = true;
				// "deadman": the server lets it go back unless told again; it springs back when let go.
				if (c.deadman) keepAlive = setInterval(() => postControl(c.key, Number(input.value)), 800);
			});
			window.addEventListener("pointerup", () => {
				if (grabbed && c.deadman) {
					clearInterval(keepAlive);
					input.value = c.default ?? 0;
					value.textContent = fmtControl(c, input.value);
					sendControl(c.key, Number(input.value));
				}
				grabbed = false;
			});
			input.addEventListener("input", () => {
				value.textContent = fmtControl(c, input.value);
				sendControl(c.key, Number(input.value));
			});
			reset.addEventListener("click", () => {
				input.value = c.default ?? c.min;
				value.textContent = fmtControl(c, input.value);
				sendControl(c.key, Number(input.value));
			});
			show = (v) => {
				// Not while the owner is dragging it (the status lags behind).
				if (grabbed || controlSends.get(c.key)?.busy) return;
				input.value = v;
				value.textContent = fmtControl(c, v);
			};
			// "wide": the range input gets a row of its own, the panel's full width (finer control on a touchscreen).
			if (c.wide) row.classList.add("wide");
			row.append(controlLabel(c, id), input, value, reset);
		} else if (c.type === "toggle") {
			const label = document.createElement("label");
			label.className = "toggle-label";
			const input = document.createElement("input");
			Object.assign(input, { type: "checkbox", id, checked: !!c.default, disabled: isViewer });
			input.addEventListener("change", () => sendControl(c.key, input.checked));
			label.append(input, c.label);
			show = (v) => {
				if (!controlSends.get(c.key)?.busy) input.checked = !!v;
			};
			row.append(label);
		} else {
			// "hold": on while held down; "button": one action per click.
			const button = document.createElement("button");
			Object.assign(button, { type: "button", id, className: "control-btn", textContent: c.label, disabled: isViewer });
			if (c.type === "hold") {
				let held = false;
				let keepAlive = null;
				const press = () => {
					if (held) return;
					held = true;
					button.classList.add("active");
					sendControl(c.key, true);
					// The server lets go unless told again within a few seconds.
					keepAlive = setInterval(() => sendControl(c.key, true), 1000);
				};
				const release = () => {
					if (!held) return;
					held = false;
					clearInterval(keepAlive);
					button.classList.remove("active");
					sendControl(c.key, false);
				};
				button.addEventListener("pointerdown", (event) => {
					if (event.button !== 0) return;
					button.setPointerCapture(event.pointerId);
					press();
				});
				for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) button.addEventListener(type, release);
				button.addEventListener("keydown", (event) => {
					if ((event.key === " " || event.key === "Enter") && !event.repeat) {
						event.preventDefault();
						press();
					}
				});
				button.addEventListener("keyup", (event) => {
					if (event.key === " " || event.key === "Enter") release();
				});
				button.addEventListener("blur", release);
				window.addEventListener("blur", release);
				document.addEventListener("visibilitychange", () => document.hidden && release());
				button.addEventListener("contextmenu", (event) => event.preventDefault()); // long press on touch
				show = (v) => {
					if (!held) button.classList.toggle("active", !!v);
				};
			} else {
				button.addEventListener("click", () => {
					postControl(c.key, null);
					button.classList.add("active");
					setTimeout(() => button.classList.remove("active"), 150);
				});
			}
			row.append(button);
		}
		form.appendChild(row);
		controlRows.set(c.key, { control: c, row, show });
	}
}

// The scene's manipulator (SceneConfig.manipulator), if it has one: see manipulator.js.
let manipulator = null;

function buildManipulator(spec, controlSpecs) {
	if (!spec || manipulator) return;
	manipulator = createManipulator({
		scene,
		renderer,
		models,
		isViewer,
		camera: () => camera,
		orbit: () => controls,
		post: (values) => fetch("api/control", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ values }),
		}).catch(() => {}),
		show: (key, value) => controlRows.get(key)?.show(value),
		scalars: () => shownScalars,
		mount: (row, side) => {
			if (row) $("bottom").prepend(row);
			if (side) document.body.appendChild(side);
		},
	}, spec, controlSpecs);
}

function updateControls(s) {
	manipulator?.update(s);
	const usable = s.status === "running";
	for (const [key, { row, show }] of controlRows) {
		row.classList.toggle("disabled", !usable);
		if (s.controls && key in s.controls) show(s.controls[key]);
	}
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
			const input = document.createElement(field.type === "choice" ? "select" : "input");
			input.id = `p-${field.key}`;
			if (field.type === "choice") for (const [value, text] of field.choices) input.add(new Option(text, value));
			else if (field.type === "bool") input.type = "checkbox";
			else if (field.type === "vector") input.type = "text";
			else {
				input.type = "number";
				input.step = field.integer ? "1" : String(10 ** -Math.max(field.decimals ?? 3, 0));
				if (field.min !== undefined) input.min = field.min / (field.scale || 1);
				if (field.max !== undefined) input.max = field.max / (field.scale || 1);
			}
			if (isViewer) input.disabled = true;
			input.addEventListener(field.type === "bool" || field.type === "choice" ? "change" : "input", () => {
				// Switching on one of a pair that can't both be on switches the other off.
				if (field.type === "bool" && input.checked) {
					for (const other of field.excludes || []) {
						const entry = fieldInputs.get(other);
						if (entry && entry.input.checked) {
							entry.input.checked = false;
							entry.input.classList.add("dirty");
							setApplyEnabled(true);
						}
					}
				}
				if (field.live) return;
				input.classList.add("dirty");
				setApplyEnabled(true);
			});
			if (field.live) input.addEventListener("change", () => submitParams([field.key]));
			else input.addEventListener("change", () => { if (autoApplying()) applyDirty(); });
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
		else if (field.type === "choice") input.value = String(value);
		else if (field.type === "vector") input.value = value.map((v) => +v.toFixed(4)).join(", ");
		else input.value = fmtNumber(value, field);
		input.classList.remove("dirty");
	}
	setApplyEnabled(false);
	applyDisplay(); // mirrored copies depend on the symmetry parameters
	matchParamsPreset();
}

function readField({ field, input }) {
	if (field.type === "bool") return input.checked;
	if (field.type === "choice") return input.value;
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

// The Apply / Defaults / Auto apply row is at both ends of the (long) panel.
function setApplyEnabled(enabled) {
	for (const button of document.querySelectorAll(".apply-params-btn")) button.disabled = !enabled;
}

function applyDirty() {
	const dirty = [...fieldInputs].filter(([, { input }]) => input.classList.contains("dirty")).map(([key]) => key);
	if (dirty.length) submitParams(dirty);
}

$("params-form").addEventListener("submit", (event) => {
	event.preventDefault();
	document.activeElement?.blur();
	applyDirty();
});

for (const button of document.querySelectorAll(".default-params-btn")) {
	button.addEventListener("click", async () => {
		selectedPreset.params = "Default";
		try {
			showParams(await (await fetch("api/params/default", { method: "POST" })).json());
		} catch (err) {
			console.error("Failed to restore defaults", err);
		}
	});
}

// Auto apply: a change rebuilds the scene at once (as Apply & restart), but
// only while the simulation is stopped -- at its start, not running -- so a
// run is never cut short. Remembered in this browser.
const AUTO_APPLY_KEY = "sofaweb-auto-apply";
let autoApply = false;
try {
	autoApply = localStorage.getItem(AUTO_APPLY_KEY) === "1";
} catch {}
const isStopped = (s) => !!s && s.status === "running" && s.paused && !!s.atStart;
const autoApplying = () => autoApply && !isViewer && isStopped(lastStatus);

function updateAutoApply(s) {
	const usable = isStopped(s);
	for (const box of document.querySelectorAll(".auto-apply-checkbox")) {
		box.checked = autoApply;
		box.disabled = !usable;
		const label = box.closest(".auto-apply");
		label.classList.toggle("disabled", !usable);
		label.title = usable
			? "Apply each change at once, while the simulation is stopped"
			: "Only while the simulation is stopped (■): it would cut the run short";
	}
}

for (const box of document.querySelectorAll(".auto-apply-checkbox")) {
	box.addEventListener("change", () => {
		autoApply = box.checked;
		try {
			localStorage.setItem(AUTO_APPLY_KEY, autoApply ? "1" : "0");
		} catch {}
		updateAutoApply(lastStatus);
		if (autoApplying()) applyDirty();
	});
}
updateAutoApply(null);

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
		if (entry.texMaterial) entry.object.material = displayState.textures !== false ? entry.texMaterial : entry.plainMaterial;
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
			copy.mesh.material = entry.object.material;
			copy.mesh.renderOrder = entry.object.renderOrder;
			copy.mesh.visible = shown && alpha > 0.001;
			if (copy.wire) copy.wire.visible = showWire && shown;
		}
	}
	updateStrainReadout();
	updateHeat(shownScalars); // (the colours above are the models' own: the heat map on top)
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
	markPresetModified("display");
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
			markPresetModified("lighting");
			saveDisplay();
			applyLights();
		});
		box.appendChild(entry.brightness.row);
		if (light.id !== "ambient") {
			const setDirection = () => {
				displayState[`light:${light.id}:dir`] = [+entry.yaw.input.value, +entry.pitch.input.value];
				markPresetModified("lighting");
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

// Back to the scene's lights, or its saved lighting defaults, with a
// preset's values on top.
function resetLighting(overrides = {}) {
	for (const key of Object.keys(displayState)) if (key.startsWith("light:")) delete displayState[key];
	for (const key of Object.keys(savedDisplay)) if (key.startsWith("light:")) delete savedDisplay[key];
	for (const [key, value] of Object.entries({ ...(sceneInfo.displayDefaults || {}), ...overrides })) {
		if (key.startsWith("light:")) displayState[key] = savedDisplay[key] = value;
	}
	saveDisplay();
	applyLights();
	syncLightSliders();
}

$("lighting-reset-btn").addEventListener("click", () => {
	resetLighting();
	selectedPreset.lighting = "Default";
	presetModified.lighting = false;
	storePresetSelection();
	renderPresets("lighting");
});

// "Set as default" (admins only): stores a panel's current state as the
// scene's default in the orchestrator, for every viewer without settings of
// their own and every simulation of this scene started from now on.
// Behind the orchestrator this page is at <orchestrator root>sim/<id>/, where
// the root may itself be under a prefix (e.g. /sofa/).
const simIdMatch = location.pathname.match(/^(.*\/)sim\/([a-z0-9]+)\//);
const orchestratorRoot = simIdMatch ? simIdMatch[1] : "/";

async function checkAdmin() {
	if (!simIdMatch) return; // not behind the orchestrator
	try {
		const response = await fetch(`${orchestratorRoot}api/admin-session`, { cache: "no-store" });
		if (response.ok && (await response.json()).admin) document.body.classList.add("admin");
	} catch {}
}

// ---------------------------------------------------------------------------
// Presets: named states of the Display, Lighting and Parameters panels
// (api/scene's `presets`): "Default" (the scene's defaults), the scene's
// built-in ones, and an admin's own. Anyone can load a Display or Lighting
// preset, for themselves; a Parameters preset rebuilds the simulation, so
// only its owner loads one. Admins save, overwrite and delete presets for
// everyone (stored per scene in the orchestrator; "Default" can't be
// deleted).
// ---------------------------------------------------------------------------

const PRESET_SECTIONS = ["display", "lighting", "params"];
const PRESET_WHAT = { display: "these display settings", lighting: "these lighting settings", params: "the running parameters" };
// The preset each panel last loaded or saved (null: none, "custom"), and
// whether it's been changed since. Display and lighting are remembered
// per browser; the parameters' preset is found by comparing values.
const selectedPreset = { display: "Default", lighting: "Default", params: null };
const presetModified = { display: false, lighting: false, params: false };
const presetStoreKey = () => `sofaweb-presets:${sceneInfo.title}`;
const presetList = (section) => (sceneInfo?.presets || {})[section] || [];
const isAdmin = () => document.body.classList.contains("admin");

function restorePresetSelection() {
	try {
		const saved = JSON.parse(localStorage.getItem(presetStoreKey()) || "{}");
		for (const section of ["display", "lighting"]) {
			if (typeof saved[section] === "string" || saved[section] === null) selectedPreset[section] = saved[section];
			presetModified[section] = !!saved[`${section}:modified`];
		}
	} catch {}
}

function storePresetSelection() {
	try {
		localStorage.setItem(presetStoreKey(), JSON.stringify({
			display: selectedPreset.display, lighting: selectedPreset.lighting,
			"display:modified": presetModified.display, "lighting:modified": presetModified.lighting,
		}));
	} catch {}
}

function renderPresets(section) {
	const row = document.querySelector(`.preset-row[data-section="${section}"]`);
	const select = $(`${section}-preset`);
	const list = presetList(section);
	let name = selectedPreset[section];
	if (!list.some((p) => p.name === name)) name = null;
	select.replaceChildren();
	if (name === null) select.appendChild(new Option("(custom)", ""));
	for (const preset of list) {
		const option = new Option(preset.name + (preset.name === name && presetModified[section] ? " (modified)" : ""), preset.name);
		if (preset.description) option.title = preset.description;
		select.appendChild(option);
	}
	select.value = name ?? "";
	const current = list.find((p) => p.name === name);
	select.title = current?.description || `Load a named preset of ${PRESET_WHAT[section]}`;
	row.querySelector(".preset-save").disabled = name === null;
	row.querySelector(".preset-delete").disabled = name === null || name === "Default";
	// Loading parameters is the owner's; a watching admin picks a preset only to save to it.
	select.disabled = section === "params" && isViewer && !isAdmin();
	const canImport = section === "params" && !isViewer && (sceneInfo?.imports || []).length > 0;
	row.querySelector(".preset-import")?.classList.toggle("hidden", !canImport);
	row.classList.toggle("hidden", list.length <= 1 && !isAdmin() && !canImport);
}

function renderAllPresets() {
	for (const section of PRESET_SECTIONS) renderPresets(section);
}

function markPresetModified(section) {
	if (presetModified[section] || selectedPreset[section] === null) return;
	presetModified[section] = true;
	storePresetSelection();
	renderPresets(section);
}

function sameValue(a, b) {
	if (Array.isArray(a) || Array.isArray(b)) {
		return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => sameValue(v, b[i]));
	}
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
	return a === b;
}

// Which parameters preset the running simulation matches: the one last
// chosen if it still does, else the first that does; or the last chosen,
// marked modified; or none.
function matchParamsPreset() {
	const list = presetList("params");
	if (!list.length) return;
	const defaults = list[0].values;
	const matches = (preset) => {
		const values = { ...defaults, ...preset.values };
		return [...fieldInputs.keys()].every((key) => !(key in values) || !(key in currentParams) || sameValue(values[key], currentParams[key]));
	};
	const chosen = list.find((p) => p.name === selectedPreset.params);
	if (chosen && matches(chosen)) presetModified.params = false;
	else {
		const found = list.find(matches);
		if (found) {
			selectedPreset.params = found.name;
			presetModified.params = false;
		} else presetModified.params = !!chosen;
	}
	renderPresets("params");
}

async function loadPreset(section, name) {
	const preset = presetList(section).find((p) => p.name === name);
	if (!preset) return;
	selectedPreset[section] = name;
	presetModified[section] = false;
	if (section === "params") {
		if (isViewer) return renderPresets(section); // a watching admin: only the target for Save
		try {
			const response = await fetch("api/params/preset", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ name }),
			});
			if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
			showParams(await response.json());
		} catch (err) {
			alert(`Couldn't load the preset: ${err.message}`);
			matchParamsPreset();
		}
		return;
	}
	if (section === "display") resetDisplay(preset.values);
	else resetLighting(preset.values);
	storePresetSelection();
	renderPresets(section);
}

function presetValues(section) {
	if (section === "params") {
		// The running simulation's parameters (not unapplied edits in the form).
		return Object.fromEntries(Object.entries(currentParams).filter(([key]) => fieldInputs.has(key)));
	}
	if (section === "lighting") {
		// Only what's been changed here: an untouched light keeps following the scene.
		return Object.fromEntries(Object.entries(displayState).filter(([key]) => key.startsWith("light:")));
	}
	// The view as it is now, if it's been moved from the scene's (the last
	// move may not be remembered yet).
	if (displayState.camera || cameraSaveTimer !== null) displayState.camera = currentCameraPose();
	clearTimeout(cameraSaveTimer);
	cameraSaveTimer = null;
	return Object.fromEntries(
		Object.entries(displayState).filter(([key]) => !key.startsWith("light:") && key !== "overlay:femStrainPeak"),
	);
}

// Importing another scene's parameters preset (api/scene's `imports`, see
// SceneConfig.import_presets): converted on the server, applied like a
// preset of this scene; the dialog then lists what couldn't be carried over.
const importSources = () => sceneInfo?.imports || [];

function fillImportPresets() {
	const source = importSources().find((s) => s.scene === $("import-scene").value);
	const select = $("import-preset");
	select.replaceChildren(...(source?.presets || []).map((p) => {
		const option = new Option(p.name, p.name);
		if (p.description) option.title = p.description;
		return option;
	}));
	showImportDescription();
}

function showImportDescription() {
	const source = importSources().find((s) => s.scene === $("import-scene").value);
	const preset = source?.presets.find((p) => p.name === $("import-preset").value);
	$("import-desc").textContent = preset?.description || "";
}

function openImportDialog() {
	const sources = importSources();
	if (!sources.length) return;
	$("import-scene").replaceChildren(...sources.map((s) => new Option(s.title, s.scene)));
	fillImportPresets();
	$("import-pick").classList.remove("hidden");
	$("import-result").classList.add("hidden");
	$("import-go").classList.remove("hidden");
	$("import-go").disabled = false;
	$("import-cancel").textContent = "Cancel";
	$("import-title").textContent = "Import a parameters preset";
	$("import-dialog").showModal();
}

function showImportResult(result) {
	const box = $("import-result");
	box.replaceChildren();
	const ok = document.createElement("p");
	ok.className = "import-ok";
	ok.textContent = `Imported "${result.preset}" from ${result.source}; the simulation was rebuilt with it.`;
	box.appendChild(ok);
	const groups = [
		["not imported", `Not imported (no counterpart in ${sceneInfo.title})`],
		["estimated", "Estimated"],
		["assumed", "Assumed"],
	];
	let any = false;
	for (const [kind, title] of groups) {
		const notes = result.notes.filter((n) => n.kind === kind);
		if (!notes.length) continue;
		any = true;
		const h = document.createElement("h4");
		h.textContent = title;
		const ul = document.createElement("ul");
		for (const note of notes) {
			const li = document.createElement("li");
			const b = document.createElement("strong");
			b.textContent = note.label;
			li.append(b, `: ${note.text}`);
			ul.appendChild(li);
		}
		box.append(h, ul);
	}
	if (!any) {
		const p = document.createElement("p");
		p.textContent = "Every parameter was carried over directly.";
		box.appendChild(p);
	}
	$("import-title").textContent = "Imported";
	$("import-pick").classList.add("hidden");
	box.classList.remove("hidden");
	$("import-go").classList.add("hidden");
	$("import-cancel").textContent = "Close";
}

async function runImport() {
	const scene = $("import-scene").value;
	const name = $("import-preset").value;
	if (!scene || !name) return;
	$("import-go").disabled = true;
	try {
		const response = await fetch("api/params/import", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ scene, name }),
		});
		const body = await response.json();
		if (!response.ok) throw new Error(body?.detail?.message || `${response.status} ${response.statusText}`);
		selectedPreset.params = null;
		showParams(body.params);
		showImportResult(body);
	} catch (err) {
		$("import-go").disabled = false;
		alert(`Couldn't import the preset: ${err.message}`);
	}
}

$("import-scene").addEventListener("change", fillImportPresets);
$("import-preset").addEventListener("change", showImportDescription);
$("import-go").addEventListener("click", runImport);
$("import-cancel").addEventListener("click", () => $("import-dialog").close());
document.querySelector('.preset-row[data-section="params"] .preset-import').addEventListener("click", openImportDialog);

async function refreshPresets() {
	try {
		const info = await (await fetch("api/scene", { cache: "no-store" })).json();
		sceneInfo.presets = info.presets;
		sceneInfo.displayDefaults = info.displayDefaults;
	} catch {}
	renderAllPresets();
	matchParamsPreset();
}

async function presetRequest(section, name, method, values) {
	const response = await fetch(
		`${orchestratorRoot}admin/api/sims/${simIdMatch[2]}/presets/${section}/${encodeURIComponent(name)}`,
		{ method, headers: { "Content-Type": "application/json" }, body: values ? JSON.stringify({ values }) : undefined },
	);
	const body = await response.json().catch(() => ({}));
	if (!response.ok) throw new Error(body?.detail?.message || `${response.status} ${response.statusText}`);
	return body;
}

async function savePreset(section, name, button) {
	const what = PRESET_WHAT[section];
	const question = name === "Default"
		? `Make ${what} the default for this scene? It applies to everyone, and to every simulation of this scene started from now on.`
		: `Save ${what} as the preset "${name}"? It changes it for everyone.`;
	if (!confirm(question)) return;
	const label = button.textContent;
	button.disabled = true;
	try {
		const body = await presetRequest(section, name, "PUT", presetValues(section));
		selectedPreset[section] = body.name || name;
		presetModified[section] = false;
		if (section !== "params") storePresetSelection();
		await refreshPresets();
		button.textContent = "Saved ✓";
		setTimeout(() => (button.textContent = label), 2000);
	} catch (err) {
		alert(`Couldn't save the preset: ${err.message}`);
	} finally {
		button.disabled = false;
		renderPresets(section);
	}
}

for (const section of PRESET_SECTIONS) {
	const row = document.querySelector(`.preset-row[data-section="${section}"]`);
	const select = $(`${section}-preset`);
	select.addEventListener("change", () => {
		if (select.value) loadPreset(section, select.value);
	});
	row.querySelector(".preset-save").addEventListener("click", (event) => {
		if (selectedPreset[section] !== null) savePreset(section, selectedPreset[section], event.currentTarget);
	});
	row.querySelector(".preset-save-as").addEventListener("click", (event) => {
		const entered = prompt(`Name of the new preset (${PRESET_WHAT[section]}):`, "");
		const name = (entered || "").trim().replace(/\s+/g, " ");
		if (!name) return;
		if (name.length > 60) return alert("A preset name has at most 60 characters.");
		const existing = presetList(section).find((p) => p.name.toLowerCase() === name.toLowerCase());
		savePreset(section, existing ? existing.name : name, event.currentTarget);
	});
	row.querySelector(".preset-delete").addEventListener("click", async () => {
		const name = selectedPreset[section];
		const preset = presetList(section).find((p) => p.name === name);
		if (!preset || name === "Default") return;
		const note = preset.builtin ? " It's built into the scene; deleting it hides it." : "";
		if (!confirm(`Delete the preset "${name}" for everyone?${note}`)) return;
		try {
			await presetRequest(section, name, "DELETE");
			selectedPreset[section] = null;
			if (section !== "params") storePresetSelection();
			await refreshPresets();
		} catch (err) {
			alert(`Couldn't delete the preset: ${err.message}`);
		}
	});
}

checkAdmin().then(renderAllPresets);

function buildDisplayPanel(spec, useSaved = true, overrides = null) {
	displaySpec = spec.map((d) => ({ ...d, re: d.models ? new RegExp(d.models) : null }));
	// The scene's defaults (an admin's "Set as default"), under this
	// browser's own saved settings; Reset goes back to just the defaults.
	let saved = { ...(sceneInfo.displayDefaults || {}) };
	if (useSaved) {
		try {
			Object.assign(saved, JSON.parse(localStorage.getItem(displayStoreKey()) || "{}"));
		} catch {}
	}
	if (overrides) Object.assign(saved, overrides);
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
	if (sceneInfo.hasGauge) {
		displayState.showGauge = "showGauge" in saved ? !!saved.showGauge : true;
		form.appendChild(checkboxRow("showGauge", "Pressure gauge (manometer, tube)"));
	}
	if (sceneInfo.view !== "2d") {
		// The scale ruler (see updateRuler), for every 3D scene.
		displayState.scaleTicks = "scaleTicks" in saved ? !!saved.scaleTicks : false;
		form.appendChild(checkboxRow("scaleTicks", "Axis scale (ticks, labels)"));
		// Surface textures, where the scene has them (shown once the setup
		// says so, see updateTextureRow); off draws plain colours, faster.
		displayState.textures = "textures" in saved ? !!saved.textures : true;
		if ((sceneInfo.heat || []).length) {
			displayState.heat = "heat" in saved ? !!saved.heat : true;
			form.appendChild(checkboxRow("heat", sceneInfo.heat[0].label || "Heat map"));
		}
		const row = checkboxRow("textures", "Textures (surfaces; off is faster)");
		row.id = "display-textures-row";
		form.appendChild(row);
		updateTextureRow();
	}
	updateDisplayAvailability();
}

$("display-reset-btn").addEventListener("click", () => {
	resetDisplay();
	selectedPreset.display = "Default";
	presetModified.display = false;
	storePresetSelection();
	renderPresets("display");
});

// The Display panel's "Textures" only for a scene with textured models.
function updateTextureRow() {
	const row = $("display-textures-row");
	if (row) row.hidden = !setup || !setup.models.some((m) => m.texture);
}

// Back to the scene's default display settings, with a preset's values on top.
function resetDisplay(overrides = null) {
	try {
		localStorage.removeItem(displayStoreKey());
	} catch {}
	buildDisplayPanel(sceneInfo.display || [], false, overrides);
	overlaySignature = null; // the Lighting panel keeps its own settings (its own Reset)
	if (setup) buildLightControls(setup);
	if (controls) {
		sceneView();
		applyCameraPose(displayState.camera); // the default view
	}
	applyDisplay();
	resetStrainPeak();
	if (overrides) saveDisplay();
}

// ---------------------------------------------------------------------------
// Playback: every run is recorded on the server (api/recording, see
// recording.py). While the simulation is paused or finished, the timeline
// plays the recording back here, in this browser only -- watchers too. A
// preset's run comes from the server's cache: then nothing is simulated
// unless the owner asks (Simulate live).
// ---------------------------------------------------------------------------

const livePositions = new Map(); // id -> latest live positions (packed)
let liveLights = null;
let liveHeader = null;
const replay = {
	rec: null, // { generation, frames: [{ t, step, scalars, lights, arrays: Map id -> Float32Array }], grid }
	active: false, // showing a recorded frame instead of the live state
	playing: false,
	index: 0,
	clock: 0, // playback position (simulated s)
	lastTick: null,
	cursorT: null,
	fetching: false,
	retryAt: 0, // after a failed fetch, not again before this (performance.now())
	shown: new Map(), // id -> the recorded (or blended) array on screen
	blend: new Map(), // id -> scratch array for positions between two frames
};

function resetReplay() {
	replay.rec = null;
	replay.active = false;
	replay.playing = false;
	replay.cursorT = null;
	replay.shown.clear();
	replay.blend.clear();
	updateTransport();
}

const timelineAvailable = () => {
	const s = lastStatus;
	return !!(s && connected && s.status === "running" && (s.paused || s.finished) && s.recording && s.recording.frames >= 2);
};

// The timeline's text while the recording isn't there yet.
function setLoading(text) {
	replay.loadingText = text;
	if (!(replay.rec && replay.rec.frames.length)) $("replay-time").textContent = text;
}

async function fetchRecording() {
	if (replay.fetching || performance.now() < replay.retryAt) return;
	replay.fetching = true;
	replay.retryAt = performance.now() + 2000; // cleared on success
	try {
		const wanted = lastStatus?.recording?.id;
		const have = replay.rec && replay.rec.generation === generation && replay.rec.id === wanted ? replay.rec.frames.length : 0;
		// While no frames are shown yet, the timeline says what's going on: the server
		// compresses the recording first (nothing arrives meanwhile), then it downloads.
		const began = performance.now();
		const waiting = setInterval(() => {
			if (!have) setLoading(`loading the recording: preparing it on the server, ${Math.round((performance.now() - began) / 1000)} s...`);
		}, 400);
		if (!have) setLoading("loading the recording: preparing it on the server...");
		let response;
		try {
			response = await fetch(`api/recording?start=${have}`, { cache: "no-store" });
		} finally {
			clearInterval(waiting);
		}
		if (response.status !== 200) return;
		const gen = Number(response.headers.get("X-Recording-Generation"));
		const start = Number(response.headers.get("X-Recording-Start"));
		const id = Number(response.headers.get("X-Recording-Id"));
		const total = Number(response.headers.get("X-Recording-Bytes") || response.headers.get("Content-Length")) || 0;
		const chunks = [];
		let received = 0;
		const reader = response.body.getReader();
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			chunks.push(value);
			received += value.length;
			if (!have) {
				const mb = (n) => (n / 1048576).toFixed(1);
				setLoading(total ? `loading the recording: ${Math.floor((100 * received) / total)} % (${mb(received)} of ${mb(total)} MB)` : `loading the recording: ${mb(received)} MB...`);
			}
		}
		if (!have) setLoading("loading the recording: unpacking...");
		const buffer = await new Response(new Blob(chunks).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
		if (gen !== generation) return; // rebuilt meanwhile
		const sameRecording = replay.rec && replay.rec.id === id;
		const wasAtEnd = !sameRecording || replay.index >= replay.rec.frames.length - 1;
		decodeRecording(buffer, gen, sameRecording ? start : 0, id);
		if (!replay.rec || !replay.rec.frames.length) return;
		replay.retryAt = 0;
		chartsFromRecording();
		if (!timelineAvailable()) return;
		// Start where the simulation is: a cached run, or the run before
		// Stop (the scene is back at its start), at its start; a run that's
		// paused or finished at its end.
		const frames = replay.rec.frames;
		const t = lastStatus.recording.cached || lastStatus.recording.previous
			? (replay.active && sameRecording ? replay.clock : frames[0].t)
			: wasAtEnd || !replay.active ? frames[frames.length - 1].t : replay.clock;
		replay.active = true;
		showRecordedTime(t);
	} catch (err) {
		console.error("Loading the recording failed", err);
		setLoading("couldn't load the recording; trying again...");
	} finally {
		replay.fetching = false;
		replay.loadingText = null;
		updateTimeline();
	}
}

function decodeRecording(buffer, gen, start, id) {
	const view = new DataView(buffer);
	const headerLength = view.getUint32(0, true);
	const header = JSON.parse(decoder.decode(new Uint8Array(buffer, 4, headerLength)));
	let offset = 4 + headerLength;
	offset += (4 - (offset % 4)) % 4;
	if (start === 0 || !replay.rec || replay.rec.generation !== gen || replay.rec.id !== id) {
		replay.rec = { generation: gen, id, frames: [], grid: new Map(), lights: null };
		replay.shown.clear();
	}
	const rec = replay.rec;
	if (start !== rec.frames.length) {
		replay.rec = null; // out of step: fetch it whole next time
		return;
	}
	for (const f of header.frames) {
		const previous = rec.frames.length ? rec.frames[rec.frames.length - 1].arrays : null;
		const arrays = new Map(previous || []);
		for (const [id, count, enc] of f.m) {
			const n = count * 3;
			let grid;
			if (enc === 0) {
				grid = new Int32Array(buffer, offset, n);
				offset += n * 4;
			} else {
				const delta = enc === 1 ? new Int16Array(buffer, offset, n) : new Int32Array(buffer, offset, n);
				offset += n * (enc === 1 ? 2 : 4);
				const before = rec.grid.get(id);
				grid = new Int32Array(n);
				for (let i = 0; i < n; i++) grid[i] = before[i] + delta[i];
			}
			offset += (4 - (offset % 4)) % 4;
			rec.grid.set(id, grid);
			const q = header.q[id];
			const positions = new Float32Array(n);
			for (let i = 0; i < n; i++) positions[i] = grid[i] * q;
			arrays.set(id, positions);
		}
		if ("l" in f) rec.lights = f.l;
		rec.frames.push({ t: f.t, step: f.k, scalars: f.s || {}, lights: rec.lights, arrays });
	}
}

// The charts from the recording: complete, also for a run this browser
// didn't watch (e.g. a preset's run from the cache).
function chartsFromRecording() {
	const frames = replay.rec.frames;
	for (const { def, chart } of charts) {
		def.series.forEach((series, i) => {
			const data = [];
			for (const f of frames) {
				const v = f.scalars[series.key];
				if (v !== undefined && Number.isFinite(v)) data.push({ x: f.t, y: v });
			}
			chart.data.datasets[i].data = data.slice(-MAX_CHART_POINTS);
		});
	}
	lastChartT = frames[frames.length - 1].t;
	chartsDirty = true;
}

// The recorded state at simulated time t. Between two stored frames the
// positions and readouts are interpolated linearly, so playback (also
// slowed down) and scrubbing move smoothly whatever the frame rate.
function showRecordedTime(t) {
	const rec = replay.rec;
	if (!rec || !rec.frames.length) return;
	const frames = rec.frames;
	const last = frames.length - 1;
	t = Math.max(frames[0].t, Math.min(frames[last].t, t));
	let lo = 0, hi = last; // the last frame at or before t
	while (lo < hi) {
		const mid = (lo + hi + 1) >> 1;
		if (frames[mid].t <= t) lo = mid;
		else hi = mid - 1;
	}
	const frame = frames[lo];
	const next = lo < last ? frames[lo + 1] : null;
	const alpha = next && next.t > frame.t ? (t - frame.t) / (next.t - frame.t) : 0;
	replay.index = lo;
	replay.clock = t;
	replay.cursorT = t;
	for (const [id, a] of frame.arrays) {
		const entry = models.get(id);
		if (!entry) continue;
		const b = next?.arrays.get(id);
		if (alpha > 0 && b && b !== a && b.length === a.length) {
			let out = replay.blend.get(id);
			if (!out || out.length !== a.length) replay.blend.set(id, (out = new Float32Array(a.length)));
			for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * alpha;
			applyPositions(entry, out);
			replay.shown.set(id, out);
		} else if (replay.shown.get(id) !== a) {
			applyPositions(entry, a);
			replay.shown.set(id, a);
		}
	}
	updateMirrorPlanes();
	if (frame.lights) updateLights(frame.lights);
	let scalars = frame.scalars;
	if (alpha > 0) {
		scalars = { ...frame.scalars };
		for (const [key, v] of Object.entries(next.scalars)) {
			const u = frame.scalars[key];
			if (Number.isFinite(u) && Number.isFinite(v)) scalars[key] = u + (v - u) * alpha;
		}
	}
	showReadouts(scalars);
	const step = next ? Math.round(frame.step + (next.step - frame.step) * alpha) : frame.step;
	$("time").textContent = `t = ${t.toFixed(3)} s · step ${step} · playback`;
	$("scrub").value = t;
	$("replay-time").textContent = `${t.toFixed(2)} / ${frames[last].t.toFixed(2)} s`;
	chartsDirty = true;
}

// Back to the simulation's own state (it's running again, or rebuilt).
function goLive() {
	if (!replay.active) return;
	replay.active = false;
	replay.playing = false;
	replay.cursorT = null;
	replay.shown.clear();
	for (const [id, packed] of livePositions) {
		const entry = models.get(id);
		if (entry) applyPositions(entry, packed);
	}
	updateMirrorPlanes();
	if (liveLights) updateLights(liveLights);
	if (liveHeader) {
		showReadouts(liveHeader.scalars || {});
		$("time").textContent = `t = ${liveHeader.t.toFixed(2)} s · step ${liveHeader.step ?? "--"}`;
	}
	chartsDirty = true;
	updateTransport();
}

function toggleReplay() {
	const rec = replay.rec;
	if (!rec || rec.frames.length < 2 || !timelineAvailable()) return;
	if (replay.playing) {
		replay.playing = false;
	} else {
		const end = rec.frames[rec.frames.length - 1].t;
		if (!replay.active || replay.clock >= end - 1e-9) showRecordedTime(rec.frames[0].t); // from the start
		replay.active = true;
		replay.playing = true;
		replay.lastTick = null;
	}
	updateTransport();
}

function replayTick() {
	if (!replay.playing) return;
	const rec = replay.rec;
	if (!rec || !replay.active) {
		replay.playing = false;
		return updateTransport();
	}
	const now = performance.now();
	const frames = rec.frames;
	const end = frames[frames.length - 1].t;
	let clock = replay.clock;
	if (replay.lastTick !== null) clock += ((now - replay.lastTick) / 1000) * Number($("replay-speed").value);
	replay.lastTick = now;
	if (clock !== replay.clock) showRecordedTime(clock);
	if (clock >= end - 1e-9) {
		if (lastStatus?.autoRestart) {
			// Auto restart: the playback loops.
			showRecordedTime(frames[0].t);
		} else {
			replay.playing = false;
			updateTransport();
		}
	}
}

// The play buttons: the HUD's controls the simulation (or, for a preset's
// recorded run, the playback); the timeline's always the playback.
function updateTransport() {
	const s = lastStatus;
	const cached = !!s?.recording?.cached;
	if (cached) showPlayState($("play-btn"), replay.playing, "Play the recorded run");
	else if (s) showPlayState($("play-btn"), !(s.paused || s.finished), s.finished ? "Continue the finished run" : "Play");
	showPlayState($("replay-btn"), replay.playing, "Play the recording");
	updatePlayHint(s, cached);
}

// First visit (per browser): the simulation opens paused, so a callout under ▶ says how to start it. It goes
// once the simulation runs or is closed, and isn't shown again.
const PLAY_HINT_KEY = "sofaweb:playHintSeen";
let playHint = null;

function playHintSeen() {
	try {
		return !!localStorage.getItem(PLAY_HINT_KEY);
	} catch {
		return false;
	}
}

function dismissPlayHint() {
	try {
		localStorage.setItem(PLAY_HINT_KEY, "1");
	} catch {}
	playHint?.remove();
	playHint = null;
}

function placePlayHint() {
	if (!playHint) return;
	const r = $("play-btn").getBoundingClientRect();
	playHint.style.left = `${Math.max(8, Math.min(r.left + r.width / 2 - 24, window.innerWidth - playHint.offsetWidth - 8))}px`;
	playHint.style.top = `${r.bottom + 10}px`;
	playHint.style.setProperty("--arrow-x", `${r.left + r.width / 2 - parseFloat(playHint.style.left)}px`);
}

function updatePlayHint(s, cached) {
	if (isViewer || !s || s.status !== "running" || cached) return;
	if (!(s.paused || s.finished)) {
		if (playHint) dismissPlayHint(); // they found it
		return;
	}
	if (playHint || playHintSeen() || s.finished) return;
	playHint = document.createElement("div");
	playHint.id = "play-hint";
	playHint.setAttribute("role", "status");
	const text = document.createElement("span");
	text.innerHTML = "The simulation is open, but <b>paused</b>: press <b>▶</b> to start it. ■ takes it back to the start.";
	const close = document.createElement("button");
	Object.assign(close, { type: "button", className: "close-btn", textContent: "×", title: "Close" });
	close.setAttribute("aria-label", "Close");
	close.addEventListener("click", dismissPlayHint);
	playHint.append(text, close);
	document.body.appendChild(playHint);
	placePlayHint();
}
window.addEventListener("resize", placePlayHint);

function updateTimeline() {
	const available = timelineAvailable();
	const timeline = $("timeline");
	timeline.classList.toggle("hidden", !available);
	if (!available) {
		goLive();
		updateTransport();
		return;
	}
	const status = lastStatus.recording;
	const rec = replay.rec;
	if (!rec || rec.generation !== lastStatus.generation || rec.id !== status.id || rec.frames.length < status.frames) {
		if (lastStatus.generation === generation) fetchRecording();
	}
	const ready = rec && rec.generation === generation && rec.id === status.id;
	const frames = ready ? rec.frames.length : 0;
	// The slider is in simulated seconds, so it can stop between frames.
	$("scrub").min = frames ? rec.frames[0].t : 0;
	$("scrub").max = frames ? rec.frames[frames - 1].t : 0;
	if (frames) $("scrub").value = replay.clock;
	$("scrub").disabled = frames < 2;
	$("replay-btn").disabled = frames < 2;
	if (!frames) $("replay-time").textContent = replay.loadingText || "loading the recording...";
	$("simulate-btn").classList.toggle("hidden", !status.cached);
	let note = "this session only";
	if (status.previous) note = "the run before ■ (until ▶)";
	else if (status.cached) note = "recorded run of a preset";
	else if (status.saved) note = "saved for this preset";
	else if (status.truncated) note = "recording stopped: too long";
	$("timeline-note").textContent = note;
	$("timeline-note").title = status.previous
		? "The run that was stopped: kept for playback until the simulation runs again or the parameters change"
		: status.cached || status.saved
		? "Kept on the server: simulations of these preset parameters play it back instead of simulating"
		: "Kept until the parameters change or the simulation ends: these parameters aren't a preset's";
	updateTransport();
}

$("scrub").addEventListener("input", () => {
	if (!replay.rec) return;
	replay.active = true;
	showRecordedTime(Number($("scrub").value));
	replay.lastTick = null;
	updateTransport();
});
$("replay-btn").addEventListener("click", toggleReplay);
$("simulate-btn").addEventListener("click", () => {
	fetch("api/simulate", { method: "POST" }).catch(() => {});
});
window.addEventListener("keydown", (event) => {
	// Space plays / pauses the playback, unless typing in a field.
	if (event.key !== " " || event.target.closest?.("input, select, textarea, button")) return;
	if (!$("timeline").classList.contains("hidden")) {
		event.preventDefault();
		toggleReplay();
	}
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
	// A preset's recorded run: ▶ plays it back (Simulate live runs it).
	if (lastStatus?.recording?.cached) return toggleReplay();
	const resume = lastStatus && (lastStatus.paused || lastStatus.finished);
	fetch(resume ? "api/resume" : "api/pause", { method: "POST" }).catch(() => {});
});
$("stop-btn").addEventListener("click", async () => {
	try {
		await fetch("api/pause", { method: "POST" });
		await fetch("api/reset", { method: "POST" });
	} catch {}
});
// On a phone the top bar keeps the title, play and stop; "..." shows the rest.
$("hud-toggle").addEventListener("click", () => {
	const open = $("hud").classList.toggle("open");
	$("hud-toggle").setAttribute("aria-expanded", String(open));
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
		add("Speed", info.rtf > 0 ? `${info.rtf.toFixed(2)}× real time${info.fps > 0 ? `, ${formatFps(info.fps)}` : ""}` : "not running");
		add("Visual models", info.models.filter((m) => m.kind !== "tetra").map((m) => `${m.name} (${m.vertices} vertices)`).join(", "));
		if (info.fem && info.fem.length) {
			add("FEM elements", info.fem.map((m) => `${m.elements.toLocaleString("en")} tetrahedra, ${m.nodes.toLocaleString("en")} nodes` + (info.fem.length > 1 ? ` (${m.name.replace(/^FEM:/, "")})` : "")).join("; "));
		}
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

function initSidePanels() {
	for (const panel of document.querySelectorAll("#side-panels details")) {
		const id = `side:${panel.id}`;
		panel.open = collapsedPanels()[id] === false; // open only if the visitor opened it before
		panel.addEventListener("toggle", () => {
			const state = collapsedPanels();
			state[id] = !panel.open;
			try {
				localStorage.setItem(panelStoreKey(), JSON.stringify(state));
			} catch {}
		});
	}
}

// Text fields: no browser autofill (on a tablet, the bar with the key, bank card and map pin icons above the
// keyboard), no auto-capitalisation or spell checking. Covers fields built later too.
function noAutofill(root) {
	const fields = root.matches?.("input, textarea") ? [root, ...root.querySelectorAll("input, textarea")] : root.querySelectorAll?.("input, textarea") || [];
	for (const el of fields) {
		if (["checkbox", "range", "color", "button"].includes(el.type)) continue;
		el.setAttribute("autocomplete", "off");
		el.setAttribute("autocapitalize", "off");
		el.setAttribute("autocorrect", "off");
		el.setAttribute("spellcheck", "false");
		el.setAttribute("data-lpignore", "true"); // password managers
		el.setAttribute("data-1p-ignore", "true");
		if (el.type === "number" && !el.hasAttribute("inputmode")) el.setAttribute("inputmode", "decimal");
	}
	if (root.tagName === "FORM") root.setAttribute("autocomplete", "off");
}
noAutofill(document.body);
new MutationObserver((records) => {
	for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1) noAutofill(node);
}).observe(document.body, { childList: true, subtree: true });

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
	restorePresetSelection();
	renderAllPresets();
	initSidePanels();
	buildControlsPanel(sceneInfo.controls || []);
	buildManipulator(sceneInfo.manipulator, sceneInfo.controls || []);
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
window.sofaweb = { scene, models, renderer, replay, get camera() { return camera; }, get setup() { return setup; }, get scalars() { return shownScalars; } };

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

// ---------------------------------------------------------------------------
// Scale ruler: ticks and labels along the three world axes through the world
// origin (where a scene's own X / Y arrows start), so the size of what's on
// screen can be judged -- PneuNets of 5 mm and of 50 cm
// look alike. The tick interval (1, 2 or 5 x a power of ten) follows the zoom:
// about eight ticks across the view. Scenes are SI (metres); labels show m, mm
// or um according to the interval. Every 3D scene gets it, with a switch in the
// Display panel; labels keep their size on screen.
// ---------------------------------------------------------------------------

const rulerGroup = new THREE.Group();
scene.add(rulerGroup);
let rulerKey = null;
const rulerTextures = new Map();
const RULER_TICKS_ACROSS = 8;
const AXIS_NAME_COLORS = ["#d64545", "#2e9e4f", "#3a5bd9"];

function niceInterval(raw) {
	const power = 10 ** Math.floor(Math.log10(raw));
	const n = raw / power;
	return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * power;
}

function rulerUnit(interval) {
	if (interval >= 0.1) return { f: 1, name: "m" };
	if (interval >= 1e-4) return { f: 1e-3, name: "mm" };
	return { f: 1e-6, name: "µm" };
}

function formatTick(value, unit, interval) {
	const decimals = Math.max(0, Math.min(6, Math.ceil(-Math.log10(interval / unit.f) - 1e-9)));
	const text = (value / unit.f).toFixed(decimals);
	return /^-0(\.0+)?$/.test(text) ? text.slice(1) : text;
}

function visibleHeight() {
	if (camera.isOrthographicCamera) return (camera.top - camera.bottom) / camera.zoom;
	return 2 * camera.position.distanceTo(controls.target) * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
}

// Dark ink with a light halo on a light background, and the other way round.
function rulerInk() {
	const bg = scene.background;
	const light = bg ? 0.2126 * bg.r + 0.7152 * bg.g + 0.0722 * bg.b > 0.5 : false;
	return light ? { fg: "#1b1d22", halo: "rgba(255,255,255,0.85)" } : { fg: "#e8e8e8", halo: "rgba(0,0,0,0.65)" };
}

function rulerLabel(text, ink, color = null) {
	const key = `${text}|${ink.fg}|${color}`;
	let entry = rulerTextures.get(key);
	if (!entry) {
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");
		const font = "bold 44px system-ui, sans-serif";
		ctx.font = font;
		canvas.width = Math.ceil(ctx.measureText(text).width) + 20;
		canvas.height = 64;
		ctx.font = font;
		ctx.textAlign = "center";
		ctx.textBaseline = "middle";
		ctx.lineJoin = "round";
		ctx.lineWidth = 8;
		ctx.strokeStyle = ink.halo;
		ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
		ctx.fillStyle = color || ink.fg;
		ctx.fillText(text, canvas.width / 2, canvas.height / 2);
		const texture = new THREE.CanvasTexture(canvas);
		texture.colorSpace = THREE.SRGBColorSpace;
		entry = { texture, aspect: canvas.width / canvas.height };
		rulerTextures.set(key, entry);
	}
	const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: entry.texture, depthTest: false, depthWrite: false, transparent: true }));
	sprite.renderOrder = 5001;
	sprite.userData.aspect = entry.aspect;
	return sprite;
}

function clearRuler() {
	for (const child of [...rulerGroup.children]) {
		rulerGroup.remove(child);
		if (child.isSprite) child.material.dispose();
		else {
			child.geometry.dispose();
			child.material.dispose();
		}
	}
}

function updateRuler() {
	const on = !!setup && !!controls && setup.view !== "2d" && displayState.scaleTicks !== false;
	const up = setup ? setup.up || [0, 0, 1] : [0, 0, 1];
	const upIdx = up.findIndex((v) => Math.abs(v) > 0.99);
	if (!on || upIdx < 0) {
		rulerGroup.visible = false;
		return;
	}
	rulerGroup.visible = true;
	const horiz = [0, 1, 2].filter((i) => i !== upIdx);
	// Each axis, and the horizontal one its ticks and labels extend along (towards negative).
	const axes = [
		{ axis: horiz[0], side: horiz[1] },
		{ axis: horiz[1], side: horiz[0] },
		{ axis: upIdx, side: horiz[0] },
	];
	const height = visibleHeight();
	if (!(height > 0) || !Number.isFinite(height)) return;
	const interval = niceInterval(height / RULER_TICKS_ACROSS);
	const unit = rulerUnit(interval);
	const ink = rulerInk();
	const reach = 0.7 * height * Math.max(1, window.innerWidth / window.innerHeight);
	const ranges = axes.map(({ axis }) => {
		const c = controls.target.getComponent(axis);
		const k0 = Math.floor((c - reach) / interval);
		return [k0, Math.min(Math.ceil((c + reach) / interval), k0 + 60)];
	});
	const key = [interval, upIdx, unit.name, ink.fg, ...ranges.flat()].join("|");
	if (key !== rulerKey) {
		rulerKey = key;
		clearRuler();
		const tick = 0.08 * interval;
		axes.forEach(({ axis, side }, n) => {
			const [k0, k1] = ranges[n];
			const point = (along, across) => {
				const v = [0, 0, 0]; // the world origin's other coordinates
				v[axis] = along;
				v[side] = across;
				return v;
			};
			const lines = [...point(k0 * interval, 0), ...point(k1 * interval, 0)];
			const dir = new THREE.Vector3();
			dir.setComponent(side, -1);
			for (let k = k0; k <= k1; k++) {
				const along = k * interval;
				lines.push(...point(along, tick), ...point(along, -tick));
				if (k === 0 && n > 0) continue; // the origin is labelled once
				const sprite = rulerLabel(formatTick(along, unit, interval), ink);
				sprite.userData.base = new THREE.Vector3(...point(along, 0));
				sprite.userData.dir = dir;
				sprite.userData.tick = tick;
				sprite.userData.axisN = n;
				rulerGroup.add(sprite);
			}
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
			const material = new THREE.LineBasicMaterial({ color: ink.fg, transparent: true, opacity: 0.75, depthTest: false, depthWrite: false });
			const line = new THREE.LineSegments(geometry, material);
			line.renderOrder = 5000;
			rulerGroup.add(line);
			// The axis name and unit, beyond the last tick.
			const name = rulerLabel(`${"XYZ"[axis]} (${unit.name})`, ink, AXIS_NAME_COLORS[axis]);
			name.userData.base = new THREE.Vector3(...point((k1 + 0.5) * interval, 0));
			name.userData.dir = new THREE.Vector3();
			name.userData.name = true;
			rulerGroup.add(name);
		});
	}
	// Labels keep their size on screen, and sit just off the axis. Where an axis is seen at a grazing angle
	// its labels would pile up: a tick label is skipped when it would land on the previous one shown.
	const labelHeight = 0.026 * height;
	const lastShown = [null, null, null];
	camera.updateMatrixWorld();
	for (const child of rulerGroup.children) {
		if (!child.isSprite) continue;
		const h = child.userData.name ? 1.15 * labelHeight : labelHeight;
		child.scale.set(h * child.userData.aspect, h, 1);
		child.position.copy(child.userData.base);
		if (!child.userData.name) child.position.addScaledVector(child.userData.dir, child.userData.tick + 0.7 * h);
		child.visible = true;
		const n = child.userData.axisN;
		if (child.userData.name || n === undefined) continue;
		const ndc = child.position.clone().project(camera);
		const x = (ndc.x * window.innerWidth) / 2;
		const y = (ndc.y * window.innerHeight) / 2;
		const min = 0.045 * Math.min(window.innerWidth, window.innerHeight); // about a label's height on screen, with a gap
		const previous = lastShown[n];
		if (previous && Math.hypot(x - previous[0], y - previous[1]) < min) child.visible = false;
		else lastShown[n] = [x, y];
	}
}

// ---------------------------------------------------------------------------
// Pressure gauge: an analog manometer on a tube into the model (see the scene
// config's `gauge`), so the pressure can be read off the 3D scene itself. The
// needle follows the scene's pressure scalar, live and in playback. Drawn here
// from the layout in the setup message; nothing is simulated.
// ---------------------------------------------------------------------------

const gaugeRoot = new THREE.Group();
scene.add(gaugeRoot);
let gauges = []; // [{ info, needle }], one per gauge of the scene (a gripper has one per finger)
// A gauge can follow a moving model of the scene (`follow`: a regex for its name, `followRest`: the centre of
// its bounding box at rest): the dials and tubes are shifted by how far that centre has moved.
let gaugeFollow = null;
const GAUGE_SWEEP = (270 * Math.PI) / 180; // the needle's travel, from lower left over the top to lower right

function gaugeFace(g) {
	const size = 1024;
	const canvas = document.createElement("canvas");
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext("2d");
	const c = size / 2;
	ctx.fillStyle = "#f7f5ee";
	ctx.beginPath();
	ctx.arc(c, c, c, 0, 2 * Math.PI);
	ctx.fill();
	// A value on the scale: the direction (from the dial's centre) of its tick.
	const dir = (value) => {
		const f = (value - g.min) / (g.max - g.min);
		const a = (0.5 - f) * GAUGE_SWEEP;
		return [-Math.sin(a), -Math.cos(a)]; // canvas y runs down
	};
	const line = (value, from, to, width, color) => {
		const [dx, dy] = dir(value);
		ctx.strokeStyle = color;
		ctx.lineWidth = width;
		ctx.beginPath();
		ctx.moveTo(c + dx * c * from, c + dy * c * from);
		ctx.lineTo(c + dx * c * to, c + dy * c * to);
		ctx.stroke();
	};
	const text = (value, radius, color, label) => {
		const [dx, dy] = dir(value);
		ctx.fillStyle = color;
		ctx.fillText(label, c + dx * c * radius, c + dy * c * radius);
	};
	const number = (v) => String(Math.round(v * 1000) / 1000);
	const eps = 1e-9 * Math.abs(g.max - g.min);
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	// Two scales, in two colours: the scene's unit (kPa) round the outside, in blue; bar (100 kPa = 1 bar)
	// inside it, in black. Red is left out: it's associated with psi.
	const KPA = "#1d5fb8";
	const BAR = "#111111";
	const barPerUnit = g.unit === "kPa" ? 0.01 : null;
	for (let v = g.min; v <= g.max + eps; v += g.minor) line(v, 0.89, 0.96, 5, KPA);
	ctx.font = `bold ${size * 0.075}px system-ui, sans-serif`;
	for (let v = g.min; v <= g.max + eps; v += g.major) {
		line(v, 0.81, 0.96, 10, KPA);
		text(v, 0.7, KPA, number(v));
	}
	if (barPerUnit) {
		for (let v = g.min; v <= g.max + eps; v += g.minor) line(v, 0.6, 0.64, 4, BAR);
		ctx.font = `bold ${size * 0.062}px system-ui, sans-serif`;
		for (let v = g.min; v <= g.max + eps; v += g.major) {
			line(v, 0.55, 0.64, 7, BAR);
			text(v, 0.45, BAR, number(v * barPerUnit));
		}
	}
	// The units, one under the other, in their scales' colours.
	ctx.font = `bold ${size * 0.07}px system-ui, sans-serif`;
	ctx.fillStyle = KPA;
	ctx.fillText(g.unit, c, c + c * 0.27);
	if (barPerUnit) {
		ctx.fillStyle = BAR;
		ctx.fillText("bar", c, c + c * 0.4);
	}
	ctx.fillStyle = "#333333";
	ctx.font = `${size * 0.04}px system-ui, sans-serif`;
	ctx.fillText("PRESSURE", c, c + c * (barPerUnit ? 0.53 : 0.4));
	const texture = new THREE.CanvasTexture(canvas);
	texture.colorSpace = THREE.SRGBColorSpace;
	texture.anisotropy = 4;
	return texture;
}

function clearGauge() {
	for (const child of [...gaugeRoot.children]) {
		gaugeRoot.remove(child);
		child.traverse((o) => {
			if (o.geometry) o.geometry.dispose();
			if (o.material) {
				if (o.material.map) o.material.map.dispose();
				o.material.dispose();
			}
		});
	}
	gauges = [];
	gaugeFollow = null;
	gaugeRoot.position.set(0, 0, 0);
}

function buildGauge(layout) {
	clearGauge();
	for (const g of Array.isArray(layout) ? layout : layout ? [layout] : []) {
		if (g && g.radius > 0 && g.max > g.min) buildOneGauge(g);
	}
	updateGauge({});
}

function buildOneGauge(g) {
	const R = g.radius;
	// The dial: local x = right, local y = up, local z = towards the viewer.
	const right = new THREE.Vector3(...g.right).normalize();
	const upv = new THREE.Vector3(...g.up).normalize();
	const normal = new THREE.Vector3().crossVectors(right, upv).normalize();
	const dial = new THREE.Group();
	dial.matrixAutoUpdate = false;
	dial.matrix.makeBasis(right, upv, normal).setPosition(...g.center);
	const casing = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.12, R * 1.12, R * 0.3, 64).rotateX(Math.PI / 2), new THREE.MeshLambertMaterial({ color: 0x3a3e47 }));
	const rim = new THREE.Mesh(new THREE.TorusGeometry(R * 1.04, R * 0.07, 12, 64), new THREE.MeshLambertMaterial({ color: 0xb8bcc6 }));
	rim.position.z = R * 0.15;
	// The face and the needle stand clear of the casing's top (z = 0.15 R), and are pulled towards the
	// camera by a polygon offset, so they don't z-fight with it when the view is zoomed out.
	const face = new THREE.Mesh(new THREE.CircleGeometry(R, 64), new THREE.MeshBasicMaterial({ map: gaugeFace(g), polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8 }));
	face.position.z = R * 0.19;
	face.renderOrder = 10;
	const needle = new THREE.Group();
	needle.position.z = R * 0.23;
	const shape = new THREE.Shape([new THREE.Vector2(0, R * 0.86), new THREE.Vector2(R * 0.045, 0), new THREE.Vector2(0, -R * 0.2), new THREE.Vector2(-R * 0.045, 0)]);
	needle.add(new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: 0xe8750a, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -16 })));
	needle.children[0].renderOrder = 11;
	const hub = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.08, R * 0.08, R * 0.05, 24).rotateX(Math.PI / 2), new THREE.MeshLambertMaterial({ color: 0x222222 }));
	hub.position.z = R * 0.03;
	needle.add(hub);
	dial.add(casing, rim, face, needle);
	gauges.push({ info: g, needle });
	if (g.follow && g.followRest && !gaugeFollow) gaugeFollow = { re: new RegExp(g.follow), rest: new THREE.Vector3(...g.followRest), entry: null };
	gaugeRoot.add(dial);
	if (g.tube && g.tube.length > 1) {
		const curve = new THREE.CatmullRomCurve3(g.tube.map((p) => new THREE.Vector3(...p)), false, "centripetal");
		const radius = g.tubeRadius || R * 0.15;
		const material = new THREE.MeshLambertMaterial({ color: 0xf2c94c });
		gaugeRoot.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 120, radius, 14, false), material));
		// Flat caps on the tube's two ends (the open end at the model would look hollow).
		for (const [t, flip] of [[0, -1], [1, 1]]) {
			const cap = new THREE.Mesh(new THREE.CircleGeometry(radius, 14), material);
			const at = curve.getPoint(t);
			cap.position.copy(at);
			cap.lookAt(at.clone().addScaledVector(curve.getTangent(t), flip));
			gaugeRoot.add(cap);
		}
	}
}

function updateGauge(scalars) {
	for (const { info, needle } of gauges) {
		const value = scalars[info.key];
		if (value === undefined || !Number.isFinite(value)) continue;
		const f = Math.min(1, Math.max(0, (value - info.min) / (info.max - info.min)));
		needle.rotation.z = (0.5 - f) * GAUGE_SWEEP; // from straight up, clockwise as the pressure rises
	}
}

const _followCentre = new THREE.Vector3();

function animate() {
	requestAnimationFrame(animate);
	replayTick();
	if (chartsDirty) {
		for (const { chart } of charts) chart.update("none");
		chartsDirty = false;
	}
	if (controls) controls.update();
	manipulator?.frame();
	updateRuler();
	gaugeRoot.visible = gauges.length > 0 && displayState.showGauge !== false;
	if (gaugeFollow) {
		if (!gaugeFollow.entry || !models.has(gaugeFollow.entry.info.id)) {
			gaugeFollow.entry = [...models.values()].find((e) => gaugeFollow.re.test(e.info.name)) || null;
		}
		const entry = gaugeFollow.entry;
		if (entry) {
			entry.geometry.computeBoundingBox();
			entry.geometry.boundingBox.getCenter(_followCentre);
			gaugeRoot.position.copy(_followCentre).sub(gaugeFollow.rest);
		}
	}
	sortTransparent();
	renderer.render(scene, camera);
}
animate();
