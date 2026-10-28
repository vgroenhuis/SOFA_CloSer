// Surface textures for a scene's visual models (MuJoCo visuals with a
// `texture`, see mjscene.MjScene.add_visual): procedural greyscale patterns
// drawn here once, so nothing is downloaded, that darken and lighten the
// model's own colour (a texel of 0.5 leaves it as it is).
//
// A texture is mapped triplanar in the model's bodies' own frames (the
// setup's `texture.local`, the vertices' positions at rest in their
// bodies): it moves with a moving part instead of sliding over it, and no
// UV coordinates are needed. Each face takes the projection along the axis
// its normal is closest to (one texture lookup a pixel: cheap).

import * as THREE from "three";

const SIZE = 256; // texels per side of every texture

function random(seed) {
	let s = seed >>> 0;
	return () => {
		s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
		return s / 4294967296;
	};
}

const smooth = (t) => t * t * (3 - 2 * t);

// Value noise in [0, 1] that tiles: `cx` x `cy` random lattice cells across
// the texture, smoothly interpolated (cx != cy stretches it, e.g. fibres).
function noise(rnd, cx, cy = cx) {
	const lattice = Float32Array.from({ length: cx * cy }, rnd);
	const out = new Float32Array(SIZE * SIZE);
	for (let y = 0; y < SIZE; y++) {
		const fy = (y / SIZE) * cy;
		const y0 = Math.floor(fy);
		const ty = smooth(fy - y0);
		const r0 = (y0 % cy) * cx;
		const r1 = ((y0 + 1) % cy) * cx;
		for (let x = 0; x < SIZE; x++) {
			const fx = (x / SIZE) * cx;
			const x0 = Math.floor(fx);
			const tx = smooth(fx - x0);
			const c0 = x0 % cx;
			const c1 = (x0 + 1) % cx;
			const a = lattice[r0 + c0] + (lattice[r0 + c1] - lattice[r0 + c0]) * tx;
			const b = lattice[r1 + c0] + (lattice[r1 + c1] - lattice[r1 + c0]) * tx;
			out[y * SIZE + x] = a + (b - a) * ty;
		}
	}
	return out;
}

// Octaves of noise from `cells` cells up, centred on 0 (about -0.5 .. 0.5).
function fbm(rnd, cells, octaves = 4) {
	const out = new Float32Array(SIZE * SIZE);
	let amp = 1;
	let total = 0;
	for (let o = 0; o < octaves; o++) {
		const n = noise(rnd, Math.min(cells << o, SIZE));
		for (let i = 0; i < out.length; i++) out[i] += (n[i] - 0.5) * amp;
		total += amp;
		amp *= 0.5;
	}
	for (let i = 0; i < out.length; i++) out[i] /= total;
	return out;
}

// `out` = 1 + sum of weight * pattern, per texel.
function mix(...terms) {
	const out = new Float32Array(SIZE * SIZE).fill(1);
	for (const [weight, pattern] of terms) for (let i = 0; i < out.length; i++) out[i] += weight * pattern[i];
	return out;
}

// Discs (wrapping around the edges) setting the texels under them to `value`.
function dots(out, centres, radius, value, soft = 1) {
	for (const [cx, cy, r = radius, v = value] of centres) {
		const reach = Math.ceil(r + soft);
		for (let dy = -reach; dy <= reach; dy++) {
			for (let dx = -reach; dx <= reach; dx++) {
				const d = Math.hypot(dx, dy);
				const k = Math.min(1, Math.max(0, (r + soft - d) / soft));
				if (k <= 0) continue;
				const x = (((Math.round(cx) + dx) % SIZE) + SIZE) % SIZE;
				const y = (((Math.round(cy) + dy) % SIZE) + SIZE) % SIZE;
				const i = y * SIZE + x;
				out[i] += (v - out[i]) * k;
			}
		}
	}
	return out;
}

function scatter(rnd, count, radius, values) {
	return Array.from({ length: count }, () => [rnd() * SIZE, rnd() * SIZE, radius * (0.5 + rnd()), values[Math.floor(rnd() * values.length)]]);
}

// Every texture: how big one repeat is [m] (unless the scene says) and its
// texels (around 1: 1 keeps the colour, 0 is black). They're scaled to a
// mean of 1, so a model looks as bright as without it, unless `keepMean`.
const TEXTURES = {
	// Powder coat: a fine orange peel.
	painted: { scale: 0.05, make: (rnd) => mix([0.06, fbm(rnd, 16, 3)]) },
	// Brushed metal: streaks along the texture's u.
	brushed: { scale: 0.08, make: (rnd) => mix([0.14, streaks(rnd)], [0.04, fbm(rnd, 32, 2)]) },
	// A shelf's back panel: perforated, 25 mm pitch.
	pegboard: {
		scale: 0.1,
		make: (rnd) => {
			const out = mix([0.05, fbm(rnd, 16, 3)]);
			const pitch = SIZE / 4;
			const holes = [];
			for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) holes.push([(i + 0.5) * pitch, (j + 0.5) * pitch]);
			return dots(out, holes, pitch * 0.13, 0.22, 1.5);
		},
	},
	// Diamond tread plate: raised ridges in alternating directions.
	treadplate: {
		scale: 0.08,
		make: (rnd) => {
			const out = mix([0.06, fbm(rnd, 8, 3)]);
			const cell = SIZE / 4;
			for (let y = 0; y < SIZE; y++) {
				for (let x = 0; x < SIZE; x++) {
					const i = Math.floor(x / cell);
					const j = Math.floor(y / cell);
					const u = (x % cell) / cell - 0.5;
					const v = (y % cell) / cell - 0.5;
					const s = (i + j) % 2 ? 1 : -1;
					const a = (u + s * v) / Math.SQRT2; // along the ridge
					const b = (u - s * v) / Math.SQRT2; // across it
					const r = (a / 0.32) ** 2 + (b / 0.07) ** 2;
					if (r < 1) out[y * SIZE + x] *= b > 0 ? 1.35 : 0.7; // lit side, shaded side
				}
			}
			return out;
		},
	},
	// Hazard stripes: dark diagonal bands (on a yellow model, yellow and black).
	hazard: {
		scale: 0.16,
		keepMean: true, // darker on the whole: the black bands
		make: (rnd) => {
			const grain = fbm(rnd, 16, 2);
			return Float32Array.from({ length: SIZE * SIZE }, (_, i) => {
				const t = (((i % SIZE) + Math.floor(i / SIZE)) / SIZE) * 2;
				return (t - Math.floor(t) < 0.5 ? 1.05 : 0.12) + 0.05 * grain[i];
			});
		},
	},
	// Rubber: a dull speckle and grooves.
	rubber: {
		scale: 0.05,
		make: (rnd) => {
			const out = mix([0.12, fbm(rnd, 64, 2)]);
			for (let y = 0; y < SIZE; y++) if (y % (SIZE / 8) < 5) for (let x = 0; x < SIZE; x++) out[y * SIZE + x] *= 0.55;
			return out;
		},
	},
	// Corrugated board: fibres, blotches and flecks.
	cardboard: {
		scale: 0.12,
		make: (rnd) => dots(mix([0.1, noise(rnd, 6, 96).map((v) => v - 0.5)], [0.12, fbm(rnd, 6, 3)], [0.05, fbm(rnd, 64, 1)]),
			scatter(rnd, 90, 0.8, [0.8, 1.15]), 1, 1, 0.8),
	},
	// Paper and printed card: almost smooth.
	paper: { scale: 0.05, make: (rnd) => mix([0.05, fbm(rnd, 64, 2)], [0.04, fbm(rnd, 6, 2)]) },
	// Moulded plastic: a faint grain.
	plastic: { scale: 0.03, make: (rnd) => mix([0.05, fbm(rnd, 64, 2)]) },
	// Cast silicone: matte, slightly cloudy.
	silicone: { scale: 0.04, make: (rnd) => mix([0.1, fbm(rnd, 4, 3)], [0.05, fbm(rnd, 96, 1)]) },
	// Woven fabric (a finger's strain-limiting layer): 16 threads per repeat.
	fabric: {
		scale: 0.012,
		make: (rnd) => {
			const grain = fbm(rnd, 64, 1);
			const n = 16;
			return Float32Array.from({ length: SIZE * SIZE }, (_, i) => {
				const x = ((i % SIZE) / SIZE) * n;
				const y = (Math.floor(i / SIZE) / SIZE) * n;
				const over = (Math.floor(x) + Math.floor(y)) % 2 === 0;
				const t = over ? Math.sin(Math.PI * (x % 1)) : Math.sin(Math.PI * (y % 1));
				return 0.72 + 0.32 * t + 0.08 * grain[i];
			});
		},
	},
	// 3D-printed part: layer lines along v.
	printed: {
		scale: 0.0096,
		make: (rnd) => {
			const grain = fbm(rnd, 32, 2);
			return Float32Array.from({ length: SIZE * SIZE }, (_, i) => 1 + 0.12 * Math.cos((2 * Math.PI * 32 * Math.floor(i / SIZE)) / SIZE) + 0.04 * grain[i]);
		},
	},
	// Terrazzo floor: chips in a mottled ground.
	terrazzo: {
		scale: 0.3,
		make: (rnd) => dots(mix([0.07, fbm(rnd, 6, 4)]), scatter(rnd, 350, 1.1, [0.86, 0.92, 1.07, 1.1]), 1, 1, 0.7),
	},
	// Eggshell: a fine mottle with brown specks.
	eggshell: { scale: 0.03, make: (rnd) => dots(mix([0.06, fbm(rnd, 8, 3)], [0.03, fbm(rnd, 96, 1)]), scatter(rnd, 60, 0.9, [0.82]), 1, 1, 0.8) },
	// Fruit skin: streaks along v, blush, pale pores.
	skin: {
		scale: 0.04,
		make: (rnd) => dots(mix([0.14, noise(rnd, 48, 3).map((v) => v - 0.5)], [0.12, fbm(rnd, 4, 3)]), scatter(rnd, 140, 0.8, [1.3]), 1, 1, 0.8),
	},
	// A banana's peel: the fruit skin with its streaks along u (lengthwise on
	// a banana lying along x).
	peel: {
		scale: 0.06,
		make: (rnd) => dots(mix([0.14, noise(rnd, 3, 48).map((v) => v - 0.5)], [0.1, fbm(rnd, 4, 3)]), scatter(rnd, 90, 0.8, [0.75]), 1, 1, 0.8),
	},
	// Foil wrapping: crinkles.
	foil: {
		scale: 0.05,
		make: (rnd) => {
			const a = fbm(rnd, 10, 4);
			const b = fbm(rnd, 24, 2);
			return Float32Array.from(a, (v, i) => 1 + 0.25 * v - 0.18 * Math.abs(b[i]));
		},
	},
};

// Streaks along u: one value per row, varying a little along it.
function streaks(rnd) {
	const rows = noise(rnd, 1, 160);
	const drift = noise(rnd, 4, 40);
	return Float32Array.from(rows, (v, i) => v - 0.5 + 0.4 * (drift[i] - 0.5));
}

const cache = new Map(); // name -> THREE texture
let anisotropy = 1;

export function setAnisotropy(value) {
	anisotropy = Math.max(1, value | 0);
}

export function hasTexture(name) {
	return Object.prototype.hasOwnProperty.call(TEXTURES, name);
}

function texture(name) {
	if (cache.has(name)) return cache.get(name);
	const def = TEXTURES[name];
	let seed = 7;
	for (const c of name) seed = (Math.imul(seed, 31) + c.charCodeAt(0)) >>> 0;
	const values = def.make(random(seed));
	if (!def.keepMean) {
		const mean = values.reduce((a, v) => a + v, 0) / values.length;
		for (let i = 0; i < values.length; i++) values[i] /= mean;
	}
	const data = new Uint8Array(SIZE * SIZE * 4);
	for (let i = 0; i < values.length; i++) {
		const v = Math.round(Math.min(1, Math.max(0, values[i] * 0.5)) * 255);
		data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = v;
		data[i * 4 + 3] = 255;
	}
	const tex = new THREE.DataTexture(data, SIZE, SIZE, THREE.RGBAFormat);
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = true;
	tex.anisotropy = anisotropy;
	tex.colorSpace = THREE.NoColorSpace;
	tex.needsUpdate = true;
	cache.set(name, tex);
	return tex;
}

function decodeFloats(base64) {
	const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
	return new Float32Array(bytes.buffer);
}

// -- labels --------------------------------------------------------------
// A package's printed label (`texture.label`: {mode, seed}): a made-up
// brand and a second line in fonts picked at random, the same for the same
// seed. Drawn on transparent, over the model's own colour; 'plane' prints it
// on the mesh's flat face (the side its thinnest extent points to, away
// from the body's origin), 'wrap' around its local z axis (twice round),
// 'box' on every face of a box (upright on tall faces, as on a carton).

const FONTS = [
	"Georgia, serif", "Impact, 'Arial Black', sans-serif", "'Courier New', monospace", "'Comic Sans MS', cursive",
	"'Trebuchet MS', sans-serif", "'Palatino Linotype', Palatino, serif", "'Brush Script MT', cursive",
	"Verdana, sans-serif", "fantasy", "'Times New Roman', serif", "'Lucida Console', monospace", "cursive",
];
const SYLLABLES = ["ka", "lo", "mi", "zu", "bre", "ta", "vo", "ri", "nel", "sa", "po", "gri", "fa", "do", "lin",
	"mar", "qui", "to", "ber", "chi", "ola", "ex", "fru", "na", "ve", "ro", "lu", "sto", "py", "ma"];
const SECOND = ["Original", "Classic", "Fresh", "Premium", "Organic", "Extra", "Natural", "Light", "Gold", "Select",
	"Finest", "Daily", "Pure", "Farm", "Deluxe", "Zesty"];

function word(rnd) {
	let w = "";
	const n = 2 + Math.floor(rnd() * 2);
	for (let i = 0; i < n; i++) w += SYLLABLES[Math.floor(rnd() * SYLLABLES.length)];
	return w[0].toUpperCase() + w.slice(1);
}

// Writes `text` centred at (x, y), as big as fits in w x h.
function fitText(ctx, text, font, x, y, w, h) {
	ctx.font = `${font.style} ${font.weight} 100px ${font.family}`;
	const m = ctx.measureText(text);
	const size = Math.max(4, Math.min(100 * (w / Math.max(m.width, 1)), h));
	ctx.font = `${font.style} ${font.weight} ${size}px ${font.family}`;
	ctx.fillText(text, x, y);
}

function labelCanvas(seed, width, height, repeat, base, upright = false) {
	const rnd = random([...String(seed)].reduce((a, c) => (Math.imul(a, 33) + c.charCodeAt(0)) >>> 0, 5381));
	const pick = (list) => list[Math.floor(rnd() * list.length)];
	const font = () => ({ family: pick(FONTS), weight: rnd() < 0.6 ? "bold" : "normal", style: rnd() < 0.25 ? "italic" : "normal" });
	const brand = word(rnd);
	const second = rnd() < 0.5 ? pick(SECOND) : word(rnd);
	const brandFont = font();
	const secondFont = font();
	// Ink that shows on the part: dark on a light part, light on a dark one.
	const light = 0.3 * base[0] + 0.59 * base[1] + 0.11 * base[2] > 0.5;
	const hue = Math.floor(rnd() * 360);
	const ink = light ? `hsl(${hue}, 75%, ${22 + Math.floor(rnd() * 14)}%)` : `hsl(${hue}, 60%, 92%)`;
	const accent = `hsl(${(hue + 150 + Math.floor(rnd() * 60)) % 360}, 70%, ${light ? 45 : 70}%)`;
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d");
	ctx.textAlign = "center";
	ctx.textBaseline = "middle";
	if (upright) {
		// Reading upwards: drawn on a canvas turned a quarter left.
		ctx.translate(0, height);
		ctx.rotate(-Math.PI / 2);
		[width, height] = [height, width];
	}
	const cell = width / repeat;
	for (let r = 0; r < repeat; r++) {
		const cx = (r + 0.5) * cell;
		ctx.fillStyle = accent;
		ctx.fillRect(cx - cell * 0.42, height * 0.14, cell * 0.84, Math.max(1, height * 0.03));
		ctx.fillRect(cx - cell * 0.42, height * 0.83, cell * 0.84, Math.max(1, height * 0.03));
		ctx.fillStyle = ink;
		fitText(ctx, brand, brandFont, cx, height * 0.43, cell * 0.8, height * 0.42);
		ctx.fillStyle = accent;
		fitText(ctx, second, secondFont, cx, height * 0.7, cell * 0.6, height * 0.16);
	}
	return canvas;
}

const labelCache = new Map();

function labelTexture(seed, width, height, repeat, base, upright = false) {
	const key = [seed, width, height, repeat, upright, ...base.map((c) => c.toFixed(2))].join("|");
	if (labelCache.has(key)) return labelCache.get(key);
	const tex = new THREE.CanvasTexture(labelCanvas(seed, width, height, repeat, base, upright));
	tex.colorSpace = THREE.NoColorSpace;
	tex.anisotropy = anisotropy;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	labelCache.set(key, tex);
	return tex;
}

// Where the label goes, from the mesh's extent in its bodies' frames:
// centre, the label's right and up directions over its width and height,
// and its face's normal; null if the mode is unknown.
function labelPlacement(label, local, base) {
	const lo = [Infinity, Infinity, Infinity];
	const hi = [-Infinity, -Infinity, -Infinity];
	for (let i = 0; i < local.length; i += 3) {
		for (let k = 0; k < 3; k++) {
			lo[k] = Math.min(lo[k], local[i + k]);
			hi[k] = Math.max(hi[k], local[i + k]);
		}
	}
	const centre = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
	const size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
	if (label.mode === "wrap") {
		const height = Math.max(size[2], 1e-6);
		const around = Math.PI * Math.max(size[0], size[1]);
		const width = 1024;
		const rows = Math.round(Math.min(512, Math.max(32, (width / around) * height * 2)));
		return { mode: 2, centre, u: new THREE.Vector3(), v: new THREE.Vector3(0, 0, 1 / height), n: new THREE.Vector3(), map: labelTexture(label.seed, width, rows, 2, base) };
	}
	if (label.mode === "box") {
		// One label per pair of opposite faces (x, y, z): w x h as the face
		// is seen (x and y faces upright along z, z faces with y up).
		const faces = [[size[1], size[2]], [size[0], size[2]], [size[0], size[1]]];
		const maps = faces.map(([w, h]) => {
			const long = 512;
			const [cols, rows] = w >= h ? [long, Math.round(Math.max(32, (long * h) / w))] : [Math.round(Math.max(32, (long * w) / h)), long];
			return labelTexture(label.seed, cols, rows, 1, base, h > 1.5 * w);
		});
		const inverse = new THREE.Vector3(...size.map((c) => 1 / Math.max(c, 1e-6)));
		return { mode: 3, centre, u: inverse, v: new THREE.Vector3(), n: new THREE.Vector3(), map: maps[0], maps };
	}
	if (label.mode !== "plane") return null;
	const thin = size.indexOf(Math.min(...size));
	const n = new THREE.Vector3().setComponent(thin, centre.getComponent(thin) >= 0 ? 1 : -1);
	const up = thin === 2 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
	const right = new THREE.Vector3().crossVectors(up, n);
	const w = Math.max(Math.abs(right.dot(new THREE.Vector3(...size))), 1e-6);
	const h = Math.max(Math.abs(up.dot(new THREE.Vector3(...size))), 1e-6);
	const width = 512;
	const rows = Math.round(Math.min(512, Math.max(32, (width * h) / w)));
	return { mode: 1, centre, u: right.divideScalar(w), v: up.divideScalar(h), n, map: labelTexture(label.seed, width, rows, 1, base) };
}

const LABEL_GLSL = `
	#if LABEL_MODE > 0
	{
		vec3 faceN = normalize(cross(dFdx(vLocalPos), dFdy(vLocalPos)));
		vec3 d = vLocalPos - labelCentre;
		#if LABEL_MODE == 1
			vec2 luv = vec2(dot(d, labelU), dot(d, labelV)) + 0.5;
			vec2 gx = dFdx(luv), gy = dFdy(luv);
			bool onFace = abs(dot(faceN, labelN)) > 0.7;
		#elif LABEL_MODE == 3
			// The face the normal points to; its side from where on the box
			// it is (labelU: 1 / the box's size).
			vec3 an = abs(faceN);
			int axis = an.x > an.y && an.x > an.z ? 0 : (an.y > an.z ? 1 : 2);
			vec3 q = d * labelU;
			vec2 luv = axis == 0 ? vec2(sign(q.x) * q.y, q.z) : (axis == 1 ? vec2(-sign(q.y) * q.x, q.z) : vec2(sign(q.z) * q.x, q.y));
			luv += 0.5;
			vec2 gx = dFdx(luv), gy = dFdy(luv);
			bool onFace = true;
		#else
			// The angle twice, with its seams on opposite sides: the
			// derivatives from the one without a seam here.
			float a1 = atan(d.y, d.x), a2 = atan(-d.y, -d.x);
			vec2 luv = vec2(a1 / 6.2831853 + 0.5, d.z * labelV.z + 0.5);
			float ax1 = dFdx(a1), ax2 = dFdx(a2), ay1 = dFdy(a1), ay2 = dFdy(a2);
			vec2 gx = vec2((abs(ax1) < abs(ax2) ? ax1 : ax2) / 6.2831853, dFdx(luv.y));
			vec2 gy = vec2((abs(ay1) < abs(ay2) ? ay1 : ay2) / 6.2831853, dFdy(luv.y));
			bool onFace = abs(faceN.z) < 0.6;
		#endif
		#if LABEL_MODE == 3
			vec4 ink = axis == 0 ? textureGrad(labelMap, luv, gx, gy) : (axis == 1 ? textureGrad(labelMapY, luv, gx, gy) : textureGrad(labelMapZ, luv, gx, gy));
		#else
			vec4 ink = textureGrad(labelMap, luv, gx, gy);
		#endif
		if (onFace && all(greaterThanEqual(luv, vec2(0.0))) && all(lessThanEqual(luv, vec2(1.0))))
			diffuseColor.rgb = mix(diffuseColor.rgb, ink.rgb, ink.a);
	}
	#endif
`;

// A textured copy of `material` (a MeshLambertMaterial) for `geometry`, from
// the setup's `info.texture` ({name, scale?, local, label?}); null when
// there's no (known) texture. Adds the geometry's `localPos` attribute.
export function texturedMaterial(material, geometry, info) {
	const spec = info.texture;
	if (!spec || !hasTexture(spec.name) || !spec.local) return null;
	const local = decodeFloats(spec.local);
	if (local.length !== info.vertexCount * 3) return null;
	geometry.setAttribute("localPos", new THREE.BufferAttribute(local, 3));
	const textured = material.clone();
	const map = texture(spec.name);
	const scale = spec.scale > 0 ? spec.scale : TEXTURES[spec.name].scale;
	const label = spec.label ? labelPlacement(spec.label, local, info.color) : null;
	textured.defines = { ...(textured.defines || {}), LABEL_MODE: label ? label.mode : 0 };
	textured.onBeforeCompile = (shader) => {
		shader.uniforms.triMap = { value: map };
		shader.uniforms.triScale = { value: scale };
		if (label) {
			shader.uniforms.labelMap = { value: label.map };
			shader.uniforms.labelCentre = { value: label.centre };
			shader.uniforms.labelU = { value: label.u };
			shader.uniforms.labelV = { value: label.v };
			shader.uniforms.labelN = { value: label.n };
			if (label.maps) {
				shader.uniforms.labelMapY = { value: label.maps[1] };
				shader.uniforms.labelMapZ = { value: label.maps[2] };
			}
		}
		shader.vertexShader = shader.vertexShader
			.replace("#include <common>", "#include <common>\nattribute vec3 localPos;\nvarying vec3 vLocalPos;")
			.replace("#include <begin_vertex>", "#include <begin_vertex>\nvLocalPos = localPos;");
		shader.fragmentShader = shader.fragmentShader
			.replace(
				"#include <common>",
				"#include <common>\nuniform sampler2D triMap;\nuniform float triScale;\nvarying vec3 vLocalPos;\n" +
					"#if LABEL_MODE > 0\nuniform sampler2D labelMap;\nuniform vec3 labelCentre, labelU, labelV, labelN;\n#endif\n" +
					"#if LABEL_MODE == 3\nuniform sampler2D labelMapY, labelMapZ;\n#endif",
			)
			.replace(
				"#include <map_fragment>",
				`#include <map_fragment>
				${LABEL_GLSL}
				{
					vec3 n = abs(cross(dFdx(vLocalPos), dFdy(vLocalPos)));
					vec3 p = vLocalPos / triScale;
					vec2 uv = n.x > n.y && n.x > n.z ? p.yz : (n.y > n.z ? p.xz : p.xy);
					diffuseColor.rgb *= 2.0 * texture2D(triMap, uv).r;
				}`,
			);
	};
	textured.customProgramCacheKey = () => `sofaweb-triplanar-${label ? label.mode : 0}`;
	return textured;
}
