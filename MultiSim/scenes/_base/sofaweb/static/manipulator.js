// The manipulator (a scene config's `manipulator`): steer an end-effector
// with the mouse, a touchscreen or the keyboard instead of (and in step
// with) its sliders in the Controls panel.
//
//  - Drag (the default): grab the end-effector itself and drag it, over a
//    horizontal plane, or up and down with Shift held (on a touchscreen:
//    the Height button). A small marker shows where the target is.
//  - Axes / Turn: a gizmo at the target pose (three.js' TransformControls):
//    its arrows and planes move the target, its rings turn it. Grabbing the
//    end-effector away from the gizmo's handles still drags it.
//  - Double-click (double-tap) a spot in the scene: the end-effector goes
//    above it, at its current height.
//  - A tall height slider at the side of the view, a Grip button, "Point
//    down" (no roll or pitch) and the keyboard: arrows / WASD move over the
//    floor (as seen from the camera), R / F up and down, Q / E turn about
//    the vertical, Space grips or lets go.
//  - `frame`: the target is relative to a moving frame (e.g. a mobile
//    robot's platform), whose x, y and yaw come from the scene's scalars.
//  - `drive`: a joystick (and the arrow keys) for a vehicle's forward and
//    sideways speeds (omniwheels; without `sideways` the joystick turns it),
//    and turn buttons (keys , and .): deadman sliders, sent again while held,
//    zero when let go.
//  - `reach`: a scalar that is 0 while the target can't be reached (the
//    marker turns red).
//  - `pick`: one click (or tap) on a model it names (e.g. a grocery item)
//    sends the clicked point to its x, y, z controls and presses its `go`
//    button (the scene acts on what's there, e.g. stocks the item); a
//    double-click still means "go there".
//
// The target is a set of the scene's slider controls (positions in control
// units, x `scale` = scene units; roll, pitch, yaw in degrees, R = Rz Ry Rx);
// every change goes to the server as one batch, and the status messages
// bring back where the owner has put them (watchers see no gizmo).

import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const DEG = Math.PI / 180;
const STATUS_HOLD_MS = 900; // a key just sent ignores the (older) status values for this long

export function createManipulator(ctx, spec, controlSpecs) {
	const byKey = new Map(controlSpecs.map((c) => [c.key, c]));
	const posKeys = spec.position || [];
	const rotKeys = spec.rotation || [];
	const scale = spec.scale ?? 1;
	const grip = spec.grip && byKey.has(spec.grip.key) ? spec.grip : null;
	const grabRe = spec.models ? new RegExp(spec.models) : null;
	const coarse = window.matchMedia("(pointer: coarse)").matches;
	const values = {};
	for (const key of [...posKeys, ...rotKeys, ...(grip ? [grip.key] : [])]) {
		const c = byKey.get(key);
		values[key] = c ? Number(c.default ?? c.min ?? 0) : 0;
	}
	const sentAt = new Map(); // key -> time of the last send
	let usable = false;
	let mode = "drag"; // or "translate" / "rotate" (the gizmo)
	let heightDrag = false; // the Height button: dragging the end-effector moves it up and down

	// -- the target and the gizmo ---------------------------------------------
	// The frame the target is given in (the world, or a moving platform), and the target in it.
	const frameObj = new THREE.Object3D();
	ctx.scene.add(frameObj);
	const proxy = new THREE.Object3D();
	frameObj.add(proxy);
	const frameSpec = spec.frame || null;
	const driveSpec = spec.drive && byKey.has(spec.drive.forward) && byKey.has(spec.drive.turn) ? spec.drive : null;
	const sideways = driveSpec && byKey.has(driveSpec.sideways) ? driveSpec.sideways : null;
	const gizmo = new TransformControls(ctx.camera(), ctx.renderer.domElement);
	gizmo.setSize(coarse ? 1.35 : 0.9);
	gizmo.setSpace("world");
	gizmo.attach(proxy);
	ctx.scene.add(gizmo);
	// Where the target is (the end-effector lags behind it while it moves): a small sphere, drawn over everything.
	const marker = new THREE.Mesh(
		new THREE.SphereGeometry(1, 16, 12),
		new THREE.MeshBasicMaterial({ color: 0xff8a1f, depthTest: false, transparent: true, opacity: 0.85 }),
		// (red while the target is out of reach: `reach`)
	);
	marker.renderOrder = 999;
	marker.scale.setScalar(spec.markerRadius ?? 0.008);
	proxy.add(marker);
	gizmo.addEventListener("dragging-changed", (event) => {
		const orbit = ctx.orbit();
		if (orbit) orbit.enabled = !event.value;
		if (!event.value) placeProxy(); // back to the (clamped) values
	});
	gizmo.addEventListener("objectChange", () => {
		if (!gizmo.dragging) return;
		if (mode === "translate") {
			setValues(Object.fromEntries(posKeys.map((k, i) => [k, proxy.position.getComponent(i) / scale])));
		} else if (rotKeys.length === 3) {
			const e = new THREE.Euler().setFromQuaternion(proxy.quaternion, "ZYX");
			setValues({ [rotKeys[0]]: e.x / DEG, [rotKeys[1]]: e.y / DEG, [rotKeys[2]]: e.z / DEG }, false);
		}
	});

	function clampValue(key, value) {
		const c = byKey.get(key);
		if (!c) return value;
		return Math.min(Math.max(value, Number(c.min)), Number(c.max));
	}

	// The moving frame's pose, from the scene's scalars (positions x `frame.scale`, yaw in degrees).
	function placeFrame() {
		if (!frameSpec) return;
		const sc = ctx.scalars() || {};
		const x = Number(sc[frameSpec.x]), y = Number(sc[frameSpec.y]), yaw = Number(sc[frameSpec.yaw]);
		if (![x, y, yaw].every(Number.isFinite)) return;
		const k = frameSpec.scale ?? scale;
		frameObj.position.set(x * k, y * k, 0);
		frameObj.rotation.set(0, 0, yaw * DEG);
		frameObj.updateMatrixWorld(true);
	}

	function placeProxy() {
		if (posKeys.length === 3) proxy.position.set(...posKeys.map((k) => values[k] * scale));
		if (rotKeys.length === 3) {
			proxy.quaternion.setFromEuler(new THREE.Euler(values[rotKeys[0]] * DEG, values[rotKeys[1]] * DEG, values[rotKeys[2]] * DEG, "ZYX"));
		}
		proxy.updateMatrixWorld();
	}

	// -- sending: one batch in flight, the latest values queued behind it ------
	let inFlight = false;
	let pending = null;
	function send(changes) {
		const now = performance.now();
		for (const key of Object.keys(changes)) sentAt.set(key, now);
		if (inFlight) {
			pending = { ...(pending || {}), ...changes };
			return;
		}
		inFlight = true;
		ctx.post(changes).finally(() => {
			inFlight = false;
			if (pending) {
				const next = pending;
				pending = null;
				send(next);
			}
		});
	}

	function setValues(changes, moveProxy = true) {
		const out = {};
		for (const [key, value] of Object.entries(changes)) {
			if (!Number.isFinite(value) || !(key in values)) continue;
			const v = clampValue(key, value);
			if (Math.abs(v - values[key]) < 1e-9) continue;
			values[key] = out[key] = v;
			ctx.show(key, v);
		}
		if (!Object.keys(out).length) return;
		if (moveProxy && !gizmo.dragging) placeProxy();
		send(out);
		updateBar();
	}

	// -- the toolbar, the hint and the height slider ---------------------------
	const bar = document.createElement("div");
	bar.id = "manip-bar";
	bar.className = "panel owner-only hidden";
	const button = (text, title, onClick, cls = "secondary") => {
		const b = document.createElement("button");
		Object.assign(b, { type: "button", className: `manip-btn ${cls}`, textContent: text, title });
		b.addEventListener("click", onClick);
		bar.appendChild(b);
		return b;
	};
	const dragBtn = button("Drag", "Grab the gripper and drag it (Shift or ↕ Height: up and down). Keyboard: T switches modes.", () => setMode("drag"));
	const moveBtn = button("✥ Axes", "A gizmo: its arrows and squares move the gripper along an axis or in a plane. Keyboard: T switches modes.", () => setMode("translate"));
	const turnBtn = rotKeys.length === 3 ? button("⟳ Turn", "A gizmo: its rings turn the gripper. Keyboard: T switches modes.", () => setMode("rotate")) : null;
	const spaceBtn = button("World axes", "The gizmo's axes: the world's, or the gripper's own", () => {
		gizmo.setSpace(gizmo.space === "world" ? "local" : "world");
		updateBar();
	});
	const heightBtn = button("↕ Height", "Dragging the gripper moves it up and down instead of over the floor (with a mouse: hold Shift)", () => {
		heightDrag = !heightDrag;
		updateBar();
	});
	const levelBtn = rotKeys.length === 3 ? button("⤓ Point down", spec.levelTitle || "No roll or pitch: the gripper points straight down (keeps its yaw)", () => {
		setValues({ [rotKeys[0]]: 0, [rotKeys[1]]: 0 });
	}) : null;
	const gripBtn = grip ? button(grip.label || "Grip", "Close the fingers / let go (keyboard: Space)", toggleGrip, "") : null;
	// The scene's own buttons (button controls, e.g. a repair): one press each.
	for (const b of spec.buttons || []) {
		if (byKey.get(b.key)?.type !== "button") continue;
		const el = button(b.label || byKey.get(b.key).label, b.title || byKey.get(b.key).description || "", () => {
			ctx.post({ [b.key]: null });
			el.classList.add("active");
			setTimeout(() => el.classList.remove("active"), 200);
		});
	}
	const reachNote = document.createElement("span");
	reachNote.className = "manip-reach hidden";
	reachNote.textContent = "Out of reach";
	reachNote.title = "The arm can't reach the target: the gripper waits where it got to (the red dot is the target)";
	bar.appendChild(reachNote);
	const helpBtn = button("?", "How to steer the gripper", () => hint.classList.toggle("hidden"));
	helpBtn.classList.add("manip-help");
	const hint = document.createElement("div");
	hint.id = "manip-hint";
	hint.className = "panel owner-only hidden";
	hint.innerHTML = coarse
		? "<b>Drag the gripper</b> with a finger to move it over the floor; with <b>↕ Height</b> on, up and down (or use the slider on the left). <b>Double-tap</b> a spot: the gripper goes above it. <b>✥ Axes</b> and <b>⟳ Turn</b> show a gizmo: drag its arrows to move along an axis, its rings to turn. Drag anywhere else to turn the view, pinch to zoom. The orange dot is where the gripper is heading."
		: "<b>Drag the gripper</b> to move it over the floor, <b>Shift-drag</b> to move it up and down. <b>Double-click</b> a spot: the gripper goes above it. <b>✥ Axes</b> and <b>⟳ Turn</b> show a gizmo: its arrows and squares move it, its rings turn it. Keys: <b>arrows / WASD</b> move (as seen from the camera), <b>R / F</b> up / down, <b>Q / E</b> turn, <b>Space</b> grip, <b>T</b> next mode. The orange dot is where the gripper is heading.";
	const height = document.createElement("div");
	height.id = "manip-height";
	height.className = "panel owner-only hidden";
	const zControl = byKey.get(posKeys[2]);
	const heightInput = document.createElement("input");
	Object.assign(heightInput, { type: "range", min: zControl?.min ?? 0, max: zControl?.max ?? 1, step: zControl?.step ?? "any", title: "Height of the gripper" });
	heightInput.setAttribute("aria-label", "Height");
	const heightLabel = document.createElement("span");
	heightInput.addEventListener("input", () => setValues({ [posKeys[2]]: Number(heightInput.value) }));
	height.append(heightLabel, heightInput);
	if (driveSpec) {
		const how = driveSpec.hint || (sideways ? "up / down forward and back, left / right sideways; ⟲ ⟳ turn it" : "up / down forward and back, left / right turn");
		hint.innerHTML += coarse
			? ` <b>Drive</b> the robot with the joystick on the left: ${how}.`
			: ` <b>Drive</b> the robot with the joystick on the left (${how}), or the <b>arrow keys</b>${sideways ? " and <b>, .</b> to turn" : ""} (then WASD moves the gripper).`;
	}
	const pickSpec = spec.pick && spec.pick.models && byKey.has(spec.pick.go) ? spec.pick : null;
	if (pickSpec?.title) hint.innerHTML += ` ${pickSpec.title}`;
	const row = document.createElement("div");
	row.id = "manip-row";
	row.append(bar, hint);
	ctx.mount(row, height);
	// The drive joystick: a pad with a knob; its offset is the forward speed (up) and the turn rate (left).
	let pad = null;
	let padKnob = null;
	let padOffset = null; // [x, y] in -1..1 while held
	let turnHeld = 0; // the turn buttons: +1 left, -1 right
	if (driveSpec) {
		pad = document.createElement("div");
		pad.id = "manip-drive";
		pad.className = "panel owner-only hidden";
		pad.title = sideways
			? "Drive the robot: up / down forward and back, left / right sideways (keyboard: the arrow keys)"
			: "Drive the robot: up / down forward and back, left / right turn (keyboard: the arrow keys)";
		const base = document.createElement("div");
		base.className = "manip-pad";
		padKnob = document.createElement("div");
		padKnob.className = "manip-knob";
		base.appendChild(padKnob);
		const label = document.createElement("span");
		label.textContent = "Drive";
		pad.append(label, base);
		if (sideways) {
			const turns = document.createElement("div");
			turns.className = "manip-turns";
			for (const [text, dir, title] of [["⟲", 1, "Turn left (keyboard: ,)"], ["⟳", -1, "Turn right (keyboard: .)"]]) {
				const b = document.createElement("button");
				Object.assign(b, { type: "button", className: "secondary manip-turn", textContent: text, title });
				b.addEventListener("pointerdown", (event) => {
					b.setPointerCapture(event.pointerId);
					turnHeld = dir;
					b.classList.add("active");
				});
				for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
					b.addEventListener(type, () => {
						if (turnHeld === dir) turnHeld = 0;
						b.classList.remove("active");
					});
				}
				b.addEventListener("contextmenu", (event) => event.preventDefault());
				turns.appendChild(b);
			}
			pad.appendChild(turns);
		}
		const moveKnob = (event) => {
			const r = base.getBoundingClientRect();
			let x = (event.clientX - (r.left + r.width / 2)) / (r.width / 2);
			let y = (event.clientY - (r.top + r.height / 2)) / (r.height / 2);
			const n = Math.hypot(x, y);
			if (n > 1) [x, y] = [x / n, y / n];
			padOffset = [x, y];
			padKnob.style.transform = `translate(${x * r.width * 0.36}px, ${y * r.height * 0.36}px)`;
		};
		const release = () => {
			padOffset = null;
			padKnob.style.transform = "";
		};
		base.addEventListener("pointerdown", (event) => {
			if (!usable) return;
			base.setPointerCapture(event.pointerId);
			moveKnob(event);
			event.preventDefault();
		});
		base.addEventListener("pointermove", (event) => base.hasPointerCapture(event.pointerId) && moveKnob(event));
		for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) base.addEventListener(type, release);
		ctx.mount(null, pad);
	}

	function setMode(m) {
		mode = m;
		if (m !== "drag") gizmo.setMode(m);
		showGizmo();
		updateBar();
	}

	function showGizmo() {
		gizmo.visible = gizmo.enabled = usable && mode !== "drag";
		marker.visible = usable;
	}

	function gripOn() {
		return grip && values[grip.key] > (Number(grip.on) + Number(grip.off ?? 0)) / 2;
	}

	function toggleGrip() {
		if (grip) setValues({ [grip.key]: gripOn() ? Number(grip.off ?? 0) : Number(grip.on) });
	}

	function updateBar() {
		dragBtn.classList.toggle("active", mode === "drag");
		moveBtn.classList.toggle("active", mode === "translate");
		turnBtn?.classList.toggle("active", mode === "rotate");
		spaceBtn.textContent = gizmo.space === "world" ? "World axes" : "Gripper axes";
		heightBtn.classList.toggle("active", heightDrag);
		if (gripBtn) {
			gripBtn.textContent = gripOn() ? grip.releaseLabel || "Let go" : grip.label || "Grip";
			gripBtn.classList.toggle("gripping", !!gripOn());
		}
		if (zControl && document.activeElement !== heightInput) heightInput.value = values[posKeys[2]];
		if (zControl) heightLabel.textContent = `${Math.round(values[posKeys[2]])}${zControl.unit ? " " + zControl.unit : ""}`;
	}

	// -- grabbing the end-effector, double-click / double-tap -----------------
	const raycaster = new THREE.Raycaster();
	const ndc = new THREE.Vector2();
	const dom = ctx.renderer.domElement;
	const host = dom.parentElement;

	function toNdc(event) {
		const r = dom.getBoundingClientRect();
		ndc.set(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
		raycaster.setFromCamera(ndc, ctx.camera());
	}

	function meshes(grabbable) {
		const out = [];
		for (const entry of ctx.models.values()) {
			const obj = entry.object;
			if (!obj?.isMesh || !obj.visible || !entry.info?.name) continue;
			if (grabRe && grabRe.test(entry.info.name) === grabbable) out.push(obj);
		}
		return out;
	}

	function overGizmo() {
		if (!gizmo.visible) return false;
		gizmo.pointerHover({ x: ndc.x, y: ndc.y, button: 0 });
		return gizmo.axis !== null;
	}

	let drag = null; // { pointerId, plane, offset, vertical }
	const hit = new THREE.Vector3();

	host.addEventListener("pointerdown", (event) => {
		if (!usable || drag || (event.pointerType === "mouse" && event.button !== 0)) return;
		toNdc(event);
		if (overGizmo()) return; // the gizmo's own drag
		const target = proxy.getWorldPosition(new THREE.Vector3());
		const found = raycaster.intersectObjects(meshes(true), false)[0];
		if (!found) return;
		event.stopPropagation(); // not a camera move
		event.preventDefault();
		const orbit = ctx.orbit();
		if (orbit) orbit.enabled = false;
		const vertical = heightDrag || event.shiftKey;
		let normal;
		if (vertical) {
			normal = ctx.camera().getWorldDirection(new THREE.Vector3()).setZ(0);
			if (normal.lengthSq() < 1e-6) normal.set(0, 1, 0);
			normal.normalize();
		} else {
			normal = new THREE.Vector3(0, 0, 1);
		}
		const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, found.point);
		drag = { pointerId: event.pointerId, plane, offset: target.sub(found.point), vertical };
		host.setPointerCapture(event.pointerId);
		host.classList.add("manip-grabbing");
	}, true);

	host.addEventListener("pointermove", (event) => {
		if (drag && event.pointerId === drag.pointerId) {
			toNdc(event);
			if (!raycaster.ray.intersectPlane(drag.plane, hit)) return;
			const p = frameObj.worldToLocal(hit.add(drag.offset));
			if (drag.vertical) setValues({ [posKeys[2]]: p.z / scale });
			else setValues({ [posKeys[0]]: p.x / scale, [posKeys[1]]: p.y / scale });
			return;
		}
		if (event.pointerType === "mouse" && usable && !event.buttons) hoverSoon(event);
	});

	function endDrag(event) {
		if (!drag || event.pointerId !== drag.pointerId) return;
		drag = null;
		host.classList.remove("manip-grabbing");
		const orbit = ctx.orbit();
		if (orbit) orbit.enabled = true;
	}
	host.addEventListener("pointerup", endDrag);
	host.addEventListener("pointercancel", endDrag);

	// The cursor shows what a press would grab (checked once per frame at most).
	let hoverEvent = null;
	function hoverSoon(event) {
		if (!hoverEvent) requestAnimationFrame(() => {
			const e = hoverEvent;
			hoverEvent = null;
			if (!e || drag) return;
			toNdc(e);
			const over = !overGizmo() && raycaster.intersectObjects(meshes(true), false).length > 0;
			host.classList.toggle("manip-grab", over);
			host.classList.toggle("manip-pick", !over && !!pickable(e));
		});
		hoverEvent = event;
	}

	function goTo(event) {
		toNdc(event);
		const found = raycaster.intersectObjects(meshes(false), false)[0];
		if (!found) return;
		const p = frameObj.worldToLocal(found.point.clone());
		setValues({ [posKeys[0]]: p.x / scale, [posKeys[1]]: p.y / scale });
	}
	dom.addEventListener("dblclick", (event) => {
		cancelPick();
		if (usable && event.pointerType !== "touch") goTo(event);
	});

	// One click on a model `pick` names (a short press that didn't move: not a
	// drag of the view), sent unless a second click follows (a double-click).
	const pickRe = pickSpec ? new RegExp(pickSpec.models) : null;
	let press = null;
	let pickTimer = null;
	function pickable(event) {
		if (!pickRe) return null;
		toNdc(event);
		const found = raycaster.intersectObjects(meshes(false), false)[0];
		if (!found) return null;
		const entry = [...ctx.models.values()].find((e) => e.object === found.object);
		return entry && pickRe.test(entry.info.name) ? found.point.clone() : null;
	}
	function cancelPick() {
		clearTimeout(pickTimer);
		pickTimer = null;
	}
	host.addEventListener("pointerdown", (event) => {
		if (!event.isPrimary || (event.pointerType === "mouse" && event.button !== 0)) return;
		if (pickTimer) cancelPick(); // a second click: a double-click, not a pick
		press = { x: event.clientX, y: event.clientY, t: performance.now() };
	}, true);
	host.addEventListener("pointerup", (event) => {
		if (!pickRe || !press || !usable || !event.isPrimary) return;
		const short = performance.now() - press.t < 350 && Math.hypot(event.clientX - press.x, event.clientY - press.y) < 6;
		press = null;
		if (!short || drag) return;
		const point = pickable(event);
		if (!point) return;
		pickTimer = setTimeout(() => {
			pickTimer = null;
			ctx.post({ [pickSpec.x]: point.x / scale, [pickSpec.y]: point.y / scale, [pickSpec.z]: point.z / scale, [pickSpec.go]: null });
		}, 320);
	});
	// Double-tap: two short taps close together (touch browsers don't all send dblclick).
	let tap = null;
	let lastTap = null;
	host.addEventListener("pointerdown", (event) => {
		if (event.pointerType !== "touch") return;
		tap = event.isPrimary ? { x: event.clientX, y: event.clientY, t: performance.now() } : null;
	}, true);
	host.addEventListener("pointerup", (event) => {
		if (event.pointerType !== "touch" || !tap || !usable) return;
		const now = performance.now();
		const short = now - tap.t < 280 && Math.hypot(event.clientX - tap.x, event.clientY - tap.y) < 12;
		if (short && lastTap && now - lastTap.t < 380 && Math.hypot(event.clientX - lastTap.x, event.clientY - lastTap.y) < 40) {
			lastTap = null;
			goTo(event);
		} else {
			lastTap = short ? { x: event.clientX, y: event.clientY, t: now } : null;
		}
		tap = null;
	});

	// -- the keyboard -----------------------------------------------------------
	const held = new Set();
	// With a drive, the arrow keys drive and WASD moves the gripper.
	const DRIVE_KEYS = new Set(driveSpec ? ["arrowup", "arrowdown", "arrowleft", "arrowright", ...(sideways ? [",", "."] : [])] : []);
	const JOG_KEYS = new Set(["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d", "r", "f", "q", "e", "pageup", "pagedown"].filter((k) => !DRIVE_KEYS.has(k)));
	const typing = (el) => el && (el.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName));
	window.addEventListener("keydown", (event) => {
		if (!usable || typing(event.target) || event.ctrlKey || event.metaKey || event.altKey) return;
		const key = event.key.toLowerCase();
		if (JOG_KEYS.has(key) || DRIVE_KEYS.has(key)) {
			held.add(key);
			event.preventDefault();
		} else if (key === " " && grip && !event.repeat) {
			toggleGrip();
			event.preventDefault();
		} else if (key === "t" && !event.repeat) {
			const modes = ["drag", "translate", ...(turnBtn ? ["rotate"] : [])];
			setMode(modes[(modes.indexOf(mode) + 1) % modes.length]);
		}
	});
	window.addEventListener("keyup", (event) => held.delete(event.key.toLowerCase()));
	window.addEventListener("blur", () => held.clear());

	let lastTick = performance.now();
	const forward = new THREE.Vector3();
	const right = new THREE.Vector3();
	function tick() {
		const now = performance.now();
		const dt = Math.min((now - lastTick) / 1000, 0.1);
		lastTick = now;
		drive(now);
		if (!usable || ![...held].some((k) => JOG_KEYS.has(k)) || posKeys.length !== 3) return;
		const has = (...keys) => keys.some((k) => held.has(k) && JOG_KEYS.has(k));
		const speed = spec.jogSpeed ?? 150; // control units per second
		const camera = ctx.camera();
		camera.getWorldDirection(forward).setZ(0);
		if (forward.lengthSq() < 1e-8) forward.set(0, 1, 0).applyQuaternion(camera.quaternion).setZ(0);
		forward.normalize();
		right.set(forward.y, -forward.x, 0);
		if (frameSpec) { // camera directions, in the moving frame
			const inv = frameObj.quaternion.clone().invert();
			forward.applyQuaternion(inv);
			right.applyQuaternion(inv);
		}
		const f = (has("arrowup", "w") ? 1 : 0) - (has("arrowdown", "s") ? 1 : 0);
		const r = (has("arrowright", "d") ? 1 : 0) - (has("arrowleft", "a") ? 1 : 0);
		const u = (has("r", "pageup") ? 1 : 0) - (has("f", "pagedown") ? 1 : 0);
		const turn = (has("q") ? 1 : 0) - (has("e") ? 1 : 0);
		const changes = {};
		if (f || r) {
			changes[posKeys[0]] = values[posKeys[0]] + (forward.x * f + right.x * r) * speed * dt;
			changes[posKeys[1]] = values[posKeys[1]] + (forward.y * f + right.y * r) * speed * dt;
		}
		if (u) changes[posKeys[2]] = values[posKeys[2]] + u * speed * dt;
		if (turn && rotKeys.length === 3) {
			let yaw = values[rotKeys[2]] + turn * (spec.jogTurn ?? 60) * dt;
			if (yaw > 180) yaw -= 360;
			if (yaw < -180) yaw += 360;
			changes[rotKeys[2]] = yaw;
		}
		setValues(changes);
	}

	// Driving: the joystick or the arrow keys give the forward speed and the turn rate, sent when they change and
	// again every 0.4 s while not zero (the server stops the vehicle if they aren't).
	let driveSent = { f: 0, s: 0, t: 0, at: 0 };
	function drive(now) {
		if (!driveSpec) return;
		const fc = byKey.get(driveSpec.forward), tc = byKey.get(driveSpec.turn), sc = sideways && byKey.get(sideways);
		let f = 0, sv = 0, t = 0;
		const key = (k) => (held.has(k) ? 1 : 0);
		if (usable) {
			if (padOffset) {
				f = -padOffset[1];
				if (sideways) sv = -padOffset[0];
				else t = -padOffset[0];
			} else {
				f = key("arrowup") - key("arrowdown");
				if (sideways) sv = key("arrowleft") - key("arrowright");
				else t = key("arrowleft") - key("arrowright");
			}
			if (sideways) t = turnHeld || key(",") - key(".");
		}
		// A little dead zone, so a thumb on the joystick doesn't creep.
		const dz = (v) => (Math.abs(v) < 0.08 ? 0 : v);
		const fv = Math.round(dz(f) * Number(fc.max) * 1000) / 1000;
		const svv = sc ? Math.round(dz(sv) * Number(sc.max) * 1000) / 1000 : 0;
		const tv = Math.round(dz(t) * Number(tc.max) * 10) / 10;
		const changed = fv !== driveSent.f || svv !== driveSent.s || tv !== driveSent.t;
		if (!changed && ((fv === 0 && svv === 0 && tv === 0) || now - driveSent.at < 400)) return;
		if (changed && now - driveSent.at < 60) return; // at most ~16 a second
		driveSent = { f: fv, s: svv, t: tv, at: now };
		const out = { [driveSpec.forward]: fv, [driveSpec.turn]: tv };
		if (sideways) out[sideways] = svv;
		ctx.post(out);
		for (const [k, v] of Object.entries(out)) ctx.show(k, v);
	}

	// -- state from the server ----------------------------------------------------
	function update(status) {
		usable = !ctx.isViewer && status.status === "running" && !status.recording?.cached;
		const now = performance.now();
		if (status.controls && !gizmo.dragging && !drag) {
			for (const key of Object.keys(values)) {
				if (!(key in status.controls)) continue;
				if (now - (sentAt.get(key) ?? -1e9) < STATUS_HOLD_MS || (inFlight && pending && key in pending)) continue;
				values[key] = Number(status.controls[key]);
			}
			placeProxy();
		}
		showGizmo();
		for (const el of [bar, hint, height]) el.classList.toggle("disabled", !usable);
		bar.classList.toggle("hidden", ctx.isViewer);
		height.classList.toggle("hidden", ctx.isViewer || !zControl);
		if (pad) {
			pad.classList.toggle("hidden", ctx.isViewer);
			pad.classList.toggle("disabled", !usable);
		}
		if (!usable) host.classList.remove("manip-grab");
		updateBar();
	}

	function frame() {
		gizmo.camera = ctx.camera();
		placeFrame();
		if (spec.reach) {
			const r = Number((ctx.scalars() || {})[spec.reach]);
			const out = Number.isFinite(r) && r < 0.5;
			marker.material.color.setHex(out ? 0xe02424 : 0xff8a1f);
			reachNote.classList.toggle("hidden", !out);
		}
		tick();
	}

	placeProxy();
	updateBar();
	if (!ctx.isViewer) {
		try {
			if (!localStorage.getItem("sofaweb:manipHintSeen")) {
				hint.classList.remove("hidden");
				localStorage.setItem("sofaweb:manipHintSeen", "1");
			}
		} catch {}
	}
	return { update, frame };
}
