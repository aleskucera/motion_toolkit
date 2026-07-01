"""Build and animate a rollout in Blender from a ``blender_export`` ``.npz``.

Run inside Blender (GUI or headless). This script imports only ``bpy``/``numpy`` (both
shipped with Blender), so it has no dependency on the rest of the package.

    # interactive: open the result and tweak camera/materials yourself
    blender --python src/kinematic_helhest/viz/blender_import.py -- --data rollout.npz

    # with your own rigged model (.blend with separate wheel objects)
    blender --python .../blender_import.py -- --data rollout.npz \
        --robot robot.blend \
        --wheel-left WheelL --wheel-right WheelR --wheel-rear WheelRear

    # headless render straight to MP4
    blender --background --python .../blender_import.py -- \
        --data rollout.npz --render out.mp4

With no ``--robot`` the script builds a box+cylinder proxy from the exported geometry
constants, so it runs standalone. Frame convention matches the sim: X-forward, Y-left,
Z-up, meters/radians; wheels spin about body Y by default (``--wheel-axis``).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Animate a sim rollout in Blender.")
    p.add_argument("--data", required=True, help="rollout .npz from blender_export")
    p.add_argument("--robot", default=None, help="optional .blend holding a rigged robot model")
    p.add_argument("--wheel-left", default=None, help="left-wheel object name inside --robot")
    p.add_argument("--wheel-right", default=None, help="right-wheel object name inside --robot")
    p.add_argument("--wheel-rear", default=None, help="rear-wheel object name inside --robot")
    p.add_argument("--wheel-axis", default="Y", choices=("X", "Y", "Z"), help="wheel spin axis")
    p.add_argument("--fps", type=int, default=30, help="playback fps (interpolates the 10 Hz sim)")
    p.add_argument("--render", default=None, help="output MP4 path; renders headless if given")
    p.add_argument("--save", default=None, help="save the built scene to a portable .blend")
    p.add_argument("--res", default="1280x720", help="render resolution WxH")
    return p.parse_args(argv)


def _material(name: str, rgba: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    return mat


def _new_mesh_object(
    name: str, verts: list, faces: list, mat: bpy.types.Material
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj


def build_terrain(data: dict) -> bpy.types.Object:
    """Heightmap [ny, nx] -> a quad mesh at cell centers (matches viz/render.build_terrain)."""
    H = np.asarray(data["terrain_H"], np.float64)
    x0, y0, cell = float(data["terrain_x0"]), float(data["terrain_y0"]), float(data["terrain_cell"])
    ny, nx = H.shape
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    XX, YY = np.meshgrid(xs, ys)
    verts = np.stack([XX, YY, H], -1).reshape(-1, 3).tolist()
    ii, jj = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    v0 = (ii * nx + jj).ravel()
    faces = np.stack([v0, v0 + 1, v0 + nx + 1, v0 + nx], 1).tolist()  # CCW quads
    obj = _new_mesh_object("terrain", verts, faces, _material("terrain", (0.30, 0.32, 0.26, 1.0)))
    obj.data.shade_smooth()
    return obj


def _box_mesh(
    cx: float, cy: float, cz: float, sx: float, sy: float, sz: float
) -> tuple[list, list]:
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [
        (cx + i * hx, cy + j * hy, cz + k * hz) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)
    ]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return verts, faces


def _cylinder_mesh(radius: float, half_w: float, segs: int = 32) -> tuple[list, list]:
    """Cylinder centered at origin, axis along body Y (so it spins about Y)."""
    ang = np.linspace(0.0, 2.0 * math.pi, segs, endpoint=False)
    ring = np.stack([radius * np.cos(ang), np.zeros_like(ang), radius * np.sin(ang)], 1)
    lo = (ring + [0.0, -half_w, 0.0]).tolist()
    hi = (ring + [0.0, +half_w, 0.0]).tolist()
    verts = lo + hi + [[0.0, -half_w, 0.0], [0.0, +half_w, 0.0]]
    cap_lo, cap_hi = 2 * segs, 2 * segs + 1
    faces = []
    for i in range(segs):
        j = (i + 1) % segs
        faces.append((i, j, segs + j, segs + i))  # side
        faces.append((cap_lo, j, i))  # bottom cap (axis -Y)
        faces.append((cap_hi, segs + i, segs + j))  # top cap (axis +Y)
    return verts, faces


def build_proxy_robot(data: dict) -> tuple[bpy.types.Object, list]:
    """Box chassis + 3 cylinder wheels from the exported geometry; returns (root, wheels)."""
    root = bpy.data.objects.new("robot_root", None)  # empty, carries the body->world transform
    bpy.context.collection.objects.link(root)

    chassis_mat = _material("chassis", (0.55, 0.57, 0.62, 1.0))
    wheel_mat = _material("wheel", (0.12, 0.12, 0.14, 1.0))
    children = []
    for box in np.asarray(data["chassis_boxes"], np.float64):
        verts, faces = _box_mesh(*box)
        children.append(_new_mesh_object("chassis", verts, faces, chassis_mat))

    radius, half_w = float(data["wheel_radius"]), float(data["wheel_width"]) / 2.0
    wheels = []
    for hub in np.asarray(data["wheel_pos"], np.float64):
        verts, faces = _cylinder_mesh(radius, half_w)
        w = _new_mesh_object("wheel", verts, faces, wheel_mat)
        w.location = Vector(hub.tolist())  # origin at the hub, so spin is about the hub
        wheels.append(w)
        children.append(w)

    for c in children:
        c.parent = root
    return root, wheels


def import_robot_blend(args: argparse.Namespace) -> tuple[bpy.types.Object, list]:
    """Append a rigged model (whole hierarchy + materials) and align it to our body frame.

    Our body-frame origin is the front-axle hub center (the sim's pose refers to it), so the
    animation root is placed at the mean of the left/right wheel positions and the model's own
    top-level objects are parented under it. Moving the root then drives the whole robot; the
    three wheel objects still spin locally on top.
    """
    if not all((args.wheel_left, args.wheel_right, args.wheel_rear)):
        raise SystemExit("--robot needs --wheel-left, --wheel-right, --wheel-rear")

    with bpy.data.libraries.load(args.robot, link=False) as (src, dst):
        dst.objects = list(src.objects)  # append every object, keeping parenting intact
    appended = [o for o in dst.objects if o is not None]
    for o in appended:
        bpy.context.collection.objects.link(o)

    by_name = {o.name: o for o in appended}
    wheels = [by_name[n] for n in (args.wheel_left, args.wheel_right, args.wheel_rear)]

    root = bpy.data.objects.new("robot_root", None)
    bpy.context.collection.objects.link(root)
    root.location = 0.5 * (wheels[0].matrix_world.translation + wheels[1].matrix_world.translation)
    for o in appended:
        if o.parent is None:  # reparent the model's own roots, keeping their world transform
            o.parent = root
            o.matrix_parent_inverse = root.matrix_world.inverted()
    return root, wheels


def animate(
    root: bpy.types.Object, wheels: list, data: dict, axis: str, dt: float, fps: int
) -> int:
    """Keyframe the body pose on `root` and the spin on each wheel. Returns the last frame.

    Each sim step (dt) is spread over `round(fps*dt)` Blender frames so the LINEAR keyframes
    interpolate smooth in-between poses (the sim's 10 Hz alone plays back choppy).
    """
    pos = np.asarray(data["pos"], np.float64)
    quat = np.asarray(data["quat"], np.float64)  # (w, x, y, z)
    spin = np.asarray(data["wheel_spin"], np.float64)  # (left, right, rear) [rad]
    n = len(pos)
    ai = _AXIS_INDEX[axis]
    step = max(1, round(fps * dt))  # sim steps -> Blender frames (real-time playback)

    root.rotation_mode = "QUATERNION"
    # Spin via a delta rotation (composed on top of each wheel's rest orientation). Blender picks
    # the delta channel by the object's rotation_mode: QUATERNION -> delta_rotation_quaternion,
    # Euler -> delta_rotation_euler. Writing the wrong one is silently ignored, so match the mode.
    # The delta is applied in the wheel's parent (body) frame, where the axle is Y.
    for f in range(n):
        bf = 1 + f * step  # Blender frames are 1-based
        root.location = Vector(pos[f].tolist())
        root.rotation_quaternion = quat[f].tolist()
        root.keyframe_insert("location", frame=bf)
        root.keyframe_insert("rotation_quaternion", frame=bf)
        for wi, w in enumerate(wheels):
            angle = float(spin[f, wi])
            if w.rotation_mode == "QUATERNION":
                q = [math.cos(angle / 2.0), 0.0, 0.0, 0.0]
                q[1 + ai] = math.sin(angle / 2.0)  # axis X/Y/Z -> quat component 1/2/3
                w.delta_rotation_quaternion = q
                w.keyframe_insert("delta_rotation_quaternion", frame=bf)
            else:
                delta = [0.0, 0.0, 0.0]
                delta[ai] = angle
                w.delta_rotation_euler = delta
                w.keyframe_insert("delta_rotation_euler", frame=bf)
    return 1 + (n - 1) * step  # last Blender frame


def setup_render(data: dict, terrain: bpy.types.Object, out_path: str, res: str, fps: int) -> None:
    """Camera + sun + world + MP4 output, framed on the terrain bounds."""
    H = np.asarray(data["terrain_H"], np.float64)
    x0, y0, cell = float(data["terrain_x0"]), float(data["terrain_y0"]), float(data["terrain_cell"])
    ny, nx = H.shape
    cx, cy = x0 + nx * cell / 2.0, y0 + ny * cell / 2.0
    span = max(nx, ny) * cell

    target = bpy.data.objects.new("cam_target", None)
    target.location = Vector((cx, cy, float(H.mean())))
    bpy.context.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = Vector((cx - span * 0.7, cy - span * 0.8, span * 0.7))
    bpy.context.collection.objects.link(cam)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.camera = cam

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 4.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(50.0), 0.0, math.radians(40.0))
    bpy.context.collection.objects.link(sun)

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"  # Blender 4.2+
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    w, h = (int(v) for v in res.lower().split("x"))
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.fps = fps

    try:  # most builds: write the MP4 directly
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.filepath = out_path
    except TypeError:  # this Blender build lacks FFMPEG -> PNG sequence next to out_path
        base = os.path.splitext(out_path)[0]
        if not base.startswith("//") and not os.path.dirname(base):
            base = os.path.abspath(base)  # a bare relative path has no dir for Blender to create
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = base + "_"
        print("note: Blender built without FFMPEG; wrote a PNG sequence instead of MP4")


def save_blend(path: str) -> None:
    """Pack external data (model textures) into the file so it's portable, then save."""
    try:
        bpy.ops.file.pack_all()
    except RuntimeError:
        pass  # nothing external to pack
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))
    print(f"saved scene -> {path}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parse_args(argv)
    data = dict(np.load(args.data))
    dt = float(data["dt"])

    if "Cube" in bpy.data.objects:  # drop the default-scene cube if present
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)

    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"  # constant-rate motion

    terrain = build_terrain(data)
    if args.robot:
        root, wheels = import_robot_blend(args)
    else:
        root, wheels = build_proxy_robot(data)
    last = animate(root, wheels, data, args.wheel_axis, dt, args.fps)

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, last

    if args.render or args.save:
        # when only saving, bake an output path relative to the .blend ("//") so a later
        # `blender -b scene.blend -a` on another box writes next to the file.
        out = args.render or "//" + os.path.splitext(os.path.basename(args.save))[0] + ".mp4"
        setup_render(data, terrain, out, args.res, args.fps)
    if args.save:
        save_blend(args.save)
    if args.render:
        bpy.ops.render.render(animation=True)
        print(f"rendered {last} frames -> {args.render}")
    if not args.render and not args.save:
        print(f"built scene: {last} frames; press space to play")


if __name__ == "__main__":
    main()
