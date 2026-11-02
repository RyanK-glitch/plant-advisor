"""Build the 3D leaf used by the interface, entirely from code (no third-party assets).

Run with Blender 4.2+ (tested on 5.2):
    blender -b -P scripts/assets/build_leaf.py

Writes, next to each other in static/3d/:
    leaf.blend     editable source (open in Blender; the texture is packed inside)
    leaf.glb       web model: blade + stem, a painted texture, and a looping "Sway" animation
                   (shape keys "Curl" and "Flutter")
    leaf.webp      transparent still render, shown when WebGL or motion is off

The texture is painted procedurally: leaf-green gradient, midrib and side veins, and a few
rust-coloured lesions with yellow halos, since the app is about diseased leaves.
"""
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "static" / "3d"
OUT.mkdir(parents=True, exist_ok=True)

LENGTH = 2.0                  # blade length (Blender units), base at y = 0, tip at y = LENGTH
NU, NV = 44, 17               # grid resolution along / across the blade (NV odd so the midrib is a row)
TEX = 512                     # texture size in pixels
FPS, LOOP = 30, 150           # Sway animation: 5 s loop
LESIONS = [                   # (u along blade, v across blade -1..1, radius in uv units)
    (0.32, 0.45, 0.035), (0.41, 0.62, 0.025), (0.48, -0.38, 0.03), (0.56, 0.30, 0.022),
    (0.63, -0.58, 0.026), (0.70, 0.52, 0.02), (0.38, -0.66, 0.02), (0.77, -0.25, 0.018),
]


def half_width(u):
    # Ovate-lanceolate outline: widest at ~35 % of the length, pointed tip, rounded base
    return 0.44 * math.sin(math.pi * u ** 0.72) ** 0.85 * (1.0 - 0.18 * u)


def blade_point(u, v, curl=0.0, flutter=0.0):
    w = half_width(u)
    x = v * w
    y = u * LENGTH
    z = 0.20 * abs(x)                                   # V-fold along the midrib
    z -= (0.32 + 0.45 * curl) * u ** 2.2                # tip droops; "Curl" droops it further
    z += 0.010 * math.sin(u * 18.0) * abs(v) ** 3       # gently wavy margin
    z += flutter * 0.10 * math.sin(u * math.pi) * v     # "Flutter": blade twists about the midrib
    x *= 1.0 - 0.18 * curl * abs(v)                     # curling narrows the blade slightly
    return x, y, z


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_blade():
    mesh = bpy.data.meshes.new("LeafBlade")
    obj = bpy.data.objects.new("LeafBlade", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    grid = []
    for i in range(NU + 1):
        u = i / NU
        row = []
        for j in range(NV):
            v = -1.0 + 2.0 * j / (NV - 1)
            row.append(bm.verts.new(blade_point(u, v)))
        grid.append(row)
    for i in range(NU):
        for j in range(NV - 1):
            face = bm.faces.new((grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]))
            for loop, (a, b) in zip(face.loops, ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))):
                loop[uv_layer].uv = (a / NU, b / (NV - 1))
    bm.to_mesh(mesh)
    bm.free()

    # Shape keys: Basis, Curl, Flutter (same vertex order as built above)
    obj.shape_key_add(name="Basis")
    for name, kwargs in (("Curl", {"curl": 1.0}), ("Flutter", {"flutter": 1.0})):
        key = obj.shape_key_add(name=name)
        for index, vert in enumerate(key.data):
            i, j = divmod(index, NV)
            key.data[index].co = blade_point(i / NU, -1.0 + 2.0 * j / (NV - 1), **kwargs)

    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def paint_texture():
    # Texture space: x = u (base -> tip), y = (v + 1) / 2 (one margin -> the other)
    u, t = np.meshgrid(np.linspace(0, 1, TEX), np.linspace(0, 1, TEX))
    v = t * 2 - 1
    dark, light = np.array([0.09, 0.33, 0.14]), np.array([0.30, 0.58, 0.22])
    shade = np.clip(0.55 + 0.35 * (1 - np.abs(v)) - 0.25 * u, 0, 1)[..., None]
    rgb = dark + (light - dark) * shade

    midrib = np.exp(-(v / 0.035) ** 2)
    phase = u * 9.0 - np.abs(v) * 1.6                    # side veins sweep towards the tip
    veins = np.exp(-((phase - np.round(phase)) / 0.06) ** 2) * (np.abs(v) > 0.04) * (1 - 0.6 * np.abs(v))
    vein_colour = np.array([0.62, 0.78, 0.45])
    rgb = rgb * (1 - 0.75 * midrib[..., None]) + vein_colour * 0.75 * midrib[..., None]
    rgb = rgb * (1 - 0.35 * veins[..., None]) + vein_colour * 0.35 * veins[..., None]

    rust, halo = np.array([0.52, 0.27, 0.09]), np.array([0.80, 0.72, 0.25])
    for cu, cv, r in LESIONS:
        ct = (cv + 1) / 2
        d = np.sqrt(((u - cu) * 1.0) ** 2 + ((t - ct) * 0.55) ** 2)   # stretch: blade is longer than wide
        halo_mask = np.clip(1 - (d - r) / (r * 1.4), 0, 1) ** 2 * (d > r * 0.6)
        core_mask = np.clip(1 - d / r, 0, 1) ** 0.6
        rgb = rgb * (1 - 0.6 * halo_mask[..., None]) + halo * 0.6 * halo_mask[..., None]
        rgb = rgb * (1 - core_mask[..., None]) + rust * core_mask[..., None]

    noise = np.random.default_rng(7).normal(0, 0.012, rgb.shape)
    rgb = np.clip(rgb + noise, 0, 1)

    image = bpy.data.images.new("LeafTexture", TEX, TEX, alpha=False)
    pixels = np.concatenate([rgb, np.ones((TEX, TEX, 1))], axis=2).astype(np.float32)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    return image


def leaf_material(image):
    mat = bpy.data.materials.new("LeafMaterial")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    mat.use_backface_culling = False
    return mat


def build_stem():
    curve = bpy.data.curves.new("LeafStem", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.022
    curve.bevel_resolution = 2
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(1)
    for point, co, handle in zip(spline.bezier_points, ((0, -0.32, -0.08), (0, 0.04, 0)),
                                 ((0, -0.22, 0.01), (0, -0.08, 0.02))):
        point.co = co
        point.handle_left_type = point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new("LeafStem", curve)
    bpy.context.collection.objects.link(obj)
    mat = bpy.data.materials.new("StemMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.20, 0.42, 0.14, 1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    obj.data.materials.append(mat)
    # Convert to mesh so the glTF export is plain geometry
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    obj.select_set(False)
    return obj


def animate(blade):
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 0, LOOP
    keys = blade.data.shape_keys
    keys.name = "LeafKeys"
    curl, flutter = keys.key_blocks["Curl"], keys.key_blocks["Flutter"]
    steps = 10
    for k in range(steps + 1):
        frame = round(LOOP * k / steps)
        phase = 2 * math.pi * k / steps
        curl.value = 0.35 + 0.35 * math.sin(phase)
        flutter.value = 0.5 + 0.5 * math.sin(2 * phase + 0.6)   # twice as fast, offset
        curl.keyframe_insert("value", frame=frame)
        flutter.keyframe_insert("value", frame=frame)
    if keys.animation_data and keys.animation_data.action:
        keys.animation_data.action.name = "Sway"
    curl.value, flutter.value = 0.35, 0.5


def render_still(path):
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (1.5, -0.55, 2.35)
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0, 0.80, -0.20)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    scene.camera = cam

    for name, rot, energy in (("Key", (0.9, 0.2, 0.6), 3.0), ("Fill", (1.2, -0.3, -2.4), 1.0)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.rotation_euler = rot
        bpy.context.collection.objects.link(light)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.9, 0.95, 0.9, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.device = "CPU"
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = 640
    scene.render.image_settings.file_format = "WEBP"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.quality = 82
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return [cam, target]


def main():
    clear_scene()
    bpy.context.preferences.filepaths.save_version = 0   # no leaf.blend1 backups
    blade = build_blade()
    blade.data.materials.append(leaf_material(paint_texture()))
    stem = build_stem()
    root = bpy.data.objects.new("Leaf", None)
    bpy.context.collection.objects.link(root)
    blade.parent = stem.parent = root
    animate(blade)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "leaf.blend"), compress=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in (root, blade, stem):
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT / "leaf.glb"), export_format="GLB", use_selection=True,
        export_animations=True, export_morph=True, export_morph_normal=False,
        export_image_format="JPEG", export_jpeg_quality=82, export_yup=True,
    )

    render_still(OUT / "leaf.webp")
    print("Wrote", *(p.name for p in sorted(OUT.iterdir())))


if __name__ == "__main__" or "bpy" in sys.modules:
    main()
