"""Open3D capsule viewer for the student data and submitted paths."""

import colorsys
import copy
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from main import load_data
from evaluate import load_submission


ROOT = Path(__file__).resolve().parent
SOLUTION_FILE = None       # ROOT / "solution.json"：同时打开原始与提交结果窗口
BUNDLE_IDS = None          # 例如 {4, 22, 31}；None 显示全部
BUNDLE_COUNT = None        # 先筛选 ID，再取前 N 条；None 显示全部
SHOW_VOXELS = False
SHOW_AXES = False
RADIUS_SCALE = 1.0         # 仅影响显示，1.0 为输入中的实际半径
CYLINDER_RESOLUTION = 16
SPHERE_RESOLUTION = 8
MAX_VOXEL_POINTS = 100_000
BACKGROUND = np.array([0.06, 0.07, 0.09])
WINDOW_WIDTH, WINDOW_HEIGHT = 900, 800
CAMERA_FILE = ROOT / "view_camera.json"
SCREENSHOT_DIR = ROOT


def bundle_color(bundle_id):
    return colorsys.hsv_to_rgb((bundle_id * 0.61803398875) % 1.0, 0.75, 0.95)


def load_bundles():
    bundles, occupancy, scene = load_data()
    paths = load_submission(SOLUTION_FILE, bundles, scene)
    if BUNDLE_IDS is not None:
        missing = set(BUNDLE_IDS) - {item["id"] for item in bundles}
        if missing:
            raise ValueError(f"unknown bundle ids: {sorted(missing)}")
        bundles = [item for item in bundles if item["id"] in BUNDLE_IDS]
    if BUNDLE_COUNT is not None:
        if not isinstance(BUNDLE_COUNT, int) or BUNDLE_COUNT < 1:
            raise ValueError("BUNDLE_COUNT must be a positive integer or None")
        bundles = bundles[:BUNDLE_COUNT]
    if not bundles:
        raise ValueError("no bundles selected")
    return bundles, paths, occupancy, scene


def rotation_from_z(direction):
    direction = direction / np.linalg.norm(direction)
    cross = np.cross([0.0, 0.0, 1.0], direction)
    dot = np.clip(direction[2], -1.0, 1.0)
    norm = np.linalg.norm(cross)
    if norm < 1e-12:
        return np.eye(3) if dot > 0 else o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0., 0.]))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(cross / norm * np.arccos(dot))


def create_capsule_polyline(points, radius):
    mesh = o3d.geometry.TriangleMesh()
    for a, b in zip(points[:-1], points[1:]):
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length,
                                                            resolution=CYLINDER_RESOLUTION, split=1)
        cylinder.rotate(rotation_from_z(direction), center=(0., 0., 0.))
        cylinder.translate((a + b) * 0.5)
        mesh += cylinder
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=SPHERE_RESOLUTION)
    for point in points:
        cap = copy.deepcopy(sphere)
        cap.translate(point)
        mesh += cap
    mesh.compute_vertex_normals()
    return mesh


def build_bundles(bundles, paths=None):
    geometries = []
    for item in bundles:
        points = item["points"] if paths is None else paths.get(item["id"], item["points"])
        radius = item["radius"] * RADIUS_SCALE
        print(f"Bundle {item['id']:3d}: points={len(points):4d}, display radius={radius:.4f}", flush=True)
        mesh = create_capsule_polyline(points, radius)
        mesh.paint_uniform_color(bundle_color(item["id"]))
        geometries.append(mesh)
    return geometries


def build_voxel_preview(occupancy, scene):
    indices = np.argwhere(occupancy == scene.get("occupied_value", 1))
    if len(indices) > MAX_VOXEL_POINTS:
        indices = indices[np.random.default_rng(42).choice(len(indices), MAX_VOXEL_POINTS, replace=False)]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(scene["origin"]) + (indices + 0.5) * scene["voxel_size"])
    cloud.paint_uniform_color([0.45, 0.46, 0.48])
    return cloud


def camera_changed(first, second):
    return (not np.allclose(first.extrinsic, second.extrinsic, rtol=1e-9, atol=1e-9) or
            not np.allclose(first.intrinsic.intrinsic_matrix, second.intrinsic.intrinsic_matrix, rtol=1e-9, atol=1e-9))


def sync_views(windows, previous):
    current = [window.get_view_control().convert_to_pinhole_camera_parameters() for window in windows]
    changed = next((i for i in range(len(windows)) if camera_changed(current[i], previous[i])), None)
    if changed is not None:
        for i, window in enumerate(windows):
            if i != changed:
                window.get_view_control().convert_from_pinhole_camera_parameters(current[changed], allow_arbitrary=True)
    return [window.get_view_control().convert_to_pinhole_camera_parameters() for window in windows]


def main(show=True):
    if o3d is None:
        raise SystemExit("Open3D is required. Run: python -m pip install -r requirements.txt")
    if not np.isfinite(RADIUS_SCALE) or RADIUS_SCALE <= 0 or MAX_VOXEL_POINTS < 1:
        raise ValueError("RADIUS_SCALE and MAX_VOXEL_POINTS must be positive")
    bundles, paths, occupancy, scene = load_bundles()
    print(f"Loaded {len(bundles)} bundles; capsule geometry: cylinders + vertex spheres", flush=True)
    scenes = [build_bundles(bundles)]
    if SOLUTION_FILE is not None:
        scenes.append(build_bundles(bundles, paths))
    cloud = build_voxel_preview(occupancy, scene) if SHOW_VOXELS else None
    all_points = [item["points"] for item in bundles] + [paths.get(item["id"], item["points"]) for item in bundles]
    if cloud is not None and len(cloud.points):
        all_points.append(np.asarray(cloud.points))
    all_points = np.concatenate(all_points)
    padding = max(item["radius"] for item in bundles) * RADIUS_SCALE
    low, high = all_points.min(axis=0) - padding, all_points.max(axis=0) + padding
    windows = []

    def capture(_=None):
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for i, window in enumerate(windows):
            filename = "preview.png" if len(windows) == 1 else ("comparison_before.png" if i == 0 else "comparison_after.png")
            output = SCREENSHOT_DIR / filename
            window.capture_screen_image(str(output), do_render=True)
            print(f"Screenshot: {output}", flush=True)
        return False

    def save_camera(window):
        o3d.io.write_pinhole_camera_parameters(str(CAMERA_FILE), window.get_view_control().convert_to_pinhole_camera_parameters())
        print(f"Camera saved: {CAMERA_FILE}", flush=True)
        return False

    try:
        for i, geometries in enumerate(scenes):
            window = o3d.visualization.VisualizerWithKeyCallback()
            title = "Original bundles" if i == 0 else "Submitted bundles"
            if not window.create_window(window_name=title, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                                        left=30 + i * (WINDOW_WIDTH + 20), top=50, visible=show):
                raise RuntimeError("Open3D could not create an OpenGL window; check the graphics/display environment")
            windows.append(window)
            for geometry in geometries:
                window.add_geometry(geometry)
            # 两个背景色锚点统一比较窗口的初始包围盒。
            anchors = o3d.geometry.PointCloud()
            anchors.points = o3d.utility.Vector3dVector(np.array([low, high]))
            anchors.paint_uniform_color(BACKGROUND)
            window.add_geometry(anchors)
            if cloud is not None:
                window.add_geometry(cloud)
            if SHOW_AXES:
                window.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=20., origin=low))
            options = window.get_render_option()
            options.background_color = BACKGROUND
            options.mesh_show_back_face = True
            options.point_size = 2.0
            options.light_on = True
            view = window.get_view_control()
            view.set_lookat((low + high) * 0.5)
            view.set_front([1., -.25, .15])
            view.set_up([0., 0., 1.])
            view.set_zoom(.75)
            window.register_key_callback(ord("S"), capture)
            window.register_key_callback(ord("C"), save_camera)
            window.poll_events()
            window.update_renderer()
        camera = (o3d.io.read_pinhole_camera_parameters(str(CAMERA_FILE)) if CAMERA_FILE.is_file()
                  else windows[0].get_view_control().convert_to_pinhole_camera_parameters())
        for window in windows:
            window.get_view_control().convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)
            window.update_renderer()
        previous = [window.get_view_control().convert_to_pinhole_camera_parameters() for window in windows]
        if not show:
            capture()
            return {"bundle_count": len(bundles), "window_count": len(windows), "mesh_count": sum(map(len, scenes))}
        print("Drag to rotate, wheel to zoom. S: screenshots; C: save camera. Comparison windows synchronize views.", flush=True)
        while all([window.poll_events() for window in windows]):
            previous = sync_views(windows, previous)
            for window in windows:
                window.update_renderer()
            time.sleep(.01)
    finally:
        for window in windows:
            window.destroy_window()


if __name__ == "__main__":
    main()
