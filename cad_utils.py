import numpy as np
import trimesh
import trimesh.repair
from scipy.ndimage import distance_transform_edt

def load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh object, got {type(mesh)}")
    return mesh

def prepare_mesh(mesh, target_faces=50000):
    if mesh.is_watertight and len(mesh.faces) <= target_faces:
        print("  Mesh is watertight and under face budget, skipping decimation")
        return mesh

    if not mesh.is_watertight:
        print(f"  Repairing non-watertight mesh...")
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        print(f"    Watertight: {mesh.is_watertight}")

    if len(mesh.faces) > target_faces:
        old = len(mesh.faces)
        mesh = mesh.simplify_quadric_decimation(target_faces)
        print(f"  Decimated: {old} → {len(mesh.faces)} faces ({100*len(mesh.faces)/old:.0f}%)")
        trimesh.repair.fix_normals(mesh)

    return mesh

def scale_and_center(mesh, grid_size, cad_chord, aoa_deg=0.0):
    extents = mesh.extents
    longest = max(extents)
    scale = cad_chord / longest
    if abs(scale - 1.0) > 1e-6:
        mesh.apply_scale(scale)

    if aoa_deg != 0.0:
        angle = np.radians(aoa_deg)
        rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        mesh.apply_transform(rot)

    center = np.array([(grid_size - 1) / 2.0] * 3)
    mesh.apply_translation(center - mesh.centroid)

def make_cad_obstacle(mesh, grid_size):
    print("  Voxelizing mesh via trimesh...")
    voxel_grid = mesh.voxelized(pitch=1.0).fill()
    indices = voxel_grid.sparse_indices

    bounds_min = np.floor(mesh.bounds[0]).astype(int)

    mask = np.zeros((grid_size,) * 3, dtype=np.uint8)
    idx = indices + bounds_min
    valid = np.all((idx >= 0) & (idx < grid_size), axis=1)
    idx = idx[valid]

    if len(idx) == 0:
        print("  WARNING: No obstacle cells placed")
        print(f"  Voxel indices: {indices.min(axis=0)} to {indices.max(axis=0)}")
        print(f"  Bounds min: {bounds_min}, Grid: {grid_size}")
    else:
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    n_obstacle = np.sum(mask)
    print(f"  Obstacle cells: {n_obstacle} / {grid_size**3} ({100.0*n_obstacle/grid_size**3:.1f}%)")

    print("  Computing SDF via distance transform...")
    dist_out = distance_transform_edt(1 - mask).astype(np.float32)
    dist_in = distance_transform_edt(mask).astype(np.float32)
    sdf = dist_out - dist_in
    sdf_range = np.max(np.abs(sdf))
    print(f"  SDF range: ±{sdf_range:.1f}")
    return mask, sdf

def summary(mesh):
    return (f"  Vertices: {len(mesh.vertices)}\n"
            f"  Faces: {len(mesh.faces)}\n"
            f"  Bounds: {mesh.bounds[0]} to {mesh.bounds[1]}\n"
            f"  Extents: {mesh.extents}\n"
            f"  Watertight: {mesh.is_watertight}")
