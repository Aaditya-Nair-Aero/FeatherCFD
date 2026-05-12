import numpy as np

def create_sphere(grid_size, center, radius):
    """Creates a spherical obstacle mask."""
    x = np.linspace(0, grid_size - 1, grid_size)
    y = np.linspace(0, grid_size - 1, grid_size)
    z = np.linspace(0, grid_size - 1, grid_size)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    
    dist_sq = (xv - center[0])**2 + (yv - center[1])**2 + (zv - center[2])**2
    mask = (dist_sq <= radius**2).astype(np.uint8)
    return mask

def create_box(grid_size, min_corner, max_corner):
    """Creates a box obstacle mask."""
    mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    x0, y0, z0 = np.maximum(0, np.array(min_corner).astype(int))
    x1, y1, z1 = np.minimum(grid_size, np.array(max_corner).astype(int))
    mask[x0:x1, y0:y1, z0:z1] = 1
    return mask

def naca_4_digit(x, t):
    """NACA 4-digit airfoil thickness distribution."""
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

def naca_half_thickness(x_norm, thickness_ratio, chord_length):
    """Returns half-thickness at normalized chord position. x_norm in [0,1]."""
    x = np.clip(x_norm, 0, 1)
    yt = naca_4_digit(x, thickness_ratio)
    return yt * chord_length

def create_naca_airfoil_sdf(grid_size, chord_start, chord_length, thickness_ratio, span_start, span_end, aoa_deg=0.0):
    """Signed distance field for a NACA 00xx airfoil extruded along Y.

    Returns float32 array of shape (grid_size, grid_size, grid_size).
    Negative = inside obstacle, Positive = fluid, Zero = surface.
    """
    z_center = (grid_size - 1) / 2.0
    c = chord_length

    X = np.arange(grid_size)
    Y = np.arange(grid_size)
    Z = np.arange(grid_size)
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

    DX = X - chord_start
    DZ = Z - z_center
    angle = np.radians(aoa_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_local = DX * cos_a - DZ * sin_a
    z_local = DX * sin_a + DZ * cos_a

    xn = x_local / c
    h = naca_half_thickness(xn, thickness_ratio, chord_length)

    dist_surface = np.abs(np.abs(z_local) - h)

    dist_le = np.sqrt(x_local**2 + z_local**2)
    dx_te = x_local - c
    dist_te = np.sqrt(dx_te**2 + z_local**2)

    sdf_profile = np.minimum(dist_surface, dist_le)
    sdf_profile = np.minimum(sdf_profile, dist_te)

    outside_left = xn < 0
    outside_right = xn > 1
    sdf_profile = np.where(outside_left, dist_le, sdf_profile)
    sdf_profile = np.where(outside_right, dist_te, sdf_profile)

    inside_profile = (xn >= 0) & (xn <= 1) & (np.abs(z_local) <= h)
    inside_span = (Y >= span_start) & (Y <= span_end)
    dist_span_bottom = np.maximum(0, span_start - Y)
    dist_span_top = np.maximum(0, Y - span_end)
    dist_span_sq = dist_span_bottom**2 + dist_span_top**2

    sdf = np.where(
        inside_span,
        sdf_profile,
        np.sqrt(sdf_profile**2 + dist_span_sq)
    )

    inside_airfoil = inside_profile & inside_span
    sdf = np.where(inside_airfoil, -sdf, np.abs(sdf))

    return sdf.astype(np.float32)

def create_cylinder_sdf(grid_size, cx, cz, radius, y_start, y_end):
    """Signed distance field for a cylinder extruded along Y.
    
    Returns float32 array of shape (grid_size, grid_size, grid_size).
    Negative = inside obstacle.
    """
    X = np.arange(grid_size)
    Y = np.arange(grid_size)
    Z = np.arange(grid_size)
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
    
    dist_xz = np.sqrt((X - cx)**2 + (Z - cz)**2) - radius
    
    inside_span = (Y >= y_start) & (Y <= y_end)
    dist_span_b = np.maximum(0, y_start - Y)
    dist_span_t = np.maximum(0, Y - y_end)
    dist_span = np.sqrt(dist_span_b**2 + dist_span_t**2)
    
    sdf = np.where(inside_span, dist_xz, np.sqrt(dist_xz**2 + dist_span**2))
    
    inside_cyl = inside_span & (dist_xz <= 0)
    sdf = np.where(inside_cyl, sdf, np.abs(sdf))
    
    return sdf.astype(np.float32)

def create_cylinder(grid_size, cx, cz, radius, y_start, y_end):
    """Creates a voxelized cylinder extruded along Y."""
    mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    y0, y1 = max(0, int(y_start)), min(grid_size, int(y_end))
    for y in range(y0, y1):
        for x in range(grid_size):
            for z in range(grid_size):
                if (x - cx)**2 + (z - cz)**2 <= radius**2:
                    mask[x, y, z] = 1
    return mask

def create_naca_airfoil(grid_size, chord_start, chord_length, thickness_ratio, span_start, span_end, aoa_deg=0.0):
    """Creates a voxelized NACA 00xx airfoil with span along Y (vertical)."""
    mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    z_center = (grid_size - 1) / 2.0
    X, Z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    
    DX = X - chord_start
    DZ = Z - z_center
    
    angle = np.radians(aoa_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    X_local = DX * cos_a - DZ * sin_a
    Z_local = DX * sin_a + DZ * cos_a
    
    x_norm = X_local / chord_length
    valid_x = (x_norm >= 0.0) & (x_norm <= 1.0)
    
    yt = naca_4_digit(np.clip(x_norm, 0, 1), thickness_ratio) * chord_length
    is_inside = valid_x & (np.abs(Z_local) <= yt)
    y_min, y_max = int(span_start), int(span_end)
    for j in range(max(0, y_min), min(grid_size, y_max)):
        mask[:, j, :][is_inside] = 1
        
    return mask
