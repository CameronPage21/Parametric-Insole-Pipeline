# ------------------------------------------------------------------------------------
# Created by Cameron Page, 05/06/2025
# Design Engineering Master's Project
# Imperial College London
# 
# INSTRUCTIONS FOR USE:
# 1) Scroll to the InsoleGenerator class constructor at the bottom of the script
# 2) Provide filepaths for the template insole STL, preprocessed density (pressure) map CSV, and logging file location
# 3) If generating the insole using a 3D foot scan, provide the foot scan STL too. It needs to contain "scan" in the filename
# ------------------------------------------------------------------------------------


import bpy
import bmesh
import numpy as np
import math
import csv
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from matplotlib.path import Path
from scipy.spatial import Delaunay, KDTree, ConvexHull
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata


class InsoleGenerator:
    def __init__(self, 
        template_insole_stl_path, 
        density_map_path, 
        log_file_path,
        foot_side,
        expansion_coefficient, 
        spacing_coefficent,
        target_insole_length = 25.0,
        insole_oversize_factor_length = 1.05,
        insole_oversize_factor_width = 1.05,
        foot_3D_scan_path = None, 
        boundary_offset=0.1, 
        rotate_pressure_CSV_180=False, 
        csv_cell_spacing=0.008382,
        grade_lattice_density_based_on_pressure=True,
        inverse_pressure_grading=False, 
        uniform_density=None, 
        nozzle_diameter=0.4, 
        thickness_multiplier=2.0, 
        ulcer_zones=None,
        contour_method="ORIGINAL_MODEL",
        pressure_points_base_height=0.5,
        pressure_points_top_height=1.5,
        pressure_points_extrapolation_factor=2.5,
        contour_grid_resolution=60,
        draping_grid_resolution=200,
        roof_thickness=0.6,
        floor_thickness=0.6
    ):
        """
        Initialise insole generator.
        Align base insole model to known position. Rotate and rescale.
        Load pressure map from CSV file.
        """
        self.mapping_expansion = 1.0
        self.fallback_threshold_multiplier = 4.0
        self.grid_multiplier = 1.2
        self.template_insole_stl_path = template_insole_stl_path
        self.density_map_path = density_map_path
        self.foot_3D_scan_path = foot_3D_scan_path
        self.log_file_path = log_file_path
        self.foot_side = foot_side
        self.target_insole_length = target_insole_length  # derived from shoe size, in cm
        self.insole_oversize_factor_x = insole_oversize_factor_width
        self.insole_oversize_factor_y = insole_oversize_factor_length
        self.expansion_coefficient = expansion_coefficient
        self.spacing_coefficent = spacing_coefficent
        self.cell_size = (csv_cell_spacing *100) / spacing_coefficent  # Base honeycomb cell size (also used for interior sampling spacing)
        self.boundary_offset = boundary_offset
        self.rotate_180 = rotate_pressure_CSV_180
        if grade_lattice_density_based_on_pressure:
            self.uniform_density = None
        else:
            self.uniform_density = uniform_density
        self.inverse_pressure_grading = inverse_pressure_grading
        self.nozzle_diameter = nozzle_diameter / 10  # Convert from mm into Blender units
        self.csv_cell_spacing = csv_cell_spacing  # in metres
        self.thickness_multiplier = thickness_multiplier
        self.ulcer_zones = ulcer_zones or []   # ulcer exclusion zones as a list of (x, y, radius) tuples
        self.contour_method = contour_method
        self.cell_height = 1.5   # Height that the infill pattern gets extruded upwards
        self.contouring_depth = abs(pressure_points_top_height - pressure_points_base_height)
        self.pressure_points_base_height = pressure_points_base_height
        self.pressure_points_extrapolation_factor = 2.5
        self.contour_grid_resolution = contour_grid_resolution
        self.draping_grid_resolution = draping_grid_resolution
        self.contour_interpolation_method = 'cubic'
        self.roof_thickness = roof_thickness    # in mm
        self.floor_thickness = floor_thickness  # in mm
        
        # Start a new log file
        self.overwrite_log("Starting insole generation script...")
        
        # Delete all objects except the foot model (if it exists from the previous run)
        foot_model_left_over = False
        for obj in bpy.data.objects:
            if self.foot_3D_scan_path is not None:
                if "scan" in obj.name or "Scan" in obj.name:
                    self.log_to_file("Leftover foot model detected")
                    self.foot_scan_model = obj 
                    foot_model_left_over = True
                    continue
            bpy.data.objects.remove(obj, do_unlink=True)
        
        if self.foot_3D_scan_path is not None and not foot_model_left_over:
            self.log_to_file("Leftover foot model not detected, adding new model")
        
        # Load density map from CSV
        self.contoured_density_map, self.density_map, self.non_inverse_preesure_map = self.load_density_map()
        
        # Import STL and prepare the model (including automatic rescaling and alignment)
        self.import_stl()
        self.prepare_model(self.obj)
        self.align_rotation(self.obj, self.rotate_180)
        
        if self.foot_side == "LEFT":
            self.mirror_insole_XZ()
        
        # Import, prepare and align the 3D scan, if one was provided
        if self.foot_3D_scan_path is not None and not foot_model_left_over:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.wm.stl_import(filepath=self.foot_3D_scan_path)
            self.foot_scan_model = bpy.context.selected_objects[0]
            self.prepare_model(self.foot_scan_model)
            self.align_rotation(self.foot_scan_model, False)
            self.align_translation(self.foot_scan_model, False)
            self.center_on_X_axis(self.foot_scan_model)
            
        self.scale_insole_to_patient()
        self.align_translation(self.obj, True)
        
        self.aligned_boundary, self.aligned_faces = self.extract_2d_boundary()
        self.log_to_file("Aligned boundary computed: " + str(len(self.aligned_boundary)) + " vertices, " +
                   str(len(self.aligned_faces)) + " faces.")

        
    def log_to_file(self, message):
        with open(self.log_file_path, "a") as file:
            file.write(message + "\n")
        
    def overwrite_log(self, message):
        with open(self.log_file_path, "w") as file:
            file.write(message + "\n")

    def load_density_map(self):
        """Load density values from CSV."""
        with open(self.density_map_path, 'r') as file:
            reader = csv.reader(file)
            data = np.array([[float(val) for val in row] for row in reader])
        data[data < 0] = 0
        data_min = data.min()
        data_max = data.max()
        if abs(data_max - data_min) < 1e-3:
            raise ValueError("Inputted pressure map contains little or no pressure variation")
        self.log_to_file(f"CSV max pressure = {data_max}, min pressure = {data_min}")
        normalized_data = (data - data_min) / (data_max - data_min)
        data_min = normalized_data.min()
        data_max = normalized_data.max()
        self.log_to_file(f"Normalised CSV max pressure = {data_max}, min pressure = {data_min}")

        if self.uniform_density is not None:
            normalized_data_copy = normalized_data.copy()
            normalized_data[:] = self.uniform_density
            return normalized_data_copy, normalized_data, None
        else:
            normalized_data_copy = normalized_data.copy()
            if not self.inverse_pressure_grading:
                for i, row in enumerate(normalized_data):
                    for j, value in enumerate(row):
                        normalized_data[i,j] = 1 - normalized_data[i,j]
            return None, normalized_data, normalized_data_copy

    def import_stl(self):
        """Import STL file into Blender."""
        bpy.ops.wm.stl_import(filepath=self.template_insole_stl_path)
        self.obj = bpy.context.selected_objects[0]
        
    def prepare_model(self, target_obj):
        target_obj.location = (0, 0, 0)
        target_obj.rotation_euler = (0, 0, 0)
        bounds = self.get_object_bounds(target_obj)
        Lx = bounds['max'].x - bounds['min'].x
        Ly = bounds['max'].y - bounds['min'].y
        longest_xy = max(Lx, Ly)
        self.log_to_file("Original XY longest dimension: " + str(longest_xy))
        
        target = 25.0
        required_factor = target / longest_xy
        exponent = round(math.log10(required_factor))
        scale_factor = 10 ** exponent
        self.log_to_file("Scaling factor: " + str(scale_factor))
        target_obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.context.view_layer.objects.active = target_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bounds = self.get_object_bounds(target_obj)
        self.log_to_file("Scaled model bounding box: min: " + str(bounds['min']) + ", max: " + str(bounds['max']))
        bpy.context.view_layer.update()
        
    def align_rotation(self, target_obj, rotate_180):
        mesh = target_obj.data
        vertices = [target_obj.matrix_world @ v.co for v in mesh.vertices]
        pts = np.array([[v.x, v.y] for v in vertices])
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        angle = math.atan2(principal[1], principal[0])
        angle_degs = math.degrees(-angle)
        snapped_degs = 90 * round(angle_degs / 90)
        snapped_rads = math.radians(snapped_degs)
        rot_mat = Matrix.Rotation(snapped_rads, 4, 'Z')
        target_obj.matrix_world = rot_mat @ target_obj.matrix_world
        self.log_to_file(f"Applied snapped rotation: original angle = {angle_degs:.2f}°, snapped to {snapped_degs}° ({snapped_rads:.2f} rad).")
        
        if not rotate_180:
            rot_180 = Matrix.Rotation(math.pi, 4, 'Z')
            target_obj.matrix_world = rot_180 @ target_obj.matrix_world
            self.log_to_file("Applied additional 180° rotation")
        
        
    def align_translation(self, target_obj, insole):
        mesh = target_obj.data
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = target_obj.evaluated_get(depsgraph)
        evaluated_verts = [eval_obj.matrix_world @ v.co for v in eval_obj.data.vertices]
        v_min = min(evaluated_verts, key=lambda v: v.x)
        self.log_to_file("Designated vertex (lowest X) after rotations: " + str(v_min))
        
        M = target_obj.matrix_world.copy()
        # Set translation so that the vertex with lowest X becomes (0,0) in XY.
        M.translation = Vector((-v_min.x, -v_min.y, M.translation.z))
        target_obj.matrix_world = M
        if insole:
            # Store the translation offset for later use.
            self.translation_offset = M.translation.copy()
        self.log_to_file("Set object translation to " + str(M.translation) + " so that designated vertex is at (0,0) in XY.")
        eval_obj.to_mesh_clear()
        
        evaluated_verts = [target_obj.matrix_world @ v.co for v in mesh.vertices]
        zs = [v.z for v in evaluated_verts]
        minZ = min(zs)
        if minZ < 0:
            M = target_obj.matrix_world.copy()
            M.translation.z -= minZ
            target_obj.matrix_world = M
            self.log_to_file("Adjusted Z by " + str(-minZ) + " so that object sits on the XY plane.")
        
        bpy.context.view_layer.update()

    def center_on_X_axis(self, target_obj):
        world_y_coords = []
        for local_corner_coords in target_obj.bound_box:
            local_corner_vector = Vector(local_corner_coords)
            world_corner_vector = target_obj.matrix_world @ local_corner_vector
            world_y_coords.append(world_corner_vector.y)

        target_obj.location.y -= (max(world_y_coords) + min(world_y_coords)) / 2
        bpy.context.view_layer.update()
    
    def mirror_insole_XZ(self):
        bpy.context.view_layer.objects.active = self.obj
        self.obj.select_set(True)
        bpy.ops.transform.mirror(orient_type='GLOBAL', constraint_axis=(False, True, False))
    
    def scale_insole_to_patient(self):
        obj_to_scale = self.obj
        
        # Get current dimensions of the object to scale
        current_dims_x = obj_to_scale.dimensions.x
        current_dims_y = obj_to_scale.dimensions.y
        
        # Scale to the 3D foot scan if it exists. If not, fallback to the specified shoe size
        if self.contour_method == "3D_SCAN":
            target_dims_x = self.foot_scan_model.dimensions.x
            target_dims_y = self.foot_scan_model.dimensions.y
        else:
            target_dims_y = self.target_insole_length
        
        # --- Calculate Y scale factor ---
        scale_factor_y = 1.0
        if current_dims_y != 0:
            scale_factor_y = target_dims_y / current_dims_y
        elif target_dims_y == 0:  # current_dims_y is 0 and target_dims_y is 0
            scale_factor_y = 1.0
        # If current_dims_y is 0 and target_dims_y is non-zero, scale_factor_y remains 1.0.
        
        if self.contour_method == "3D_SCAN":
            # --- Calculate X scale factor ---
            scale_factor_x = 1.0
            if current_dims_x != 0:
                scale_factor_x = target_dims_x / current_dims_x
            elif target_dims_x == 0:  # current_dims_x is 0 and target_dims_x is 0
                scale_factor_x = 1.0
            # If current_dims_x is 0 and target_dims_x is non-zero, scale_factor_x remains 1.0.   
        else:
            scale_factor_x = scale_factor_y
        
        scale_factor_x *= self.insole_oversize_factor_x
        scale_factor_y *= self.insole_oversize_factor_y

        # Apply the scale factors to the object's existing local scale
        obj_to_scale.scale.x *= scale_factor_x
        obj_to_scale.scale.y *= scale_factor_y

        # Adjust the object's location.x and location.y components.
        obj_to_scale.location.x *= scale_factor_x
        obj_to_scale.location.y *= scale_factor_y
        
        bpy.context.view_layer.update()
        self.log_to_file(f"Scaled insole X by {scale_factor_x}x and Y by {scale_factor_y}x to fit footprint of 3D foot STL")

        # Translate insole in Z it sits on the XY plane ----------------------

        # Get the 8 corners of the object's local bounding box
        local_bbox_corners = [Vector(corner) for corner in obj_to_scale.bound_box]

        # Transform these corners to world space using the object's current world matrix
        world_bbox_corners_z = [(obj_to_scale.matrix_world @ corner).z for corner in local_bbox_corners]

        min_world_z = min(world_bbox_corners_z)

        # Move the insole lowest point to Z = 0
        z_offset = -min_world_z
        new_world_z = obj_to_scale.matrix_world.translation.z + z_offset
        obj_to_scale.matrix_world.translation.z = new_world_z
        bpy.context.view_layer.update()


    def get_object_bounds(self, target_obj):
        return {
            'min': Vector(target_obj.bound_box[0]),
            'max': Vector(target_obj.bound_box[6])
        }
    
    def extract_2d_boundary(self, alpha=0.1):
        """
        Directly extract the 2D boundary of the model’s projection using an alpha shape
        (concave hull) approach. The resulting boundary will follow the actual silhouette
        more closely than a convex hull.
        
        Returns:
            final_vertices, final_faces: vertices (as Vectors) and face indices from a Delaunay
            triangulation over the boundary and interior sample points.
        """
        mapping_expansion = self.mapping_expansion
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(temp_mesh)
        coords = []
        for v in bm.verts:
            co_world = self.obj.matrix_world @ v.co
            coords.append((co_world.x, co_world.y))
        bm.free()
        eval_obj.to_mesh_clear()
        
        points = np.array(coords)
        if len(points) < 4:
            raise ValueError("Not enough points for alpha shape extraction.")
        
        # Compute the alpha shape (concave hull) as a set of boundary edges.
        boundary_edges = self.alpha_shape(points, alpha)
        # Order the boundary edges to form a continuous polygon.
        ordered_indices = self.order_boundary_edges(boundary_edges)
        boundary_coords = [points[i] for i in ordered_indices]
        
        # --- New step: apply the boundary_offset ---
        # For each boundary vertex, compute an approximate inward normal.
        # (One way is to compute the vector from the vertex to the centroid of the boundary.)
        centroid = np.mean(boundary_coords, axis=0)
        offset_boundary_coords = []
        for pt in boundary_coords:
            direction = centroid - pt
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = np.array([0, 0])
            offset_pt = pt + direction * self.boundary_offset
            offset_boundary_coords.append(offset_pt.tolist())
        
        # Continue using offset_boundary_coords as the final boundary.
        final_boundary = [Vector((pt[0], pt[1], 0)) for pt in offset_boundary_coords]
        self.log_to_file("Alpha shape boundary extracted with " + str(len(final_boundary)) + " vertices.")
        
        # Optionally, sample interior points from the polygon.
        poly_path = Path(boundary_coords)
        
        # Determine original bounds from the boundary coordinates.
        xs, ys = zip(*boundary_coords)
        orig_xmin, orig_xmax = min(xs), max(xs)
        orig_ymin, orig_ymax = min(ys), max(ys)
        cx = (orig_xmin + orig_xmax) / 2.0
        cy = (orig_ymin + orig_ymax) / 2.0
        # Expand bounds using mapping_expansion factor.
        width = (orig_xmax - orig_xmin) * mapping_expansion
        height = (orig_ymax - orig_ymin) * mapping_expansion
        x_min_exp = cx - width / 2.0
        x_max_exp = cx + width / 2.0
        y_min_exp = cy - height / 2.0
        y_max_exp = cy + height / 2.0

        # To avoid an extremely dense grid, limit the number of sampling steps.
        MAX_STEPS = 100  # maximum steps allowed in either direction
        # Use self.cell_size as the desired sampling resolution.
        num_steps_x = (x_max_exp - x_min_exp) / self.cell_size
        if num_steps_x > MAX_STEPS:
            step_x = (x_max_exp - x_min_exp) / MAX_STEPS
        else:
            step_x = self.cell_size

        num_steps_y = (y_max_exp - y_min_exp) / self.cell_size
        if num_steps_y > MAX_STEPS:
            step_y = (y_max_exp - y_min_exp) / MAX_STEPS
        else:
            step_y = self.cell_size

        sample_points = []
        for x in np.arange(x_min_exp, x_max_exp, step_x):
            for y in np.arange(y_min_exp, y_max_exp, step_y):
                # Only add the sample point if it is within the original polygon (defined by boundary_coords).
                if poly_path.contains_point((x, y)):
                    sample_points.append((x, y))
        self.log_to_file("Sampled " + str(len(sample_points)) + " interior points from the expanded region.")
        
        combined_points = boundary_coords + sample_points
        combined_points = np.array(combined_points)
        
        # Perform Delaunay triangulation on all points.
        delaunay = Delaunay(combined_points)
        all_triangles = delaunay.simplices
        self.log_to_file("Delaunay produced " + str(len(all_triangles)) + " triangles before filtering.")
        
        final_faces = []
        for tri in all_triangles:
            pts = combined_points[tri]
            centroid_tri = (pts[0] + pts[1] + pts[2]) / 3.0
            if poly_path.contains_point(centroid_tri):
                final_faces.append(list(tri))
        final_vertices = [Vector((pt[0], pt[1], 0)) for pt in combined_points]
        # Store mapping field data for later visualization.
        self.mapping_points = final_vertices
        self.mapping_faces = final_faces
        return final_vertices, final_faces

    def alpha_shape(self, points, alpha):
        """
        Compute the alpha shape (concave hull) of a set of 2D points.
        Returns a set of edges (as tuples of point indices) that lie on the boundary.
        
        Args:
            points: (N,2) numpy array of points.
            alpha: alpha parameter (a smaller alpha yields a more concave hull).
        """
        if len(points) < 4:
            return set()
        tri = Delaunay(points)
        edges = {}
        for simplex in tri.simplices:
            pa, pb, pc = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            a = np.linalg.norm(pb - pc)
            b = np.linalg.norm(pa - pc)
            c = np.linalg.norm(pa - pb)
            s = (a + b + c) / 2.0
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
            if area == 0:
                circum_r = float('inf')
            else:
                circum_r = a * b * c / (4.0 * area)
            if circum_r < 1.0 / alpha:
                for i in range(3):
                    edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                    edges[edge] = edges.get(edge, 0) + 1
        boundary_edges = {edge for edge, count in edges.items() if count == 1}
        return boundary_edges

    def order_boundary_edges(self, boundary_edges):
        """
        Order a set of boundary edges (given as pairs of indices) into a continuous polygon.
        Returns a list of vertex indices in order.
        """
        # if there are no boundary edges, return an empty loop
        if not boundary_edges:
            return []
        
        connectivity = {}
        for edge in boundary_edges:
            i, j = edge
            connectivity.setdefault(i, []).append(j)
            connectivity.setdefault(j, []).append(i)
            
        # if for some reason we still have no vertices, bail
        if not connectivity:
            return []
            
        # Choose a starting vertex (if open, one with a single neighbor; otherwise arbitrary).
        start = None
        for vertex, neighbors in connectivity.items():
            if len(neighbors) == 1:
                start = vertex
                break
        if start is None:
            start = next(iter(connectivity))
        ordered = [start]
        current = start
        prev = None
        while True:
            neighbors = connectivity[current]
            next_vertex = None
            for n in neighbors:
                if n != prev:
                    next_vertex = n
                    break
            if next_vertex is None or next_vertex == start:
                break
            ordered.append(next_vertex)
            prev, current = current, next_vertex
        return ordered
    
    def segment_intersection(self, p, p2, q, q2, eps=1e-6):
        """
        Compute the intersection between two line segments [p, p2] and [q, q2] in 2D.
        Returns (t, u, intersection_point) where t is the parameter along [p,p2] and u along [q,q2],
        or None if there is no valid intersection.
        """
        # Convert vectors to 2D by taking only x and y components.
        p2d   = Vector((p.x, p.y))
        p2d2  = Vector((p2.x, p2.y))
        q2d   = Vector((q.x, q.y))
        q2d2  = Vector((q2.x, q2.y))
        
        r = p2d2 - p2d
        s = q2d2 - q2d
        r_cross_s = r.x * s.y - r.y * s.x
        if abs(r_cross_s) < eps:
            # The lines are parallel or collinear.
            return None
        diff = q2d - p2d
        t = (diff.x * s.y - diff.y * s.x) / r_cross_s
        u = (diff.x * r.y - diff.y * r.x) / r_cross_s
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point_2d = p2d + r * t
            # Return as a 3D vector with z=0.
            return (t, u, Vector((intersection_point_2d.x, intersection_point_2d.y, 0)))
        else:
            return None
        
    def get_outer_clipping_boundary(self, alpha=0.1):
        """
        Recompute the outer boundary for clipping by using only the original projected 
        vertices and applying the alpha shape and ordering routines (with the boundary offset).
        Returns a list of Vectors (with z = 0) representing the outer contour used for clipping.
        """
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(temp_mesh)
        coords = []
        for v in bm.verts:
            co_world = self.obj.matrix_world @ v.co
            coords.append((co_world.x, co_world.y))
        bm.free()
        eval_obj.to_mesh_clear()

        points = np.array(coords)
        if len(points) < 4:
            raise ValueError("Not enough points for alpha shape boundary extraction.")
        boundary_edges = self.alpha_shape(points, alpha)
        ordered_indices = self.order_boundary_edges(boundary_edges)
        boundary_coords = [points[i] for i in ordered_indices]

        # Apply the boundary offset so the boundary is shifted inward.
        centroid = np.mean(boundary_coords, axis=0)
        offset_boundary_coords = []
        for pt in boundary_coords:
            direction = centroid - pt
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = np.array([0, 0])
            offset_pt = pt + direction * self.boundary_offset
            offset_boundary_coords.append(offset_pt.tolist())

        outer_boundary = [Vector((pt[0], pt[1], 0)) for pt in offset_boundary_coords]
        return outer_boundary

    def clip_hexagon_cell(self, cell, clip_polygon, clip_path, max_divisions=10, tol=1e-6):
        """
        Clip one hexagon cell against both the alpha‐shape boundary and ulcer circles,
        subdividing edges at every intersection and retaining only the pieces whose
        midpoints are inside the insole (clip_path) and outside all ulcer_zones.
        """
        clipped_edges = []
        for i in range(len(cell)):
            A = cell[i]
            B = cell[(i+1) % len(cell)]
            d = B - A

            # 1) collect subdivision parameters t in [0,1]
            t_vals = {0.0, 1.0}

            # 1a) alpha‐shape boundary intersections
            for j in range(len(clip_polygon)):
                C = Vector(clip_polygon[j])
                D = Vector(clip_polygon[(j+1) % len(clip_polygon)])
                # intersect segment A->B with C->D
                res = self.segment_intersection(A, B, C, D, eps=tol)
                if res:
                    t, _, _ = res
                    t_vals.add(max(0.0, min(1.0, t)))

            # 1b) ulcer‐circle intersections
            for (ux, uy, ur) in self.ulcer_zones:
                # solve |A + d*t - U|^2 = ur^2
                fx = A.x - ux
                fy = A.y - uy
                dx = d.x
                dy = d.y
                a = dx*dx + dy*dy
                b = 2*(fx*dx + fy*dy)
                c = fx*fx + fy*fy - ur*ur
                disc = b*b - 4*a*c
                if disc <= 0:
                    continue
                sqrtD = math.sqrt(disc)
                for sign in (+1, -1):
                    t = (-b + sign*sqrtD) / (2*a)
                    if tol < t < 1-tol:
                        t_vals.add(t)

            # limit total subdivisions
            if len(t_vals) > max_divisions + 1:
                # keep only the smallest (max_divisions+1) Ts by sampling
                t_vals = set(sorted(t_vals)[: max_divisions+1])

            # 2) build and test each sub‐segment
            segs = []
            t_list = sorted(t_vals)
            for t0, t1 in zip(t_list, t_list[1:]):
                P0 = A + d * t0
                P1 = A + d * t1
                mid = A + d * ((t0 + t1)/2)

                # must lie inside the insole
                if not clip_path.contains_point((mid.x, mid.y), radius=tol):
                    continue
                # must lie outside all ulcer circles
                if any((mid.x-ux)**2 + (mid.y-uy)**2 < ur*ur
                       for ux, uy, ur in self.ulcer_zones):
                    continue

                segs.append((P0, P1))
            clipped_edges.append(segs)

        # if all edges vanished, return None
        if all(len(segs)==0 for segs in clipped_edges):
            return None
        return {"original_cell": cell, "clipped_edges": clipped_edges}
    
    def compute_mapping_matrices(self, vertices, faces):
        n_vertices = len(vertices)
        rows = []
        cols = []
        data = []
        b_p_list = []
        b_q_list = []
        row_counter = 0

        for face in faces:
            if len(face) != 3:
                continue
            i, j, k = face
            v1 = vertices[i]
            v2 = vertices[j]
            v3 = vertices[k]
            x1, y1 = v1.x, v1.y
            x2, y2 = v2.x, v2.y
            x3, y3 = v3.x, v3.y

            det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(det) < 1e-12:
                continue

            b1 = (y2 - y3) / det
            c1 = (x3 - x2) / det
            b2 = (y3 - y1) / det
            c2 = (x1 - x3) / det
            b3 = (y1 - y2) / det
            c3 = (x2 - x1) / det

            rows.extend([row_counter, row_counter, row_counter])
            cols.extend([i, j, k])
            data.extend([b1, b2, b3])
            rows.extend([row_counter+1, row_counter+1, row_counter+1])
            cols.extend([i, j, k])
            data.extend([c1, c2, c3])
            
            center = (v1 + v2 + v3) / 3.0
            density = self.get_density_at_point(center)
            density = np.clip(density, 0, 1)
            expansion = math.exp(self.expansion_coefficient * density)
            
            b_p_list.extend([expansion, 0])
            b_q_list.extend([0, expansion])
            
            row_counter += 2

        A = csr_matrix((data, (rows, cols)), shape=(row_counter, n_vertices))
        b_p_arr = np.array(b_p_list)
        b_q_arr = np.array(b_q_list)
        return A, b_p_arr, b_q_arr
    
    def get_density_at_point(self, point):
        spacing = self.csv_cell_spacing * 100
        M, N = self.density_map.shape
        r_index = M - 1 - int(round(point.x / spacing))
        c_index = int(round(point.y / spacing + (N - 1) / 2))
        r_index = max(0, min(r_index, M - 1))
        c_index = max(0, min(c_index, N - 1))
        return self.density_map[r_index, c_index]
    
    def generate_hexagon_vertices(self, center_x, center_y, s):
        verts = []
        angles_deg = [90, 150, 210, 270, 330, 30]
        for angle_deg in angles_deg:
            angle = math.radians(angle_deg)
            x = center_x + s * math.cos(angle)
            y = center_y + s * math.sin(angle)
            verts.append(Vector((x, y, 0)))
        return verts

    def generate_honeycomb_pattern(self, mapped_vertices, cell_size):
        """
        Generate a uniform honeycomb (hexagon) pattern in the deformed domain.
        
        Args:
            mapped_vertices: List of Vector objects representing the deformed domain points.
            cell_size: Base size for the hexagon cells.
            grid_multiplier: Factor by which to expand the bounding box used for generating the grid.
                             For example, grid_multiplier=2 will approximately double the number of rows
                             and columns compared to the natural bounds of mapped_vertices.
                             
        Returns:
            vertices, edges: The list of hexagon vertices and corresponding edge connectivity.
        """
        grid_multiplier = self.grid_multiplier
        
        if not mapped_vertices:
            raise ValueError("Mapped vertices list is empty.")
        
        # Get the original bounding box from the deformed vertices.
        pts = np.array([[v.x, v.y] for v in mapped_vertices])
        orig_min_x = pts[:, 0].min()
        orig_max_x = pts[:, 0].max()
        orig_min_y = pts[:, 1].min()
        orig_max_y = pts[:, 1].max()
        
        # Compute the center of the original bounding box.
        cx = (orig_min_x + orig_max_x) / 2.0
        cy = (orig_min_y + orig_max_y) / 2.0
        
        # Expand the bounding box symmetrically.
        width = (orig_max_x - orig_min_x) * grid_multiplier
        height = (orig_max_y - orig_min_y) * grid_multiplier
        new_min_x = cx - width / 2.0
        new_max_x = cx + width / 2.0
        new_min_y = cy - height / 2.0
        new_max_y = cy + height / 2.0
        
        # Define grid spacing for a hexagon grid.
        x_spacing = cell_size * math.sqrt(3)
        y_spacing = 1.5 * cell_size
        
        vertices = []
        edges = []
        
        # Instead of using the original polygon test to restrict hexagons,
        # we generate hexagon centers over the full expanded bounding box.
        x_start = new_min_x
        x_end = new_max_x
        y_start = new_min_y
        y_end = new_max_y
        
        row = 0
        y = y_start
        while y <= y_end:
            # For staggered rows, offset the x-position.
            x_offset = x_spacing / 2 if (row % 2 == 1) else 0
            x = x_start
            
            while x <= x_end:
                center = Vector((x + x_offset, y, 0))
                # Always add the hexagon center—later stages (clipping, inverse mapping)
                # will handle those that fall too far outside.
                hex_verts = self.generate_hexagon_vertices(center.x, center.y, cell_size)
                start_idx = len(vertices)
                vertices.extend(hex_verts)
                for i in range(6):
                    edges.append((start_idx + i, start_idx + ((i + 1) % 6)))
                x += x_spacing
            row += 1
            y += y_spacing
        
        return vertices, edges
    
    def map_vertices_back(self, pattern_verts, original_vertices, original_faces, mapped_vertices):
        """
        Inverse-map each vertex in pattern_verts from the deformed domain back to the original domain.
        
        For each vertex, this function attempts to find a containing triangle in the Delaunay triangulation
        (represented by original_faces over 'mapped_vertices') and uses barycentric interpolation.
        If a vertex falls outside the mapping domain (i.e. no containing triangle is found), it is temporarily
        marked as a fallback.
        
        After processing all vertices, for each fallback vertex:
          - Find the nearest neighbour (in the deformed domain, from pattern_verts) that was successfully mapped.
          - If that neighbour is within a threshold distance (fallback_threshold_multiplier * cell_size),
            compute the local transformation vector (mapped_neighbour - neighbour_original) and apply it to the fallback vertex.
          - Otherwise, leave the fallback vertex at its original (deformed) location.
        
        Returns:
            A list of transformed vertices.
        """
        fallback_threshold_multiplier=self.fallback_threshold_multiplier
        
        mapped_back = []
        fallback_indices = []  # Indices where mapping failed.

        # First pass: attempt to map every vertex via barycentrics.
        for idx, vert in enumerate(pattern_verts):
            face_idx, bary = self.find_containing_face(vert, mapped_vertices, original_faces)
            if face_idx is not None:
                tri = [original_vertices[i] for i in original_faces[face_idx]]
                mapped_vert = tri[0] * bary[0] + tri[1] * bary[1] + tri[2] * bary[2]
                mapped_back.append(mapped_vert)
            else:
                mapped_back.append(None)
                fallback_indices.append(idx)
        self.fallback_indices = fallback_indices  # Optionally store for debugging.

        # Define the threshold distance.
        threshold_distance = self.cell_size * fallback_threshold_multiplier

        # Second pass: for each fallback, find the nearest successfully mapped neighbor.
        for idx in fallback_indices:
            fallback_vert = pattern_verts[idx]
            best_distance = float('inf')
            best_neighbor = None
            # Compare fallback vertex with all vertices that were successfully mapped.
            for j, candidate in enumerate(pattern_verts):
                if mapped_back[j] is not None:
                    dist = (fallback_vert - candidate).length
                    if dist < best_distance:
                        best_distance = dist
                        best_neighbor = j
            if best_neighbor is not None and best_distance <= threshold_distance:
                # Compute the neighbor's local transformation vector.
                transformation = mapped_back[best_neighbor] - pattern_verts[best_neighbor]
                # Apply the same transformation to the fallback vertex.
                mapped_back[idx] = fallback_vert + transformation
            else:
                # No suitably close neighbor; leave the vertex unmapped.
                mapped_back[idx] = fallback_vert.copy()

        return mapped_back
    
    def find_containing_face(self, point, mapped_vertices, faces):
        for idx, face in enumerate(faces):
            if len(face) != 3:
                continue
            v1, v2, v3 = mapped_vertices[face[0]], mapped_vertices[face[1]], mapped_vertices[face[2]]
            bary = self.calculate_barycentric(point, v1, v2, v3)
            if bary is not None and all(-1e-4 <= b <= 1+1e-4 for b in bary):
                return idx, bary
        return None, None
    
    def calculate_barycentric(self, p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return None
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return (u, v, w)

    def generate_2d_infill(self):
        """
        Generates a single 2D mesh containing edges for the graded infill pattern,
        the outer boundary, and the ulcer cutouts, suitable for direct processing
        (e.g., extrusion and thickening) without intermediate curve conversion
        or complex booleans.

        Workflow:
        1. Calculate lattice pattern edges after clipping (as before).
        2. Calculate outer boundary edges (including ulcer intersections).
        3. Calculate interior ulcer hole edges.
        4. Combine all vertices and edges into a single BMesh.
        5. Remove duplicate vertices to connect the network cleanly.
        6. Output the single mesh object containing only the combined edges.
        """
        self.log_to_file("Generating unified 2D edge mesh (Lattice + Boundary + Ulcers)...")

        # --- Steps 1-5: Calculate clipped lattice segments ---
        # (Same as before, up to calculating clipped_cells)
        vertices, faces = self.extract_2d_boundary() # Needed for mapping
        # ... (Compute mapping fields phi_p, phi_q) ...
        A, b_p, b_q = self.compute_mapping_matrices(vertices, faces)
        reg_term = diags([1e-6] * A.shape[1])
        A_reg = A.T @ A + reg_term
        phi_p = spsolve(A_reg, A.T @ b_p)
        phi_q = spsolve(A_reg, A.T @ b_q)
        self.log_to_file("Computed phi fields.")
        mapped_vertices = [Vector((p, q, 0)) for p, q in zip(phi_p, phi_q)]
        mapped_vertices = [v + self.translation_offset for v in mapped_vertices] # Apply offset if used

        # Get clipping boundary polygon and path
        outer_boundary_verts = self.get_outer_clipping_boundary(alpha=0.1)
        clip_polygon = [(v.x, v.y) for v in outer_boundary_verts]
        clip_path = Path(clip_polygon)

        # Generate honeycomb pattern
        pattern_verts, _ = self.generate_honeycomb_pattern(mapped_vertices, self.cell_size)
        if not pattern_verts: raise ValueError("Failed to generate honeycomb pattern.")

        # Inverse map pattern
        original_pattern_verts = self.map_vertices_back(pattern_verts, vertices, faces, mapped_vertices)
        if not original_pattern_verts: raise ValueError("Inverse mapping failed.")

        # Group into hexagon cells
        num_cells = len(original_pattern_verts) // 6
        if len(original_pattern_verts) % 6 != 0: self.log_to_file("Warning: Hex grid vertex count not multiple of 6.")
        cells = [original_pattern_verts[i*6:(i+1)*6] for i in range(num_cells)]

        # Clip hexagon cells
        clipped_lattice_segments = [] # Store tuples of (Vector, Vector)
        for cell in cells:
            clipped = self.clip_hexagon_cell(cell, clip_polygon, clip_path, max_divisions=5)
            if clipped:
                for edge_list in clipped["clipped_edges"]:
                    clipped_lattice_segments.extend(edge_list) # Add (v1, v2) tuples
        self.log_to_file(f"Generated {len(clipped_lattice_segments)} clipped lattice edge segments.")

        self.boundary_curve = None # Initialize class attribute
        temp_curve_gen_objects = [] # Track objects ONLY for curve generation cleanup

        # --- Calculate Boundary/Ulcer Components (for curve AND final mesh) ---
        self.log_to_file("Calculating boundary and ulcer components...")
        boundary_ulcer_segments = [] # For final unified mesh
        temp_boundary_vertices = [] # For temp boundary mesh obj
        temp_boundary_edges = []    # For temp boundary mesh obj
        ulcer_curves_dict = {}      # For mesh_and_ulcers_to_boundary_curve
        num_zones = len(self.ulcer_zones) if self.ulcer_zones else 0
        circle_edge_angles = { zi: [] for zi in range(num_zones) }
        insole_loop = outer_boundary_verts
        insole_path_for_check = clip_path # Use path object created earlier

        # 1) Process insole_loop edges
        for i in range(len(insole_loop)):
             # ... [Logic to calculate intersections, store angles] ...
             a = insole_loop[i]; b = insole_loop[(i + 1) % len(insole_loop)]; d = b - a
             if d.length_squared < 1e-12: continue
             t_vals = {0.0, 1.0}
             for zone_i, (ux, uy, ur) in enumerate(self.ulcer_zones):
                  f = Vector((a.x - ux, a.y - uy)); A = d.length_squared; B = 2 * f.dot(d); C = f.length_squared - ur**2
                  disc = B**2 - 4*A*C
                  if disc > 1e-12:
                       sqrtD = math.sqrt(disc)
                       for sign in (1, -1):
                            t = (-B + sign*sqrtD) / (2*A)
                            if 0 < t < 1:
                                 t_vals.add(t); P_int = a + d * t
                                 ang = math.atan2(P_int.y - uy, P_int.x - ux) % (2*math.pi)
                                 circle_edge_angles[zone_i].append(ang)
             # Build sub-segments for BOTH outputs
             t_list = sorted(t_vals)
             for j in range(len(t_list)-1):
                  t0, t1 = t_list[j], t_list[j+1]; p0 = a + d * t0; p1 = a + d * t1; mid = (p0 + p1) / 2.0
                  is_outside_all = not any((mid.x-ux)**2 + (mid.y-uy)**2 < ur**2 for ux, uy, ur in self.ulcer_zones)
                  if is_outside_all:
                       boundary_ulcer_segments.append((p0, p1)) # For final mesh
                       idx0 = len(temp_boundary_vertices); temp_boundary_vertices.append(p0) # For temp mesh
                       idx1 = len(temp_boundary_vertices); temp_boundary_vertices.append(p1) # For temp mesh
                       temp_boundary_edges.append((idx0, idx1)) # For temp mesh

        # 2) Generate ulcer arc segments AND temporary curve objects
        samples = 64
        two_pi = 2*math.pi
        for i, (ux, uy, ur) in enumerate(self.ulcer_zones):
             # ... [Logic to calculate ulcer arcs (runs list)] ...
             angs = set(circle_edge_angles[i]); angs.add(0.0)
             # (Add circle-circle intersections to angs)
             for j, (ux2, uy2, ur2) in enumerate(self.ulcer_zones):
                if j == i: continue
                dx, dy = ux2 - ux, uy2 - uy
                d2 = dx*dx + dy*dy
                d  = math.sqrt(d2)
                if d > ur + ur2 or d < abs(ur - ur2):
                    continue
                base  = math.atan2(dy, dx)
                cos_t = (ur*ur + d2 - ur2*ur2) / (2*ur*d)
                cos_t = max(-1, min(1, cos_t))
                delta = math.acos(cos_t)
                angs.add((base + delta) % two_pi)
                angs.add((base - delta) % two_pi)
             # (Adaptive sampling)
             if len(angs) > 1:
                 ang_list = sorted(list(angs)); extra = []
                 for a0, a1 in zip(ang_list, ang_list[1:] + [ang_list[0] + two_pi]):
                      span = (a1 - a0) % two_pi
                      if span > math.radians(10):
                           steps = max(1, int(span / math.radians(10)))
                           for m in range(1, steps): extra.append((a0 + span * m/steps) % two_pi)
                 angs.update(extra)
             # Generate segments and runs
             ang_list = sorted(list(angs)); ang_list.append(ang_list[0] + 2*math.pi)
             runs = []; current_run = None
             for k in range(len(ang_list) - 1):
                  a0, a1 = ang_list[k], ang_list[k+1]; mid_ang = (a0 + a1) / 2.0
                  mx = ux + ur * math.cos(mid_ang); my = uy + ur * math.sin(mid_ang)
                  is_inside_insole = insole_path_for_check.contains_point((mx, my))
                  is_outside_others = not any((mx - vx)**2 + (my - vy)**2 < vr**2 for j_other, (vx, vy, vr) in enumerate(self.ulcer_zones) if j_other != i)
                  if is_inside_insole and is_outside_others:
                       p0 = Vector((ux + ur * math.cos(a0), uy + ur * math.sin(a0), 0.0))
                       # Add potentially subsampled segments to boundary_ulcer_segments
                       num_sub_samples = max(1, int(abs(a1-a0) / math.radians(15)))
                       last_p = p0
                       if current_run is None: current_run = [last_p] # Start run for spline
                       for s_idx in range(1, num_sub_samples + 1):
                            t_sub = s_idx / num_sub_samples; a_sub = a0 + (a1 - a0) * t_sub
                            next_p = Vector((ux + ur * math.cos(a_sub), uy + ur * math.sin(a_sub), 0.0))
                            if (last_p - next_p).length_squared > 1e-12: boundary_ulcer_segments.append((last_p, next_p))
                            last_p = next_p
                       current_run.append(last_p) # Add final point to spline run
                  else: # Segment invalid, end current run for spline
                       if current_run: runs.append(current_run); current_run = None
             if current_run: runs.append(current_run)

             # Create temporary Curve object for this ulcer if runs exist
             if runs:
                  curve_data = bpy.data.curves.new(f"Temp_UlcerCurve_{i}", 'CURVE')
                  curve_data.dimensions = '2D'; curve_data.fill_mode = 'NONE'
                  for run_pts in runs:
                       spl = curve_data.splines.new('POLY'); spl.use_cyclic_u = (len(run_pts) > 1 and (run_pts[0] - run_pts[-1]).length < 1e-5)
                       spl.points.add(len(run_pts) - 1)
                       for j_pt, v in enumerate(run_pts): spl.points[j_pt].co = (v.x, v.y, 0, 1)
                  obj = bpy.data.objects.new(f"Temp_UlcerCurve_{i}", curve_data)
                  bpy.context.collection.objects.link(obj); temp_curve_gen_objects.append(obj)
                  # Check overlap flag
                  overlapped = False; sample_dirs = [k * math.pi/4 for k in range(8)]
                  for θ in sample_dirs:
                       x_s = ux + ur * math.cos(θ); y_s = uy + ur * math.sin(θ)
                       if not insole_path_for_check.contains_point((x_s, y_s)): overlapped = True; break
                  ulcer_curves_dict[obj] = overlapped
        self.log_to_file(f"Generated {len(boundary_ulcer_segments)} boundary/ulcer edge segments for mesh.")
        self.log_to_file(f"Created {len(ulcer_curves_dict)} temporary ulcer curve objects.")


        # --- Create Temporary Boundary Mesh Object ---
        boundary_obj_temp = None
        if temp_boundary_vertices and temp_boundary_edges:
             temp_boundary_mesh = bpy.data.meshes.new("Temp_Boundary_Mesh")
             temp_boundary_mesh.from_pydata([v.to_tuple() for v in temp_boundary_vertices], temp_boundary_edges, [])
             temp_boundary_mesh.update()
             boundary_obj_temp = bpy.data.objects.new("Temp_Boundary", temp_boundary_mesh)
             bpy.context.collection.objects.link(boundary_obj_temp)
             temp_curve_gen_objects.append(boundary_obj_temp)
             self.log_to_file(f"Created temporary boundary mesh '{boundary_obj_temp.name}'.")
        else:
             self.log_to_file("Warning: No boundary segments generated for curve creation.")


        # --- Call mesh_and_ulcers_to_boundary_curve ---
        self.log_to_file("Generating final combined boundary curve...")
        if not hasattr(self, 'mesh_and_ulcers_to_boundary_curve') or not callable(self.mesh_and_ulcers_to_boundary_curve):
             raise AttributeError("Helper function 'mesh_and_ulcers_to_boundary_curve' not found.")
        if not boundary_obj_temp:
             raise ValueError("Temporary boundary mesh object was not created.")

        combined_curve_obj = self.mesh_and_ulcers_to_boundary_curve(
            boundary_obj_temp, ulcer_curves_dict
        )
        if not combined_curve_obj:
             raise ValueError("mesh_and_ulcers_to_boundary_curve failed.")


        # --- Store as self.boundary_curve ---
        self.boundary_curve = combined_curve_obj
        self.boundary_curve.name = "CombinedBoundaryCurve"
        self.log_to_file(f"Stored combined boundary curve as self.boundary_curve ('{self.boundary_curve.name}')")


        # --- Combine ALL Geometry and Clean with BMesh ---
        self.log_to_file("Combining and cleaning all edge segments for final mesh...")
        bm = bmesh.new()
        vert_map = {} # Map Vector coord tuple -> BMVert

        def get_or_create_vert(coord_vec):
            coord_tuple = tuple(round(c, 6) for c in coord_vec.to_tuple())
            vert = vert_map.get(coord_tuple)
            if vert is None or not vert.is_valid:
                vert = bm.verts.new(coord_vec.to_tuple())
                vert_map[coord_tuple] = vert
            return vert

        all_segments = clipped_lattice_segments + boundary_ulcer_segments
        if not all_segments: raise ValueError("No segments generated to create final mesh.")

        edge_creation_errors = 0
        for v1_vec, v2_vec in all_segments:
            if (v1_vec - v2_vec).length_squared > 1e-12:
                v1_bm = get_or_create_vert(v1_vec); v2_bm = get_or_create_vert(v2_vec)
                if v1_bm != v2_bm and bm.edges.get((v1_bm, v2_bm)) is None:
                    try: bm.edges.new((v1_bm, v2_bm))
                    except ValueError: edge_creation_errors += 1
        if edge_creation_errors > 0: self.log_to_file(f"Warning: {edge_creation_errors} errors during edge creation.")
        self.log_to_file(f"Added {len(bm.verts)} vertices and {len(bm.edges)} edges to final BMesh.")

        self.log_to_file("Running Remove Doubles on final combined geometry...")
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        self.log_to_file("Remove Doubles complete.")


        # --- Create Final Mesh Object ---
        final_mesh = bpy.data.meshes.new(name="Unified_Infill_Pattern_Mesh")
        bm.to_mesh(final_mesh); final_mesh.update(); bm.free(); bm = None

        final_obj = bpy.data.objects.new(name="Unified_Infill_Pattern", object_data=final_mesh)
        bpy.context.collection.objects.link(final_obj)
        self.log_to_file(f"Created final unified edge mesh object '{final_obj.name}'.")


        self.log_to_file("Cleaning up temporary objects used for curve generation...")
        cleaned_count = 0
        for obj in temp_curve_gen_objects:
            if obj and obj.name in bpy.data.objects:
                # Double-check it's not the final curve we want to keep
                if not (hasattr(self, 'boundary_curve') and self.boundary_curve and obj == self.boundary_curve):
                    self.log_to_file(f" Removing temp object: {obj.name}")
                    mesh_data = obj.data
                    try:
                        bpy.data.objects.remove(obj, do_unlink=True)
                        cleaned_count += 1
                        # Remove data block if it has no users
                        if mesh_data and mesh_data.users == 0:
                            if isinstance(mesh_data, bpy.types.Mesh): bpy.data.meshes.remove(mesh_data)
                            elif isinstance(mesh_data, bpy.types.Curve): bpy.data.curves.remove(mesh_data)
                    except ReferenceError: pass # Object might be gone already
        self.log_to_file(f"Cleaned up {cleaned_count} temporary objects.")


        # --- Return ONLY the combined mesh object ---
        self.log_to_file("generate_2d_infill finished successfully.")
        return final_obj
    
    
    # ------------------------------------
    # END OF 2D PATTERN GENERATION
    # ------------------------------------
    
        
    def visualize_csv_data(self):
        M, N = self.density_map.shape
        spacing = self.csv_cell_spacing * 100
        points = []
        for r in range(M):
            for c in range(N):
                if self.density_map[r, c] != 0:
                    X = (M - 1 - r) * spacing
                    Y = (c - (N - 1) / 2) * spacing
                    points.append((X, Y, 5))
        self.log_to_file("Visualized CSV: " + str(len(points)) + " points created from non-zero cells.")
        mesh = bpy.data.meshes.new("CSVPoints")
        mesh.from_pydata(points, [], [])
        mesh.update()
        obj = bpy.data.objects.new("CSVPoints", mesh)
        bpy.context.collection.objects.link(obj)
        self.log_to_file("CSV point cloud object created.")
        return obj

    def visualize_phi_fields(self, phi_p, phi_q):
        points = [ (p, q, 0) for p, q in zip(phi_p, phi_q) ]
        mesh = bpy.data.meshes.new("PhiFields")
        mesh.from_pydata(points, [], [])
        mesh.update()
        obj = bpy.data.objects.new("PhiFields", mesh)
        bpy.context.collection.objects.link(obj)
        self.log_to_file("PhiFields point cloud created with " + str(len(points)) + " points.")
        return obj
    
    def visualize_deformed_hexagon_grid(self):
        """
        Generate and create a Blender mesh object showing the deformed hexagon grid 
        prior to clipping. This mesh is built from:
          1. Extracting the 2D boundary.
          2. Computing the mapping (phi) fields and generating the deformed domain.
          3. Generating a uniform hexagon grid in the deformed domain.
          4. Inverse-mapping each grid vertex back to the original domain.
          5. Grouping the vertices into hexagon cells (assumed 6 per cell)
             and creating closed-loop edge connectivity for each cell.
        The resulting mesh is linked into the Blender scene.
        """
        # 1. Extract the 2D boundary.
        vertices, faces = self.extract_2d_boundary()
        
        # 2. Compute the mapping matrices and phi fields.
        A, b_p, b_q = self.compute_mapping_matrices(vertices, faces)
        reg_term = diags([1e-6] * A.shape[1])
        A_reg = A.T @ A + reg_term
        phi_p = spsolve(A_reg, A.T @ b_p)
        phi_q = spsolve(A_reg, A.T @ b_q)
        
        # Create deformed domain vertices and account for any translation offset.
        mapped_vertices = [Vector((p, q, 0)) for p, q in zip(phi_p, phi_q)]
        mapped_vertices = [v + self.translation_offset for v in mapped_vertices]
        
        # 3. Generate a uniform hexagon grid in the deformed domain.
        pattern_verts, pattern_edges = self.generate_honeycomb_pattern(mapped_vertices, self.cell_size)
        if not pattern_verts:
            raise ValueError("Error: No hexagon grid generated in the deformed domain.")
        
        # 4. Inverse-map the deformed grid vertices back to the original domain.
        original_pattern = self.map_vertices_back(pattern_verts, vertices, faces, mapped_vertices)
        if not original_pattern:
            raise ValueError("Error: Inverse-mapped hexagon grid is empty.")
        if len(original_pattern) % 6 != 0:
            self.log_to_file("Warning: Number of inverse-mapped vertices is not a multiple of 6.")
        num_cells = len(original_pattern) // 6
        cells = [original_pattern[i*6:(i+1)*6] for i in range(num_cells)]
        
        # 5. Build the final mesh data from the cells.
        final_vertices = []
        final_edges = []
        idx_offset = 0
        for cell in cells:
            # Build a closed loop for each hexagon cell.
            for i in range(len(cell)):
                final_vertices.append(cell[i])
                final_edges.append((idx_offset + i, idx_offset + ((i + 1) % len(cell))))
            idx_offset += len(cell)
        
        # Create a new Blender mesh and object.
        mesh = bpy.data.meshes.new("DeformedHexagonGrid")
        mesh.from_pydata(final_vertices, final_edges, [])
        mesh.update()
        obj = bpy.data.objects.new("DeformedHexagonGrid", mesh)
        bpy.context.collection.objects.link(obj)
        self.log_to_file("Deformed hexagon grid mesh created for comparison.")
        return obj
    
    def visualize_outer_clipping_boundary(self, alpha=0.1):
        """
        Re-extract the outer boundary (used for clipping) from the object's 2D projection.
        This function uses only the alpha shape and ordering steps (and applies the boundary offset)
        to compute a closed polygon representing the outer boundary without the interior sample points.
        It then creates and links a Blender mesh object showing this boundary.
        """
        # Get the object's world-space XY coordinates.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(temp_mesh)
        coords = []
        for v in bm.verts:
            co_world = self.obj.matrix_world @ v.co
            coords.append((co_world.x, co_world.y))
        bm.free()
        eval_obj.to_mesh_clear()

        points = np.array(coords)
        if len(points) < 4:
            raise ValueError("Not enough points for alpha shape boundary extraction.")

        # Compute the alpha shape edges (concave hull) from the points.
        boundary_edges = self.alpha_shape(points, alpha)
        # Order these edges to get a continuous boundary loop.
        ordered_indices = self.order_boundary_edges(boundary_edges)
        # The outer boundary coordinates (in order) are taken from the original points.
        boundary_coords = [points[i] for i in ordered_indices]

        # Apply the boundary offset: move each point slightly inward.
        centroid = np.mean(boundary_coords, axis=0)
        offset_boundary_coords = []
        for pt in boundary_coords:
            direction = centroid - pt
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = np.array([0, 0])
            offset_pt = pt + direction * self.boundary_offset
            offset_boundary_coords.append(offset_pt)
        # Convert to Blender Vectors (with z = 0)
        final_boundary = [Vector((pt[0], pt[1], 0)) for pt in offset_boundary_coords]

        # Build edge connectivity to form a closed loop.
        edges = []
        num_vertices = len(final_boundary)
        for i in range(num_vertices):
            edges.append((i, (i + 1) % num_vertices))

        # Create a new mesh object for the outer clipping boundary.
        mesh = bpy.data.meshes.new("OuterClippingBoundary")
        mesh.from_pydata(final_boundary, edges, [])
        mesh.update()
        obj = bpy.data.objects.new("OuterClippingBoundary", mesh)
        bpy.context.collection.objects.link(obj)
        self.log_to_file("Outer clipping boundary mesh created with " + str(len(final_boundary)) + " vertices.")
        return obj
    
    def visualize_fallback_vertices(self):
        """
        Create and link a Blender mesh object showing as a point cloud
        all the hexagon vertices for which the mapping fallback was used.
        
        This function assumes that self.fallback_vertices has been set (e.g., by map_vertices_back).
        """
        if not hasattr(self, 'fallback_vertices') or not self.fallback_vertices:
            raise ValueError("No fallback vertices found. Make sure map_vertices_back() has been run.")
        
        # Create a point cloud mesh from the fallback vertices.
        mesh = bpy.data.meshes.new("FallbackVertices")
        # The fallback vertices are already Vector objects.
        mesh.from_pydata(self.fallback_vertices, [], [])
        mesh.update()
        obj = bpy.data.objects.new("FallbackVertices", mesh)
        bpy.context.collection.objects.link(obj)
        self.log_to_file("Fallback vertices point cloud created with " + str(len(self.fallback_vertices)) + " points.")
        return obj
    
    def visualize_mapping_field_boundary(self):
        """
        Visualize the effective boundary of the mapping field.
        
        This boundary is computed from the Delaunay triangulation used to
        solve the mapping fields (φ). Vertices placed within this boundary
        are expected to be inverse-mapped normally, while vertices outside
        will not be captured (leading to fallback behavior).
        
        It assumes that self.mapping_points (a list of Vectors) and
        self.mapping_faces (a list of triangle indices) have been stored
        in extract_2d_boundary().
        """
        if not hasattr(self, 'mapping_points') or not hasattr(self, 'mapping_faces'):
            raise ValueError("Mapping field data not found. Ensure extract_2d_boundary() has been run "
                             "and stores self.mapping_points and self.mapping_faces.")
        
        mapping_points = self.mapping_points  # List of Vectors (with z = 0)
        mapping_faces = self.mapping_faces    # List of triangles (each a list of 3 indices)

        # Create a dictionary counting edges from all triangles.
        edge_count = {}
        for face in mapping_faces:
            # Each face is a triangle; add its three edges.
            for i in range(3):
                a = face[i]
                b = face[(i + 1) % 3]
                edge = tuple(sorted((a, b)))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear only once.
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        # Build a connectivity dictionary from boundary_edges.
        connectivity = {}
        for edge in boundary_edges:
            a, b = edge
            connectivity.setdefault(a, []).append(b)
            connectivity.setdefault(b, []).append(a)

        # Order the boundary edges into a continuous loop.
        # Start with a vertex that has only one neighbor if an open boundary, else any vertex.
        start = None
        for vertex, neighbors in connectivity.items():
            if len(neighbors) == 1:
                start = vertex
                break
        if start is None:
            start = boundary_edges[0][0]

        ordered_indices = [start]
        current = start
        prev = None
        while True:
            neighbors = connectivity[current]
            next_vertex = None
            for candidate in neighbors:
                if candidate != prev:
                    next_vertex = candidate
                    break
            if next_vertex is None or next_vertex == start:
                break
            ordered_indices.append(next_vertex)
            prev = current
            current = next_vertex

        # Use the ordered indices to get the boundary vertices (as Vectors).
        ordered_boundary = [mapping_points[idx] for idx in ordered_indices]

        # Build edge connectivity for the ordered boundary.
        ordered_edges = []
        for i in range(len(ordered_boundary)):
            ordered_edges.append((i, (i + 1) % len(ordered_boundary)))

        # Create a new mesh for the mapping field boundary.
        mesh = bpy.data.meshes.new("MappingFieldBoundary")
        mesh.from_pydata(ordered_boundary, ordered_edges, [])
        mesh.update()
        obj = bpy.data.objects.new("MappingFieldBoundary", mesh)
        bpy.context.collection.objects.link(obj)
        self.log_to_file("Mapping field boundary mesh created with " + str(len(ordered_boundary)) + " vertices.")
        return obj
    
    # ------------------------------------
    # END OF VISUALISATION FUNCTIONS
    # ------------------------------------

    def mesh_and_ulcers_to_boundary_curve(self, boundary_obj, ulcer_curves_dict, tol=0.0005):

        # 1) Gather all raw segments in world‑space
        segments = []
        mesh = boundary_obj.data
        mat  = boundary_obj.matrix_world
        for e in mesh.edges:
            A = mat @ mesh.vertices[e.vertices[0]].co
            B = mat @ mesh.vertices[e.vertices[1]].co
            segments.append((A, B))

        for curve_obj, overlapped in ulcer_curves_dict.items():
            if not overlapped:
                continue
            for spl in curve_obj.data.splines:
                pts = [Vector((p.co.x, p.co.y, p.co.z)) for p in spl.points]
                for i in range(len(pts) - 1):
                    segments.append((pts[i], pts[i+1]))

        # 2) Quantize each endpoint to a grid of cell size = tol
        scale = 1.0 / tol
        unique_pts = []
        key_to_idx = {}
        seg_ids    = []

        for A, B in segments:
            pair = []
            for P in (A, B):
                key = (round(P.x * scale), round(P.y * scale))
                idx = key_to_idx.get(key)
                if idx is None:
                    idx = len(unique_pts)
                    unique_pts.append(P.copy())
                    key_to_idx[key] = idx
                pair.append(idx)
            seg_ids.append(tuple(pair))

        # 3) Build adjacency
        adj = {i: [] for i in range(len(unique_pts))}
        for a, b in seg_ids:
            adj[a].append(b)
            adj[b].append(a)

        # 4) Extract loops
        loops = []
        visited = set()
        for start in adj:
            if start in visited or not adj[start]:
                continue
            loop = [start]
            visited.add(start)
            prev, cur = None, start
            while True:
                nbrs = [n for n in adj[cur] if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                if nxt == start:
                    break
                loop.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
            loops.append(loop)

        if not loops:
            return None

        # 5) Pick outer loop by area
        def area(idx_loop):
            a = 0.0
            pts = unique_pts
            n = len(idx_loop)
            for i in range(n):
                x0,y0 = pts[idx_loop[i]].x, pts[idx_loop[i]].y
                x1,y1 = pts[idx_loop[(i+1)%n]].x, pts[idx_loop[(i+1)%n]].y
                a += x0*y1 - x1*y0
            return abs(a) * 0.5

        areas = [area(l) for l in loops]
        outer_i = areas.index(max(areas))

        # 6) Build the Curve
        curve_data = bpy.data.curves.new("InsoleWithUlcers", 'CURVE')
        curve_data.dimensions = '2D'
        curve_data.fill_mode  = 'NONE'

        # outer (closed)
        pts = [unique_pts[i] for i in loops[outer_i]]
        spl = curve_data.splines.new('POLY')
        spl.use_cyclic_u = True
        spl.points.add(len(pts)-1)
        for j, v in enumerate(pts):
            spl.points[j].co = (v.x, v.y, 0, 1)

        # indents (open)
        for idx, loop in enumerate(loops):
            if idx == outer_i: continue
            pts = [unique_pts[i] for i in loop]
            spl = curve_data.splines.new('POLY')
            spl.use_cyclic_u = False
            spl.points.add(len(pts)-1)
            for j, v in enumerate(pts):
                spl.points[j].co = (v.x, v.y, 0, 1)

        # 7) Link & return
        curve_obj = bpy.data.objects.new("InsoleBoundaryCurve", curve_data)
        bpy.context.collection.objects.link(curve_obj)
        return curve_obj
    
    def convert_infill_to_3d(self, infill_obj):
        """
        1) Build two Curve objects:
           - boundary_curve: outer insole + boundary‑overlapping ulcers
           - lattice_curve : just the lattice struts
        2) Solidify each with proper compensation:
           - lattice: multiplier_lattice - 1.0
           - interior ulcers: multiplier_lattice
           - boundary: multiplier_boundary - 1.0
        3) Boolean‑UNION all solids into one final mesh.
        """

        # --- B1) Process Lattice Mesh Directly using Geometry Nodes ---
        self.log_to_file("Processing lattice edge mesh via Geometry Nodes...")
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = infill_obj; infill_obj.select_set(True)

        # 1. Clean Edge Mesh (Remove Doubles)
        self.log_to_file(" Cleaning lattice edge mesh (Remove Doubles)...")
        bpy.ops.object.mode_set(mode='EDIT'); needs_object_mode = True
        bm = bmesh.from_edit_mesh(infill_obj.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.update_edit_mesh(infill_obj.data); bm.free(); bm = None
        bpy.ops.object.mode_set(mode='OBJECT'); needs_object_mode = False
        self.log_to_file(" Finished removing doubles on lattice edges.")

        # 2. Add and Configure Geometry Nodes Modifier
        self.log_to_file(" Setting up Geometry Nodes for lattice thickening...")
        gn_mod = infill_obj.modifiers.new(name="LatticeThickenExtrude", type='NODES')

        # Create Node Group or get existing one
        node_group_name = "LatticeThickenExtrudeNodes"
        if node_group_name in bpy.data.node_groups:
            node_group = bpy.data.node_groups[node_group_name]
            self.log_to_file(f"Reusing existing node group '{node_group_name}'.")
            for node in list(node_group.nodes): node_group.nodes.remove(node)
            node_group.links.clear(); node_group.interface.clear()
        else:
            node_group = bpy.data.node_groups.new(node_group_name, 'GeometryNodeTree')
            self.log_to_file(f"Created new node group '{node_group_name}'.")

        gn_mod.node_group = node_group # Assign group to modifier

        # --- <<< CORRECTED: Assign nodes and links HERE >>> ---
        nodes = node_group.nodes
        links = node_group.links
        # --- <<< END CORRECTION >>> ---

        # --- Define Node Group Interface (Sockets) ---
        input_socket_name = "Geometry"; output_socket_name = "Geometry"
        if not any(item.name == input_socket_name for item in node_group.interface.items_tree if item.item_type == 'SOCKET' and item.in_out == 'INPUT'):
            node_group.interface.new_socket(name=input_socket_name, in_out='INPUT', socket_type='NodeSocketGeometry')
        if not any(item.name == output_socket_name for item in node_group.interface.items_tree if item.item_type == 'SOCKET' and item.in_out == 'OUTPUT'):
            node_group.interface.new_socket(name=output_socket_name, in_out='OUTPUT', socket_type='NodeSocketGeometry')
        self.log_to_file(f"Defined Interface Sockets: '{input_socket_name}', '{output_socket_name}'")


        # --- Create/Get Internal Group Nodes ---
        node_input = nodes.get("Group Input") or nodes.new(type='NodeGroupInput')
        node_input.location = (-600, 0); node_input.name = "Group Input"
        node_output = nodes.get("Group Output") or nodes.new(type='NodeGroupOutput')
        node_output.location = (800, 0); node_output.name = "Group Output"
        if output_socket_name not in node_output.inputs: node_output.inputs.new('NodeSocketGeometry', output_socket_name)
        self.log_to_file("Created/found Group Input/Output nodes.")

        # Create Processing Nodes
        node_m2c = nodes.new(type='GeometryNodeMeshToCurve'); node_m2c.name = "Mesh to Curve"; # ... location ...
        node_profile = nodes.new(type='GeometryNodeCurvePrimitiveLine'); node_profile.name = "Profile Curve Line"; # ... location ...
        # ... (Set profile width) ...
        profile_width = self.nozzle_diameter * self.thickness_multiplier; half_width = profile_width / 2.0
        node_profile.inputs["Start"].default_value = (-half_width, 0.0, 0.0); node_profile.inputs["End"].default_value = (half_width, 0.0, 0.0)

        node_c2m = nodes.new(type='GeometryNodeCurveToMesh'); node_c2m.name = "Curve to Mesh"; node_c2m.location = (-150, 0)

        # --- <<< NEW: Flip Faces Node for Bottom Cap >>> ---
        node_flip_bottom = nodes.new(type='GeometryNodeFlipFaces')
        node_flip_bottom.name = "Flip Bottom Cap Faces"
        node_flip_bottom.location = (50, -100) # Position after CurveToMesh
        # --- <<< END NEW >>> ---

        node_extrude = nodes.new(type='GeometryNodeExtrudeMesh'); node_extrude.name = "Extrude Mesh"; node_extrude.location = (250, 100) # Shifted Y
        node_extrude.mode = 'FACES'
        node_join = nodes.new(type='GeometryNodeJoinGeometry')
        node_join.name = "Join Caps"; node_join.location = (500, 0) # Shifted X

        # --- Link Nodes ---
        links.clear()
        input_geo_socket = node_input.outputs.get(input_socket_name)
        output_geo_socket = node_output.inputs.get(output_socket_name)
        # ... (Check sockets exist) ...

        links.new(input_geo_socket, node_m2c.inputs["Mesh"])
        links.new(node_profile.outputs["Curve"], node_c2m.inputs["Profile Curve"])
        links.new(node_m2c.outputs["Curve"], node_c2m.inputs["Curve"])

        mesh_after_profile = node_c2m.outputs["Mesh"] # This is the bottom ribbon

        # Link ribbon to Flip Faces node
        links.new(mesh_after_profile, node_flip_bottom.inputs["Mesh"])

        # Link original ribbon AND flipped ribbon to Join Geometry
        # Correction: Only link the FLIPPED bottom faces to Join
        # links.new(mesh_after_profile, node_join.inputs["Geometry"]) # No - Join flipped instead
        links.new(node_flip_bottom.outputs["Mesh"], node_join.inputs["Geometry"]) # Link FLIPPED bottom faces

        # Link original ribbon to Extrude Mesh
        links.new(mesh_after_profile, node_extrude.inputs["Mesh"])

        # Set Extrude offset
        node_extrude.inputs["Offset Scale"].default_value = self.cell_height
        node_extrude.inputs["Offset"].default_value = (0.0, 0.0, 0.0)
        # --- <<< LOGGING FOR HEIGHT >>> ---
        self.log_to_file(f"DEBUG: Using self.cell_height = {self.cell_height} for extrusion.")
        scene_unit_scale = bpy.context.scene.unit_settings.scale_length
        if not math.isclose(scene_unit_scale, 1.0):
             self.log_to_file(f"WARNING: Scene Unit Scale is {scene_unit_scale}, not 1.0. This might affect extrusion distance.")
        # --- <<< END LOGGING >>> ---


        # Link Extrude output to the Join Geometry node
        links.new(node_extrude.outputs["Mesh"], node_join.inputs["Geometry"])

        # Link final Join result to Group Output
        links.new(node_join.outputs["Geometry"], output_geo_socket) # Output joined geometry

        self.log_to_file(" Geometry Nodes tree created/updated and linked (with bottom face flip).")


        # 3. Apply Modifier
        self.log_to_file(" Applying Geometry Nodes modifier...")
        if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = infill_obj # Make sure correct obj is active
        bpy.ops.object.modifier_apply(modifier=gn_mod.name)
        bpy.ops.object.shade_flat()
        lattice_solid = infill_obj # Object is modified in place
        lattice_solid.name = "LatticeSolid_GN"
        self.log_to_file(" Lattice thickening via Geometry Nodes complete.")
        
        return lattice_solid
    
    # ------------------------------------
    # END OF 3D EXTRUSION AND THICKENING
    # ------------------------------------
    
    def visualize_point_cloud(self, points_list, object_name="PointCloud_Visualization"):
        """
        Creates a Blender object displaying a list of points as a point cloud.

        Args:
            points_list (list): A list of 3D points (Vector, list, or tuple).
            object_name (str): The desired name for the new Blender object.

        Returns:
            bpy.types.Object: The created point cloud object, or None if input is invalid.
        """
        if not points_list or not isinstance(points_list, list):
            self.log_to_file(f"Error visualizing point cloud: Invalid input list provided for '{object_name}'.")
            return None

        self.log_to_file(f"Visualizing {len(points_list)} points as '{object_name}'...")

        vertices_for_mesh = points_list

        # Create a new mesh with only vertices
        mesh_data = bpy.data.meshes.new(name=f"{object_name}_Mesh")
        mesh_data.from_pydata(vertices_for_mesh, [], []) # No edges, no faces
        mesh_data.update()

        # Create a new object linked to the mesh data
        point_cloud_obj = bpy.data.objects.new(object_name, mesh_data)

        # Link the object to the scene collection
        bpy.context.collection.objects.link(point_cloud_obj)

        self.log_to_file(f"Created point cloud object '{point_cloud_obj.name}'.")
        return point_cloud_obj
    

    def apply_contour_shrinkwrap(self, infill_object, target_surface_object):
        """
        Contours the top surface of the infill_object to match the target_surface_object
        using a Shrinkwrap modifier. Uses self.cell_height for selection.

        Args:
            infill_object (bpy.types.Object): The generated infill solid mesh.
            target_surface_object (bpy.types.Object): The smooth surface to project onto.

        Returns:
            bpy.types.Object: The contoured infill object.
        """
        self.log_to_file(f"Applying Shrinkwrap contouring to '{infill_object.name}' using target '{target_surface_object.name}'.")

        # Basic validation remains useful
        if not infill_object or not target_surface_object:
            self.log_to_file("Error: Invalid objects passed to apply_contour_shrinkwrap.")
            raise ValueError("Infill object or target surface object is missing.")
        if infill_object.type != 'MESH' or target_surface_object.type != 'MESH':
             self.log_to_file("Error: Objects must be of type MESH for Shrinkwrap.")
             raise ValueError("Objects for Shrinkwrap must be MESH type.")

        target_surface_object.hide_set(False)
        target_surface_object.hide_select = False
        bpy.context.view_layer.objects.active = infill_object # Make active for operators
        infill_object.select_set(True)


        vg_name = "TopSurface"
        existing_vg = infill_object.vertex_groups.get(vg_name)
        if existing_vg:
            infill_object.vertex_groups.remove(existing_vg)
        vg = infill_object.vertex_groups.new(name=vg_name)

        bpy.ops.object.mode_set(mode='OBJECT')
        mesh = infill_object.data
        top_verts_indices = []

        # Use self.cell_height (assuming it's reliably set in __init__)
        world_matrix = infill_object.matrix_world
        max_z_world = max((world_matrix @ v.co).z for v in mesh.vertices) if mesh.vertices else 0
        # Calculate threshold relative to object origin Z
        z_threshold_local = max_z_world - infill_object.location.z - 0.1 # Tolerance
        top_verts_indices = [v.index for v in mesh.vertices if v.co.z >= z_threshold_local]
        self.log_to_file(f"Selecting top vertices based on max Z heuristic (Local Z >= {z_threshold_local:.3f})")

        if not top_verts_indices:
            self.log_to_file("Warning: Primary vertex selection failed. Trying face normals.")
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            top_face_verts = set()
            for face in bm.faces:
                 world_normal = (infill_object.matrix_world.inverted_safe().transposed().to_3x3() @ face.normal).normalized()
                 if world_normal.z > 0.95:
                      for vert in face.verts:
                           top_face_verts.add(vert.index)
            bm.free()
            top_verts_indices = list(top_face_verts)

            if not top_verts_indices:
                 self.log_to_file("ERROR: Failed to select any top vertices/faces for shrinkwrapping.")
                 # Raise error or return unmodified object
                 raise ValueError("Could not select vertices for shrinkwrapping.")

        vg.add(top_verts_indices, 1.0, 'REPLACE')
        self.log_to_file(f"Assigned {len(top_verts_indices)} vertices to group '{vg_name}'.")

        mod = infill_object.modifiers.new(name="ContourShrinkwrap", type='SHRINKWRAP')
        mod.target = target_surface_object
        mod.vertex_group = vg_name
        mod.wrap_method = 'PROJECT'
        mod.use_project_z = True
        mod.use_negative_direction = True
        mod.use_positive_direction = True
        self.log_to_file("Configured Shrinkwrap modifier.")

        # Apply Modifier - this can raise RuntimeError
        bpy.ops.object.modifier_apply(modifier=mod.name)
        self.log_to_file("Applied Shrinkwrap modifier.")

        return infill_object
    
    def contouring_points_from_pressure_csv(self):
        """
        Generates 3D contour points from the normalized pressure map CSV.
        Only non-zero pressures produce points.  Z goes from:
          • pressure = 1 → self.pressure_points_base_height
          • pressure = 0 → self.pressure_points_base_height + self.contouring_depth
        """
        self.log_to_file("Generating contour points from pressure map...")

        # Preconditions
        if self.density_map is None:
            self.log_to_file("Error: Density map not loaded.")
            return []
        if not hasattr(self, 'aligned_boundary') or not self.aligned_boundary:
            self.log_to_file("Error: Aligned boundary required.")
            return []
        if not hasattr(self, 'contouring_depth') or self.contouring_depth is None:
            self.log_to_file("Error: contouring_depth not set.")
            return []
        if not hasattr(self, 'pressure_points_base_height') or self.pressure_points_base_height is None:
            self.log_to_file("Error: pressure_points_base_height not set.")
            return []
        
        # Workaround for if a uniform lattice is set. We don't want that to carry forward to the pressure-derived contouring
        if self.contoured_density_map is not None:
            mapping_points = self.contoured_density_map
        elif self.non_inverse_preesure_map is not None:
            mapping_points = self.non_inverse_preesure_map
        else:
            mapping_points = self.density_map
        
        M, N = mapping_points.shape
        self.log_to_file(f"Pressure map size: {M}×{N}")

        # Get XY bounds from the aligned insole boundary
        coords = np.array([(v.x, v.y) for v in self.aligned_boundary])
        min_x, max_x = coords[:,0].min(), coords[:,0].max()
        min_y, max_y = coords[:,1].min(), coords[:,1].max()
        x_range = max(max_x - min_x, 1e-6)
        y_range = max(max_y - min_y, 1e-6)
        self.log_to_file(f"Boundary XY: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}]")

        x_div = M - 1 if M > 1 else 1
        y_div = N - 1 if N > 1 else 1

        base_z = self.pressure_points_base_height
        depth  = self.contouring_depth
        
        self.log_to_file(str(mapping_points))
        contour_points = []
        for r in range(M):
            for c in range(N):
                p = mapping_points[r, c]

                # map array indices to XY
                x = max_x - (r / x_div) * x_range
                y = min_y + (c / y_div) * y_range

                # Z mapping: p=1 → base_z; p=0 → base_z+depth
                z = base_z + (1.0 - p) * depth
                
                if p <= 1e-3:
                    if x == max_x or x == min_x:
                        z = base_z + depth * self.pressure_points_extrapolation_factor
                    elif y == max_y or y == min_y:
                        z = base_z + depth * self.pressure_points_extrapolation_factor
                    else:
                        continue

                contour_points.append(Vector((x, y, z)))

        self.log_to_file(f"Generated {len(contour_points)} non-zero contour points.")
        return contour_points
    
    def contouring_points_from_stl_top_surface_floodfill(self,
                                                         seed_normal_z_min=0.90, # Keep seed strict
                                                         expand_normal_z_min=0.50, # Relaxed Z check for expansion
                                                         max_angle_deg=15.0): # ** Primary stopping criterion **
        """
        Extracts 3D points representing the top surface of the aligned STL model
        using a flood-fill selection starting from the highest faces.
        Primarily uses the angle between adjacent faces to stop at sharp edges
        (like transitions to vertical side walls).

        Args:
            seed_normal_z_min (float): Minimum world Z normal for a face to be
                                       considered a potential starting seed (close to 1.0).
            expand_normal_z_min (float): Minimum world Z normal for a neighboring
                                         face to be considered *at all* during flood fill.
                                         Set lower to allow slightly sloped top faces.
            max_angle_deg (float): Maximum angle (degrees) between adjacent face
                                   normals to continue the flood fill. Stops selection
                                   at edges sharper than this angle (e.g., 45 degrees).

        Returns:
            list: A list of Vector points representing the top surface,
                  or an empty list on failure.
        """
        self.log_to_file(f"Extracting contour points from STL top surface (Flood Fill - Angle Stop: {max_angle_deg} deg)...") # Log the key param

        if not self.obj or self.obj.type != 'MESH':
            self.log_to_file("Error: Invalid or missing self.obj (aligned STL mesh)."); return []

        max_angle_rad = math.radians(max_angle_deg) # Convert angle to radians

        # Use BMesh for efficient topology traversal
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(depsgraph)
        bm.from_mesh(eval_obj.to_mesh(depsgraph=depsgraph))
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.faces.ensure_lookup_table()

        world_matrix = self.obj.matrix_world
        try: normal_matrix = world_matrix.inverted_safe().transposed().to_3x3()
        except ValueError: self.log_to_file("Error: World matrix non-invertible."); bm.free(); return []

        # --- Find Seed Face(s) (same logic as before: highest Z faces with steep UP normal) ---
        seed_faces = []
        max_z_found = -float('inf')
        for face in bm.faces:
            try:
                world_normal = (normal_matrix @ face.normal).normalized()
                if world_normal.z >= seed_normal_z_min:
                     world_center = world_matrix @ face.calc_center_median()
                     if world_center.z > max_z_found: max_z_found = world_center.z
            except Exception as e: self.log_to_file(f"Warning: Error processing face {face.index} during seed search: {e}"); continue
        if max_z_found == -float('inf'): self.log_to_file("Error: Could not find any potential seed faces."); bm.free(); return []
        z_tolerance = 0.05
        for face in bm.faces:
            try:
                 world_normal = (normal_matrix @ face.normal).normalized()
                 if world_normal.z >= seed_normal_z_min:
                      world_center = world_matrix @ face.calc_center_median()
                      if world_center.z >= max_z_found - z_tolerance: seed_faces.append(face)
            except Exception: continue
        if not seed_faces: self.log_to_file(f"Error: No seed faces found near max Z ({max_z_found:.3f})."); bm.free(); return []
        self.log_to_file(f"Found {len(seed_faces)} potential seed face(s) near Z={max_z_found:.3f}")
        # --- End Seed Finding ---


        # --- Flood Fill ---
        visited_faces = set()
        queue = seed_faces[:]
        selected_faces = set(seed_faces)
        for face in seed_faces: visited_faces.add(face.index)

        processed_count = 0
        while queue:
            current_face = queue.pop(0)
            processed_count += 1
            # if processed_count % 500 == 0: self.log_to_file(f"Flood fill processing... Queue size: {len(queue)}") # Optional progress log

            try: current_world_normal = (normal_matrix @ current_face.normal).normalized()
            except Exception: continue # Skip if current face normal fails

            # Check neighbors via edges
            for edge in current_face.edges:
                for neighbor_face in edge.link_faces:
                    if neighbor_face.index not in visited_faces:
                        visited_faces.add(neighbor_face.index) # Mark visited

                        try: neighbor_world_normal = (normal_matrix @ neighbor_face.normal).normalized()
                        except Exception: continue # Skip neighbor if normal fails

                        # *** Refined Criteria Check ***
                        # 1. Basic check: Is the neighbor generally upward facing?
                        if neighbor_world_normal.z >= expand_normal_z_min:
                            # 2. Primary check: Is the angle between normals within the limit?
                            angle = current_world_normal.angle(neighbor_world_normal)
                            if angle <= max_angle_rad:
                                # Criteria met: add to selection and queue
                                selected_faces.add(neighbor_face)
                                queue.append(neighbor_face)
                            #else: self.log_to_file(f"Debug: Stopped at angle: {math.degrees(angle):.1f} > {max_angle_deg}") # DEBUG
                        #else: self.log_to_file(f"Debug: Stopped at Z normal: {neighbor_world_normal.z:.2f} < {expand_normal_z_min}") # DEBUG

        self.log_to_file(f"Flood fill complete using angle threshold. Selected {len(selected_faces)} faces.")
        # --- End Flood Fill ---

        # --- Extract Vertices (same as before) ---
        top_vertex_coords = set()
        for face in selected_faces:
            for vert in face.verts:
                world_co = world_matrix @ vert.co
                top_vertex_coords.add( (world_co.x, world_co.y, world_co.z) )
        bm.free() # Release BMesh
        if not top_vertex_coords: self.log_to_file("Warning: No vertices extracted after flood fill."); return []
        contour_points = [Vector(coord) for coord in top_vertex_coords]
        self.log_to_file(f"Extracted {len(contour_points)} unique contour points using flood fill.")
        return contour_points

    
    def create_smooth_foot_surface(self, input_points_list, extension_factor=0.05):
        """
        Creates a raw, rectangular grid surface from a list of 3D points using
        interpolation and k-NN fill for exterior points. Adds an outward brim
        by offsetting boundary vertices using robust normal calculation.
        Does NOT apply subdivision.
        Uses parameters stored in self: contour_grid_resolution, contour_interpolation_method.

        Args:
            input_points_list (list): A list of 3D points.
            extension_factor (float): Factor to extend the boundary outwards.

        Returns:
            bpy.types.Object: The generated raw grid surface object with brim, or None on failure.
        """
        self.log_to_file("Creating RAW interpolated grid surface WITH BRIM (User Normal Helper - Ref Based)...")

        # --- Initial setup and Input Processing ---
        # ... (Same as before) ...
        if not input_points_list or not isinstance(input_points_list, list): self.log_to_file("Error: input_points_list must be a non-empty list."); return None
        try:
            points_arr = np.array([(p[0], p[1], p[2]) if not isinstance(p, Vector) else (p.x, p.y, p.z) for p in input_points_list], dtype=float)
            if points_arr.ndim != 2 or points_arr.shape[1] != 3: raise ValueError("Processed array is not Nx3.")
            input_points_xy = points_arr[:, :2]; input_z_values = points_arr[:, 2]
            if input_points_xy.shape[0] < 3: raise ValueError("Need at least 3 points for interpolation.")
            self.log_to_file(f"Processed {len(input_points_list)} input points.")
        except (IndexError, TypeError, ValueError) as e: self.log_to_file(f"Error processing input_points_list: {e}"); return None

        # --- Calculate Convex Hull Path for Normal Checking ---
        boundary_path = None
        try:
            if input_points_xy.shape[0] >= 3:
                hull = ConvexHull(input_points_xy); hull_points = input_points_xy[hull.vertices]
                boundary_path = Path(hull_points); self.log_to_file("Calculated convex hull path.")
            else: self.log_to_file("Warning: Not enough points for convex hull.")
        except Exception as hull_e: self.log_to_file(f"Warning: Convex hull calculation failed: {hull_e}.")

        # Calculate bounds and grid parameters
        # ... (Same as before) ...
        min_xy = input_points_xy.min(axis=0); max_xy = input_points_xy.max(axis=0)
        center_xy = (min_xy + max_xy) / 2.0; size_xy = max_xy - min_xy
        if size_xy[0] <= 1e-6 or size_xy[1] <= 1e-6: self.log_to_file("Error: Input points have zero XY extent."); return None
        longest_dim = max(size_xy)
        if not hasattr(self, 'contour_grid_resolution') or self.contour_grid_resolution <= 0: cell_s = 1.0
        elif longest_dim <= 1e-9: cell_s = 1.0
        else: cell_s = longest_dim / self.contour_grid_resolution
        x_subdivisions = max(1, int(round(size_xy[0] / cell_s))); y_subdivisions = max(1, int(round(size_xy[1] / cell_s)))
        self.log_to_file(f"Target grid dimensions: {x_subdivisions+1} x {y_subdivisions+1}")

        # --- Create Base Grid ---
        # ... (Same as before) ...
        grid_obj = None
        try:
            bpy.ops.mesh.primitive_grid_add(x_subdivisions=x_subdivisions, y_subdivisions=y_subdivisions, size=2.0, enter_editmode=False, align='WORLD', location=(0, 0, 0))
            grid_obj = bpy.context.object; grid_obj.name = "Raw_Interpolated_Grid_With_Brim"
        except RuntimeError as e: self.log_to_file(f"Error creating grid: {e}"); return None
        grid_obj.scale = (size_xy[0] / 2.0, size_xy[1] / 2.0, 1.0); bpy.ops.object.transform_apply(scale=True)
        grid_obj.location = (center_xy[0], center_xy[1], np.mean(input_z_values)); bpy.ops.object.transform_apply(location=True)
        self.log_to_file("Created and positioned base grid.")

        # --- Interpolate Z values ---
        # ... (Same as before) ...
        mesh = grid_obj.data; vertices = mesh.vertices
        if not vertices: self.log_to_file("Error: Grid has no vertices."); bpy.data.objects.remove(grid_obj, do_unlink=True); return None
        world_matrix = grid_obj.matrix_world; grid_verts_xy = np.array([(world_matrix @ v.co).xy for v in vertices])
        if not hasattr(self, 'contour_interpolation_method'): self.contour_interpolation_method = 'linear'
        self.log_to_file(f"Interpolating Z values using method: {self.contour_interpolation_method}")
        interpolated_z = None
        try:
            if self.contour_interpolation_method == 'nearest':
                if 'SciPyKDTree' not in locals(): from scipy.spatial import KDTree as SciPyKDTree
                tree = SciPyKDTree(input_points_xy); _, indices = tree.query(grid_verts_xy); interpolated_z = input_z_values[indices]
            elif griddata is not None:
                interpolated_z = griddata(input_points_xy, input_z_values, grid_verts_xy, method=self.contour_interpolation_method, fill_value=np.nan)
                nan_indices = np.isnan(interpolated_z)
                if np.any(nan_indices):
                    self.log_to_file(f"Found {np.sum(nan_indices)} exterior points. Filling using k-NN (k=3)...")
                    if 'SciPyKDTree' not in locals(): from scipy.spatial import KDTree as SciPyKDTree
                    tree = SciPyKDTree(input_points_xy); k = 3
                    distances, indices = tree.query(grid_verts_xy[nan_indices], k=k)
                    if k > input_points_xy.shape[0]: k = input_points_xy.shape[0]; distances, indices = tree.query(grid_verts_xy[nan_indices], k=k)
                    if distances.ndim == 1: distances = distances[:, np.newaxis]; indices = indices[:, np.newaxis]
                    distances = np.maximum(distances, 1e-9); weights = 1.0 / distances; sum_weights = np.sum(weights, axis=1, keepdims=True); sum_weights = np.maximum(sum_weights, 1e-9); weights /= sum_weights
                    neighbor_z_values = input_z_values[indices]; weighted_avg_z = np.sum(neighbor_z_values * weights, axis=1); interpolated_z[nan_indices] = weighted_avg_z
            else: raise ImportError("scipy.interpolate.griddata not available")
        except ImportError: self.log_to_file("Error: SciPy not available."); bpy.data.objects.remove(grid_obj, do_unlink=True); return None
        except Exception as e: self.log_to_file(f"Error during interpolation/fill: {e}"); bpy.data.objects.remove(grid_obj, do_unlink=True); raise
        if interpolated_z is None or len(interpolated_z) != len(vertices): raise ValueError("Interpolation failed.")
        bpy.ops.object.mode_set(mode='OBJECT'); obj_origin_z = grid_obj.location.z
        for i, v in enumerate(vertices): v.co.z = interpolated_z[i] - obj_origin_z
        mesh.update(); self.log_to_file("Applied interpolated Z heights to grid.")


        # --- Add Outward Brim ---
        if extension_factor > 1e-6 and boundary_path is not None:
            self.log_to_file(f"Adding outward brim (factor: {extension_factor})...")
            bm = None
            needs_object_mode = False
            try:
                if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
                bpy.context.view_layer.objects.active = grid_obj
                grid_obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')
                needs_object_mode = True
                bm = bmesh.from_edit_mesh(mesh)
                bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()

                boundary_verts = [v for v in bm.verts if v.is_boundary]
                if not boundary_verts:
                     self.log_to_file("Warning: No boundary vertices found for brim.")
                else:
                    self.log_to_file(f"Found {len(boundary_verts)} boundary vertices for brim.")
                    # Order boundary vertices
                    ordered_boundary_verts_bm = []
                    # ... [Boundary ordering logic - same as before] ...
                    if boundary_verts:
                         start_vert = boundary_verts[0]; current_vert = start_vert; visited = {start_vert}; ordered_boundary_verts_bm.append(start_vert)
                         for _ in range(len(boundary_verts)):
                              found_next = False
                              for edge in current_vert.link_edges:
                                   if edge.is_boundary:
                                       next_vert = edge.other_vert(current_vert)
                                       if next_vert in boundary_verts and next_vert not in visited:
                                           ordered_boundary_verts_bm.append(next_vert); visited.add(next_vert); current_vert = next_vert; found_next = True; break
                              if not found_next or current_vert == start_vert: break
                         if len(ordered_boundary_verts_bm) != len(boundary_verts): self.log_to_file("Warning: Could not order all boundary vertices cleanly.")

                    if not ordered_boundary_verts_bm:
                         self.log_to_file("Error: Failed to create ordered boundary loop for brim.")
                    else:
                         self.log_to_file(f"Ordered {len(ordered_boundary_verts_bm)} vertices for brim loop.")
                         # Calculate world bounds and extension distance
                         world_matrix = grid_obj.matrix_world # Re-get matrix
                         world_verts_co = [world_matrix @ v.co for v in bm.verts]
                         world_min_x = min(v.x for v in world_verts_co); world_max_x = max(v.x for v in world_verts_co)
                         world_min_y = min(v.y for v in world_verts_co); world_max_y = max(v.y for v in world_verts_co)
                         world_size_x = world_max_x - world_min_x; world_size_y = world_max_y - world_min_y
                         avg_delta = max(world_size_x, world_size_y) * extension_factor
                         self.log_to_file(f" Brim extension distance: {avg_delta:.4f}")

                         # --- <<< Use Reference-Based Approach >>> ---
                         # Calculate normals and store by original vertex reference
                         vert_normal_map = {}
                         num_v = len(ordered_boundary_verts_bm)
                         for i, v_orig in enumerate(ordered_boundary_verts_bm):
                              v_prev = ordered_boundary_verts_bm[(i - 1 + num_v) % num_v]
                              v_next = ordered_boundary_verts_bm[(i + 1) % num_v]
                              avg_normal_xy = self.calculate_outward_normal(v_orig, v_prev, v_next, boundary_path)
                              vert_normal_map[v_orig] = avg_normal_xy # Store normal keyed by BMVert

                         # Create new vertices and store mapping: orig_BMVert -> new_BMVert
                         new_vert_map = {}
                         new_brim_verts = [] # Keep track of new verts if needed later
                         for v_orig, normal_xy in vert_normal_map.items():
                              if not v_orig.is_valid: continue # Skip if original vert somehow invalid
                              if normal_xy.length > 1e-6:
                                   new_local_co = v_orig.co + Vector((normal_xy.x * avg_delta, normal_xy.y * avg_delta, 0.0))
                                   new_v = bm.verts.new(new_local_co)
                              else:
                                   new_v = bm.verts.new(v_orig.co) # Fallback
                              new_brim_verts.append(new_v)
                              new_vert_map[v_orig] = new_v # Map original vert to new vert
                         self.log_to_file(f" Created {len(new_brim_verts)} new brim vertices.")

                         # Update lookup tables AFTER adding all new vertices
                         bm.verts.ensure_lookup_table()
                         bm.edges.ensure_lookup_table()
                         bm.faces.ensure_lookup_table()
                         self.log_to_file("Updated lookup tables after adding brim vertices.")

                         # Create Connecting Faces using references
                         faces_created = 0
                         if len(ordered_boundary_verts_bm) == len(new_brim_verts) and num_v > 1:
                              for i in range(num_v):
                                   # Get original vertices directly from the ordered list
                                   v1_orig = ordered_boundary_verts_bm[i]
                                   v2_orig = ordered_boundary_verts_bm[(i + 1) % num_v]

                                   # Get corresponding new vertices from the map
                                   v1_new = new_vert_map.get(v1_orig)
                                   v2_new = new_vert_map.get(v2_orig)

                                   # Check validity before creating face
                                   if v1_orig and v1_orig.is_valid and \
                                      v2_orig and v2_orig.is_valid and \
                                      v1_new and v1_new.is_valid and \
                                      v2_new and v2_new.is_valid:
                                        try:
                                            bm.faces.new((v1_orig, v2_orig, v2_new, v1_new))
                                            faces_created += 1
                                        except ValueError as e:
                                             self.log_to_file(f"Warning: Could not create brim quad face {i}. Error: {e}")
                                   else:
                                       self.log_to_file(f"Warning: Skipping brim face {i} due to missing/invalid vertices.")
                              self.log_to_file(f"Created {faces_created} brim quad faces.")
                         # --- <<< End Reference-Based Approach >>> ---

                # Update mesh data from BMesh
                bmesh.update_edit_mesh(mesh)
                self.log_to_file("Finished adding brim.")

            except Exception as e:
                 self.log_to_file(f"Error during brim creation: {e}")
            finally:
                 if bm and bm.is_valid: bm.free()
                 if needs_object_mode and bpy.context.mode == 'EDIT':
                      bpy.ops.object.mode_set(mode='OBJECT')
                      needs_object_mode = False
        elif boundary_path is None:
             self.log_to_file("Skipping brim creation (boundary path calculation failed).")
        else:
             self.log_to_file("Skipping brim creation (extension_factor <= 0).")


        # --- Recalculate normals ---
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            self.log_to_file("Recalculated normals on final grid with brim.")
        except Exception as e:
            self.log_to_file(f"Warning: Failed to recalculate final normals: {e}")
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')

        self.log_to_file("Smooth grid surface with brim created successfully.")
        return grid_obj

    
    def create_surface_from_alpha_shrinkwrap(self,
                                             input_points_list,
                                             alpha=0.1,
                                             inset_distance=0.05,
                                             brim_distance=1.0, # Outward brim distance
                                             subdivision_levels=2,
                                             vertical_offset=0.1):
        """
        Creates a smooth target surface for shrinkwrapping the infill.

        Workflow includes boundary inset, interior sampling, triangulation,
        shrinkwrap to self.obj, origin reset, adding an outward brim,
        and final subdivision smoothing.

        Args:
            input_points_list (list): List of 3D points defining footprint shape.
            alpha (float): Parameter for the alpha shape calculation.
            inset_distance (float): Distance to inset the boundary inwards before triangulation.
            brim_distance (float): Distance to extend the boundary outwards after shrinkwrap.
            subdivision_levels (int): Levels for the final Subdivision Surface modifier.
            vertical_offset (float): How far above self.obj's max Z to place the
                                     flat mesh before shrinkwrapping.

        Returns:
            bpy.types.Object: The generated smooth surface object, or None on failure.
        """
        self.log_to_file(f"Creating surface: Inset Alpha (a={alpha}, i={inset_distance}), Brim (d={brim_distance}) + Shrinkwrap...")
        base_mesh_obj = None # Initialize for cleanup
        shrinkwrapped_obj = None # Initialize

        # --- Steps 1-5 (Create flat, triangulated, locally centered base mesh) ---
        # (Same as the previous working version - results in base_mesh_obj)
        try:
            # ... (Input validation, XY extraction) ...
            if not input_points_list or not isinstance(input_points_list, list): raise ValueError("...")
            if len(input_points_list) < 4: raise ValueError("...")
            points_arr = np.array([(p[0], p[1], p[2]) if not isinstance(p, Vector) else (p.x, p.y, p.z) for p in input_points_list], dtype=float)
            input_points_xy = points_arr[:, :2]

            # Calculate Alpha Shape Boundary
            boundary_edges = self.alpha_shape(input_points_xy, alpha)
            if not boundary_edges: raise ValueError(f"...")
            ordered_indices = self.order_boundary_edges(boundary_edges)
            if not ordered_indices: raise ValueError("...")
            original_boundary_pts = np.array([input_points_xy[i] for i in ordered_indices])
            original_alpha_path = Path(original_boundary_pts)
            self.log_to_file(f"Original boundary count={len(original_boundary_pts)}.")

            # Calculate Inset Boundary
            inset_boundary_pts = original_boundary_pts
            inset_path = original_alpha_path
            if inset_distance > 1e-6:
                centroid = np.mean(original_boundary_pts, axis=0)
                inset_boundary_pts_list = []
                for pt in original_boundary_pts:
                    direction = centroid - pt; norm = np.linalg.norm(direction)
                    if norm > 1e-6: inset_boundary_pts_list.append(pt + (direction / norm) * inset_distance)
                    else: inset_boundary_pts_list.append(pt)
                inset_boundary_pts = np.array(inset_boundary_pts_list)
                if len(inset_boundary_pts) < 3: raise ValueError("...")
                inset_path = Path(inset_boundary_pts)
                self.log_to_file(f"Inset boundary count={len(inset_boundary_pts)}.")
            else: self.log_to_file("Skipping inset.")

            # Sample Interior Points
            xs, ys = original_boundary_pts[:, 0], original_boundary_pts[:, 1]
            min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
            longest_dim = max(max_x - min_x, max_y - min_y)
            if self.contour_grid_resolution <= 0 or longest_dim <= 1e-9: step = 1.0
            else: step = longest_dim / self.contour_grid_resolution
            step = max(step, 1e-6)
            sample_points_xy_list = []
            epsilon = step * 0.01
            for x in np.arange(min_x, max_x + epsilon, step):
                for y in np.arange(min_y, max_y + epsilon, step):
                    if original_alpha_path.contains_point((x, y)): sample_points_xy_list.append((x, y))
            sample_points_arr = np.array(sample_points_xy_list)
            self.log_to_file(f"Sampled {len(sample_points_xy_list)} interior points.")

            # Combine Points, Calculate Center, Triangulate, Filter
            if sample_points_arr.ndim == 2 and sample_points_arr.shape[0] > 0:
                combined_points_arr = np.vstack((inset_boundary_pts, sample_points_arr))
            else: combined_points_arr = inset_boundary_pts
            if len(combined_points_arr) < 3: raise ValueError("...")
            mesh_center_x = np.mean(combined_points_arr[:, 0])
            mesh_center_y = np.mean(combined_points_arr[:, 1])
            self.log_to_file(f"Mesh vertex center (World XY): ({mesh_center_x:.4f}, {mesh_center_y:.4f})")

            delaunay = Delaunay(combined_points_arr)
            final_faces = []
            kept_vertices_indices = set()
            for tri_indices in delaunay.simplices:
                pts = combined_points_arr[tri_indices]
                centroid_tri = np.mean(pts, axis=0)
                if inset_path.contains_point(centroid_tri):
                    final_faces.append(list(tri_indices))
                    kept_vertices_indices.update(tri_indices)
            if not final_faces: raise ValueError("...")
            self.log_to_file(f"Filtered to {len(final_faces)} triangles.")

            final_vertices_map = {old_idx: i for i, old_idx in enumerate(sorted(list(kept_vertices_indices)))}
            vertex_coordinates_for_mesh = []
            for old_idx in sorted(list(kept_vertices_indices)):
                 pt = combined_points_arr[old_idx]
                 local_x = pt[0] - mesh_center_x
                 local_y = pt[1] - mesh_center_y
                 vertex_coordinates_for_mesh.append((local_x, local_y, 0.0))

            final_faces_remapped = []
            for face in final_faces:
                remapped_face = [final_vertices_map[old_idx] for old_idx in face if old_idx in final_vertices_map]
                if len(remapped_face) == 3: final_faces_remapped.append(remapped_face)
            final_faces = final_faces_remapped
            if not final_faces: raise ValueError("...")

            # Create Base Mesh
            mesh_data = bpy.data.meshes.new(name="Triangulated_Inset_Base_Mesh")
            mesh_data.from_pydata(vertex_coordinates_for_mesh, [], final_faces)
            mesh_data.update()
            base_mesh_obj = bpy.data.objects.new("Triangulated_Inset_Base_Object", mesh_data)
            bpy.context.collection.objects.link(base_mesh_obj)
            self.log_to_file("Created flat base mesh.")

        except Exception as e:
             self.log_to_file(f"Error during base mesh creation steps: {e}")
             if base_mesh_obj: bpy.data.objects.remove(base_mesh_obj, do_unlink=True)
             return None

        # --- Step 6: Position Base Mesh OBJECT ORIGIN ---
        # (Same as before - positions origin using mesh_center_x/y)
        target_insole_obj = self.obj
        if not target_insole_obj or target_insole_obj.type != 'MESH':
            self.log_to_file("Error: self.obj (target insole) is not valid.")
            bpy.data.objects.remove(base_mesh_obj, do_unlink=True); return None
        try:
            target_bbox_world = [target_insole_obj.matrix_world @ Vector(corner) for corner in target_insole_obj.bound_box]
            max_z_insole = max(v.z for v in target_bbox_world)
            base_mesh_obj.location = (mesh_center_x, mesh_center_y, max_z_insole + vertical_offset)
            bpy.context.view_layer.update()
            self.log_to_file(f"Positioned base mesh origin at World: ({base_mesh_obj.location.x:.4f}, ...)")
        except Exception as e:
            self.log_to_file(f"Error positioning base mesh: {e}")
            bpy.data.objects.remove(base_mesh_obj, do_unlink=True); return None

        # --- Step 7: Apply Shrinkwrap Modifier ---
        # (Same as before)
        try:
            bpy.context.view_layer.objects.active = base_mesh_obj; base_mesh_obj.select_set(True)
            target_insole_obj.select_set(False)
            sw_mod = base_mesh_obj.modifiers.new(name="ShrinkwrapToInsole", type='SHRINKWRAP')
            sw_mod.target = target_insole_obj; sw_mod.wrap_method = 'PROJECT'
            sw_mod.use_project_z = True; sw_mod.use_negative_direction = True; sw_mod.use_positive_direction = False
            bpy.ops.object.modifier_apply(modifier=sw_mod.name)
            self.log_to_file("Applied Shrinkwrap modifier.")
            base_mesh_obj.name = "Shrinkwrapped_Surface"; shrinkwrapped_obj = base_mesh_obj
        except Exception as e:
            self.log_to_file(f"Error applying Shrinkwrap modifier: {e}")
            bpy.data.objects.remove(base_mesh_obj, do_unlink=True); return None

        # --- Step 8: Apply Object Transform to Bake Origin ---
        # (Same as before)
        try:
            bpy.context.view_layer.objects.active = shrinkwrapped_obj; shrinkwrapped_obj.select_set(True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            self.log_to_file("Object transform applied.")
        except Exception as e:
             self.log_to_file(f"Error applying object transform: {e}")
             bpy.data.objects.remove(shrinkwrapped_obj, do_unlink=True); return None

        # --- Step 9: Add Outward Brim ---
        if brim_distance > 1e-6:
            self.log_to_file(f"Adding outward brim (distance={brim_distance})...")
            bm = None
            needs_object_mode = False
            try:
                if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
                bpy.context.view_layer.objects.active = shrinkwrapped_obj
                shrinkwrapped_obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')
                needs_object_mode = True
                bm = bmesh.from_edit_mesh(shrinkwrapped_obj.data)

                # Ensure tables are fresh at the start of this edit session
                bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()

                # --- Find & Order Boundary Loop ---
                boundary_verts_ordered = [] # List to store BMVert objects
                all_boundary_edges = {e for e in bm.edges if e.is_boundary}
                if not all_boundary_edges:
                    self.log_to_file("No boundary edges found for brim creation.")
                else:
                    # Build connectivity map using BMVert objects directly
                    adj = {v: [] for e in all_boundary_edges for v in e.verts}
                    start_node = None
                    for e in all_boundary_edges:
                        v1, v2 = e.verts
                        adj[v1].append(v2)
                        adj[v2].append(v1)
                        if start_node is None: start_node = v1

                    if start_node:
                        current_loop = []
                        visited_verts = set()
                        prev_vert = None
                        current_vert = start_node

                        # Walk the loop using BMVert references
                        for _ in range(len(adj) + 1):
                            if current_vert in visited_verts:
                                if current_vert == start_node and len(current_loop) > 0: break # Closed loop
                                else: self.log_to_file(f"Warning: Boundary walk error - hit visited vert {current_vert.index} prematurely."); break

                            current_loop.append(current_vert)
                            visited_verts.add(current_vert)
                            neighbors = adj.get(current_vert, [])
                            next_vert = None
                            for n_vert in neighbors:
                                if n_vert != prev_vert: next_vert = n_vert; break

                            if next_vert is None: self.log_to_file(f"Warning: Loop walk broke at vertex {current_vert.index}."); break

                            prev_vert = current_vert
                            current_vert = next_vert
                            if len(current_loop) > len(bm.verts): self.log_to_file("Error: Loop walk safety break."); break

                        if current_loop and current_vert == start_node:
                             boundary_verts_ordered = current_loop
                             self.log_to_file(f"Found and ordered boundary loop with {len(boundary_verts_ordered)} vertices.")
                        else: self.log_to_file(f"Warning: Could not close boundary loop cleanly. Using partial loop of {len(current_loop)} verts.")

                # --- Proceed only if boundary_verts_ordered is valid ---
                if boundary_verts_ordered and len(boundary_verts_ordered) >= 3:

                    # --- Create a Path object from the boundary loop for normal checking ---
                    # Use vertex coordinates directly since origin is (0,0,0) now
                    boundary_loop_coords = [v.co.to_tuple()[:2] for v in boundary_verts_ordered]
                    boundary_path_for_normals = Path(boundary_loop_coords)
                    self.log_to_file("Created Path object for boundary normal check.")
                    # --- End Path Creation ---

                    # --- Calculate outward normals and create new vertices ---
                    new_brim_verts = []
                    vert_data_for_brim = [] # Stores {'orig': BMVert, 'normal': Vector}

                    for i, v_orig in enumerate(boundary_verts_ordered):
                        num_v = len(boundary_verts_ordered)
                        v_prev = boundary_verts_ordered[(i - 1 + num_v) % num_v]
                        v_next = boundary_verts_ordered[(i + 1 + num_v) % num_v]
                        # Pass the boundary path to the helper function
                        avg_normal_xy = self.calculate_outward_normal(v_orig, v_prev, v_next, boundary_path_for_normals)
                        vert_data_for_brim.append({'orig': v_orig, 'normal': avg_normal_xy})

                    # Create new vertices
                    new_vert_map = {} # Map original BMVert to new brim BMVert
                    for data in vert_data_for_brim:
                        v_orig = data['orig']
                        normal_xy = data['normal']
                        # Check original vertex is still valid before using its coordinate
                        if not v_orig.is_valid:
                            self.log_to_file(f"Warning: Original boundary vertex {v_orig.index} became invalid before brim vert creation.")
                            continue
                        if normal_xy.length > 1e-6:
                            new_co = v_orig.co + Vector((normal_xy.x * brim_distance, normal_xy.y * brim_distance, 0.0))
                            new_v = bm.verts.new(new_co)
                        else:
                            new_v = bm.verts.new(v_orig.co) # Fallback
                        new_brim_verts.append(new_v)
                        new_vert_map[v_orig] = new_v # Use BMVert object as key

                    # --- Create Connecting Faces ---
                    faces_created = 0
                    # Ensure tables are updated AFTER adding all new vertices and BEFORE creating faces
                    bm.verts.ensure_lookup_table()
                    bm.edges.ensure_lookup_table()
                    self.log_to_file("Refreshed lookup tables before creating brim faces.")

                    if len(boundary_verts_ordered) == len(new_brim_verts) and len(boundary_verts_ordered) > 1:
                        num_v = len(boundary_verts_ordered)
                        for i in range(num_v):
                            v1_orig = boundary_verts_ordered[i]
                            v2_orig = boundary_verts_ordered[(i + 1) % num_v]
                            # Get new verts using the original BMVert reference as the key
                            v1_new = new_vert_map.get(v1_orig)
                            v2_new = new_vert_map.get(v2_orig)

                            # Check all vertex references are valid BMesh objects
                            if v1_orig and v2_orig and v1_new and v2_new and \
                               all(isinstance(v, bmesh.types.BMVert) and v.is_valid for v in [v1_orig, v2_orig, v1_new, v2_new]):
                                try:
                                    # Use direct BMVert references
                                    bm.faces.new((v1_orig, v2_orig, v2_new, v1_new))
                                    faces_created += 1
                                except ValueError as e:
                                     self.log_to_file(f"Warning: Could not create brim quad face {i}. Error: {e}")
                            else: self.log_to_file(f"Warning: Skipping brim face {i} due to missing/invalid vertices.")
                        self.log_to_file(f"Created {faces_created} brim quad faces.")

                    # --- Update Mesh *before* exiting Edit Mode ---
                    self.log_to_file("Updating mesh data from BMesh after brim creation...")
                    bmesh.update_edit_mesh(shrinkwrapped_obj.data)
                    self.log_to_file("BMesh update complete.")
                else:
                     self.log_to_file("Skipping brim face/vert creation due to invalid boundary loop.")

                # --- Exit Edit mode ---
                bpy.ops.object.mode_set(mode='OBJECT')
                needs_object_mode = False # Reset flag
                self.log_to_file("Exited Edit Mode after brim attempt.")

            except Exception as e:
                 self.log_to_file(f"Error during brim creation: {e}")
                 # Fallback / Cleanup
                 if bm: bm.free()
                 if needs_object_mode: # If we entered Edit mode, try to exit
                      try:
                          if bpy.context.mode == 'EDIT': bpy.ops.object.mode_set(mode='OBJECT')
                      except Exception as exit_e:
                           self.log_to_file(f"Error trying to exit Edit mode during error handling: {exit_e}")
            finally:
                 # Ensure BMesh is freed if it exists and is valid
                 if bm and bm.is_valid: bm.free()
                 # Ensure we are in Object mode before proceeding
                 try:
                     if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
                 except Exception as final_exit_e:
                      self.log_to_file(f"Error ensuring Object mode in finally block: {final_exit_e}")


        # --- Step 10: Apply Subdivision Surface Modifier ---
        if subdivision_levels > 0:
            try:
                # Ensure correct context for modifier apply
                bpy.context.view_layer.objects.active = shrinkwrapped_obj
                shrinkwrapped_obj.select_set(True)
                if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')

                self.log_to_file(f"Applying Subdivision Surface (Levels: {subdivision_levels})...")
                subdiv_mod = shrinkwrapped_obj.modifiers.new(name="Subdivision", type='SUBSURF')
                subdiv_mod.levels = subdivision_levels
                subdiv_mod.render_levels = subdivision_levels
                bpy.ops.object.modifier_apply(modifier=subdiv_mod.name) # Apply should work now
                self.log_to_file("Applied Subdivision modifier.")
                shrinkwrapped_obj.name = "Smooth_Surface_Inset_Brim_Shrinkwrap"
            except Exception as e:
                self.log_to_file(f"Warning: Failed to apply Subdivision modifier: {e}. Returning surface with brim.")
        else:
            self.log_to_file("Skipping subdivision (levels=0).")

        # --- Step 11: Recalculate Normals ---
        # (Same as before)
        try:
            bpy.context.view_layer.objects.active = shrinkwrapped_obj
            shrinkwrapped_obj.select_set(True)
            if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            self.log_to_file("Recalculated final normals.")
        except Exception as e:
            self.log_to_file(f"Warning: Failed to recalculate final normals: {e}")
            if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')


        self.log_to_file("Surface with inset alpha, brim, and shrinkwrap created successfully.")
        return shrinkwrapped_obj


    def calculate_outward_normal(self, v_curr, v_prev, v_next, boundary_path):
        """
        Calculates an outward pointing XY normal at v_curr based on neighbors.
        Uses boundary_path to robustly check outward direction.
        """
        outward_normal_xy = Vector((0.0, 0.0))
        edge_count = 0

        # Normal from edge v_prev -> v_curr
        vec1 = v_curr.co - v_prev.co
        if vec1.length_squared > 1e-12:
            perp1 = Vector((-vec1.y, vec1.x)).normalized()
            outward_normal_xy += perp1
            edge_count += 1

        # Normal from edge v_curr -> v_next
        vec2 = v_next.co - v_curr.co
        if vec2.length_squared > 1e-12:
            perp2 = Vector((-vec2.y, vec2.x)).normalized()
            outward_normal_xy += perp2
            edge_count += 1

        if edge_count > 0:
            avg_normal_xy = (outward_normal_xy / edge_count).normalized()

            # --- <<< Robust Outward Check using Path >>> ---
            # Test point slightly along the calculated normal direction
            test_dist = 0.001 # Small distance
            test_point_out = v_curr.co.xy + avg_normal_xy * test_dist
            # Check if this test point is INSIDE the boundary path
            # If it IS inside, the normal is actually pointing inwards, so flip it.
            if boundary_path.contains_point(test_point_out.to_tuple()):
                 avg_normal_xy *= -1.0 # Flip the normal
            # --- <<< End Robust Check >>> ---

            return avg_normal_xy
        else:
            # Fallback... (same as before)
            self.log_to_file(f"Warning: Could not calculate normal for vertex {v_curr.index}")
            v_dir = v_curr.co.xy
            if v_dir.length_squared > 1e-12: return v_dir.normalized()
            else: return Vector((1.0, 0.0))
        

    # ------------------------------------
    # END OF TOP SURFACE CONTOURING
    # ------------------------------------
    
    def add_roof_and_floor(self,
                           contoured_lattice_obj,
                           contour_surface_obj,
                           alpha=0.1, # Alpha for footprint shape
                           xy_plane_tolerance=1e-4,
                           cleanup_merge_dist=1e-5):
        """
        Adds roof/floor. Creates footprint from lattice alpha shape, uses Boolean
        Modifiers + evaluated mesh copy to cut ulcer holes from floor base before
        solidifying floor. DEFERRED DELETION of cutter objects. Uses safer BMesh workflow.

        Args:
            contoured_lattice_obj (bpy.types.Object): Final contoured 3D lattice.
            contour_surface_obj (bpy.types.Object): Smooth surface for contouring roof.
            roof_thickness (float): Thickness of the roof layer.
            floor_thickness (float): Thickness of the floor layer.
            alpha (float): Alpha value for calculating the footprint shape.
            xy_plane_tolerance (float): Z tolerance for identifying bottom faces.
            cleanup_merge_dist (float): Merge distance for final cleanup.

        Returns:
            bpy.types.Object: Final combined object, or None on failure.
        """
        self.log_to_file(f"Starting roof/floor addition for '{contoured_lattice_obj.name}' (Evaluated Mesh Ulcer Cutout)...")

        roof_thickness = self.roof_thickness / 10
        floor_thickness = self.floor_thickness / 10

        # --- Input Validation ---
        if not contoured_lattice_obj or contoured_lattice_obj.type != 'MESH':
            self.log_to_file("Error: Invalid contoured_lattice_obj.")
            return None
        if not contour_surface_obj or contour_surface_obj.type != 'MESH':
            self.log_to_file("Error: Invalid contour_surface_obj.")
            return None
        # Removed check for self.boundary_curve
        if contoured_lattice_obj.name not in bpy.context.scene.objects:
            self.log_to_file(f"Error: Object '{contoured_lattice_obj.name}' not found.")
            return None
        if contour_surface_obj.name not in bpy.context.scene.objects:
            self.log_to_file(f"Error: Object '{contour_surface_obj.name}' not found.")
            return None
        if not hasattr(self, 'ulcer_zones'):
            self.log_to_file("Warning: self.ulcer_zones not defined.")
            self.ulcer_zones = [] # Default to empty list
        if not hasattr(self, 'nozzle_diameter') or not hasattr(self, 'thickness_multiplier'):
            self.log_to_file("Error: Missing nozzle_diameter or thickness_multiplier attribute for expansion.")
            return None


        original_name = contoured_lattice_obj.name
        temp_objects = [] # Tracks objects to be deleted LATER
        final_obj = contoured_lattice_obj # Start with the lattice
        bm = None
        bm_footprint = None # Define bm_footprint here
        needs_object_mode = False
        footprint_mesh_data = None # Store the initial footprint mesh data
        floor_base_obj = None # Object holding footprint BEFORE boolean eval
        floor_base_with_holes_obj = None # Object holding footprint AFTER boolean eval
        roof_obj = None
        floor_obj = None # This will be the floor AFTER cutting holes AND solidifying
        roof_base_obj = None
        footprint_cutter_obj = None
        ulcer_cutter_objs = [] # Store ulcer cutters for deferred deletion
        footprint_sample_density = 1.0

        try:
            # --- Ensure Object Mode ---
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')

            # --- Step 1: Fix Lattice Bottom Normals ---
            self.log_to_file("Step 1: Fixing normals on lattice bottom faces...")
            bpy.context.view_layer.objects.active = final_obj
            final_obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            needs_object_mode = True
            bm = bmesh.from_edit_mesh(final_obj.data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            bm.normal_update()
            faces_flipped = 0
            bottom_faces_found = 0
            for f in bm.faces:
                 if not f.is_valid: continue
                 center_z = f.calc_center_median().z
                 normal_z = f.normal.normalized().z
                 if abs(center_z) < xy_plane_tolerance * 10 and abs(normal_z) > 0.9:
                      bottom_faces_found += 1
                      if normal_z > 0:
                           f.normal_flip()
                           faces_flipped += 1
            bmesh.update_edit_mesh(final_obj.data)
            self.log_to_file(f"Checked {bottom_faces_found} bottom faces, flipped {faces_flipped}.")
            bm.free()
            bm = None
            bpy.ops.object.mode_set(mode='OBJECT')
            needs_object_mode = False
            
            footprint_boundary_verts_xy = []
            footprint_alpha_path = None

            # --- Step 2: Create Alpha Shape Footprint Mesh (Filled) ---
            self.log_to_file(f"Step 2: Calculating footprint alpha shape (alpha={alpha})...")
            try:
                mesh = final_obj.data
                if not mesh.vertices: raise ValueError("Lattice object has no vertices.")
                matrix_world = final_obj.matrix_world
                points_xy = np.array([(matrix_world @ v.co).xy for v in mesh.vertices])
                if len(points_xy) < 4: raise ValueError("Not enough vertices for alpha shape.")
                boundary_edges = self.alpha_shape(points_xy, alpha)
                if not boundary_edges: raise ValueError("Alpha shape failed.")
                ordered_indices = self.order_boundary_edges(boundary_edges)
                if not ordered_indices: raise ValueError("Failed to order alpha shape.")

                # Store boundary points for sampling and roof cap base
                footprint_boundary_verts_xy = [tuple(points_xy[i]) for i in ordered_indices]
                footprint_alpha_path = Path(footprint_boundary_verts_xy)

                bm_footprint = bmesh.new()
                bmesh_verts = []
                for idx in ordered_indices:
                     coord = (points_xy[idx][0], points_xy[idx][1], 0.0)
                     bmesh_verts.append(bm_footprint.verts.new(coord))
                if len(bmesh_verts) < 3: raise ValueError("Not enough valid boundary vertices.")
                for i in range(len(bmesh_verts)):
                     v1 = bmesh_verts[i]
                     v2 = bmesh_verts[(i + 1) % len(bmesh_verts)]
                     if bm_footprint.edges.get((v1, v2)) is None:
                         bm_footprint.edges.new((v1, v2))
                bm_footprint.edges.ensure_lookup_table()
                bmesh.ops.triangle_fill(bm_footprint, edges=list(bm_footprint.edges), use_beauty=True)
                for f in bm_footprint.faces:
                     if f.is_valid and f.normal.z > 0:
                         f.normal_flip()
                bm_footprint.normal_update()
                
                bm_footprint.verts.ensure_lookup_table()
                for v_fp in bm_footprint.verts:
                    v_fp.normal_update()

                footprint_mesh_data = bpy.data.meshes.new(name="Footprint_Alpha_Mesh")
                bm_footprint.to_mesh(footprint_mesh_data)
                footprint_mesh_data.validate(verbose=True)
                footprint_mesh_data.update() 
                # Create the initial object that will have modifiers added
                floor_base_obj = bpy.data.objects.new(name="FloorBase_Uncut_Orig", object_data=footprint_mesh_data)
                bpy.context.collection.objects.link(floor_base_obj)
                temp_objects.append(floor_base_obj) # Manage this original base
                self.log_to_file(f"Created alpha shape footprint mesh object '{floor_base_obj.name}'.")
            finally:
                if bm_footprint and bm_footprint.is_valid:
                    bm_footprint.free()

            if not floor_base_obj: raise ValueError("Footprint mesh object creation failed.")


            # --- Step 3: Cut Ulcer Holes from Floor Base using Boolean Modifier ---
            self.log_to_file("Step 3: Cutting ulcer holes from floor base using Boolean Modifier...")
            if self.ulcer_zones:
                # Ensure floor_base_obj is the active object
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = floor_base_obj
                floor_base_obj.select_set(True)

                cut_success_count = 0
                cutters_created = 0
                for i, (ux, uy, ur) in enumerate(self.ulcer_zones):
                    self.log_to_file(f" Creating cutter object for ulcer zone {i}...")
                    cutter_obj = None # Define cutter_obj before try block
                    try:
                        # Create Cylinder Cutter Object using bpy.ops
                        cutter_depth = floor_thickness + 2.0 # Make cutter slightly thicker than floor
                        cutter_z_center = 0 - (floor_thickness / 2.0) # Center in solidified floor
                        bpy.ops.mesh.primitive_cylinder_add(
                            vertices = 32,
                            radius = ur - (self.nozzle_diameter * self.thickness_multiplier / 2),
                            depth = cutter_depth,
                            location = (ux, uy, cutter_z_center),
                            scale = (1, 1, 1))
                        cutter_obj = bpy.context.object
                        cutter_obj.name = f"UlcerCutter_{i}"
                        temp_objects.append(cutter_obj) # Manage temporary cutter
                        cutters_created += 1

                        # Add Boolean Modifier to floor_base_obj
                        self.log_to_file(f" Applying boolean difference modifier for ulcer {i}...")
                        bool_mod = floor_base_obj.modifiers.new(name=f"CutUlcer_{i}", type='BOOLEAN')
                        bool_mod.operation = 'DIFFERENCE'
                        bool_mod.object = cutter_obj
                        bool_mod.solver = 'EXACT' # Use EXACT solver

                        # Apply the modifier immediately
                        bpy.context.view_layer.objects.active = floor_base_obj
                        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
                        self.log_to_file(f" Boolean difference applied for ulcer {i}.")
                        cut_success_count += 1

                        # Remove the cutter object AFTER applying the modifier
                        bpy.data.objects.remove(cutter_obj, do_unlink=True)
                        temp_objects.remove(cutter_obj)
                        cutter_obj = None # Reset variable

                    except Exception as e:
                         self.log_to_file(f"Warning: Boolean difference failed for ulcer {i}: {e}")
                         # Clean up modifier and cutter if they still exist
                         if cutter_obj and cutter_obj.name in bpy.data.objects:
                              if f"CutUlcer_{i}" in floor_base_obj.modifiers:
                                   floor_base_obj.modifiers.remove(floor_base_obj.modifiers[f"CutUlcer_{i}"])
                              if cutter_obj in temp_objects: temp_objects.remove(cutter_obj)
                              bpy.data.objects.remove(cutter_obj, do_unlink=True)
                self.log_to_file(f"Attempted to cut {cutters_created} ulcer zones using Boolean Modifier, {cut_success_count} succeeded.")
                # Rename the object now that holes are cut
                floor_base_obj.name = "FloorBase_WithHoles"
            else:
                self.log_to_file(" No ulcer zones defined, skipping hole cutting.")
            # floor_base_obj now contains the footprint with holes


            # --- Step 4: Creating floor solid (Manual BMesh Solidify) ---
            self.log_to_file("Step 4: Creating floor solid (Manual BMesh Solidify)...")
            floor_obj = None
            bm_floor = None # Define for finally block
            try:
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = floor_base_obj
                floor_base_obj.select_set(True)
                
                bpy.ops.object.duplicate(linked=False)
                floor_obj = bpy.context.object
                floor_obj.name = original_name + "_Floor"
                temp_objects.append(floor_obj)

                if floor_obj.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')

                if floor_obj.data.has_custom_normals:
                    self.log_to_file(f"'{floor_obj.name}' has custom normals. Clearing them...")
                    bpy.context.view_layer.objects.active = floor_obj 
                    floor_obj.select_set(True)
                    bpy.ops.mesh.customdata_custom_splitnormals_clear()
                    self.log_to_file(f"Cleared custom normals from '{floor_obj.name}'.")
                else:
                    self.log_to_file(f"'{floor_obj.name}' has no custom normals to clear.")

                self.log_to_file(f"Manually solidifying '{floor_obj.name}' using bmesh.ops.solidify...")
                
                bpy.context.view_layer.objects.active = floor_obj
                floor_obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')

                bm_floor = bmesh.from_edit_mesh(floor_obj.data)
                bm_floor.faces.ensure_lookup_table() 

                geom_to_solidify = [f for f in bm_floor.faces] 

                if not geom_to_solidify:
                    # bm_floor.free() # Will be freed in finally
                    # bpy.ops.object.mode_set(mode='OBJECT') # Will be set after try
                    raise ValueError(f"No faces found in '{floor_obj.name}' to solidify.")

                # Perform the solidify operation using bmesh.ops
                # The face normals of floor_obj (derived from floor_base_obj) should be pointing -Z.
                # A positive thickness will extrude along these normals (i.e., downwards).
                bmesh.ops.solidify(bm_floor, geom = geom_to_solidify, thickness = -abs(floor_thickness))
                
                for f_solid in bm_floor.faces: # Iterate over all faces in the modified bmesh
                    if f_solid.is_valid:
                        f_solid.normal_flip()
                
                bmesh.update_edit_mesh(floor_obj.data)
                self.log_to_file(f"'{floor_obj.name}' manually solidified using bmesh.ops.solidify.")

            except Exception as e:
                self.log_to_file(f"Error manually solidifying floor: {e}")
                if floor_obj and floor_obj.name in bpy.data.objects:
                    if floor_obj in temp_objects: temp_objects.remove(floor_obj)
                    bpy.data.objects.remove(floor_obj, do_unlink=True)
                raise
            finally:
                if bm_floor and bm_floor.is_valid: # Check if bm_floor was assigned and is valid
                    bm_floor.free()
                # Ensure we are back in object mode if an error didn't occur before mode_set
                if bpy.context.active_object == floor_obj and floor_obj.mode == 'EDIT':
                     bpy.ops.object.mode_set(mode='OBJECT')
                elif bpy.context.active_object and bpy.context.active_object.mode == 'EDIT': # Failsafe
                     bpy.ops.object.mode_set(mode='OBJECT')

            
            # --- Step 5: Create Roof using Shrinkwrap ---
            self.log_to_file("Step 5: Creating roof using Shrinkwrap...")
            roof_base_obj = None # Thickened contour surface
            roof_cap_base_obj = None # Duplicate of floor base for shrinkwrap
            roof_obj = None # Final roof object
            try:
                # 1. Create TESSELLATED Roof Cap Base
                self.log_to_file(" Creating tessellated roof cap base...")
                if not footprint_boundary_verts_xy or not footprint_alpha_path:
                     raise ValueError("Footprint boundary data missing for roof cap base.")
                xs, ys = zip(*footprint_boundary_verts_xy)
                min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
                size_x = max_x - min_x; size_y = max_y - min_y
                longest_dim = max(size_x, size_y)
                if hasattr(self, 'contour_grid_resolution') and self.contour_grid_resolution > 0 and longest_dim > 1e-9:
                     step = longest_dim / self.contour_grid_resolution
                else: step = 1.0
                step = max(step, 1e-6) * (1.0 / max(footprint_sample_density, 0.1))
                sample_points_xy_list = []
                epsilon = step * 0.01
                for x in np.arange(min_x, max_x + epsilon, step):
                    for y in np.arange(min_y, max_y + epsilon, step):
                        if footprint_alpha_path.contains_point((x, y)):
                            sample_points_xy_list.append((x, y))
                self.log_to_file(f" Sampled {len(sample_points_xy_list)} interior points for roof cap base.")
                combined_points_xy = footprint_boundary_verts_xy + sample_points_xy_list
                if len(combined_points_xy) < 3: raise ValueError("Not enough points for roof base triangulation.")
                combined_points_arr = np.array(combined_points_xy)
                delaunay = Delaunay(combined_points_arr)
                final_faces = []
                kept_vertices_indices = set()
                for tri_indices in delaunay.simplices:
                    pts = combined_points_arr[tri_indices]; centroid_tri = np.mean(pts, axis=0)
                    if footprint_alpha_path.contains_point(centroid_tri):
                        final_faces.append(list(tri_indices)); kept_vertices_indices.update(tri_indices)
                if not final_faces: raise ValueError("No valid triangles found for roof base after filtering.")
                final_vertices_map = {old_idx: i for i, old_idx in enumerate(sorted(list(kept_vertices_indices)))}
                vertex_coordinates_for_mesh = [(combined_points_arr[old_idx][0], combined_points_arr[old_idx][1], 0.0) for old_idx in sorted(list(kept_vertices_indices))]
                final_faces_remapped = []
                for face in final_faces:
                    remapped_face = [final_vertices_map[old_idx] for old_idx in face if old_idx in final_vertices_map]
                    if len(remapped_face) == 3: final_faces_remapped.append(remapped_face)
                if not final_faces_remapped: raise ValueError("No valid faces after remapping for roof base.")
                roof_cap_mesh_data = bpy.data.meshes.new(name=original_name + "_RoofCapBase_Mesh")
                roof_cap_mesh_data.from_pydata(vertex_coordinates_for_mesh, [], final_faces_remapped)
                roof_cap_mesh_data.update()
                roof_cap_base_obj = bpy.data.objects.new(original_name + "_RoofCapBase", roof_cap_mesh_data)
                bpy.context.collection.objects.link(roof_cap_base_obj)
                temp_objects.append(roof_cap_base_obj)
                
                # 2. Ensure normals point UP for shrinkwrap projection
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = roof_cap_base_obj
                roof_cap_base_obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')
                needs_object_mode = True
                bm = bmesh.from_edit_mesh(roof_cap_base_obj.data)
                for f in bm.faces:
                    if f.is_valid and f.normal.z < 0: f.normal_flip()
                bmesh.update_edit_mesh(roof_cap_base_obj.data)
                bm.free(); bm = None
                bpy.ops.object.mode_set(mode='OBJECT')
                needs_object_mode = False
                self.log_to_file(" Created tessellated roof cap base with upward normals.")
                
                
                self.log_to_file(" Cutting ulcer holes from roof cap base using Boolean Modifier...")
                if self.ulcer_zones:
                    # Ensure floor_base_obj is the active object
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.context.view_layer.objects.active = roof_cap_base_obj
                    roof_cap_base_obj.select_set(True)

                    cut_success_count = 0
                    cutters_created = 0
                    for i, (ux, uy, ur) in enumerate(self.ulcer_zones):
                        self.log_to_file(f" Creating cutter object for ulcer zone {i}...")
                        cutter_obj = None # Define cutter_obj before try block
                        try:
                            # Create Cylinder Cutter Object using bpy.ops
                            cutter_depth = floor_thickness + 2.0 # Make cutter slightly thicker than floor
                            cutter_z_center = 0 - (floor_thickness / 2.0) # Center in solidified floor
                            bpy.ops.mesh.primitive_cylinder_add(
                                vertices = 32,
                                radius = ur - (self.nozzle_diameter * self.thickness_multiplier / 2),
                                depth = cutter_depth,
                                location = (ux, uy, cutter_z_center),
                                scale = (1, 1, 1))
                            cutter_obj = bpy.context.object
                            cutter_obj.name = f"UlcerCutter_{i}"
                            temp_objects.append(cutter_obj) # Manage temporary cutter
                            cutters_created += 1

                            # Add Boolean Modifier to floor_base_obj
                            self.log_to_file(f" Applying boolean difference modifier for ulcer {i}...")
                            bool_mod = roof_cap_base_obj.modifiers.new(name=f"CutUlcer_{i}", type='BOOLEAN')
                            bool_mod.operation = 'DIFFERENCE'
                            bool_mod.object = cutter_obj
                            bool_mod.solver = 'EXACT' # Use EXACT solver

                            # Apply the modifier immediately
                            bpy.context.view_layer.objects.active = roof_cap_base_obj
                            bpy.ops.object.modifier_apply(modifier=bool_mod.name)
                            self.log_to_file(f" Boolean difference applied for ulcer {i}.")
                            cut_success_count += 1

                            # Remove the cutter object AFTER applying the modifier
                            bpy.data.objects.remove(cutter_obj, do_unlink=True)
                            temp_objects.remove(cutter_obj)
                            cutter_obj = None # Reset variable

                        except Exception as e:
                             self.log_to_file(f"Warning: Boolean difference failed for ulcer {i}: {e}")
                             # Clean up modifier and cutter if they still exist
                             if cutter_obj and cutter_obj.name in bpy.data.objects:
                                  if f"CutUlcer_{i}" in floor_base_obj.modifiers:
                                       floor_base_obj.modifiers.remove(roof_cap_base_obj.modifiers[f"CutUlcer_{i}"])
                                  if cutter_obj in temp_objects: temp_objects.remove(cutter_obj)
                                  bpy.data.objects.remove(cutter_obj, do_unlink=True)
                    self.log_to_file(f"Attempted to cut {cutters_created} ulcer zones using Boolean Modifier, {cut_success_count} succeeded.")
                    # Rename the object now that holes are cut
                    roof_cap_base_obj.name = "RoofCapBase_WithHoles"
                else:
                    self.log_to_file(" No ulcer zones defined, skipping hole cutting.")
                
                
                # 3. Shrinkwrap Roof Cap Base onto Thickened Roof Base
                self.log_to_file(" Shrinkwrapping roof cap base...")
                mod_roof_sw = roof_cap_base_obj.modifiers.new(name="RoofContour", type='SHRINKWRAP')
                #mod_roof_sw.target = roof_base_obj # Target the thick roof
                mod_roof_sw.target = contour_surface_obj
                mod_roof_sw.wrap_method = 'PROJECT'
                mod_roof_sw.use_project_z = True
                mod_roof_sw.use_negative_direction = False # Project UPWARDS
                mod_roof_sw.use_positive_direction = True
                bpy.ops.object.modifier_apply(modifier=mod_roof_sw.name)
                self.log_to_file(" Roof cap base shrinkwrapped.")

                # 4. Solidify the Contoured Roof Cap
                self.log_to_file(" Applying Solidify to roof cap...")
                mod_roof_cap_solid = roof_cap_base_obj.modifiers.new(name="RoofCapSolidify", type='SOLIDIFY')
                mod_roof_cap_solid.thickness = roof_thickness
                mod_roof_cap_solid.offset = 1.0 # Thicken along positive normal (+Z)
                mod_roof_cap_solid.use_even_offset = True
                mod_roof_cap_solid.use_quality_normals = True
                bpy.ops.object.modifier_apply(modifier=mod_roof_cap_solid.name)
                roof_obj = roof_cap_base_obj # This is now the final roof object
                roof_obj.name = original_name + "_Roof"
                self.log_to_file(" Roof cap created.")

                # Cleanup the thickened roof base target
                bpy.data.objects.remove(contour_surface_obj, do_unlink=True)
                
            except Exception as e:
                self.log_to_file(f"Error creating roof: {e}")
                if bm and bm.is_valid: 
                    bm.free()
                if roof_base_obj and roof_base_obj.name in bpy.data.objects: 
                    bpy.data.objects.remove(roof_base_obj, do_unlink=True)
                if roof_cap_base_obj and roof_cap_base_obj.name in bpy.data.objects: 
                    bpy.data.objects.remove(roof_cap_base_obj, do_unlink=True)
                if roof_cap_top_obj and roof_cap_top_obj.name in bpy.data.objects: 
                    bpy.data.objects.remove(roof_cap_top_obj, do_unlink=True)
                raise
            
            # --- Step 6: Join Lattice, Roof, and Floor ---
            self.log_to_file("Step 6: Joining Lattice, Roof, and Floor...")
            objects_to_join = [obj for obj in [final_obj, roof_obj, floor_obj] if obj and obj.name in bpy.data.objects]
            if len(objects_to_join) > 1: # Need at least 2 objects to join
                 bpy.ops.object.select_all(action='DESELECT')
                 # Make the original lattice object the active one for the join result
                 bpy.context.view_layer.objects.active = final_obj
                 for obj in objects_to_join:
                      obj.select_set(True)

                 bpy.ops.object.join()
                 final_obj = bpy.context.object # Result is the previously active object
                 final_obj.name = "InsoleObject"
                 self.log_to_file(f"Joined components into '{final_obj.name}'.")

                 # Remove original references from temp list if they were joined
                 if roof_obj in temp_objects: temp_objects.remove(roof_obj)
                 if floor_obj in temp_objects: temp_objects.remove(floor_obj)
            else:
                 self.log_to_file("Warning: Not enough valid objects to join (Lattice/Roof/Floor).")


            # --- Step 8: Output ---
            self.log_to_file("Roof and floor addition successful.")
            # ... (Cleanup footprint data, boundary curve, intermediate floor base) ...
            if footprint_mesh_data and footprint_mesh_data.users == 0: bpy.data.meshes.remove(footprint_mesh_data)
            if 'mesh_data_with_holes' in locals() and mesh_data_with_holes and mesh_data_with_holes.users == 0: bpy.data.meshes.remove(mesh_data_with_holes)
            if hasattr(self, 'boundary_curve') and self.boundary_curve and self.boundary_curve.name in bpy.data.objects:
                 curve_data = self.boundary_curve.data; bpy.data.objects.remove(self.boundary_curve, do_unlink=True)
                 if curve_data and curve_data.users == 0: bpy.data.curves.remove(curve_data)
                 self.boundary_curve = None
            if floor_base_with_holes_obj and floor_base_with_holes_obj.name in bpy.data.objects:
                 if floor_base_with_holes_obj in temp_objects: temp_objects.remove(floor_base_with_holes_obj)
                 bpy.data.objects.remove(floor_base_with_holes_obj, do_unlink=True)

            return final_obj
            
        # --- Error Handling & Cleanup ---
        finally:
             # Final safety net
             if bm and bm.is_valid: bm.free()
             try:
                 if needs_object_mode and bpy.context.object and bpy.context.object.mode != 'OBJECT':
                     bpy.ops.object.mode_set(mode='OBJECT')
                 # Clean up temporary objects
                 active_obj_name = final_obj.name if final_obj and final_obj.name in bpy.data.objects else ""
                 objs_to_remove_on_error = []
                 for obj in temp_objects:
                      if obj and obj.name != active_obj_name and obj.name in bpy.data.objects:
                           objs_to_remove_on_error.append(obj)
                 if objs_to_remove_on_error:
                     self.log_to_file(f"Cleaning up {len(objs_to_remove_on_error)} temporary objects in finally block...")
                     bpy.ops.object.select_all(action='DESELECT')
                     for obj in objs_to_remove_on_error:
                          if obj.name in bpy.data.objects: obj.select_set(True)
                     if bpy.context.selected_objects: bpy.ops.object.delete()
                 # Cleanup footprint data
                 if 'footprint_mesh_data' in locals() and \
                    footprint_mesh_data and \
                    footprint_mesh_data.users == 0:
                     try: bpy.data.meshes.remove(footprint_mesh_data)
                     except: pass
                 if 'mesh_data_with_holes' in locals() and \
                    mesh_data_with_holes and \
                    mesh_data_with_holes.users == 0:
                     try: bpy.data.meshes.remove(mesh_data_with_holes)
                     except: pass
                 # Cleanup boundary curve
                 if hasattr(self, 'boundary_curve') and self.boundary_curve and self.boundary_curve.name in bpy.data.objects:
                      try:
                           curve_data = self.boundary_curve.data
                           bpy.data.objects.remove(self.boundary_curve, do_unlink=True)
                           if curve_data and curve_data.users == 0: bpy.data.curves.remove(curve_data)
                           self.boundary_curve = None
                      except: pass

             except Exception as final_e:
                  self.log_to_file(f"Error during final cleanup/mode check: {final_e}")
        
                
    # --------------------------------------------------------
    # END OF ROOF AND FLOOR ADDTION
    # --------------------------------------------------------
                
                
    def get_drape_grid_from_initial(self, context, base_grid_name, initial_surface_obj, 
                                    x_vertices, y_vertices):
        """
        Creates a NEW grid object based on the bounds and sampled Z heights of the initial_surface_obj.
        Requires x_vertices and y_vertices to define the new grid's resolution.
        Returns the newly created grid object (Mesh type) or None on failure.
        """
        self.log_to_file(f"Creating new grid '{base_grid_name}' based on initial surface '{initial_surface_obj.name}'.")

        # Ensure mesh data exists for calculations
        temp_mesh_data = None
        is_temp_mesh = False
        depsgraph = context.evaluated_depsgraph_get()
        mesh_data_owner = initial_surface_obj.evaluated_get(depsgraph)

        # --- Calculate bounds, average Z, and MAX Z ---
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        avg_z_sum = 0.0
        vert_count = len(mesh_data_owner.data.vertices)

        if vert_count == 0:
             raise ValueError(self.log_to_file(f"Error: Initial surface '{initial_surface_obj.name}' (or its mesh conversion) has no vertices."))

        initial_surf_matrix_world = mesh_data_owner.matrix_world
        for v in mesh_data_owner.data.vertices:
            world_co = initial_surf_matrix_world @ v.co
            min_x = min(min_x, world_co.x)
            max_x = max(max_x, world_co.x)
            min_y = min(min_y, world_co.y)
            max_y = max(max_y, world_co.y)
            min_z = min(min_z, world_co.z)
            max_z = max(max_z, world_co.z)
            avg_z_sum += world_co.z

        avg_z_world = avg_z_sum / vert_count
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        width = max_x - min_x
        height = max_y - min_y

        self.log_to_file(f"Initial surface bounds (world): X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}], Z[{min_z:.3f}, {max_z:.3f}]")
        self.log_to_file(f"Initial surface center XY: ({center_x:.3f}, {center_y:.3f}), Avg Z: {avg_z_world:.3f}")
        self.log_to_file(f"Initial surface dimensions W: {width:.3f}, H: {height:.3f}")

        # --- Create the new grid ---
        self.log_to_file(f"Creating new grid '{base_grid_name}' with dimensions {x_vertices}x{y_vertices}.")

        bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=x_vertices - 1,
            y_subdivisions=y_vertices - 1,
            size=1.0,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0)
        )
        drape_grid_obj = context.active_object
        drape_grid_obj.name = base_grid_name

        # Scale and position the new grid
        scale_x = width / 1.0 if width > 1e-6 else 1.0
        scale_y = height / 1.0 if height > 1e-6 else 1.0
        drape_grid_obj.scale = (scale_x, scale_y, 1.0)
        drape_grid_obj.location = (center_x, center_y, avg_z_world)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        self.log_to_file(f"Scaled grid and moved to match initial surface. Scale applied.")

        # --- Sample Z heights ---
        self.log_to_file(f"Sampling Z heights for '{drape_grid_obj.name}' from '{initial_surface_obj.name}'.")
        initial_surf_matrix_world_inv = initial_surf_matrix_world.inverted()
        ray_start_z_world = max_z + 1.0

        for v_grid in drape_grid_obj.data.vertices:
            grid_vert_world_xy = drape_grid_obj.matrix_world @ v_grid.co
            origin_world = Vector((grid_vert_world_xy.x, grid_vert_world_xy.y, ray_start_z_world))
            direction_world = Vector((0, 0, -1))
            origin_local_init_surf = initial_surf_matrix_world_inv @ origin_world
            direction_local_init_surf = initial_surf_matrix_world_inv.to_3x3() @ direction_world
            direction_local_init_surf.normalize()

            hit, loc_local, norm_local, face_idx = mesh_data_owner.ray_cast(
                origin_local_init_surf, direction_local_init_surf
            )

            if hit:
                hit_loc_world = initial_surf_matrix_world @ loc_local
                hit_loc_local_grid = drape_grid_obj.matrix_world.inverted() @ hit_loc_world
                v_grid.co.z = hit_loc_local_grid.z
            else:
                 avg_z_local_grid = (drape_grid_obj.matrix_world.inverted() @ Vector((0,0,avg_z_world))).z
                 v_grid.co.z = avg_z_local_grid

        drape_grid_obj.data.update()
        self.log_to_file("Finished sampling Z heights.")

        return drape_grid_obj


    def raycast_and_identify_hits(self, drape_obj, target_obj):
        """
        Performs raycasting from drape_obj vertices upwards to target_obj.
        Handles coordinate space transformations correctly.
        Modifies drape_obj vertices ONLY for hits that occur BELOW the vertex's original Z height.
        Non-hit vertices and hits above original Z retain their original Z.
        Rays start from below both objects.
        Returns: hit_indices, non_hit_indices
        """
        context = bpy.context # Get context locally
        depsgraph = context.evaluated_depsgraph_get()
        target_obj_eval = target_obj.evaluated_get(depsgraph)
        drape_obj_eval = drape_obj.evaluated_get(depsgraph)

        drape_obj_matrix_world = drape_obj_eval.matrix_world
        target_matrix_world = target_obj_eval.matrix_world
        target_matrix_world_inv = target_matrix_world.inverted()
        drape_obj_matrix_world_inv = drape_obj_matrix_world.inverted()

        min_z_target = float('inf')
        min_z_drape = float('inf')
        try:
            target_bbox_corners = [target_matrix_world @ Vector(corner) for corner in target_obj.bound_box]
            min_z_target = min(corner.z for corner in target_bbox_corners)
            if drape_obj_eval.data.vertices:
                 min_z_drape = min((drape_obj_matrix_world @ v.co).z for v in drape_obj_eval.data.vertices)
            else: min_z_drape = 0.0
        except Exception as e:
            self.log_to_file(f"Could not calculate bounding boxes for ray start Z: {e}")
            min_z_target = -1000.0
            min_z_drape = -1000.0

        ray_start_z = min(min_z_target, min_z_drape) - 0.1
        self.log_to_file(f"Ray start Z calculated: {ray_start_z:.4f}")

        hit_indices = []
        non_hit_indices = []

        self.log_to_file(f"Starting raycasts from Z={ray_start_z:.4f} towards target '{target_obj.name}'...")
        for i, v_local in enumerate(drape_obj.data.vertices):
            original_vertex_world_co = drape_obj_matrix_world @ v_local.co
            original_vertex_world_z = original_vertex_world_co.z
            origin_world = Vector((original_vertex_world_co.x, original_vertex_world_co.y, ray_start_z))
            direction_world = Vector((0, 0, 1))
            origin_local_target = target_matrix_world_inv @ origin_world
            direction_local_target = target_matrix_world_inv.to_3x3() @ direction_world
            direction_local_target.normalize()

            hit, location_local_target, normal_local_target, face_index = target_obj_eval.ray_cast(
                origin_local_target, direction_local_target, distance=10000.0
            )

            if hit:
                location_world = target_matrix_world @ location_local_target
                if location_world.z <= original_vertex_world_z:
                    hit_indices.append(i)
                    location_local_drape = drape_obj_matrix_world_inv @ location_world
                    v_local.co.z = location_local_drape.z
                else:
                    non_hit_indices.append(i)
            else:
                non_hit_indices.append(i)

        drape_obj.data.update()
        self.log_to_file(f"Raycasting complete: {len(hit_indices)} valid hits, {len(non_hit_indices)} non-hits (including ignored hits).")
        return hit_indices, non_hit_indices


    def apply_smoothing(self, drape_obj, iterations, factor):
        """Applies standard Laplacian smoothing to the Z-coordinates of ALL vertices."""
        if iterations == 0:
            raise ValueError("No smoothing applied (iterations is 0).")
        if not drape_obj.data.vertices:
            raise ValueError("No vertices to smooth.")
             
        mesh = drape_obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()

        self.log_to_file(f"Applying {iterations} smoothing passes to all {len(bm.verts)} vertices...")
        for i in range(iterations):
            new_z_values_local = {}
            for v_idx, v_bm in enumerate(bm.verts):
                if not v_bm.link_edges: continue
                neighbor_z_sum_local = 0.0
                neighbor_count = 0
                for edge in v_bm.link_edges:
                    other_v = edge.other_vert(v_bm)
                    neighbor_z_sum_local += other_v.co.z
                    neighbor_count += 1
                avg_neighbor_z_local = v_bm.co.z
                if neighbor_count > 0:
                    avg_neighbor_z_local = neighbor_z_sum_local / neighbor_count
                smoothed_z_local = v_bm.co.z * (1.0 - factor) + avg_neighbor_z_local * factor
                new_z_values_local[v_idx] = smoothed_z_local
            for v_idx, new_z_local in new_z_values_local.items():
                bm.verts[v_idx].co.z = new_z_local
        bm.to_mesh(mesh)
        bm.free()
        drape_obj.data.update()
        self.log_to_file(f"Finished applying smoothing passes.")


    def create_output_mesh(self, context, drape_obj_final_mesh, output_mesh_name):
        """Creates the final output mesh by duplicating the smoothed drape grid."""
        self.log_to_file(f"Creating final output mesh '{output_mesh_name}' by duplicating '{drape_obj_final_mesh.name}'.")
        
        # Ensure object mode and selection for duplication
        current_mode = context.object.mode if context.object else 'OBJECT'
        if current_mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        drape_obj_final_mesh.select_set(True)
        context.view_layer.objects.active = drape_obj_final_mesh

        bpy.ops.object.duplicate(linked=False)
        output_mesh_obj = context.active_object
        output_mesh_obj.name = output_mesh_name
        
        # Restore original mode if needed
        if current_mode != 'OBJECT': bpy.ops.object.mode_set(mode=current_mode)

        self.log_to_file(f"Successfully created output mesh: {output_mesh_obj.name}")
        return output_mesh_obj


    def drape_surface(self,
        target_obj,
        initial_surface_obj,
        smoothing_iterations=10,
        smoothing_factor=0.5
        ):
        """
        Main function to drape a surface.

        Args:
            target_object: Target mesh object to drape over.
            initial_surface: Initial mesh/surface object defining the base shape.

        Returns:
            bpy.types.Object: The final draped mesh object, or None on failure.
        """
        context = bpy.context
        
        # --- Create Working Grid based on Initial Surface ---
        drape_grid_obj = self.get_drape_grid_from_initial(
            context, "BaseGrid", initial_surface_obj,
            self.draping_grid_resolution, self.draping_grid_resolution
        )
        if not drape_grid_obj:
            self.log_to_file("Failed to create drape grid from initial surface. Aborting.")
            return None

        # --- Raycast and Modify Grid ---
        hit_indices, non_hit_indices = self.raycast_and_identify_hits(
            drape_grid_obj, target_obj
        )
        
        # --- Apply Smoothing ---
        self.apply_smoothing(
            drape_grid_obj,
            smoothing_iterations,
            smoothing_factor
        )
        
        # --- Create Final Output Mesh ---
        final_mesh_surface = self.create_output_mesh(
            context, drape_grid_obj, "Draped_Foot_Surface"
        )

        # --- Cleanup Intermediate Grid ---
        if drape_grid_obj and drape_grid_obj.name == "BaseGrid":
             if drape_grid_obj.name in bpy.data.objects: # Check it exists
                 self.log_to_file(f"Deleting intermediate grid object '{drape_grid_obj.name}'.")
                 bpy.data.objects.remove(drape_grid_obj, do_unlink=True)
        
        # Cleanup initial surface
        bpy.data.objects.remove(initial_surface_obj, do_unlink=True)
        
        if final_mesh_surface:
            self.log_to_file(f"--- Draping Process Complete. Output Mesh: '{final_mesh_surface.name}' ---")
        else:
            self.log_to_file(f"--- Draping Process Failed to create output mesh. ---")

        return final_mesh_surface


    # --------------------------------------------------------
    # END OF IMPRINTING USING 3D FOOT SCAN
    # --------------------------------------------------------







def main():   
    
    generator = InsoleGenerator(
        template_insole_stl_path = "",
        density_map_path = "",
        log_file_path = "",
        
        # Optional, set as None if not using a 3D scan. STL file should contain the word "scan".
        # If no foot scan model exists within the Blender scene, the model will be imported from the specified filepath.
        # You may align the foot scan mesh manually if the auto-alignment procedure fails.
        # Run the script again after manually aligning and the updated foot scan position will be used.
        foot_3D_scan_path = "",
        
        # Which side foot the pressure map and 3D scan are for
        foot_side = "RIGHT",   # "LEFT" or "RIGHT"
        
        rotate_pressure_CSV_180 = False,  # used in cases where the patient stood on the pressure pad upside-down
        
        # In cm, used to scale the base insole model. If a 3D foot model is provided, that will be used instead.
        target_insole_length = 28.0,  
        
        # How much bigger to make the insole than the foot dimensions
        insole_oversize_factor_length = 1.02,
        insole_oversize_factor_width = 1.05,
        
        
        grade_lattice_density_based_on_pressure = True,   # Set to False to generate an internal lattice with a global density of uniform_density
        inverse_pressure_grading = True,                  # Inverses the grading when True to map higher pressure regions to higher density infill
        uniform_density = 0.15,                           # Set as float from 0.1-1.0. Sparse lattice = 0.15, dense lattice = 0.4
        
        # Controls the magnitude of difference in graded lattice density between high and low pressure regions
        expansion_coefficient = 1.3,  # Recommended value 1.6 for normal grading and 1.3 for inverse grading
        
        # Controls the overall size of lattice cells before grading
        # Honeycomb lattice cell size is calculated as csv_cell_spacing divided by spacing_coefficent
        spacing_coefficent = 1.2,     # Recommended value 0.42 for normal grading and 1.2 for inverse grading
        
        csv_cell_spacing = 0.008382,  # hor/vert spacing between pressure sensors, from pressure pad metadata. IN METRES
        nozzle_diameter = 0.4,        # 3D printer nozzle diameter, in mm
        
        thickness_multiplier = 1.0,   # thickness of lattice/boundary walls in multiples of the nozzle diameter
        
        # Circular exclusions zones for ulcers, list of (x_coord, y_coord, radius) tuples, can overlap
        ulcer_zones = [
            #(18.0, -2.0, 1.25),
            #(17.0, -2.5, 1.0),
            #(4.0, -0.3, 1.5)
            #(18.0, 4.0, 1.5),
            #(10.0, 0.0, 1.0)
        ],
        
        # Options for contouring the top surface of the insole:
        #   "ORIGINAL_MODEL" - Extracts and uses the top surface of the base insole model from stl_path
        #   "PRESSURE"       - Uses the csv pressure map to indent the top surface of the insole according to weight distribution
        #   "3D_SCAN"        - Extracts and uses the plantar surface geometry of the foot scan from foot_model_path
        contour_method = "3D_SCAN",
        
        # ------- Top surface contouring parameters for PRESSURE option -----------
        # The top surface foot impression is created by interpolating a surface between the points in the
        # pressure map, where the region of highest pressure is mapped to pressure_points_base_height above
        # the lattice floor and the regions of zero pressure are mapped pressure_points_top_height above the
        # lattice floor.
        pressure_points_base_height = 0.2,            # min height (cm) for contouring points
        pressure_points_top_height  = 1.0,            # max height (cm) for contouring points
        pressure_points_extrapolation_factor = 1.5,   # extrapolates the remaining foot surface by curving the points
                                                      # up to pressure_points_top_height * pressure_points_extrapolation_factor
        
        contour_grid_resolution = 60,       # only used for PRESSURE option
        draping_grid_resolution = 200,      # only used for 3D_SCAN option
        
        # Thickness of insole roof and floor (in mm)
        roof_thickness = 1.0,
        floor_thickness = 0.6
    )
    

    
    # VISUALISATION FUNCTIONS FOR DEBUG ---------------------------------------------------------------------------------
    
    #csv_obj = generator.visualize_csv_data()
    
    #deformed_grid_obj = generator.visualize_deformed_hexagon_grid()
    
    #fallback_obj = generator.visualize_fallback_vertices()
    
    #boundary_obj = generator.visualize_outer_clipping_boundary()
    
    #mapping_field_obj = generator.visualize_mapping_field_boundary()
    
    # -------------------------------------------------------------------------------------------------------------------


    
    # generate density-graded infill pattern based on pressure map distribution
    infill_2d = generator.generate_2d_infill()
    
    # extrude the graded infill pattern into a 3D lattice
    infill_3d = generator.convert_infill_to_3d(infill_2d)
    
    # Extract the top surface of the base insole model
    foot_contour_points_list_insole = generator.contouring_points_from_stl_top_surface_floodfill()
    #insole_surface_point_cloud = generator.visualize_point_cloud(foot_contour_points_list_insole)
    base_insole_top_surface = generator.create_surface_from_alpha_shrinkwrap(foot_contour_points_list_insole, brim_distance=3.0)
        
    # generate a surface to be used for contouring
    if generator.contour_method == "ORIGINAL_MODEL":
        new_insole_top_surface = base_insole_top_surface
    
    elif generator.contour_method == "PRESSURE":
        foot_contour_points_list_csv = generator.contouring_points_from_pressure_csv()
        #csv_surface_point_cloud = generator.visualize_point_cloud(foot_contour_points_list_csv)
        foot_model_analogue = generator.create_smooth_foot_surface(foot_contour_points_list_csv)
        new_insole_top_surface = generator.drape_surface(foot_model_analogue, base_insole_top_surface)
    
    elif generator.contour_method == "3D_SCAN":
        new_insole_top_surface = generator.drape_surface(generator.foot_scan_model, base_insole_top_surface)
    
    
    # use a shrinkwrap modifier to deform the top surface of the lattice to the contours of the surface object
    infill_3d_contoured = generator.apply_contour_shrinkwrap(infill_3d, new_insole_top_surface)
    
    # generate a floor and (contoured) roof for the lattice
    final_insole_obj = generator.add_roof_and_floor(infill_3d_contoured, new_insole_top_surface)
    
    # clean up
    bpy.data.objects.remove(generator.obj, do_unlink=True)
    if generator.contour_method == "PRESSURE":
        bpy.data.objects.remove(foot_model_analogue, do_unlink=True)
    

if __name__ == "__main__":
    main()