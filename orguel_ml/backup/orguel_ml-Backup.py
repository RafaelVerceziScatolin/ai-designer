from typing import List, Tuple
import cupy
import cudf
import math
from torch.utils.dlpack import from_dlpack
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo # DEBUG

class Graph:
    _attributes = cupy.arange(10, dtype=cupy.int32)
    
    parallel = 0
    colinear = 1
    perpendicular_distance = 2
    overlap_ratio = 3
    point_intersection = 4
    segment_intersection = 5
    angle_difference = 6
    angle_difference_sin = 7
    angle_difference_cos = 8
    perimeter_intersection = 9
    
    def __init__(self, dataframe, edges_per_element=200):
        self._dataframe = dataframe
        
        start_x, start_y = dataframe['start_x'], dataframe['start_y']
        end_x, end_y = dataframe['end_x'], dataframe['end_y']
        length = dataframe['length']
        angle = dataframe['angle']
        circle = dataframe['circle']
        arc = dataframe['arc']
        
        # Normalize coordinates
        coordinates_x = cupy.concatenate([start_x.to_cupy(), end_x.to_cupy()])
        coordinates_y = cupy.concatenate([start_y.to_cupy(), end_y.to_cupy()])
        
        normalized_start_x = (start_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_start_y = (start_y - coordinates_y.mean()) / coordinates_y.std()
        normalized_end_x = (end_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_end_y = (end_y - coordinates_y.mean()) / coordinates_y.std()
        
        normalized_coordinates = cupy.stack([normalized_start_x.to_cupy(), normalized_start_y.to_cupy(),
                                            normalized_end_x.to_cupy(), normalized_end_y.to_cupy()], axis=1)
        
        # Normalize angles and lengths
        normalized_angles = cupy.stack([cupy.sin(angle.to_cupy()), cupy.cos(angle.to_cupy())], axis=1)
        normalized_lengths = (length / length.max()).to_cupy().reshape(-1, 1)
        
        # Flags
        circle_flags = circle.to_cupy().reshape(-1, 1)
        arc_flags = arc.to_cupy().reshape(-1, 1)
        
        # Classification labels
        self.classificationLabels = (dataframe['layer']
            .map({"beam": 0, "column": 1, "eave": 2, "slab_hole": 3, "stair": 4, "section": 5, "info": 6, "0": 7})
            .to_cupy() if "layer" in dataframe.columns else None)
        
        # Node attributes
        self.nodeAttributes = from_dlpack(cupy.hstack([
            normalized_coordinates,
            normalized_angles,
            normalized_lengths,
            circle_flags,
            arc_flags
        ]).toDlpack())
        
        edge_capacity = len(dataframe) * edges_per_element
        
        self.size = 0
        self.edges = cupy.empty((2, edge_capacity), dtype=cupy.int32)
        self.edgeAttributes = cupy.empty((edge_capacity, len(self._attributes)), dtype=cupy.float32)
    
    @staticmethod
    def create_arc(center, radius, startAngle, endAngle, resolution=64):
        
        startAngle = math.radians(startAngle)
        endAngle   = math.radians(endAngle)
        angle = endAngle - startAngle
        
        if angle < 0: angle += 2*math.pi
        
        segments = math.ceil((angle / (2*math.pi))*resolution)
        
        points = []
        step = angle / segments
        for segment in range(segments+1):
            stepAngle = startAngle + segment * step
            px = center.x + radius * math.cos(stepAngle)
            py = center.y + radius * math.sin(stepAngle)
            points.append((px, py))
        return points
         
    @staticmethod
    def overlap_ratios(start_xA, start_yA, end_xA, end_yA, lengthA,
                       start_xB, start_yB, end_xB, end_yB, lengthB):
        # Compute direction vector
        dxA, dyA = end_xA - start_xA, end_yA - start_yA
        dir_lengthA = cupy.sqrt(dxA**2 + dyA**2)
        
        # Check if the line has zero length
        validA = dir_lengthA >= 1e-12
        
        # Compute unit vector
        ndxA = cupy.where(validA, dxA / dir_lengthA, 0.0)
        ndyA = cupy.where(validA, dyA / dir_lengthA, 0.0)
        
        # Project B's endpoints onto A
        tB1 = ((start_xB - start_xA) * ndxA + (start_yB - start_yA) * ndyA)
        tB2 = ((end_xB - start_xA) * ndxA + (end_yB - start_yA) * ndyA)
        
        # Get the projected range
        tmin, tmax = cupy.minimum(tB1, tB2), cupy.maximum(tB1, tB2)
        
        # Clip the projection range within [0, lengthA]
        overlap_start = cupy.maximum(0, tmin)
        overlap_end = cupy.minimum(lengthA, tmax)
        overlap_len = cupy.maximum(0, overlap_end - overlap_start)
        
        # fraction of A that’s overlapped by B
        overlapAinB = cupy.where(lengthA > 0, overlap_len / lengthA, 0.0)
        
        # Now do the same from B’s perspective
        dxB, dyB = end_xB - start_xB, end_yB - start_yB
        dir_lengthB = cupy.sqrt(dxB**2 + dyB**2)
        
        validB = dir_lengthB >= 1e-12
        
        ndxB = cupy.where(validB, dxB / dir_lengthB, 0.0)
        ndyB = cupy.where(validB, dyB / dir_lengthB, 0.0)
        
        # Project A’s endpoints onto B
        tA1 = ((start_xA - start_xB) * ndxB + (start_yA - start_yB) * ndyB)
        tA2 = ((end_xA - start_xB) * ndxB + (end_yA - start_yB) * ndyB)
        smin, smax = cupy.minimum(tA1, tA2), cupy.maximum(tA1, tA2)
        
        overlap_start_B = cupy.maximum(0, smin)
        overlap_end_B = cupy.minimum(lengthB, smax)
        overlap_len_B = cupy.maximum(0, overlap_end_B - overlap_start_B)

        overlapBinA = cupy.where(lengthB > 0, overlap_len_B / lengthB, 0.0)
        
        return overlapAinB, overlapBinA
    
    @staticmethod
    def create_obb(start_x, start_y, end_x, end_y, width):
        # Compute line directions and lengths
        dx = end_x - start_x
        dy = end_y - start_y
        length = cupy.sqrt(dx**2 + dy**2)

        # Unit direction vectors
        ux = dx / length
        uy = dy / length

        # Perpendicular vectors (unit)
        perp_x = -uy
        perp_y = ux

        # Half dimensions
        half_length = (length / 2) + width
        half_width = width / 2

        # Midpoints of the lines
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        # Corners (4 per OBB)
        corner1_x = mid_x - ux * half_length - perp_x * half_width
        corner1_y = mid_y - uy * half_length - perp_y * half_width

        corner2_x = mid_x + ux * half_length - perp_x * half_width
        corner2_y = mid_y + uy * half_length - perp_y * half_width

        corner3_x = mid_x + ux * half_length + perp_x * half_width
        corner3_y = mid_y + uy * half_length + perp_y * half_width

        corner4_x = mid_x - ux * half_length + perp_x * half_width
        corner4_y = mid_y - uy * half_length + perp_y * half_width

        # Stack corners: shape (n_lines, 4, 2)
        obb = cupy.stack([
            cupy.stack([corner1_x, corner1_y], axis=1),
            cupy.stack([corner2_x, corner2_y], axis=1),
            cupy.stack([corner3_x, corner3_y], axis=1),
            cupy.stack([corner4_x, corner4_y], axis=1)
        ], axis=1)

        return obb
    
    @staticmethod
    def hilbert_sort(mid_x, mid_y, resolution=1024, bits=10):
        # Normalize to [0, resolution)
        x = ((mid_x - mid_x.min()) / (mid_x.max() - mid_x.min()) * (resolution - 1)).astype(cupy.uint32)
        y = ((mid_y - mid_y.min()) / (mid_y.max() - mid_y.min()) * (resolution - 1)).astype(cupy.uint32)

        # Compute Hilbert indices using bitwise math
        n = x.shape[0]
        indices = cupy.zeros(n, dtype=cupy.uint32)

        for i in range(bits - 1, -1, -1):
            xi = (x >> i) & 1
            yi = (y >> i) & 1
            indices |= ((3 * xi) ^ yi) << (2 * i)
        
        # Argsort to reorder lines
        indices = cupy.argsort(indices)
        
        return indices
    
    @staticmethod
    def create_bins(hilbert_indices, num_bins):
        total = hilbert_indices.shape[0]
        bin_size = total // num_bins
        
        # Compute the bin indices for each element
        bin_indices = cupy.arange(total) // bin_size
        bin_indices = cupy.minimum(bin_indices, num_bins - 1)
        
        # Sort indices by bin id
        sort_order = cupy.lexsort(cupy.stack([bin_indices, hilbert_indices]))
        sorted_indices = hilbert_indices[sort_order]
        sorted_bins = bin_indices[sort_order]
        
        # Count how many elements per bin
        bin_counts = cupy.bincount(sorted_bins, minlength=num_bins)
        max_bin_size = bin_counts.max().item()
        
        # Prepare output matrix with -1 padding
        bins_matrix = cupy.full((num_bins, max_bin_size), -1, dtype=hilbert_indices.dtype)
        
        # Compute flattened row and column indices for scatter assignment
        bin_repeat = cupy.repeat(cupy.arange(num_bins), bin_counts.tolist())
        bin_positions = cupy.concatenate([cupy.arange(count) for count in bin_counts.tolist()])
        bins_matrix[bin_repeat, bin_positions] = sorted_indices
        
        return bins_matrix, bin_counts
    
    @staticmethod
    def _get_axes(corners):
        print("[DEBUG] _get_axes corners shape:", corners.shape)
        # Returns the two edge directions as axes to test (unit vectors)
        edge1 = corners[:, 1] - corners[:, 0]
        edge2 = corners[:, 3] - corners[:, 0]
        print("[DEBUG] edge1 shape:", edge1.shape)
        print("[DEBUG] edge2 shape:", edge2.shape)
        
        axes = cupy.stack([edge1, edge2], axis=1)
        print("[DEBUG] stacked axes shape:", axes.shape)
        lengths = cupy.linalg.norm(axes, axis=2, keepdims=True)
        print("[DEBUG] lengths shape:", lengths.shape)
        
        return axes / lengths
    
    @staticmethod
    def _project(corners, axes):
        print("[DEBUG] _project corners shape:", corners.shape)
        print("[DEBUG] _project axes shape:", axes.shape)
        projections = cupy.einsum('nij,nkj->nik', corners, axes)
        print("[DEBUG] _project projections shape:", projections.shape)
        return projections
    
    @staticmethod
    def check_overlap_sat(obbs_a, obbs_b):
        print("[DEBUG] Entering check_overlap_sat")
        print("[DEBUG] obbs_a shape:", obbs_a.shape)
        print("[DEBUG] obbs_b shape:", obbs_b.shape)
         
        Na, Nb = obbs_a.shape[0], obbs_b.shape[0]
        
        # Compute AABBs (Axis-Aligned Bounding Boxes)
        min_x_a = obbs_a[..., 0].min(axis=1)
        max_x_a = obbs_a[..., 0].max(axis=1)
        min_y_a = obbs_a[..., 1].min(axis=1)
        max_y_a = obbs_a[..., 1].max(axis=1)

        min_x_b = obbs_b[..., 0].min(axis=1)
        max_x_b = obbs_b[..., 0].max(axis=1)
        min_y_b = obbs_b[..., 1].min(axis=1)
        max_y_b = obbs_b[..., 1].max(axis=1)
        
        # Computing AABB intersection matrix
        print("[DEBUG] Computing AABB intersection matrix")
        aabb_x = (min_x_a[:, None] <= max_x_b[None, :]) & (max_x_a[:, None] >= min_x_b[None, :])
        aabb_y = (min_y_a[:, None] <= max_y_b[None, :]) & (max_y_a[:, None] >= min_y_b[None, :])
        aabb_overlap = aabb_x & aabb_y
        print("[DEBUG] AABB_overlap shape:", aabb_overlap.shape)
        pairs = cupy.argwhere(aabb_overlap)
        
        print("[DEBUG] aabb_overlap type:", type(aabb_overlap))
        print("[DEBUG] aabb_overlap shape:", aabb_overlap.shape if isinstance(aabb_overlap, cupy.ndarray) else "invalid")
        
        if pairs.shape[0] == 0: cupy.zeros((Na, Nb), dtype=bool)
        
        i, j = pairs[:, 0], pairs[:, 1]
        
        pairs_a, pairs_b = obbs_a[i], obbs_b[j]
        
        print("[DEBUG] pairs_a shape:", pairs_a.shape)
        print("[DEBUG] pairs_b shape:", pairs_b.shape)
        
        # axes per box
        print("[DEBUG] Computing axes...")
        axes_a = Graph._get_axes(pairs_a)
        axes_b = Graph._get_axes(pairs_b)
        axes = cupy.concatenate([axes_a, axes_b], axis=1)
        print("[DEBUG] axes shape:", axes.shape)
        
        # project & SAT
        print("[DEBUG] Projecting...")
        projections_a = Graph._project(pairs_a, axes)
        projections_b = Graph._project(pairs_b, axes)
        print("[DEBUG] projections_a shape:", projections_a.shape)
        print("[DEBUG] projections_b shape:", projections_b.shape)
        
        if projections_a.shape != projections_b.shape:
            print("[DEBUG] Shape mismatch detected, dumping pair indices and exiting...")
            print("[DEBUG] i shape:", i.shape)
            print("[DEBUG] j shape:", j.shape)
            print("[DEBUG] axes shape:", axes.shape)
            print("[DEBUG] projections_a:", projections_a)
            print("[DEBUG] projections_b:", projections_b)
            raise ValueError(f"Shape mismatch: projections_a {projections_a.shape}, projections_b {projections_b.shape}")
        
        max_a = projections_a.max(2)
        min_b = projections_b.min(2)
        max_b = projections_b.max(2)
        min_a = projections_a.min(2)

        print("[DEBUG] max_a shape:", max_a.shape)
        print("[DEBUG] min_b shape:", min_b.shape)
        print("[DEBUG] max_b shape:", max_b.shape)
        print("[DEBUG] min_a shape:", min_a.shape)
        
        separating_axis = (projections_a.max(2) < projections_b.min(2)) |\
                          (projections_b.max(2) < projections_a.min(2))
        
        overlap = cupy.zeros((Na, Nb), dtype=bool)
        overlap[i, j] = ~cupy.any(separating_axis, axis=1)
        
        print("[DEBUG] overlap shape:", overlap.shape)
        print("[DEBUG] Exiting check_overlap_sat")
        
        return overlap
         
    @staticmethod
    def _cross(vector_a, vector_b):
        return vector_a[:, 0] * vector_b[:, 1] - vector_a[:, 1] * vector_b[:, 0]
    
    @staticmethod
    def _point_to_segment_closest(p, a, b):
        """Returns the closest point on segment ab to point p."""
        ap = p - a
        ab = b - a
        ab_length_squared = cupy.sum(ab ** 2, axis=1)
        
        # fractional position of the projection along segment ab
        t = cupy.sum(ap * ab, axis=1) / ab_length_squared
        t = cupy.clip(t, 0, 1).reshape(-1, 1)
        
        closest = a + t * ab
        
        return closest
    
    @staticmethod
    def is_point_intersection_CPU(intersection, endpoints, threshold):
        ix, iy = intersection
        for ex, ey in endpoints:
            distance = math.dist((ix, iy), (ex, ey))  # Euclidean distance
            if distance <= threshold:
                return True
        return False
    
    @staticmethod
    def is_point_intersection_GPU(p1, p2, q1, q2, threshold):
        
        # Compute closest points for both p1 and p2 to segment q1–q2
        closest_1 = Graph._point_to_segment_closest(p1, q1, q2)
        closest_2 = Graph._point_to_segment_closest(p2, q1, q2)
        
        distance_1 = cupy.sum((p1 - closest_1) ** 2, axis=1)
        distance_2 = cupy.sum((p2 - closest_2) ** 2, axis=1)
        
        hits = (distance_1 <= threshold**2) | (distance_2 <= threshold**2)
        
        return hits
        
    def ParallelDetection(self, max_threshold=50, colinear_threshold=0.5,
                          min_overlap_ratio=0.2, angle_tolerance=cupy.radians(0.01)):
        
        dataframe = self._dataframe.copy()
        
        # Filter valid line indices
        is_line = (dataframe['circle'] == 0) & (dataframe['arc'] == 0)
        lines = dataframe[is_line].reset_index(drop=True)
        lines['angle'] = lines['angle'] % cupy.pi  # collapse symmetrical directions
        
        # bin settings
        bin_factor = 10**5
        bin_offset_size = 25
        bin_angle_size = cupy.radians(0.25)
        
        # Extract coordinates
        start_x = lines['start_x'].to_cupy()
        start_y = lines['start_y'].to_cupy()
        end_x = lines['end_x'].to_cupy()
        end_y = lines['end_y'].to_cupy()
        length = lines['length'].to_cupy()
        angle = lines['angle'].to_cupy()
        offset = lines['offset'].to_cupy()
        max_length = dataframe['length'].to_cupy().max()
        
        angle_bin = cupy.floor(angle / bin_angle_size).astype(cupy.int32)
        offset_bin = cupy.floor(offset / bin_offset_size).astype(cupy.int32)
        element_keys = angle_bin * bin_factor + offset_bin
        bin_keys = cupy.unique(element_keys)
        
        for key in bin_keys:
            angle_key = key // bin_factor
            offset_key = key % bin_factor
            
            neighbor_keys = [(angle_key + da) * bin_factor + (offset_key + do)
                             for da in (-1, 0, 1) for do in (-1, 0, 1)]
            
            # Filter all elements in the drawing that fall in the current bin or in neighboring bins
            mask = cupy.isin(element_keys, cupy.array(neighbor_keys, dtype=cupy.int32))
            element_indices = cupy.where(mask)[0]
            
            if element_indices.shape[0] < 2: continue
            
            # Get coordinates
            subset_start_x = start_x[element_indices]
            subset_start_y = start_y[element_indices]
            subset_end_x = end_x[element_indices]
            subset_end_y = end_y[element_indices]
            subset_length = length[element_indices]
            subset_angle = angle[element_indices]
            subset_offset = offset[element_indices]
            
            coordinates = cupy.stack([subset_angle, subset_offset], axis=1)
            
            difference_matrix = coordinates[:, None, :] - coordinates[None, :, :]
            distance_squared = cupy.sum(difference_matrix**2, axis=-1)
            pairs = (distance_squared <= max_threshold**2) & cupy.triu(cupy.ones_like(distance_squared, dtype=bool), k=1)
            i, j = cupy.where(pairs)
            
            if i.size == 0: continue
            
            # Retrieve true angles from dataframe before mod % π
            original_angle = dataframe['angle'].to_cupy()[element_indices]
            angle_i = original_angle[i]
            angle_j = original_angle[j]
            angle_difference = cupy.abs(angle_i - angle_j)
            angle_difference_min = cupy.minimum(angle_difference, 2 * cupy.pi - angle_difference)
            valid = angle_difference_min < angle_tolerance
            
            if not valid.any(): continue
            
            i, j = i[valid], j[valid]
            
            xA1 = subset_start_x[i]
            yA1 = subset_start_y[i]
            xA2 = subset_end_x[i]
            yA2 = subset_end_y[i]
            lenA = subset_length[i]
            
            xB1 = subset_start_x[j]
            yB1 = subset_start_y[j]
            xB2 = subset_end_x[j]
            yB2 = subset_end_y[j]
            lenB = subset_length[j]
            
            overlapA, overlapB = self.overlap_ratios(xA1, yA1, xA2, yA2, lenA, xB1, yB1, xB2, yB2, lenB)
            valid = (overlapA > min_overlap_ratio) | (overlapB > min_overlap_ratio)
            
            if not valid.any(): continue
            
            i, j = i[valid], j[valid]
            
            overlapA = overlapA[valid]
            overlapB = overlapB[valid]
            angle_i = subset_angle[i]
            offset_i = subset_offset[i]
            offset_j = subset_offset[j]
            
            angle_difference = angle_difference[i]
            angle_difference_min = angle_difference_min[i]
            
            distance = cupy.abs(offset_j - offset_i) / cupy.cos(angle_i)
            edges_ij = cupy.stack([element_indices[i], element_indices[j]], axis=0)
            edges_ji = cupy.stack([element_indices[j], element_indices[i]], axis=0)
            edges = cupy.concatenate([edges_ij, edges_ji], axis=1)
            
            attributes = cupy.zeros((i.shape[0], len(self._attributes)), dtype=cupy.float32)
            attributes[:, self.parallel] = 1.0
            attributes[:, self.colinear] = (distance < colinear_threshold).astype(cupy.float32)
            attributes[:, self.perpendicular_distance] = distance / max_length
            attributes[:, self.angle_difference] = angle_difference_min / (cupy.pi / 2)
            attributes[:, self.angle_difference_sin] = cupy.sin(angle_difference)
            attributes[:, self.angle_difference_cos] = cupy.cos(angle_difference)
            
            attributes_ij = attributes.copy()
            attributes_ij[:, self.overlap_ratio] = overlapA
            
            attributes_ji = attributes.copy()
            attributes_ji[:, self.overlap_ratio] = overlapB
            
            attributes = cupy.concatenate([attributes_ij, attributes_ji], axis=0)
            
            start = self.size
            end = start + edges.shape[1]
            self.edges[:, start:end] = edges
            self.edgeAttributes[start:end, :] = attributes
            self.size = end
                        
    def IntersectionDetection(self, obb_width=0.5, co_linear_tolerance=cupy.radians(0.01)):
        print("[DEBUG] Starting IntersectionDetection")
        
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        def print_gpu_mem(label):
            info = nvmlDeviceGetMemoryInfo(handle)
            print(f"[DEBUG GPU] {label}: used = {info.used // 1024**2} MB, free = {info.free // 1024**2} MB")

        print_gpu_mem("At start")

        dataframe = self._dataframe.copy()
        
        # bin settings
        min_num_bins = 1
        lines_per_bin = 150
        depth_percentage = 1.0
        min_neighbor_deph = 10
        
        # Step 1: Filter valid line indices
        is_line = (dataframe['circle'] == 0) & (dataframe['arc'] == 0)
        lines = dataframe[is_line].reset_index(drop=True)
        circular_elements = dataframe[~is_line].reset_index(drop=True)
        
        if len(lines) == 0: return
        
        # Extract coordinates
        start_x = lines['start_x'].to_cupy()
        start_y = lines['start_y'].to_cupy()
        end_x = lines['end_x'].to_cupy()
        end_y = lines['end_y'].to_cupy()
        angle = lines['angle'].to_cupy()
        
        # Compute line midpoints and OBBs
        obbs = self.create_obb(start_x, start_y, end_x, end_y, width=obb_width)
        print_gpu_mem("After OBB creation")
        print_gpu_mem("Before OBB mean")
        obb_centroids = obbs.mean(axis=1)
        print("[DEBUG] obb_centroids shape:", obb_centroids.shape)
        mid_x = obb_centroids[:, 0]
        mid_y = obb_centroids[:, 1]
        print("[DEBUG] mid_x shape:", mid_x.shape)
        
        # Hilbert sort and dynamic binning
        print_gpu_mem("Before hilbert_sort")
        hilbert_order = self.hilbert_sort(mid_x, mid_y)
        print("[DEBUG] hilbert_order shape:", hilbert_order.shape)
        num_bins = max(min_num_bins, int(len(lines) / lines_per_bin))
        print("[DEBUG] num_bins:", num_bins)
        print_gpu_mem("Before create_bins")
        bins_matrix, bin_counts = self.create_bins(hilbert_order, num_bins=num_bins)
        print("[DEBUG] bins_matrix shape:", bins_matrix.shape)
        
        # Define neighbor depth based on percentage of total bins
        neighbor_depth = max(min_neighbor_deph, int(num_bins * depth_percentage))
        print("[DEBUG] neighbor_depth:", neighbor_depth)
        
        # Step 2: Check pairwise overlaps within bins and neighbor bins (forward only)
        for bin_i in range(num_bins):
            for offset in range(neighbor_depth + 1):
                bin_j = bin_i + offset
                if bin_j >= num_bins: continue
                
                print(f"[DEBUG] Checking bin pair: ({bin_i}, {bin_j})")
                
                indices_i = bins_matrix[bin_i]
                indices_j = bins_matrix[bin_j]

                indices_i = indices_i[indices_i != -1]
                indices_j = indices_j[indices_j != -1]
                
                print(f"[DEBUG] bin_i size: {indices_i.size}, bin_j size: {indices_j.size}")
                print_gpu_mem("Before meshgrid")
                
                if indices_i.shape[0] == 0 or indices_j.shape[0] == 0: continue
                
                indices_i, indices_j = cupy.meshgrid(indices_i, indices_j, indexing='ij')
                print_gpu_mem("After meshgrid")
                indices_i = indices_i.flatten()
                indices_j = indices_j.flatten()
                
                # Skip self-pairs and enforce order to avoid duplicates
                valid = indices_i < indices_j
                indices_i = indices_i[valid]
                indices_j = indices_j[valid]
                
                print(f"[DEBUG] Valid pairs after flattening: {indices_i.size}")
                print_gpu_mem("Before obb slicing")
                
                if indices_i.shape[0] == 0: continue
                
                # Extract OBBs and coordinates
                obbs_i = obbs[indices_i]
                obbs_j = obbs[indices_j]
                
                print("[DEBUG] obbs_i shape:", obbs_i.shape)
                print("[DEBUG] obbs_j shape:", obbs_j.shape)
                print_gpu_mem("Before check_overlap_sat")
                
                # SAT check
                overlap = self.check_overlap_sat(obbs_i, obbs_j)
                print_gpu_mem("After check_overlap_sat")
                overlap = cupy.argwhere(overlap)
                indices_i = indices_i[overlap[:, 0]]
                indices_j = indices_j[overlap[:, 1]]
                
                if indices_i.shape[0] == 0: continue
                
                # Angle difference
                angle_i = angle[indices_i]
                angle_j = angle[indices_j]
                angle_difference = (angle_j - angle_i) % (2 * cupy.pi)
                angle_difference_min = cupy.minimum(angle_difference, 2 * cupy.pi - angle_difference)
                
                # Filter colinear lines based on angle tolerance
                colinear = (angle_difference_min % cupy.pi) >= co_linear_tolerance
                indices_i = indices_i[colinear]
                indices_j = indices_j[colinear]
                angle_difference = angle_difference[colinear]
                angle_difference_min = angle_difference_min[colinear]
                
                print_gpu_mem("Before segment intersection")
                
                # Segment intersection
                p1 = cupy.stack([start_x[indices_i], start_y[indices_i]], axis=1)
                p2 = cupy.stack([end_x[indices_i], end_y[indices_i]], axis=1)
                q1 = cupy.stack([start_x[indices_j], start_y[indices_j]], axis=1)
                q2 = cupy.stack([end_x[indices_j], end_y[indices_j]], axis=1)
                
                print("[DEBUG] p1 shape:", p1.shape)
                print("[DEBUG] q1 shape:", q1.shape)
                
                print("[DEBUG] Starting is_point_intersection_ij")
                is_point_intersection_ij = self.is_point_intersection_GPU(p1, p2, q1, q2, threshold=obb_width)
                print("[DEBUG] Finished is_point_intersection_ij")
                print("[DEBUG] Starting is_point_intersection_ji")
                is_point_intersection_ji = self.is_point_intersection_GPU(q1, q2, p1, p2, threshold=obb_width)
                print("[DEBUG] Finished is_point_intersection_ji")
                is_segment_intersection_ij = ~is_point_intersection_ij
                is_segment_intersection_ji = ~is_point_intersection_ji
                
                i_nodes = cupy.array(indices_i, dtype=cupy.int32)
                j_nodes = cupy.array(indices_j, dtype=cupy.int32)
                edges_ij = cupy.stack([i_nodes, j_nodes], axis=0)
                edges_ji = cupy.stack([j_nodes, i_nodes], axis=0)
                edges = cupy.concatenate([edges_ij, edges_ji], axis=1)
                num_edges = i_nodes.shape[0]
                
                print_gpu_mem("Before attribute allocation")
                
                attributes = cupy.zeros((num_edges, len(self._attributes)), dtype=cupy.float32)
                attributes[:, self.angle_difference] = angle_difference_min / (cupy.pi / 2)
                attributes[:, self.angle_difference_sin] = cupy.sin(angle_difference)
                attributes[:, self.angle_difference_cos] = cupy.cos(angle_difference)
                
                attributes_ij = attributes.copy()
                attributes_ij[:, self.point_intersection] = is_point_intersection_ij.astype(cupy.float32)
                attributes_ij[:, self.segment_intersection] = is_segment_intersection_ij.astype(cupy.float32)
                
                attributes_ji = attributes.copy()
                attributes_ji[:, self.point_intersection] = is_point_intersection_ji.astype(cupy.float32)
                attributes_ji[:, self.segment_intersection] = is_segment_intersection_ji.astype(cupy.float32)
                
                attributes = cupy.concatenate([attributes_ij, attributes_ji], axis=0)
                
                print("[DEBUG] Writing edges and attributes")
                print(f"[DEBUG] self.size: {self.size}")
                print(f"[DEBUG] edges.shape: {edges.shape}")
                print(f"[DEBUG] attributes.shape: {attributes.shape}")
                print(f"[DEBUG] self.edges shape: {self.edges.shape}")
                print(f"[DEBUG] self.edgeAttributes shape: {self.edgeAttributes.shape}")
                
                start = self.size
                end = start + edges.shape[1]
                self.edges[:, start:end] = edges
                self.edgeAttributes[start:end, :] = attributes
                self.size = end
                print_gpu_mem("After writing edges and attributes")
        
        if len(circular_elements) == 0: return
        
        dataframe = dataframe.to_pandas()
        lines = lines.to_pandas()
        circular_elements = circular_elements.to_pandas()

        geometryLines: List[Tuple[int, LineString]] = []
        for i, row in lines.iterrows():
            line = LineString([(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])])
            geometryLines.append((i, line))

        geometryCircular: List[Tuple[int, LineString]] = []
        for i, row in circular_elements.iterrows():
            center = Point(row["start_x"], row["start_y"])
            radius = row["radius"]
            if row["circle"]:
                circle = center.buffer(radius).boundary
                geometryCircular.append((i, circle))
            else:
                startAngle = row["start_angle"]
                endAngle = row["end_angle"]
                arcPoints = self.create_arc(center, radius, startAngle, endAngle)
                arc = LineString(arcPoints)
                geometryCircular.append((i, arc))
                
        # Build STRtree for lines
        space2D = STRtree([geometry for _, geometry in geometryLines])

        edges: List[Tuple[int, int]] = []
        attributes: List[List[float]] = []
        # Circle-line perimeter intersections
        for i, circularElement in geometryCircular:
            nearbyLines = space2D.query(circularElement.buffer(threshold:=obb_width))
            
            for k in nearbyLines:
                j, line = geometryLines[k]
                
                # Check actual intersection
                intersection = circularElement.intersection(line)
                if intersection.is_empty:
                    if circularElement.distance(line) < threshold:
                        pA, pB = nearest_points(circularElement, line)
                        intersection_x, intersection_y = (pA.x + pB.x) / 2, (pA.y + pB.y) / 2
                        intersection = Point(intersection_x, intersection_y)
                    else: continue
                
                # Ensure intersection is always stored as a point
                if intersection.geom_type == "Point": intersection = (intersection.x, intersection.y)
                elif intersection.geom_type == "MultiPoint":intersection = list(intersection.geoms)[0].coords[0]
                else: intersection = intersection.interpolate(0.5, normalized=True).coords[0]
                
                row = dataframe.iloc[j]
                rowPoints = [(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])]
                isPointIntersection = self.is_point_intersection_CPU(intersection, rowPoints, threshold)
                
                edges.append((i, j))
                attributes_ij = [0.0] * len(self._attributes)
                attributes_ij[self.perimeter_intersection] = 1
                attributes.append(attributes_ij)
                
                edges.append((j, i))
                attributes_ji = [0.0] * len(self._attributes)
                attributes_ji[self.point_intersection] = 1 if isPointIntersection else 0
                attributes_ji[self.segment_intersection] = 0 if isPointIntersection else 1
                attributes.append(attributes_ji)
        
        edges = cupy.asarray(edges).T
        attributes = cupy.asarray(attributes)
        
        start = self.size
        end = start + edges.shape[1]
        self.edges[:, start:end] = edges
        self.edgeAttributes[start:end, :] = attributes
        self.size = end
            
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

def LaplacianEigenvectors(graph, k=2):
    try:
        # Step 1: Create scipy sparse Laplacian matrix
        L = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes).astype(float)
        L = csgraph.laplacian(L, normed=True)
        
        # Step 2: Compute the first k+1 eigenvectors (skip the trivial 0-th)
        eigenvals, eigenvecs = eigsh(L, k=k + 1, which='SM')
        eigenvecs = eigenvecs[:, 1:]  # Skip first trivial one
        
    except ArpackNoConvergence as e:
        print("[Warning] ARPACK did not fully converge. Using partial results.")
        if e.eigenvectors is not None and e.eigenvectors.shape[1] > k: eigenvecs = e.eigenvectors[:, 1:k + 1]
        else: eigenvecs = numpy.zeros((graph.num_nodes, k))
    
    except Exception as e:
        print(f"[Error] Laplacian eigenvector computation failed: {e}")
        eigenvecs = igenvecs = numpy.zeros((graph.num_nodes, k))
    
    # Step 3: Convert to torch and concatenate with node features
    eigenvecs = torch.from_numpy(eigenvecs).float()
    graph.x = torch.cat([graph.x, eigenvecs], dim=1)
    
    return graph

import ezdxf
import torch
import numpy
from torch_geometric.data import Data

def CreateGraph(dxf_file):
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    
    entities = [entity for entity in modelSpace if entity.dxftype() in ('LINE', 'CIRCLE', 'ARC')]
    
    if not entities: return
    
    count = len(entities)
    
    start_x = numpy.zeros(count, dtype=numpy.float32)
    start_y = numpy.zeros(count, dtype=numpy.float32)
    end_x = numpy.zeros(count, dtype=numpy.float32)
    end_y = numpy.zeros(count, dtype=numpy.float32)
    length = numpy.zeros(count, dtype=numpy.float32)
    angle = numpy.zeros(count, dtype=numpy.float32)
    layer = numpy.empty(count, dtype=object)
    circle_flag = numpy.zeros(count, dtype=numpy.int8)
    arc_flag = numpy.zeros(count, dtype=numpy.int8)
    radius = numpy.zeros(count, dtype=numpy.float32)
    start_angle = numpy.zeros(count, dtype=numpy.float32)
    end_angle = numpy.zeros(count, dtype=numpy.float32)
    
    for i, entity in enumerate(entities):
        entity_type = entity.dxftype()
        layer[i] = entity.dxf.layer
        
        if entity_type == 'LINE':
            sx, sy, _ = entity.dxf.start
            ex, ey, _ = entity.dxf.end
            dx, dy = ex - sx, ey - sy
            
            start_x[i], start_y[i] = sx, sy
            end_x[i], end_y[i] = ex, ey
            length[i] = numpy.hypot(dx, dy)
            angle[i] = numpy.arctan2(dy, dx) % (2*numpy.pi)
        
        elif entity_type == 'CIRCLE':
            cx, cy, _ = entity.dxf.center
            r = entity.dxf.radius
            l = 2 * numpy.pi * r
            
            circle_flag[i] = 1
            start_x[i], start_y[i] = cx, cy
            length[i] = l
            radius[i] = r
        
        elif entity_type == 'ARC':
            cx, cy, _ = entity.dxf.center
            r = entity.dxf.radius
            sa = entity.dxf.start_angle
            ea = entity.dxf.end_angle
            l = r * numpy.radians((ea - sa) % 360)
            
            arc_flag[i] = 1
            start_x[i], start_y[i] = cx, cy
            length[i] = l
            radius[i] = r
            start_angle[i] = sa
            end_angle[i] = ea
    
    dataframe = cudf.DataFrame(
        {
            "start_x": cupy.asarray(start_x),
            "start_y": cupy.asarray(start_y),
            "end_x": cupy.asarray(end_x),
            "end_y": cupy.asarray(end_y),
            "length": cupy.asarray(length),
            "angle": cupy.asarray(angle),
            "layer": layer,
            "circle": cupy.asarray(circle_flag),
            "arc": cupy.asarray(arc_flag),
            "radius": cupy.asarray(radius),
            "start_angle": cupy.asarray(start_angle),
            "end_angle": cupy.asarray(end_angle)
        }
    )
        
    # Define anchor point
    anchor_x = cupy.float32(dataframe[['start_x', 'end_x']].to_cupy().min().item() - 100000)
    anchor_y = cupy.float32(dataframe[['start_y', 'end_y']].to_cupy().min().item() - 100000)
     
    # Compute offset
    start_x, start_y, end_x, end_y = dataframe['start_x'], dataframe['start_y'], dataframe['end_x'], dataframe['end_y']
    
    # Set a safe length (to avoid division by zero)
    mask = (dataframe["circle"] == 0) & (dataframe["arc"] == 0) & (dataframe["length"] > 1e-12)
    length = dataframe["length"].where(mask, 1.0)
    
    # Perpendicular offset (same formula, but mask invalid ones to zero)
    offset = cupy.abs((start_x - anchor_x) * (-(end_y - start_y) / length) +
                      (start_y - anchor_y) * ((end_x - start_x) / length))
    
    # Apply masking: invalid offsets get zero
    offset = cupy.where(mask.to_cupy(), offset, 0.0)
    
    # Assign back
    dataframe["offset"] = offset
        
    graph = Graph(dataframe)
    graph.ParallelDetection()
    graph.IntersectionDetection()
    
    # Truncate after edge collection
    graph.edges = graph.edges[:, :graph._size]
    graph.edgeAttributes = graph.edgeAttributes[:graph._size, :]
    
    graph.classificationLabels = torch.tensor(graph.classificationLabels, dtype=torch.long)
    graph.nodeAttributes = torch.tensor(graph.nodeAttributes, dtype=torch.float)
    graph.edges = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
    graph.edgeAttributes = torch.tensor(graph.edgeAttributes, dtype=torch.float)
    
    graph = Data(x=graph.nodeAttributes, edge_index=graph.edges, edge_attr=graph.edgeAttributes, y=graph.classificationLabels)
    
    #graph: Data = LaplacianEigenvectors(graph)
    return graph

import os
from tqdm import tqdm
from multiprocessing import Pool

class CreateGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dxf_files, chunksize=4):
        self.graphs = []
        with Pool(processes=os.cpu_count()) as pool:
            for graph in tqdm(pool.imap_unordered(CreateGraph, dxf_files, chunksize=chunksize), total=len(dxf_files), desc="Creating graphs"):
                self.graphs.append(graph)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphGPSNetwork(nn.Module):
    def __init__(self, edge_attributes=8, transformer_heads=4, transformer_layers=2):
        super().__init__()
        
        # Define input and embedding dimensions
        coordinates_dimensions = 9  # (start_x, start_y, end_x, end_y, sin(angle), cos(angle), length, circle, arc)
        embedding_dimensions = 64  # We map 4D coordinates into a 32D embedding
        
        # Small MLP to learn embeddings for coordinate features
        self.embedding_layer = nn.Sequential(nn.Linear(coordinates_dimensions, 32),
                                             nn.ReLU(),
                                             nn.LayerNorm(32),
                                             nn.Linear(32, embedding_dimensions),
                                             nn.ReLU(),
                                             nn.LayerNorm(embedding_dimensions))
        
        # First NNConv layer: Transforms node features from embedding_dimensions → 128
        self.conv1 = NNConv(in_channels=embedding_dimensions, out_channels=128,
            nn=nn.Sequential(nn.Linear(edge_attributes, 32),
                             nn.ReLU(),
                             nn.Linear(32, embedding_dimensions * 128)
            )
        )
        self.norm1 = nn.LayerNorm(128)  # Add normalization after conv1
        
        # Second NNConv layer: Reduces node features from 128 → 32
        self.conv2 = NNConv(in_channels=128, out_channels=32,
            nn=nn.Sequential(nn.Linear(edge_attributes, 32),
                             nn.ReLU(),
                             nn.Linear(32, 128 * 32)
            )
        )
        self.norm2 = nn.LayerNorm(32)  # Add normalization after conv2
        
        # -------------------- Transformer Encoder -------------------- #
        # The Transformer Encoder helps in learning global dependencies
        encoder_layer = TransformerEncoderLayer(
            d_model=32,  # Must match NNConv output dimension
            nhead=transformer_heads,
            dim_feedforward=64,  # Can be tuned
            dropout=0.2,
            batch_first=True)
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Fully connected layers for classification
        self.fully_connected = nn.Linear(32, 32)
        self.out = nn.Linear(32, 7) # classes: beam, column, eave, slab_hole, stair, section, info
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Pass (start_x, start_y, end_x, end_y) coordinates through embedding MLP
        x = self.embedding_layer(x)  # Output shape: [num_nodes, 16]
        
        # Apply first NNConv: Expands node features from 16 → 32
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.norm1(x)  # Apply LayerNorm
        
        # Apply second NNConv: Reduces node features from 32 → 16
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.norm2(x)  # Apply LayerNorm
        
        # -------------------- Apply Transformer -------------------- #
        x = x.unsqueeze(0)  # Transformer expects input of shape [batch, num_nodes, features]
        x = self.transformer(x)  # Learn global dependencies
        x = x.squeeze(0)  # Back to shape [num_nodes, features]
        
        # Fully connected layers for classification
        x = F.relu(self.fully_connected(x))
        
        # Apply dropout before final layer
        x = F.dropout(x, p=0.4, training=self.training) # Dropout to prevent overfitting
        
        # Raw logits
        output = self.out(x)
        
        return output

from collections import Counter

def BalanceClassWeights(dataset, device="cpu", smoothing_factor=0.2, classification_labels=7):
    labels = [graph.y.tolist() for graph in dataset]
    labels_flattened = [label for sublist in labels for label in sublist]
    total_class = Counter(labels_flattened)
    total_samples = sum(total_class.values())
    
    class_weights = [(total_samples / (2 * total_class[i])) ** smoothing_factor for i in range(classification_labels)]
    
    return torch.tensor(class_weights, dtype=torch.float).to(device)

