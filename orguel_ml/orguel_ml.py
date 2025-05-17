from typing import List, Tuple
import math
import cupy
import cuspatial
from torch.utils.dlpack import from_dlpack
from cupyx.scipy.spatial import cKDTree
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

class Graph:
    _edge_attributes = cupy.arange(10, dtype=cupy.int32)
    
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
    
    def __init__(self, dataframe, edges=10**6):
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
            .map({"beam": 0, "column": 1, "eave": 2, "slab_hole": 3, "stair": 4, "section": 5, "info": 6})
            .to_cupy() if "layer" in dataframe.columns else None)
        
        # Node attributes
        self.nodeAttributes = from_dlpack(cupy.hstack([
            normalized_coordinates,
            normalized_angles,
            normalized_lengths,
            circle_flags,
            arc_flags
        ]).toDlpack())
        
        self._size = 0
        self.edges = cupy.empty((2, edge_capacity), dtype=cupy.int32)
        self.edgeAttributes = cupy.empty((edge_capacity, len(self._edge_attributes)), dtype=cupy.float32)
           
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
        sort_order = cupy.lexsort((hilbert_indices, bin_indices))
        sorted_indices = hilbert_indices[sort_order]
        sorted_bins = bin_indices[sort_order]
        
        # Count how many elements per bin
        bin_counts = cupy.bincount(sorted_bins, minlength=num_bins)
        max_bin_size = bin_counts.max()
        
        # Prepare output matrix with -1 padding
        bins_matrix = cupy.full((num_bins, max_bin_size), -1, dtype=hilbert_indices.dtype)
        
        # Compute flattened row and column indices for scatter assignment
        bin_repeat = cupy.repeat(cupy.arange(num_bins), bin_counts)
        bin_positions = cupy.concatenate([cupy.arange(count) for count in bin_counts.tolist()])
        bins_matrix[bin_repeat, bin_positions] = sorted_indices
        
        return bins_matrix, bin_counts
    
    @staticmethod
    def _get_axes(corners):
        # Returns the two edge directions as axes to test (unit vectors)
        edge1 = corners[:, 1] - corners[:, 0]
        edge2 = corners[:, 3] - corners[:, 0] 
        axes = cupy.stack([edge1, edge2], axis=1) 
        lengths = cupy.linalg.norm(axes, axis=2, keepdims=True)
        return axes / lengths
    
    @staticmethod
    def _project(corners, axes):
        return cupy.einsum('nij,nkj->nki', corners, axes)
    
    @staticmethod
    def check_overlap_sat(obbs_a, obbs_b):
        # Get the 4 corners of each box
        axes_a = Graph._get_axes(obbs_a)
        axes_b = Graph._get_axes(obbs_b)
        axes = cupy.concatenate([axes_a, axes_b], axis=1)
        
        projections_a = Graph._project(obbs_a, axes)
        projections_b = Graph._project(obbs_b, axes)
        
        proj_a_min = projections_a.min(axis=2)
        proj_a_max = projections_a.max(axis=2)
        proj_b_min = projections_b.min(axis=2)
        proj_b_max = projections_b.max(axis=2)
        
        separating_axes = (proj_a_max < proj_b_min) | (proj_b_max < proj_a_min)
        overlap = ~cupy.any(separating_axes, axis=1)
        
        return overlap  # Boolean mask for overlapping pairs
        
    @staticmethod
    def _cross(vector_a, vector_b):
        return vector_a[:, 0] * vector_b[:, 1] - vector_a[:, 1] * vector_b[:, 0]
    
    @staticmethod
    def intersection(p1, p2, q1, q2):
        " Return the intersection points "
        # Define vectors
        va = p2 - p1
        vb = q2 - q1
        qp = q1 - p1
        
        denominator = Graph._cross(va, vb)
        
        # Avoid division by zero: mask where denom != 0
        valid = denominator != 0
        ta = cupy.zeros_like(denominator)
        tb = cupy.zeros_like(denominator)
        intersections = cupy.full_like(p1, cupy.nan)
        
        ta[valid] = Graph._cross(qp[valid], vb[valid]) / denominator[valid]
        tb[valid] = Graph._cross(qp[valid], va[valid]) / denominator[valid]
        
        # Condition for segments intersecting
        condition = (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1) & valid
        
        # Compute intersection points only where condition is True
        intersections[condition] = p1[condition] + ta[condition].reshape(-1, 1) * va[condition]
        return intersections, condition
    
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
    def is_point_intersection(p1, p2, q1, q2, threshold):
        batch_size = p1.shape[0]
        
        # Check p1 and p2 against q1–q2
        points = cupy.stack([p1, p2], axis=1).reshape(-1, 2) 
        q1 = cupy.repeat(q1, 2, axis=0)
        q2 = cupy.repeat(q2, 2, axis=0)
        
        # Compute closest points on each segment
        closest = Graph._point_to_segment_closest(points, q1, q2)
        distance_squared = cupy.sum((points - closest) ** 2, axis=1)
        hits = distance_squared <= threshold**2
        
        # Reshape hits to (2, batch_size)
        hits = hits.reshape(2, batch_size)
        
        return hits.any(axis=0)
    
    @staticmethod
    def explode_circle_to_lines(circular_elements, segments=32):
        circular_elements = circular_elements.reset_index(drop=True)
        
        # Convert column data to CuPy arrays
        center_x = cupy.asarray(circular_elements['start_x'].to_cupy())
        center_y = cupy.asarray(circular_elements['start_y'].to_cupy())
        radius = cupy.asarray(circular_elements['radius'].to_cupy())
        is_circle = cupy.asarray(circular_elements['circle'].to_cupy())
        start_angle = cupy.where(is_circle, 0, cupy.asarray(circular_elements['start_angle'].to_cupy()))
        end_angle = cupy.where(is_circle, 360, cupy.asarray(circular_elements['end_angle'].to_cupy()))
        
        # Create angle segments (same for all elements)
        base_theta = cupy.linspace(0, 1, segments)
        theta = (end_angle - start_angle).reshape(-1, 1) * base_theta + start_angle.reshape(-1, 1)
        theta = cupy.radians(theta)
        
        # Compute coordinates for all segments
        x = cupy.cos(theta) * radius[:, None] + center_x[:, None]
        y = cupy.sin(theta) * radius[:, None] + center_y[:, None]
        
        x_start = x[:, :-1].reshape(-1)
        y_start = y[:, :-1].reshape(-1)
        x_end = x[:, 1:].reshape(-1)
        y_end = y[:, 1:].reshape(-1)

        dx = x_end - x_start
        dy = y_end - y_start
        length = cupy.sqrt(dx**2 + dy**2)
        angle = cupy.arctan2(dy, dx)
        
        parent_id = cupy.repeat(cupy.arange(len(circular_elements)), segments - 1)
        
        line_segments = cudf.DataFrame(
            {
                'parent_id':parent_id,
                'start_x': x_start,
                'start_y': y_start,
                'end_x': x_end,
                'end_y': y_end,
                'length': length,
                'angle': angle,
            }
        )
        
        return line_segments
        
    def ParallelDetection(self, max_threshold=50, colinear_threshold=0.5,
                          min_overlap_ratio=0.2, angle_tolerance=cupy.radians(0.01)):
        
        dataframe = self._dataframe.copy()
        dataframe = dataframe[(dataframe['circle'] == 0) & (dataframe['arc'] == 0)]
        dataframe['angle'] = dataframe['angle'] % cupy.pi  # collapse symmetrical directions
        
        # bin settings
        bin_factor = 10**5
        bin_offset_size = 25
        bin_angle_size = cupy.radians(0.25)
        
        # Extract coordinates
        start_x = dataframe['start_x'].to_cupy()
        start_y = dataframe['start_y'].to_cupy()
        end_x = dataframe['end_x'].to_cupy()
        end_y = dataframe['end_y'].to_cupy()
        length = dataframe['length'].to_cupy()
        angle = dataframe['angle'].to_cupy()
        offset = dataframe['offset'].to_cupy()
        max_length = self._dataframe['length'].to_cupy().max()
        
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
            
            angle_i = subset_angle[i]
            angle_j = subset_angle[j]
            angle_difference = cupy.abs(angle_i - angle_j)
            valid = angle_difference < angle_tolerance
            
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
            valid = cupy.minimum(overlapA, overlapB) > min_overlap_ratio
            
            if not valid.any(): continue
            
            i, j = i[valid], j[valid]
            
            overlapA = overlapA[valid]
            overlapB = overlapB[valid]
            angle_i = subset_angle[i]
            offset_i = subset_offset[i]
            offset_j = subset_offset[j]
            
            distance = cupy.abs(offset_j - offset_i) / cupy.cos(angle_i)
            
            edges_ij = cupy.stack([element_indices[i], element_indices[j]], axis=0)
            edges_ji = cupy.stack([element_indices[j], element_indices[i]], axis=0)
            edges = cupy.concatenate([edges_ij, edges_ji], axis=1)
            
            attributes = cupy.zeros((i.shape[0], len(self._edge_attributes)), dtype=cupy.float32)
            attributes[:, self.parallel] = 1.0
            attributes[:, self.colinear] = (distance < colinear_threshold).astype(cupy.float32)
            attributes[:, self.perpendicular_distance] = distance / max_length
            
            attributes_ij = attributes.copy()
            attributes_ij[:, self.overlap_ratio] = overlapA
            
            attributes_ji = attributes.copy()
            attributes_ji[:, self.overlap_ratio] = overlapB
            
            attributes = cupy.concatenate([attributes_ij, attributes_ji], axis=0)
            
            start = self._size
            end = start + edges.shape[1]
            self.edges[:, start:end] = edges
            self.edgeAttributes[start:end, :] = attributes
            self._size = end
                        
    def IntersectionDetection(self, obb_width=0.5, co_linear_tolerance=cupy.radians(0.01)):
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
        
        # Extract coordinates
        start_x = lines['start_x'].to_cupy()
        start_y = lines['start_y'].to_cupy()
        end_x = lines['end_x'].to_cupy()
        end_y = lines['end_y'].to_cupy()
        angle = lines['angle'].to_cupy()
        
        # Compute line midpoints and OBBs
        obbs = self.create_obb(start_x, start_y, end_x, end_y, width=obb_width)
        obb_centroids = obbs.mean(axis=1)
        mid_x = obb_centroids[:, 0]
        mid_y = obb_centroids[:, 1]
        
        # Hilbert sort and dynamic binning
        hilbert_order = self.hilbert_sort(mid_x, mid_y)
        num_bins = max(min_num_bins, int(len(lines) / lines_per_bin))
        bins_matrix, bin_counts = self.create_bins(hilbert_order, num_bins=num_bins)
        
        # Define neighbor depth based on percentage of total bins
        neighbor_depth = max(min_neighbor_deph, int(num_bins * depth_percentage))
        
        # Step 2: Check pairwise overlaps within bins and neighbor bins (forward only)
        for bin_i in range(num_bins):
            for offset in range(neighbor_depth + 1):
                bin_j = bin_i + offset
                if bin_j >= num_bins: continue
                
                indices_i = bins_matrix[bin_i]
                indices_j = bins_matrix[bin_j]

                indices_i = indices_i[indices_i != -1]
                indices_j = indices_j[indices_j != -1]
                
                if indices_i.shape[0] == 0 or indices_j.shape[0] == 0: continue
                
                indices_i, indices_j = cupy.meshgrid(indices_i, indices_j, indexing='ij')
                indices_i = indices_i.flatten()
                indices_j = indices_j.flatten()
                
                # Skip self-pairs and enforce order to avoid duplicates
                valid = indices_i < indices_j
                indices_i = indices_i[valid]
                indices_j = indices_j[valid]
                
                if indices_i.shape[0] == 0: continue
                
                # Extract OBBs and coordinates
                obbs_i = obbs[indices_i]
                obbs_j = obbs[indices_j]
                
                # SAT check
                overlap = self.check_overlap_sat(obbs_i, obbs_j)
                indices_i = indices_i[overlap]
                indices_j = indices_j[overlap]
                
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
                
                # Segment intersection
                p1 = cupy.stack([start_x[indices_i], start_y[indices_i]], axis=1)
                p2 = cupy.stack([end_x[indices_i], end_y[indices_i]], axis=1)
                q1 = cupy.stack([start_x[indices_j], start_y[indices_j]], axis=1)
                q2 = cupy.stack([end_x[indices_j], end_y[indices_j]], axis=1)
                
                is_point_intersection_ij = self.is_point_intersection(p1, p2, q1, q2, obb_width)
                is_point_intersection_ji = self.is_point_intersection(q1, q2, p1, p2, obb_width)
                is_segment_intersection_ij = ~is_point_intersection_ij
                is_segment_intersection_ji = ~is_point_intersection_ji
                
                i_nodes = cupy.array(indices_i, dtype=cupy.int32)
                j_nodes = cupy.array(indices_j, dtype=cupy.int32)
                edges_ij = cupy.stack([i_nodes, j_nodes], axis=0)
                edges_ji = cupy.stack([j_nodes, i_nodes], axis=0)
                edges = cupy.concatenate([edges_ij, edges_ji], axis=1)
                num_edges = i_nodes.shape[0]
                
                attributes = cupy.zeros((num_edges, len(self._edge_attributes)), dtype=cupy.float32)
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
                
                start = self._size
                end = start + edges.shape[1]
                self.edges[:, start:end] = edges
                self.edgeAttributes[start:end, :] = attributes
                self._size = end
        
        # Explode circular elements into line segments with parent_id
        exploded = self.explode_circle_to_lines(circular_elements)
        
        lines = lines.reset_index(drop=True)
        exploded = exploded.reset_index(drop=True)
        
        # Create CuPy arrays from circular segments
        circular_start_x = exploded['start_x'].to_cupy()
        circular_start_y = exploded['start_y'].to_cupy()
        circular_end_x = exploded['end_x'].to_cupy()
        circular_end_y = exploded['end_y'].to_cupy()
        circular_parent_id = exploded['parent_id'].to_cupy()
        
        # Compute OBBs
        obb_lines = self.create_obb(start_x, start_y, end_x, end_y, width=obb_width)
        obb_circles = self.create_obb(circular_start_x, circular_start_y, circular_end_x, circular_end_y, width=obb_width)
        
        # SAT overlap
        pair = self.check_overlap_sat(obb_circles, obb_lines) # returns (N_circ, N_lines) mask
        circular_indices, line_indices = cupy.where(pair)
        
        if circular_indices.size == 0: return # No matches
        
        # Build edge index arrays
        circles = circular_parent_id[circular_indices]  # circular element ID (original)
        lines = cupy.asarray(line_indices, dtype=cupy.int32)
        
        # Deduplicate (circle, line) pairs
        pairs = cupy.stack([circles, lines], axis=1)
        pairs, indices = cupy.unique(pairs, axis=0, return_index=True)
        circles = pairs[:, 0]
        lines = pairs[:, 1]
        
        circular_indices = circular_indices[indices]
        line_indices = line_indices[indices]
        
        # Extract coordinates for segment intersection
        p1 = cupy.stack([circular_start_x[circular_indices], circular_start_y[circular_indices]], axis=1)
        p2 = cupy.stack([circular_end_x[circular_indices], circular_end_y[circular_indices]], axis=1)
        q1 = cupy.stack([start_x[line_indices], start_y[line_indices]], axis=1)
        q2 = cupy.stack([end_x[line_indices], end_y[line_indices]], axis=1)
        
        is_point_intersection = self.is_point_intersection(q1, q2, p1, p2, obb_width)
        is_segment_intersection = ~is_point_intersection
        
        # Prepare edge attributes for both directions (circle → line, line → circle)
        num_edges = circles.shape[0]
        attributes = cupy.zeros((num_edges, len(self._edge_attributes)), dtype=cupy.float32)
        
        attributes_circle_line = attributes.copy()
        attributes_circle_line[:, self.perimeter_intersection] = 1.0
        
        attributes_line_circle = attributes.copy()
        attributes_line_circle[:, self.point_intersection] = is_point_intersection.astype(cupy.float32)
        attributes_line_circle[:, self.segment_intersection] = is_segment_intersection.astype(cupy.float32)
        
        # Build edges
        edge_circle_line = cupy.stack([circles, lines], axis=0)
        edge_line_circle = cupy.stack([lines, circles], axis=0)
        
        edges = cupy.concatenate([edge_circle_line, edge_line_circle], axis=1)
        attributes = cupy.concatenate([attributes_circle_line, attributes_line_circle], axis=0)
        
        # Store into the graph
        start = self._size
        end = start + edges.shape[1]
        self.edges[:, start:end] = edges
        self.edgeAttributes[start:end, :] = attributes
        self._size = end
                         
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

from typing import Dict
import ezdxf
import torch
import cudf
from torch_geometric.data import Data

def CreateGraph(dxf_file):
    dataframe: List[Dict] = [ ]
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    lineCollector = [line for line in modelSpace if line.dxftype() == 'LINE']
    circleCollector = [circle for circle in modelSpace if circle.dxftype() == 'CIRCLE']
    arcCollector = [arc for arc in modelSpace if arc.dxftype() == 'ARC']
    
    # create a dataframe with line information
    for line in lineCollector:
        start_x, start_y, _ = line.dxf.start
        end_x, end_y, _ = line.dxf.end
        layer = line.dxf.layer
        
        length = math.hypot(end_x - start_x, end_y - start_y)
        angle = math.atan2(end_y - start_y, end_x - start_x) % (2*math.pi)
        
        dataframe.append(
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "length": length,
                "angle": angle,
                "layer": layer,
                "circle": 0,
                "arc": 0,
                "radius": 0,
                "start_angle": 0,
                "end_angle": 0
            }
        )
    
    for circle in circleCollector:
        center_x, center_y, _ = circle.dxf.center
        radius = circle.dxf.radius
        perimeter = 2 * math.pi * radius
        layer = circle.dxf.layer
        
        dataframe.append(
            {
                "start_x": center_x,
                "start_y": center_y,
                "end_x": center_x,
                "end_y": center_y,
                "length": perimeter,
                "angle": 0.0,
                "layer": layer,
                "circle": 1,
                "arc": 0,
                "radius": radius,
                "start_angle": 0,
                "end_angle": 0
            }
        )
    
    for arc in arcCollector:
        center_x, center_y, _ = arc.dxf.center
        radius = arc.dxf.radius
        startAngle = arc.dxf.start_angle
        endAngle = arc.dxf.end_angle
        arcLength = radius * math.radians((endAngle - startAngle) % 360)
        layer = arc.dxf.layer
        
        dataframe.append(
            {
                "start_x": center_x,
                "start_y": center_y,
                "end_x": center_x,
                "end_y": center_y,
                "length": arcLength,
                "angle": 0.0,
                "layer": layer,
                "circle": 0,
                "arc": 1,
                "radius": radius,
                "start_angle": startAngle,
                "end_angle": endAngle
            }
        )
    
    dataframe = cudf.DataFrame(dataframe)
    
    # Define anchor point
    anchor_x = cupy.float32(cupy.concatenate([dataframe['start_x'].to_cupy(), dataframe['end_x'].to_cupy()]).min() - 100000)
    anchor_y = cupy.float32(cupy.concatenate([dataframe['start_y'].to_cupy(), dataframe['end_y'].to_cupy()]).min() - 100000)
    
    # Compute offset
    start_x, start_y, end_x, end_y = dataframe['start_x'], dataframe['start_y'], dataframe['end_x'], dataframe['end_y']
    
    # Set a safe length (to avoid division by zero)
    mask = (dataframe["circle"] == 0) & (dataframe["arc"] == 0) & dataframe["length"] > 1e-12
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

"""
1 - build an OBB with threshold width for every line
2 - Hilbert-sort the OBB centers and then group then into bins
2 - apply BVH per bin, run a tiny SAT kernel for quick finding
3 - for each pair of lines in which its OBBs overlap we use Segment-to-Segment to compute its intersection
4 - in case it doesn't find the intersection, check the threshold radius on the endpoints via vector math


hilbert_sort - uses the full OBB centroid (obbs.mean(axis=1)) for sorting

create_bins - query neighbor bins combining only future bins

"""