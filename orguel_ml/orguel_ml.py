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
        
        self.edges = self.edges[:, :edges]
        self.edgeAttributes = self.edgeAttributes[:edges, :]
           
    @staticmethod
    def overlap_ratios(lineA, lineB):
        xA1, yA1, xA2, yA2, lengthA = lineA[['start_x', 'start_y', 'end_x', 'end_y', 'length']]
        xB1, yB1, xB2, yB2, lengthB = lineB[['start_x', 'start_y', 'end_x', 'end_y', 'length']]
        
        # Compute direction vector
        dxA, dyA = xA2 - xA1, yA2 - yA1
        dir_lengthA = math.sqrt(dxA**2 + dyA**2)
        
        # Check if the line has zero length
        if dir_lengthA < 1e-12: return 0.0, 0.0
        
        # Compute unit vector
        ndxA, ndyA = dxA / dir_lengthA, dyA / dir_lengthA  # Unit vector

        # Project B's endpoints onto A
        tB1 = ( (xB1 - xA1) * ndxA + (yB1 - yA1) * ndyA )
        tB2 = ( (xB2 - xA1) * ndxA + (yB2 - yA1) * ndyA )

        # Get the projected range
        tmin, tmax = sorted([tB1, tB2])
        
        # Clip the projection range within [0, lengthA]
        overlap_start = max(0, tmin)
        overlap_end = min(lengthA, tmax)
        overlap_len   = max(0, overlap_end - overlap_start)
        
        # fraction of A that’s overlapped by B
        overlapAinB = overlap_len / lengthA
        
        # Now do the same from B’s perspective
        dxB, dyB = xB2 - xB1, yB2 - yB1
        dir_lengthB = math.sqrt(dxB**2 + dyB**2)
        
        if dir_lengthB < 1e-12: return 0.0, 0.0
        
        ndxB, ndyB = dxB / dir_lengthB, dyB / dir_lengthB
        
        # Project A’s endpoints onto B
        tA1 = ((xA1 - xB1) * ndxB + (yA1 - yB1) * ndyB)
        tA2 = ((xA2 - xB1) * ndxB + (yA2 - yB1) * ndyB)
        smin, smax = sorted([tA1, tA2])
        
        overlap_start_B = max(0, smin)
        overlap_end_B   = min(lengthB, smax)
        overlap_len_B   = max(0, overlap_end_B - overlap_start_B)
        
        overlapBinA = overlap_len_B / lengthB
        
        return overlapAinB, overlapBinA
    
    @staticmethod
    def is_point_intersection(self, threshold, ix, iy, x1, y1, x2, y2):
        d1 = cupy.hypot(ix - x1, iy - y1)
        d2 = cupy.hypot(ix - x2, iy - y2)
        return (d1 <= threshold) | (d2 <= threshold)
    
    @staticmethod
    def create_arc(center, radius, startAngle, endAngle, segments=16):
        points = []
        startAngle = math.radians(startAngle)
        endAngle   = math.radians(endAngle)

        arcRange = endAngle - startAngle
        if arcRange < 0:
            arcRange += 2*math.pi

        step = arcRange / segments
        for s in range(segments+1):
            angle = startAngle + s*step
            px = center.x + radius * math.cos(angle)
            py = center.y + radius * math.sin(angle)
            points.append((px, py))
        return points

    @staticmethod
    def create_bins_angle_offset(dataframe, angle_size, offset_size):
        # Bin keys for each line
        dataframe['angle_bin'] = cupy.floor((dataframe['angle'] % cupy.pi) / angle_size).astype(cupy.int32)
        dataframe['offset_bin'] = cupy.floor(dataframe['offset'] / offset_size).astype(cupy.int32)
        return dataframe.groupby(['angle_bin', 'offset_bin'])
    
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
    def create_bins_spatial(hilbert_indices, num_bins):
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
    def segment_intersection(p1, p2, q1, q2):
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
    def endpoint_threshold_check(p1, p2, q1, q2, threshold=0.5):
        batch_size = p1.shape[0]
        
        # Stack all endpoints and their corresponding opposite segments
        points = cupy.stack([p1, p2, q1, q2], axis=1).reshape(-1, 2) 
        segments_a = cupy.repeat(cupy.concatenate([q1, p1]), 2, axis=0)
        segments_b = cupy.repeat(cupy.concatenate([q2, p2]), 2, axis=0)
        
        # Compute closest points on each segment
        closest = Graph._point_to_segment_closest(points, segments_a, segments_b)
        distance_squared = cupy.sum((points - closest) ** 2, axis=1)
        hits = distance_squared <= threshold**2
        
        # Compute midpoints
        midpoints = (points + closest) / 2
        
        # Reshape results to (4, N, 2)
        midpoints = midpoints.reshape(4, batch_size, 2)
        hits = hits.reshape(4, batch_size)
        
        # Select the first valid midpoint per line pair
        first_hit = hits.argmax(axis=0)
        valid = hits.any(axis=0)
        
        midpoint_result = midpoints[first_hit, cupy.arange(batch_size)]
        midpoint_result[~valid] = cupy.nan
        
        return midpoint_result
    
    def ParallelDetection(self, max_threshold=50, colinear_threshold=0.5, min_overlap_ratio=0.2,
                          angle_tolerance=cupy.radians(0.01), bin_angle_size=cupy.radians(0.25)):
        dataframe = self._dataframe.copy()
        dataframe['angle'] = dataframe['angle'] % cupy.pi  # collapse symmetrical directions
        
        bins = self.create_bins_angle_offset(dataframe, bin_angle_size, max_threshold)
        
        # Include neighboring bins for robustness
        for (angle, offset), _ in bins:
            neighboring_bins = [(angle + da, offset + do) for da in (-1, 0, 1) for do in (-1, 0, 1)]
            neighboringBinsIndices = [bins.groups[bin] for bin in neighboring_bins if bin in bins.groups]
            
            lineIndices = cupy.concatenate([i.to_cupy() for i in neighboringBinsIndices])
            
            if len(lineIndices) < 2: continue
            
            subset = dataframe.take(lineIndices).reset_index(drop=True)
            
            coordinates = cupy.stack([subset['angle'].to_cupy(), subset['offset'].to_cupy()], axis=1)
            space2D = cKDTree(coordinates)
            
            # Query for nearby parallel lines in (angle, offset) space
            for i in range(len(subset)):
                if subset['circle'].iloc[i] or subset['arc'].iloc[i]: continue
                
                angle_i, offset_i = subset['angle'].iloc[i], subset['offset'].iloc[i]
                nearbyLines = space2D.query_ball_point([angle_i, offset_i], r=max_threshold)
                
                for j in nearbyLines:
                    if i >= j: continue
                    if subset['circle'].iloc[j] or subset['arc'].iloc[j]: continue
                    
                    line_i, line_j = subset.iloc[i], subset.iloc[j]
                    overlapA, overlapB = self.overlap_ratios(line_i, line_j)
                    if min(overlapA, overlapB) <= min_overlap_ratio: continue
                    
                    angle_j = subset['angle'].iloc[j]
                    if cupy.abs(angle_i - angle_j) >= angle_tolerance: continue
                    
                    # perpendicular distance between the two parallel lines
                    distance = cupy.abs(subset['offset'].iloc[j] - offset_i) / cupy.cos(angle_i)
                    
                    # Assign normalized distances to edges
                    for (a, b, overlapRatio) in [(i, j, overlapA), (j, i, overlapB)]:
                        edgeAttributes = cupy.zeros(len(self._edge_attributes), dtype=cupy.float32)
                        edgeAttributes[self.parallel] = 1
                        edgeAttributes[self.colinear] = 1 if distance < colinear_threshold else 0
                        edgeAttributes[self.perpendicular_distance] = distance / dataframe["length"].max()
                        edgeAttributes[self.overlap_ratio] = overlapRatio
                        self.edges.append(int((lineIndices[a]), int(lineIndices[b])))
                        self.edgeAttributes.append(edgeAttributes)
                                   
    def IntersectionDetection(self, threshold=0.5, co_linear_tolerance=cupy.radians(0.01)):
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
        angle = lines['angle'].to_cupy()
        start_x = lines['start_x'].to_cupy()
        start_y = lines['start_y'].to_cupy()
        end_x = lines['end_x'].to_cupy()
        end_y = lines['end_y'].to_cupy()
        
        # Compute line midpoints and OBBs
        obbs = self.create_obb(start_x, start_y, end_x, end_y, width=threshold)
        obb_centroids = obbs.mean(axis=1)
        mid_x = obb_centroids[:, 0]
        mid_y = obb_centroids[:, 1]
        
        # Hilbert sort and dynamic binning
        hilbert_order = self.hilbert_sort(mid_x, mid_y)
        num_bins = max(min_num_bins, int(len(lines) / lines_per_bin))
        bins_matrix, bin_counts = self.create_bins_spatial(hilbert_order, num_bins=num_bins)
        
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
                
                # Segment intersection
                p1 = cupy.stack([start_x[indices_i], start_y[indices_i]], axis=1)
                p2 = cupy.stack([end_x[indices_i], end_y[indices_i]], axis=1)
                q1 = cupy.stack([start_x[indices_j], start_y[indices_j]], axis=1)
                q2 = cupy.stack([end_x[indices_j], end_y[indices_j]], axis=1)
                
                intersections, condition = self.segment_intersection(p1, p2, q1, q2)
                
                midpoints_ij = self.endpoint_threshold_check(p1, p2, q1, q2, threshold)
                midpoints_ji = self.endpoint_threshold_check(q1, q2, p1, p2, threshold)
                midpoints_ij = ~cupy.isnan(midpoints_ij).any(axis=1)
                midpoints_ji = ~cupy.isnan(midpoints_ji).any(axis=1)
                
                # Angle difference
                angle_i = angle[indices_i]
                angle_j = angle[indices_j]
                angle_difference = (angle_j - angle_i) % (2 * cupy.pi)
                angle_difference_min = cupy.minimum(angle_difference, 2 * cupy.pi - angle_difference)
                
                # Filter colinear lines based on angle tolerance
                colinear = (angle_difference_min % cupy.pi) >= co_linear_tolerance
                condition = condition[colinear]
                indices_i = indices_i[colinear]
                indices_j = indices_j[colinear]
                midpoints_ij = midpoints_ij[colinear]
                midpoints_ji = midpoints_ji[colinear]
                angle_difference = angle_difference[colinear]
                angle_difference_min = angle_difference_min[colinear]
                
                is_point_intersection_ij = midpoints_ij | (condition & ~midpoints_ij)
                is_point_intersection_ji = midpoints_ji | (condition & ~midpoints_ji)
                is_segment_intersection_ij = ~is_point_intersection_ij
                is_segment_intersection_ji = ~is_point_intersection_ji
                
                edge_attributes_ij = cupy.zeros(len(self._edge_attributes), dtype=cupy.float32)
                edge_attributes_ij[:, self.point_intersection] = is_point_intersection_ij.astype(cupy.float32)
                edge_attributes_ij[:, self.segment_intersection] = is_segment_intersection_ij.astype(cupy.float32)
                edge_attributes_ij[:, self.angle_difference] = angle_difference_min / (cupy.pi / 2)
                edge_attributes_ij[:, self.angle_difference_sin] = cupy.sin(angle_difference)
                edge_attributes_ij[:, self.angle_difference_cos] = cupy.cos(angle_difference)

                edge_attributes_ji = cupy.zeros(len(self._edge_attributes), dtype=cupy.float32)
                edge_attributes_ji[:, self.point_intersection] = is_point_intersection_ji.astype(cupy.float32)
                edge_attributes_ji[:, self.segment_intersection] = is_segment_intersection_ji.astype(cupy.float32)
                edge_attributes_ji[:, self.angle_difference] = angle_difference_min / (cupy.pi / 2)
                edge_attributes_ji[:, self.angle_difference_sin] = cupy.sin(angle_difference)
                edge_attributes_ji[:, self.angle_difference_cos] = cupy.cos(angle_difference)
                
                
                
                
                
                
                
                

                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
                
                
               
           
            
            
            
            
        
        
        
        
        
        # Step 1: Separate lines vs circles
        geometryCircular: List[Tuple[int, LineString]] = []
        geometryLines: List[Tuple[int, LineString]] = []
        for i, row in self._dataframe.iterrows():
            if row["circle"]:
                center = Point(row["start_x"], row["start_y"])
                radius = row["radius"]
                circle = center.buffer(radius).boundary
                geometryCircular.append((i, circle))
            elif row["arc"]:
                center = Point(row["start_x"], row["start_y"])
                radius = row["radius"]
                startAngle = row["start_angle"]
                endAngle = row["end_angle"]
                arcPoints = self.create_arc(center, radius, startAngle, endAngle, segments=16)
                arc = LineString(arcPoints)
                geometryCircular.append((i, arc))      
            else:
                line = LineString([(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])])
                geometryLines.append((i, line))
        
        # Build STRtree for lines
        space2D = STRtree([geometry for _, geometry in geometryLines])
        
        # Step 2: Compute all angle differences
        for i, lineA in geometryLines:
            nearbyLines = space2D.query(lineA)  # Fetch only nearby lines
            
            for k in nearbyLines:
                j, lineB = geometryLines[k]
                
                if i >= j: continue  # Avoid duplicate processing
                
                # Angle check (skip nearly identical lines)
                angle_i = self._dataframe.iloc[i]["angle"]
                angle_j = self._dataframe.iloc[j]["angle"]
                minAngleDifference = (angle_j - angle_i) % math.pi
                minAngleDifference = min(minAngleDifference, math.pi - minAngleDifference)
                if minAngleDifference < co_linear_tolerance: continue
                
                # Step 3: Check actual intersection
                intersection = lineA.intersection(lineB)
                if intersection.is_empty:
                    if lineA.distance(lineB) < threshold:
                        pA, pB = nearest_points(lineA, lineB)
                        intersection_x, intersection_y = (pA.x + pB.x) / 2, (pA.y + pB.y) / 2
                        intersection = Point(intersection_x, intersection_y)
                    else: continue
                
                # Step 4: Ensure intersection is always stored as a point
                if intersection.geom_type == "Point": intersection = (intersection.x, intersection.y)
                elif intersection.geom_type == "MultiPoint":intersection = list(intersection.geoms)[0].coords[0]
                else: intersection = intersection.interpolate(0.5, normalized=True).coords[0]
                
                angleDifference = (angle_j - angle_i) % (2 * math.pi)
                
                # Step 5: Add edges with normalized angleDifference
                for a, b in [(i, j), (j, i)]:
                    row = self._dataframe.iloc[a]
                    rowPoints = [(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])]
                    isPointIntersection = self.is_point_intersection(intersection, rowPoints, threshold)
                    edgeAttributes = self._edge_attributes.copy()
                    edgeAttributes["point_intersection"] = 1 if isPointIntersection else 0
                    edgeAttributes["segment_intersection"] = 0 if isPointIntersection else 1
                    edgeAttributes["angle_difference"] = minAngleDifference / (math.pi / 2)
                    edgeAttributes["angle_difference_sin"] = math.sin(angleDifference)
                    edgeAttributes["angle_difference_cos"] = math.cos(angleDifference)
                    self.edges.append((a, b))
                    self.edgeAttributes.append(list(edgeAttributes.values()))
        
        # Step 6: circle-line perimeter intersections
        for i, circularElement in geometryCircular:
            nearbyLines = space2D.query(circularElement)
            
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
                
                row = self._dataframe.iloc[j]
                rowPoints = [(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])] 
                isPointIntersection = self.is_point_intersection(intersection, rowPoints, threshold)
                
                edgeAttributes = self._edge_attributes.copy()
                edgeAttributes["perimeter_intersection"] = 1
                self.edges.append((i, j))
                self.edgeAttributes.append(list(edgeAttributes.values()))
                
                edgeAttributes = self._edge_attributes.copy()
                edgeAttributes["point_intersection"] = 1 if isPointIntersection else 0
                edgeAttributes["segment_intersection"] = 0 if isPointIntersection else 1
                self.edges.append((j, i))
                self.edgeAttributes.append(list(edgeAttributes.values()))
                         
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

create_bins_spatial - query neighbor bins combining only future bins

"""