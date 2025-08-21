from enum import IntEnum

class _DataframeField(IntEnum):
    original_index = 0
    layer = 1
    line_flag = 2
    circle_flag = 3
    arc_flag = 4
    point_flag = 5
    start_x = 6
    start_y = 7
    end_x = 8
    end_y = 9
    mid_x = 10
    mid_y = 11
    angle = 12
    length = perimeter = 13
    d_x = 14
    d_y = 15
    u_x = t_x = 16
    u_y = t_y = 17
    n_x = r_x = 18
    n_y = r_y = 19
    center_x = 20
    center_y = 21
    radius = 22
    start_angle = 23
    end_angle = 24
    arc_span = 25
    
    @classmethod
    def count(cls) -> int: return len(cls)

class _EdgeAttribute(IntEnum):
    parallel = 0
    offset = 1
    overlap_ratio = 2
    oblique = 3
    intersection_min = 4
    intersection_max = 5
    angle_difference_sin_min = 6
    angle_difference_cos_min = 7
    angle_difference_sin_max = 8
    angle_difference_cos_max = 9
    
    @classmethod
    def count(cls) -> int: return len(cls)

from typing import Tuple, List
import math
import torch
from torch import Tensor, newaxis

class Graph:
    def __init__(self, dataframe:Tensor):
        
        F = _DataframeField
        Att = _EdgeAttribute
        
        self._dataframe = dataframe
        self._device = dataframe.device
        self.edge_pairs = torch.empty([2, 0], dtype=torch.long, device=self._device)
        self.edge_attributes = torch.empty([0, Att.count()], dtype=torch.float32, device=self._device)
        
        functions = [
            self.detect_parallel,
            self.detect_intersection
        ]
        
        for function in functions:
            pairs, attributes = function()
            self.edge_pairs = torch.hstack([self.edge_pairs, pairs])
            self.edge_attributes = torch.vstack([self.edge_attributes, attributes])
        
        # Update flags and reshape for ML model
        is_point = dataframe[F.point_flag] == 1
        is_circle = dataframe[F.circle_flag] == 1
        
        dataframe[F.line_flag] = torch.where(is_point, 1, dataframe[F.line_flag])
        dataframe[F.arc_flag] = torch.where(is_circle, 1, dataframe[F.arc_flag])
        
        line_flag = dataframe[F.line_flag].reshape([-1, 1]) # shape (N, 1)
        arc_flag = dataframe[F.arc_flag].reshape([-1, 1]) # shape (N, 1)
        
        # Normalize coordinates
        coordinates_x = torch.hstack([dataframe[F.start_x], dataframe[F.end_x]])
        coordinates_y = torch.hstack([dataframe[F.start_y], dataframe[F.end_y]])
        coordinates_x_mean, coordinates_x_std = coordinates_x.mean(), coordinates_x.std()
        coordinates_y_mean, coordinates_y_std = coordinates_y.mean(), coordinates_y.std()
        
        normalized_start_x = (dataframe[F.start_x] - coordinates_x_mean) / coordinates_x_std
        normalized_start_y = (dataframe[F.start_y] - coordinates_y_mean) / coordinates_y_std
        normalized_end_x = (dataframe[F.end_x] - coordinates_x_mean) / coordinates_x_std
        normalized_end_y = (dataframe[F.end_y] - coordinates_y_mean) / coordinates_y_std
        
        normalized_coordinates = torch.stack([normalized_start_x, normalized_start_y, 
                                              normalized_end_x, normalized_end_y], dim=1) # shape (N, 4)
        
        # Normalize length and offset
        normalized_length = (dataframe[F.length] / self.p95_length).clamp(min=0, max=1).reshape([-1, 1]) # shape (N, 1)
        self.edge_attributes[:, Att.offset] = (self.edge_attributes[:, Att.offset] / self.parallel_max_offset).clamp(min=-1, max=1)
        
        # Stack direction unit vectors
        direction = torch.stack([dataframe[F.u_x], dataframe[F.u_y]], dim=1) # shape (N, 2)
        
        # Node attributes
        self.node_attributes = torch.hstack([line_flag, arc_flag, normalized_coordinates, normalized_length, direction])
    
    @staticmethod
    def create_obbs(elements:Tensor, width:float, length_extension:float=0.) -> Tuple[Tensor, Tensor]:
        
        F = _DataframeField
        n_elements = elements.size(1)
        
        # Filter supported elements
        is_line = elements[F.line_flag] == 1
        is_circle_or_arc = (elements[F.circle_flag] == 1) | (elements[F.arc_flag] == 1)
        
        filter = is_line | is_circle_or_arc
        
        obbs = torch.empty((n_elements, 4, 2), dtype=torch.float32, device=elements.device)
        
        if is_line.any():
            # === LINE ELEMENTS ===
            lines = elements[:, is_line]
            
            # Half dimensions
            half_width = width / 2
            half_length = (lines[F.length] + length_extension) / 2
            
            # Compute displacements
            dx_length = lines[F.u_x] * half_length
            dy_length = lines[F.u_y] * half_length
            dx_width = lines[F.n_x] * half_width
            dy_width = lines[F.n_y] * half_width
            
            # Corners (4 per OBB)
            corner1 = torch.stack([lines[F.mid_x] - dx_length - dx_width,
                                   lines[F.mid_y] - dy_length - dy_width], dim=1)
            
            corner2 = torch.stack([lines[F.mid_x] + dx_length - dx_width,
                                   lines[F.mid_y] + dy_length - dy_width], dim=1)
            
            corner3 = torch.stack([lines[F.mid_x] + dx_length + dx_width,
                                   lines[F.mid_y] + dy_length + dy_width], dim=1)
            
            corner4 = torch.stack([lines[F.mid_x] - dx_length + dx_width,
                                   lines[F.mid_y] - dy_length + dy_width], dim=1)
            
            # Stack corners
            obbs[is_line] = torch.stack([corner1, corner2, corner3, corner4], dim=1)
        
        if is_circle_or_arc.any():
            # === ARC ELEMENTS ===
            arcs = elements[:, is_circle_or_arc]
            
            margin = width / 2
            
            r_x, r_y, t_x, t_y = arcs[F.r_x], arcs[F.r_y], arcs[F.t_x], arcs[F.t_y]
            arc_span, radius = arcs[F.arc_span], arcs[F.radius]

            dx_length = torch.where(arc_span < torch.pi, t_x * (radius * torch.sin(arc_span/2) + margin), t_x * (radius + margin))
            dy_length = torch.where(arc_span < torch.pi, t_y * (radius * torch.sin(arc_span/2) + margin), t_y * (radius + margin))

            dx_width = r_x * (radius * (1 - torch.cos(arc_span/2)) + margin)
            dy_width = r_y * (radius * (1 - torch.cos(arc_span/2)) + margin)

            dx_margin = r_x * margin
            dy_margin = r_y * margin
            
            corner1 = torch.stack([arcs[F.mid_x] - dx_length + dx_margin,
                                   arcs[F.mid_y] - dy_length + dy_margin], dim=1)
            
            corner2 = torch.stack([arcs[F.mid_x] + dx_length + dx_margin,
                                   arcs[F.mid_y] + dy_length + dy_margin], dim=1)
            
            corner3 = torch.stack([arcs[F.mid_x] + dx_length - dx_width,
                                   arcs[F.mid_y] + dy_length - dy_width], dim=1)
            
            corner4 = torch.stack([arcs[F.mid_x] - dx_length - dx_width,
                                   arcs[F.mid_y] - dy_length - dy_width], dim=1)
            
            obbs[is_circle_or_arc] = torch.stack([corner1, corner2, corner3, corner4], dim=1)
        
        elements = elements[:, filter]
        obbs = obbs[filter]
        
        return elements, obbs # Shape obbs (n_elements, 4, 2)
    
    @staticmethod
    def find_overlaping_pairs(elements:Tensor, obbs:Tensor) -> Tuple[Tensor, Tensor]:
        
        F = _DataframeField
        
        # Compute AABBs (Axis-Aligned Bounding Boxes)
        min_x = obbs[..., 0].min(dim=1).values
        max_x = obbs[..., 0].max(dim=1).values
        min_y = obbs[..., 1].min(dim=1).values
        max_y = obbs[..., 1].max(dim=1).values
        
        # Computing AABB intersection matrix
        mask_x = (min_x[:, newaxis] <= max_x[newaxis, :]) & (min_x[newaxis, :] <= max_x[:, newaxis])
        mask_y = (min_y[:, newaxis] <= max_y[newaxis, :]) & (min_y[newaxis, :] <= max_y[:, newaxis])
        aabb_overlap = mask_x & mask_y
        
        # Candidate pairs
        pairs = torch.argwhere(aabb_overlap) # shape (n, 2)
        i, j = pairs[:, 0], pairs[:, 1]
        
        # Remove self-comparisons and duplicates
        mask = (i < j)
        i, j = i[mask], j[mask]
        
        # Get OBBs for pairs
        obbs_i, obbs_j = obbs[i], obbs[j]
        
        # axes per box
        axes_i = torch.stack([elements[F.u_x, i], elements[F.u_y, i], elements[F.n_x, i], elements[F.n_y, i]], dim=1).reshape(-1,2,2)
        axes_j = torch.stack([elements[F.u_x, j], elements[F.u_y, j], elements[F.n_x, j], elements[F.n_y, j]], dim=1).reshape(-1,2,2)
        axes = torch.cat([axes_i, axes_j], axis=1) # shape (n_pairs, 4, 2)
        
        # Project corners onto axes
        projections_i = torch.einsum('nij,nkj->nik', axes, obbs_i) # Shape (n_pairs, 4, 4)
        projections_j = torch.einsum('nij,nkj->nik', axes, obbs_j)
        
        # Interval comparisons on each axis
        min_i = projections_i.min(dim=2).values
        max_i = projections_i.max(dim=2).values
        min_j = projections_j.min(dim=2).values
        max_j = projections_j.max(dim=2).values
        
        separating_axis = (max_i < min_j) | (max_j < min_i) # True if a separating axis exists
        obb_overlap = ~torch.any(separating_axis, dim=1) # True if overlap
        
        return i[obb_overlap], j[obb_overlap]
    
    @staticmethod
    def get_overlap_ratios(lines_a:Tensor, lines_b:Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        F = _DataframeField
        
        # Project B's endpoints onto A
        t1 = (lines_b[F.start_x] - lines_a[F.start_x]) * lines_a[F.u_x] + \
             (lines_b[F.start_y] - lines_a[F.start_y]) * lines_a[F.u_y]

        t2 = (lines_b[F.end_x] - lines_a[F.start_x]) * lines_a[F.u_x] + \
             (lines_b[F.end_y] - lines_a[F.start_y]) * lines_a[F.u_y]

        # Get the projected range
        tmin, tmax = torch.minimum(t1, t2), torch.maximum(t1, t2)

        # Clip the projection range within [0, length_a]
        overlap_start = torch.clamp(tmin, min=0)
        overlap_end = torch.clamp(tmax, max=lines_a[F.length])
        overlap_mid = (overlap_start + overlap_end) / 2

        # Overlap ratio
        overlap_length = torch.clamp(overlap_end - overlap_start, min=0)
        overlap_a_b = overlap_length / lines_a[F.length]
        overlap_b_a = overlap_length / lines_b[F.length]

        # Midpoint of projected overlap on line A
        overlap_mid_xa = lines_a[F.start_x] + lines_a[F.u_x] * overlap_mid
        overlap_mid_ya = lines_a[F.start_y] + lines_a[F.u_y] * overlap_mid

        # Perpendicular distance to line B
        dx = overlap_mid_xa - lines_b[F.mid_x]
        dy = overlap_mid_ya - lines_b[F.mid_y]

        # Perpendicular distances relative to overlaping segment midpoint
        distance_a_b = dx * lines_b[F.n_x] + dy * lines_b[F.n_y]
        distance_b_a = -dx * lines_a[F.n_x] - dy * lines_a[F.n_y]
        
        return overlap_a_b, distance_a_b, overlap_b_a, distance_b_a # Each: shape (n_pairs,)
    
    @staticmethod
    def _get_line_arc_intersections(lines:Tensor, arcs:Tensor, margin:float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, 
                                                                                      Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        F = _DataframeField
        
        dx_start = arcs[F.center_x] - lines[F.start_x]
        dy_start = arcs[F.center_y] - lines[F.start_y]
        dx_end = arcs[F.center_x] - lines[F.end_x]
        dy_end = arcs[F.center_y] - lines[F.end_y]

        distance_start = torch.sqrt(dx_start**2 + dy_start**2)
        distance_end = torch.sqrt(dx_end**2 + dy_end**2)

        # Remove pairs in which the lines falls completelly within the circle inner perimeter
        inner_radius = arcs[F.radius] - margin

        start_in_inner = distance_start < inner_radius
        end_in_inner = distance_end < inner_radius

        mask1 = ~(start_in_inner & end_in_inner)

        arcs, lines = arcs[:, mask1], lines[:, mask1]
        dx_start, dy_start = dx_start[mask1], dy_start[mask1]
        inner_radius = inner_radius[mask1]
        start_in_inner, end_in_inner = start_in_inner[mask1], end_in_inner[mask1]

        # Compute closest distance betweem the center of the circle and the line
        t = torch.clamp((dx_start * lines[F.d_x] + dy_start * lines[F.d_y]) / lines[F.length]**2, min=0, max=1)
        closest_x, closest_y = lines[F.start_x] + t * lines[F.d_x], lines[F.start_y] + t * lines[F.d_y]

        closest_distance = torch.sqrt((closest_x - arcs[F.center_x])**2 + (closest_y - arcs[F.center_y])**2)

        # Remove pairs in which the lines falls completelly out of the circle outer perimeter
        outer_radius = arcs[F.radius] + margin

        mask2 = ~(closest_distance > outer_radius)

        arcs, lines = arcs[:, mask2], lines[:, mask2]
        dx_start, dy_start = dx_start[mask2], dy_start[mask2]
        inner_radius = inner_radius[mask2]
        start_in_inner, end_in_inner = start_in_inner[mask2], end_in_inner[mask2]
        closest_distance = closest_distance[mask2]

        # Compute min and max intersections
        a = lines[F.u_x]**2 + lines[F.u_y]**2
        b = 2 * (dx_start * lines[F.u_x] + dy_start * lines[F.u_y])
        c = dx_start**2 + dy_start**2 - arcs[F.radius]**2
        discriminant = torch.clamp(b**2 - 4 * a * c, min=0)

        sqrt_discriminant = torch.sqrt(discriminant)

        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        t1, t2 = -t1, -t2

        t_min = torch.minimum(t1, t2)
        t_max = torch.maximum(t1, t2)

        t_min = torch.where(start_in_inner, t_max, t_min)
        t_max = torch.where(end_in_inner, t_min, t_max)

        closest_out_inner = closest_distance >= inner_radius

        t_min = torch.where(closest_out_inner & (t_min < 0), t_max, t_min)
        t_max = torch.where(closest_out_inner & (t_max > lines[F.length]), t_min, t_max)

        # Check whether the intersections fall on arc range
        ix = lines[F.start_x] + t_min * lines[F.u_x]
        iy = lines[F.start_y] + t_min * lines[F.u_y]
        jx = lines[F.start_x] + t_max * lines[F.u_x]
        jy = lines[F.start_y] + t_max * lines[F.u_y]

        angle1 = torch.atan2(iy - arcs[F.center_y], ix - arcs[F.center_x]) % (2 * torch.pi)
        angle2 = torch.atan2(jy - arcs[F.center_y], jx - arcs[F.center_x]) % (2 * torch.pi)

        start_angle, end_angle = arcs[F.start_angle], arcs[F.end_angle]

        on_arc_range1 = torch.where(start_angle < end_angle, (start_angle <= angle1) & (angle1 <= end_angle), (start_angle <= angle1) | (angle1 <= end_angle))
        on_arc_range2 = torch.where(start_angle < end_angle, (start_angle <= angle2) & (angle2 <= end_angle), (start_angle <= angle2) | (angle2 <= end_angle))

        on_arc_range1 |= closest_out_inner
        on_arc_range2 |= closest_out_inner

        # Keep only pairs where there are at least one intersection on the arc range
        mask3 = (on_arc_range1 | on_arc_range2)

        arcs, lines = arcs[:, mask3], lines[:, mask3]
        t_min, t_max = t_min[mask3], t_max[mask3]
        ix, iy, jx, jy = ix[mask3], iy[mask3], jx[mask3], jy[mask3]
        angle1, angle2 = angle1[mask3], angle2[mask3]
        start_angle, end_angle = start_angle[mask3], end_angle[mask3]
        on_arc_range1, on_arc_range2 = on_arc_range1[mask3], on_arc_range2[mask3]

        ix = torch.where(on_arc_range1, ix, jx)
        iy = torch.where(on_arc_range1, iy, jy)
        jx = torch.where(on_arc_range2, jx, ix)
        jy = torch.where(on_arc_range2, jy, iy)
        t_min = torch.where(on_arc_range1, t_min, t_max)
        t_max = torch.where(on_arc_range2, t_max, t_min)
        angle1 = torch.where(on_arc_range1, angle1, angle2)
        angle2 = torch.where(on_arc_range2, angle2, angle1)

        # Compute the angle difference sin and cos

        # radius unit vectors at each intersection
        rx1 = (ix - arcs[F.center_x]) / arcs[F.radius]
        ry1 = (iy - arcs[F.center_y]) / arcs[F.radius]
        rx2 = (jx - arcs[F.center_x]) / arcs[F.radius]
        ry2 = (jy - arcs[F.center_y]) / arcs[F.radius]

        # sin and cos of angle between line and tangent
        sin1 = lines[F.u_x] * rx1 + lines[F.u_y] * ry1
        cos1 = -(lines[F.u_x] * ry1 - lines[F.u_y] * rx1)
        sin2 = lines[F.u_x] * rx2 + lines[F.u_y] * ry2
        cos2 = -(lines[F.u_x] * ry2 - lines[F.u_y] * rx2)
        
        angle_difference_sin_min = torch.clamp(sin1, min=-1, max=1)
        angle_difference_sin_max = torch.clamp(sin2, min=-1, max=1)
        angle_difference_cos_min = torch.clamp(cos1, min=-1, max=1)
        angle_difference_cos_max = torch.clamp(cos2, min=-1, max=1)
        
        # Compute intersections on arcs and lines
        on_arc_range = torch.where(start_angle < end_angle, (start_angle <= angle1) & (angle1 <= end_angle), (start_angle <= angle1) | (angle1 <= end_angle))
        
        arc_intersection_min = torch.where(on_arc_range, (angle1 - start_angle), (angle1 - start_angle).clamp(min=0).clamp(max=arcs[F.arc_span]))
        arc_intersection_max = torch.where(on_arc_range, (angle2 - start_angle), (angle2 - start_angle).clamp(min=0).clamp(max=arcs[F.arc_span]))
        arc_intersection_min = ((arc_intersection_min % (2*torch.pi)) / (arcs[F.arc_span] / 2)) -1
        arc_intersection_max = ((arc_intersection_max % (2*torch.pi)) / (arcs[F.arc_span] / 2)) -1

        line_intersection_min = torch.clamp(t_min / lines[F.length], min=0, max=1)
        line_intersection_max = torch.clamp(t_max / lines[F.length], min=0, max=1)
        
        mask = torch.zeros_like(mask1, dtype=torch.bool, device=lines.device)
        mask[mask1] = mask2
        mask[mask.clone()] = mask3
        
        return lines, line_intersection_min, line_intersection_max, arcs, arc_intersection_min, arc_intersection_max, \
               angle_difference_sin_min, angle_difference_cos_min, angle_difference_sin_max, angle_difference_cos_max, mask
    
    @staticmethod
    def get_intersection_positions(elements_a:Tensor, elements_b:Tensor, margin:float, angle_tolerance:float) -> \
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        F = _DataframeField
        device = elements_a.device
        n_pairs = elements_a.size(1)
        
        # Filter supported pairs
        is_line_a = elements_a[F.line_flag] == 1
        is_line_b = elements_b[F.line_flag] == 1
        is_arc_a = (elements_a[F.circle_flag] == 1) | (elements_a[F.arc_flag] == 1)
        is_arc_b = (elements_b[F.circle_flag] == 1) | (elements_b[F.arc_flag] == 1)
        
        line_line_pair = is_line_a & is_line_b
        line_arc_pair, arc_line_pair = is_line_a & is_arc_b, is_arc_a & is_line_b
        
        filter = line_line_pair | (line_arc_pair | arc_line_pair)
        
        intersection_a_min = torch.empty(n_pairs, dtype=torch.float32, device=device)
        intersection_a_max = torch.empty(n_pairs, dtype=torch.float32, device=device)
        intersection_b_min = torch.empty(n_pairs, dtype=torch.float32, device=device)
        intersection_b_max = torch.empty(n_pairs, dtype=torch.float32, device=device)
        
        angle_difference_b_a_sin_min = torch.empty(n_pairs, dtype=torch.float32, device=device)
        angle_difference_b_a_cos_min = torch.empty(n_pairs, dtype=torch.float32, device=device)
        angle_difference_b_a_sin_max = torch.empty(n_pairs, dtype=torch.float32, device=device)
        angle_difference_b_a_cos_max = torch.empty(n_pairs, dtype=torch.float32, device=device)
        
        if line_line_pair.any():
            lines_a = elements_a[:, line_line_pair]
            lines_b = elements_b[:, line_line_pair]
            
            # Compute absolute angle difference in [0, pi]
            angle_difference_b_a = (lines_b[F.angle] - lines_a[F.angle]) % (2*torch.pi)
            min_angle_difference = angle_difference_b_a % torch.pi
            min_angle_difference = torch.minimum(min_angle_difference, torch.pi - min_angle_difference)
            
            # Keep only pairs with angle difference above threshold
            oblique = min_angle_difference > angle_tolerance
            lines_a, lines_b = lines_a[:, oblique], lines_b[:, oblique]
            angle_difference_b_a = angle_difference_b_a[oblique]
            filter[line_line_pair] = oblique
            
            # Update line_line_pair
            line_line_pair &= filter
            
            # Vector from B start to A start
            w_x = lines_a[F.start_x] - lines_b[F.start_x]
            w_y = lines_a[F.start_y] - lines_b[F.start_y]

            # Core dot products
            b = lines_a[F.u_x] * lines_b[F.u_x] + lines_a[F.u_y] * lines_b[F.u_y]
            d = lines_a[F.u_x] * w_x + lines_a[F.u_y] * w_y
            e = lines_b[F.u_x] * w_x + lines_b[F.u_y] * w_y

            denominator = 1 - b * b

            # Get closest point parameters along each line
            t = (b * e - d) / denominator  # Along A
            s = (e - b * d) / denominator  # Along B

            # Convert to relative position from center [-1, 1]
            intersection_a_min[line_line_pair] = intersection_a_max[line_line_pair] = ((t / lines_a[F.length]).clamp(0, 1)) * 2 - 1
            intersection_b_min[line_line_pair] = intersection_b_max[line_line_pair] = ((s / lines_b[F.length]).clamp(0, 1)) * 2 - 1
            
            # get angle difference sin and cos
            angle_difference_b_a_sin_min[line_line_pair] = angle_difference_b_a_sin_max[line_line_pair] = torch.sin(angle_difference_b_a)
            angle_difference_b_a_cos_min[line_line_pair] = angle_difference_b_a_cos_max[line_line_pair] = torch.cos(angle_difference_b_a)
            
        if (line_arc_pair | arc_line_pair).any():
            if line_arc_pair.any():
                lines_a, arcs_b = elements_a[:, line_arc_pair], elements_b[:, line_arc_pair]
                
                lines_a, line_intersection_min, line_intersection_max, arcs_b, arc_intersection_min, arc_intersection_max, \
                angle_b_a_sin_min, angle_b_a_cos_min, angle_b_a_sin_max, angle_b_a_cos_max, filter[line_arc_pair] = Graph._get_line_arc_intersections(lines_a, arcs_b, margin)
                
                # Update line_arc_pair
                line_arc_pair &= filter
                
                # Save the min and max
                intersection_a_min[line_arc_pair], intersection_a_max[line_arc_pair] = line_intersection_min, line_intersection_max
                intersection_b_min[line_arc_pair], intersection_b_max[line_arc_pair] = arc_intersection_min, arc_intersection_max
                
                angle_difference_b_a_sin_min[line_arc_pair], angle_difference_b_a_cos_min[line_arc_pair] = angle_b_a_sin_min, angle_b_a_cos_min
                angle_difference_b_a_sin_max[line_arc_pair], angle_difference_b_a_cos_max[line_arc_pair] = angle_b_a_sin_max, angle_b_a_cos_max
            
            if arc_line_pair.any():
                arcs_a, lines_b = elements_a[:, arc_line_pair], elements_b[:, arc_line_pair]
                
                lines_b, line_intersection_min, line_intersection_max, arcs_a, arc_intersection_min, arc_intersection_max, \
                angle_b_a_sin_min, angle_b_a_cos_min, angle_b_a_sin_max, angle_b_a_cos_max, filter[arc_line_pair] = Graph._get_line_arc_intersections(lines_b, arcs_a, margin)
                
                # Update arc_line_pair
                arc_line_pair &= filter
                
                # Save the min and max
                intersection_a_min[arc_line_pair], intersection_a_max[arc_line_pair] = arc_intersection_min, arc_intersection_max
                intersection_b_min[arc_line_pair], intersection_b_max[arc_line_pair] = line_intersection_min, line_intersection_max
                
                angle_difference_b_a_sin_min[arc_line_pair], angle_difference_b_a_cos_min[arc_line_pair] = angle_b_a_sin_min, angle_b_a_cos_min
                angle_difference_b_a_sin_max[arc_line_pair], angle_difference_b_a_cos_max[arc_line_pair] = angle_b_a_sin_max, angle_b_a_cos_max
                
        elements_a, elements_b = elements_a[:, filter], elements_b[:, filter]
        intersection_a_min, intersection_a_max = intersection_a_min[filter], intersection_a_max[filter]
        intersection_b_min, intersection_b_max = intersection_b_min[filter], intersection_b_max[filter]
        angle_difference_b_a_sin_min, angle_difference_b_a_cos_min = angle_difference_b_a_sin_min[filter], angle_difference_b_a_cos_min[filter]
        angle_difference_b_a_sin_max, angle_difference_b_a_cos_max = angle_difference_b_a_sin_max[filter], angle_difference_b_a_cos_max[filter]
        
        return elements_a, intersection_a_min, intersection_a_max, elements_b, intersection_b_min, intersection_b_max, \
               angle_difference_b_a_sin_min, angle_difference_b_a_cos_min, angle_difference_b_a_sin_max, angle_difference_b_a_cos_max
    
    # Parameters
    p95_length=450.
    line_obb_width=0.5
    parallel_max_offset=25.
    parallel_angle_tolerance=0.01
    
    def detect_parallel(self, max_offset=None, angle_tolerance=None) -> Tuple[Tensor, Tensor]:
        F = _DataframeField
        dataframe = self._dataframe
        max_offset = max_offset or self.parallel_max_offset
        angle_tolerance = math.radians(angle_tolerance or self.parallel_angle_tolerance)
        
        # Filter valid line indices
        is_line = dataframe[F.line_flag] == 1
        lines = dataframe[:, is_line]
        
        # Compute OBBs
        lines, obbs = self.create_obbs(elements=lines, width=max_offset, length_extension=self.line_obb_width) # Shape obbs (n_lines, 4, 2)
        
        # Get the pairs of overlapping obbs
        i, j = self.find_overlaping_pairs(lines, obbs)
        lines_a, lines_b = lines[:, i], lines[:, j]
        
        # Compute absolute angle difference in [0, pi]
        angle_difference_b_a = (lines_b[F.angle] - lines_a[F.angle]) % (2*torch.pi)
        min_angle_difference = angle_difference_b_a % torch.pi
        min_angle_difference = torch.minimum(min_angle_difference, torch.pi - min_angle_difference)
        
        # Keep only pairs with angle difference below threshold
        parallel = min_angle_difference <= angle_tolerance
        lines_a, lines_b = lines_a[:, parallel], lines_b[:, parallel]
        angle_difference_b_a = angle_difference_b_a[parallel]
        
        # get the angle difference sin and cos
        angle_difference_b_a_sin = torch.sin(angle_difference_b_a)
        angle_difference_b_a_cos = torch.cos(angle_difference_b_a)
        
        # Compute overlap ratio and perpendicular distance
        overlap_a_b, distance_a_b, overlap_b_a, distance_b_a = self.get_overlap_ratios(lines_a, lines_b)
        
        # Create edges
        Att = _EdgeAttribute
        
        i, j = lines_a[F.original_index].long(), lines_b[F.original_index].long()
        
        edge_pairs = torch.hstack([torch.vstack([i, j]), torch.vstack([j, i])])
        
        n_edges = edge_pairs.size(1)
        edges_i_j = edges_j_i = int(n_edges / 2)
        
        attributes = torch.zeros((n_edges, Att.count()), dtype=torch.float32, device=self._device)
        
        attributes[:, Att.parallel] = 1.0
        
        attributes[:edges_i_j, Att.offset] = distance_a_b
        attributes[:edges_i_j, Att.overlap_ratio] = overlap_a_b
        attributes[:edges_i_j, Att.angle_difference_sin_min] = -angle_difference_b_a_sin
        attributes[:edges_i_j, Att.angle_difference_cos_min] = angle_difference_b_a_cos
        attributes[:edges_i_j, Att.angle_difference_sin_max] = -angle_difference_b_a_sin
        attributes[:edges_i_j, Att.angle_difference_cos_max] = angle_difference_b_a_cos
        
        attributes[edges_j_i:, Att.offset] = distance_b_a
        attributes[edges_j_i:, Att.overlap_ratio] = overlap_b_a
        attributes[edges_j_i:, Att.angle_difference_sin_min] = angle_difference_b_a_sin
        attributes[edges_j_i:, Att.angle_difference_cos_min] = angle_difference_b_a_cos
        attributes[edges_j_i:, Att.angle_difference_sin_max] = angle_difference_b_a_sin
        attributes[edges_j_i:, Att.angle_difference_cos_max] = angle_difference_b_a_cos
        
        return edge_pairs, attributes
     
    def detect_intersection(self, obb_width=None, angle_tolerance=None) -> Tuple[Tensor, Tensor]:
        F = _DataframeField
        dataframe = self._dataframe
        obb_width = obb_width or self.line_obb_width
        angle_tolerance = math.radians(angle_tolerance or self.parallel_angle_tolerance)
        
        # Compute OBBs
        elements, obbs = self.create_obbs(elements=dataframe, width=obb_width, length_extension=obb_width) # Shape obbs (n_lines, 4, 2)
        
        # Get the pairs of overlapping obbs
        i, j = self.find_overlaping_pairs(elements, obbs)
        elements_a, elements_b = elements[:, i], elements[:, j]
        
        # Compute intersection positions
        elements_a, intersection_a_min, intersection_a_max, \
        elements_b, intersection_b_min, intersection_b_max, \
        angle_difference_b_a_sin_min, angle_difference_b_a_cos_min, \
        angle_difference_b_a_sin_max, angle_difference_b_a_cos_max \
        = self.get_intersection_positions(elements_a, elements_b, margin=obb_width, angle_tolerance=angle_tolerance)
        
        # Create edges
        Att = _EdgeAttribute
        
        i, j = elements_a[F.original_index].long(), elements_b[F.original_index].long()
        
        edge_pairs = torch.hstack([torch.vstack([i, j]), torch.vstack([j, i])])
        
        n_edges = edge_pairs.size(1)
        edges_i_j = edges_j_i = int(n_edges / 2)
        
        attributes = torch.zeros((n_edges, Att.count()), dtype=torch.float32, device=self._device)
        
        attributes[:, Att.oblique] = 1.0

        attributes[:edges_i_j, Att.intersection_min] = intersection_a_min
        attributes[:edges_i_j, Att.intersection_max] = intersection_a_max
        attributes[:edges_i_j, Att.angle_difference_sin_min] = -angle_difference_b_a_sin_min
        attributes[:edges_i_j, Att.angle_difference_cos_min] = angle_difference_b_a_cos_min
        attributes[:edges_i_j, Att.angle_difference_sin_max] = -angle_difference_b_a_sin_max
        attributes[:edges_i_j, Att.angle_difference_cos_max] = angle_difference_b_a_cos_max

        attributes[edges_j_i:, Att.intersection_min] = intersection_b_min
        attributes[edges_j_i:, Att.intersection_max] = intersection_b_max
        attributes[edges_j_i:, Att.angle_difference_sin_min] = angle_difference_b_a_sin_min
        attributes[edges_j_i:, Att.angle_difference_cos_min] = angle_difference_b_a_cos_min
        attributes[edges_j_i:, Att.angle_difference_sin_max] = angle_difference_b_a_sin_max
        attributes[edges_j_i:, Att.angle_difference_cos_max] = angle_difference_b_a_cos_max
        
        return edge_pairs, attributes
     
import ezdxf

supported_entities = ('LINE', 'POINT', 'CIRCLE', 'ARC')
supported_layers = {"beam": 0, "column": 1, "eave": 2, "hole_slab": 3, "stair": 4, "section": 5, "info": 6}

def extract_coordinates(dxf_file) -> Tensor:
    
    doc = ezdxf.readfile(dxf_file)
    modelspace = doc.modelspace()
    
    entities = [entity for entity in modelspace if entity.dxftype() in supported_entities]
    
    F = _DataframeField
    dataframe = torch.zeros((F.count(), len(entities)), dtype=torch.float32, device='cpu')
    
    for i, entity in enumerate(entities):
        dataframe[F.original_index,i] = i
        entity_type = entity.dxftype()
        entity_layer = entity.dxf.layer
        
        if entity_layer in supported_layers: dataframe[F.layer,i] = supported_layers[entity_layer]
        else: dataframe[F.layer,i] = -1; print(f"[Unsupported layer] '{entity_layer}' in file '{dxf_file}'")
        
        if entity_type == 'LINE':
            dataframe[F.line_flag,i] = 1
            dataframe[F.start_x,i], dataframe[F.start_y,i], _ = entity.dxf.start
            dataframe[F.end_x,i], dataframe[F.end_y,i], _ = entity.dxf.end
        
        elif entity_type == 'POINT':
            dataframe[F.point_flag,i] = 1
            dataframe[F.start_x,i], dataframe[F.start_y,i], _ = entity.dxf.location
            
        elif entity_type in ('CIRCLE', 'ARC'):
            dataframe[F.circle_flag,i] = 1 if entity_type == 'CIRCLE' else 0
            dataframe[F.arc_flag,i] = 1 if entity_type == 'ARC' else 0
            dataframe[F.radius,i] = entity.dxf.radius
            dataframe[F.center_x,i], dataframe[F.center_y,i], _ = entity.dxf.center
            dataframe[F.start_angle,i] = 0 if entity_type == 'CIRCLE' else entity.dxf.start_angle
            dataframe[F.end_angle,i] = 360 if entity_type == 'CIRCLE' else entity.dxf.end_angle
        
    return dataframe

from torch_geometric.data import Data

def create_graph(dataframe:Tensor, rotation=0., mirror_axis=None) -> Data:
    
    F = _DataframeField
    
    rotation = math.radians(rotation)
    
    # Apply mirror
    if mirror_axis == 'x':
        dataframe[F.start_y], dataframe[F.end_y], dataframe[F.center_y] = -dataframe[F.start_y], -dataframe[F.end_y], -dataframe[F.center_y]
        dataframe[F.start_angle], dataframe[F.end_angle] = (-dataframe[F.end_angle] % 360, -dataframe[F.start_angle] % 360)
    elif mirror_axis == 'y':
        dataframe[F.start_x], dataframe[F.end_x], dataframe[F.center_x] = -dataframe[F.start_x], -dataframe[F.end_x], -dataframe[F.center_x]
        dataframe[F.start_angle], dataframe[F.end_angle] = ((180 - dataframe[F.end_angle]) % 360, (180 - dataframe[F.start_angle]) % 360)
    
    # Rotate coordinates
    rotation_sin, rotation_cos = math.sin(rotation), math.cos(rotation)

    dataframe[F.start_x], dataframe[F.start_y] = (dataframe[F.start_x] * rotation_cos - dataframe[F.start_y] * rotation_sin, \
                                                  dataframe[F.start_x] * rotation_sin + dataframe[F.start_y] * rotation_cos)

    dataframe[F.end_x], dataframe[F.end_y] = (dataframe[F.end_x] * rotation_cos - dataframe[F.end_y] * rotation_sin, \
                                              dataframe[F.end_x] * rotation_sin + dataframe[F.end_y] * rotation_cos)

    dataframe[F.center_x], dataframe[F.center_y] = (dataframe[F.center_x] * rotation_cos - dataframe[F.center_y] * rotation_sin, \
                                                    dataframe[F.center_x] * rotation_sin + dataframe[F.center_y] * rotation_cos)

    # Convert angles to radians and rotate
    dataframe[F.start_angle] = (torch.deg2rad(dataframe[F.start_angle]) + rotation) % (2*torch.pi)
    dataframe[F.end_angle] = (torch.deg2rad(dataframe[F.end_angle]) + rotation) % (2*torch.pi)
    
    # === LINES ===
    is_line = dataframe[F.line_flag] == 1
    
    min_length = 1e-3
    
    dataframe[F.mid_x] = torch.where(is_line, (dataframe[F.start_x] + dataframe[F.end_x]) / 2, dataframe[F.mid_x])
    dataframe[F.mid_y] = torch.where(is_line, (dataframe[F.start_y] + dataframe[F.end_y]) / 2, dataframe[F.mid_y])
    
    dataframe[F.d_x] = torch.where(is_line, dataframe[F.end_x] - dataframe[F.start_x], dataframe[F.d_x])
    dataframe[F.d_y] = torch.where(is_line, dataframe[F.end_y] - dataframe[F.start_y], dataframe[F.d_y])
    dataframe[F.angle] = torch.where(is_line, torch.arctan2(dataframe[F.d_y], dataframe[F.d_x]) % (2*torch.pi), dataframe[F.angle])
    dataframe[F.length] = torch.where(is_line, torch.sqrt(dataframe[F.d_x]**2 + dataframe[F.d_y]**2), dataframe[F.length])
    
    is_line &= (dataframe[F.length] > min_length)
    
    # Unit direction and unit perpendicular vectors
    dataframe[F.u_x] = torch.where(is_line, dataframe[F.d_x] / dataframe[F.length], dataframe[F.u_x])
    dataframe[F.u_y] = torch.where(is_line, dataframe[F.d_y] / dataframe[F.length], dataframe[F.u_y])
    dataframe[F.n_x] = torch.where(is_line, -dataframe[F.u_y], dataframe[F.n_x])
    dataframe[F.n_y] = torch.where(is_line, dataframe[F.u_x], dataframe[F.n_y])
    
    # === POINTS ===
    is_point = dataframe[F.point_flag] == 1
    
    for field in [F.end_x, F.mid_x]: dataframe[field] = torch.where(is_point, dataframe[F.start_x], dataframe[field])
    for field in [F.end_y, F.mid_y]: dataframe[field] = torch.where(is_point, dataframe[F.start_y], dataframe[field])
    
    is_line_point = (dataframe[F.line_flag] == 1) & (dataframe[F.length] <= min_length)
    
    dataframe[F.point_flag] = torch.where(is_line_point, 1, dataframe[F.point_flag])
    dataframe[F.line_flag] = torch.where(is_line_point, 0, dataframe[F.line_flag])
    
    # === CIRCLES AND ARCS ===
    is_circle = dataframe[F.circle_flag] == 1
    is_arc = dataframe[F.arc_flag] == 1
    is_circle_or_arc = is_circle | is_arc
    
    dataframe[F.arc_span] = torch.where(is_circle, 2 * torch.pi, dataframe[F.arc_span])
    dataframe[F.arc_span] = torch.where(is_arc, ((dataframe[F.end_angle] - dataframe[F.start_angle]) % (2*torch.pi)), dataframe[F.arc_span])
    
    mid_angle = torch.where(is_circle_or_arc, (dataframe[F.start_angle] + dataframe[F.arc_span] / 2) % (2*torch.pi), 0)
    
    dataframe[F.start_x] = torch.where(is_circle_or_arc, dataframe[F.center_x] + dataframe[F.radius] * torch.cos(dataframe[F.start_angle]), dataframe[F.start_x])
    dataframe[F.start_y] = torch.where(is_circle_or_arc, dataframe[F.center_y] + dataframe[F.radius] * torch.sin(dataframe[F.start_angle]), dataframe[F.start_y])
    dataframe[F.end_x] = torch.where(is_circle_or_arc, dataframe[F.center_x] + dataframe[F.radius] * torch.cos(dataframe[F.end_angle]), dataframe[F.end_x])
    dataframe[F.end_y] = torch.where(is_circle_or_arc, dataframe[F.center_y] + dataframe[F.radius] * torch.sin(dataframe[F.end_angle]), dataframe[F.end_y])
    dataframe[F.mid_x] = torch.where(is_circle_or_arc, dataframe[F.center_x] + dataframe[F.radius] * torch.cos(mid_angle), dataframe[F.mid_x])
    dataframe[F.mid_y] = torch.where(is_circle_or_arc, dataframe[F.center_y] + dataframe[F.radius] * torch.sin(mid_angle), dataframe[F.mid_y])
    
    dataframe[F.r_x] = torch.where(is_circle_or_arc, (dataframe[F.mid_x] - dataframe[F.center_x]) / dataframe[F.radius], dataframe[F.r_x])
    dataframe[F.r_y] = torch.where(is_circle_or_arc, (dataframe[F.mid_y] - dataframe[F.center_y]) / dataframe[F.radius], dataframe[F.r_y])
    dataframe[F.t_x] = torch.where(is_circle_or_arc, -dataframe[F.r_y], dataframe[F.t_x])
    dataframe[F.t_y] = torch.where(is_circle_or_arc, dataframe[F.r_x], dataframe[F.t_y])
    
    dataframe[F.perimeter] = torch.where(is_circle_or_arc, dataframe[F.radius] * dataframe[F.arc_span], dataframe[F.perimeter])
    
    graph = Graph(dataframe)
    
    return Data(x=graph.node_attributes, edge_index=graph.edge_pairs, edge_attr=graph.edge_attributes, y=dataframe[F.layer].long())

import os
from tqdm import tqdm
from multiprocessing import Pool

def _worker(): torch.set_num_threads(1); os.environ.setdefault("OMP_NUM_THREADS", "1")

class CoordinateDataset(torch.utils.data.Dataset):
    def __init__(self, dxf_files):
        
        self.dataset:List[Tensor] = []
        
        n_cpu = os.cpu_count()
        n_files = len(dxf_files)
        chunksize = max(1, n_files // (n_cpu * 8))
        
        with Pool(processes=n_cpu, initializer=_worker, maxtasksperchild=100) as pool:
            for dataframe in tqdm(pool.imap_unordered(extract_coordinates, dxf_files, chunksize=chunksize), total=n_files, desc="Extracting coordinates"):
                self.dataset.append(dataframe)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

from itertools import product

def _build_graph(arguments:List[Tuple]) -> Data: dataframe, angle, axis = arguments; return create_graph(dataframe, angle, axis)

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, coordinate_dataset, rotations=[0], mirror_axes=[None]):
        
        self.dataset:list[Data] = []
        
        # Build argument list
        arguments = [(dataframe, angle, axis) for dataframe, angle, axis in product(coordinate_dataset, rotations, mirror_axes)]
        
        n_cpu = os.cpu_count()
        n_tasks = len(arguments)
        chunksize = max(1, n_tasks // (n_cpu * 8))
        
        with Pool(processes=n_cpu, initializer=_worker, maxtasksperchild=100) as pool:
            for data in tqdm(pool.imap_unordered(_build_graph, arguments, chunksize=chunksize), total=n_tasks, desc="Building graphs"):
                self.dataset.append(data)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

import torch.nn as nn
from torch_geometric.nn import PNAConv, TransformerConv, GraphNorm

class _GNNLayerBlock(nn.Module):
    def __init__(self, hidden_dimensions:int, dropout:float, conv:nn.Module):
        super().__init__()
        self.conv = conv
        self.graph_norm = GraphNorm(hidden_dimensions)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.graph_norm(x, batch); h = self.conv(h, edge_index, edge_attr); h = self.drop(h)
        return x + h
        
class GraphNeuralNetwork(nn.Module):
    def __init__(self, pna_degree):
        super().__init__()
        
        node_attributes = 9
        edge_attributes = 10
        hidden_dimensions = 128
        hidden_dimensions_edge = 64
        dropout = 0.1
        targets = 7
        heads = 4
        
        # Small MLP to learn embeddings for coordinate features
        self.embedding_layer = nn.Sequential(nn.Linear(node_attributes, hidden_dimensions), nn.ReLU(), nn.LayerNorm(hidden_dimensions))
        
        # Edge encoders (shared or per-block; here per block, small)
        self.edge_layer = nn.Sequential(nn.Linear(edge_attributes, hidden_dimensions_edge), nn.ReLU(), 
                                        nn.Linear(hidden_dimensions_edge, hidden_dimensions_edge), nn.ReLU(), nn.LayerNorm(hidden_dimensions_edge))
        
        # Block 1: Local (PNA) then Global (TransformerConv)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.pna1 = _GNNLayerBlock(hidden_dimensions, dropout,
             PNAConv(in_channels=hidden_dimensions, out_channels=hidden_dimensions,
                     aggregators=aggregators, scalers=scalers, deg=pna_degree, edge_dim=hidden_dimensions_edge))
        
        self.transformer1 = _GNNLayerBlock(hidden_dimensions, dropout,
             TransformerConv(in_channels=hidden_dimensions, out_channels=hidden_dimensions, heads=heads, 
                             concat=False, edge_dim=hidden_dimensions_edge, beta=True, dropout=dropout))
        
        # Block 2: Local + Global again
        self.pna2 = _GNNLayerBlock(hidden_dimensions, dropout,
             PNAConv(in_channels=hidden_dimensions, out_channels=hidden_dimensions,
                     aggregators=aggregators, scalers=scalers, deg=pna_degree, edge_dim=hidden_dimensions_edge))
        
        self.transformer2 = _GNNLayerBlock(hidden_dimensions, dropout,
             TransformerConv(in_channels=hidden_dimensions, out_channels=hidden_dimensions, heads=heads, 
                             concat=False, edge_dim=hidden_dimensions_edge, beta=True, dropout=dropout))
        
        # Head
        self.head = nn.Sequential(nn.LayerNorm(hidden_dimensions), 
                                  nn.Linear(hidden_dimensions, 64), nn.ReLU(), 
                                  nn.Dropout(0.3), nn.Linear(64, targets))
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.embedding_layer(x)
        e = self.edge_layer(edge_attr)
        
        x = self.pna1(x, edge_index, e, batch)
        x = self.transformer1(x, edge_index, e, batch)
        x = self.pna2(x, edge_index, e, batch)
        x = self.transformer2(x, edge_index, e, batch)
        
        return self.head(x)  # [num_nodes_total, classes]
