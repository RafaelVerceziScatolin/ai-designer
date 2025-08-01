
from enum import IntEnum

class _dataframe_field(IntEnum):
    original_index = 0
    line_flag = 1
    circle_flag = 2
    arc_flag = 3
    point_flag = 4
    start_x = 5
    start_y = 6
    end_x = 7
    end_y = 8
    mid_x = 9
    mid_y = 10
    length = perimeter = 11
    angle = 12
    u_x = 13
    u_y = 14
    n_x = 15
    n_y = 16
    center_x = 17
    center_y = 18
    radius = 19
    start_angle = 20
    end_angle = 21
    arc_span = 22
    
    @classmethod
    def count(cls) -> int: return len(cls)

class _edge_attribute(IntEnum):
    parallel = 0
    colinear = 1
    perpendicular_distance = 2
    overlap_ratio = 3
    oblique = 4
    intersection_position = 5
    angle_difference_sin = 6
    angle_difference_cos = 7
    
    @classmethod
    def count(cls) -> int: return len(cls)

from typing import Tuple
import torch
from torch import Tensor
from torch import newaxis

class Graph:
    def __init__(self, dataframe:Tensor):
        
        F = _dataframe_field
        
        start_x, start_y = dataframe[F.start_x], dataframe[F.start_y]
        end_x, end_y = dataframe[F.end_x], dataframe[F.end_y]
        length = dataframe[F.length]
        angle = dataframe[F.angle]
        
        # Normalize coordinates
        coordinates_x = torch.hstack([start_x, end_x])
        coordinates_y = torch.hstack([start_y, end_y])
        coordinates_x_mean, coordinates_x_std = coordinates_x.mean(), coordinates_x.std()
        coordinates_y_mean, coordinates_y_std = coordinates_y.mean(), coordinates_y.std()
        
        normalized_start_x = (start_x - coordinates_x_mean) / coordinates_x_std
        normalized_start_y = (start_y - coordinates_y_mean) / coordinates_y_std
        normalized_end_x = (end_x - coordinates_x_mean) / coordinates_x_std
        normalized_end_y = (end_y - coordinates_y_mean) / coordinates_y_std
        
        normalized_coordinates = torch.stack([normalized_start_x, normalized_start_y,
                                              normalized_end_x, normalized_end_y], dim=1) # shape (N, 4) for ML model
        
        # Normalize angles and lengths
        normalized_angle = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1) # shape (N, 2)
        normalized_length = (length / length.max()).reshape([-1, 1]) # shape (N, 1)
        
        # Flags
        line_flag = dataframe[F.line_flag].reshape([-1, 1]) # shape (N, 1)
        circle_flag = dataframe[F.circle_flag].reshape([-1, 1]) # shape (N, 1)
        arc_flag = dataframe[F.arc_flag].reshape([-1, 1]) # shape (N, 1)
        
        # Node attributes
        self.nodeAttributes = torch.hstack([normalized_coordinates, normalized_angle,
                                            normalized_length, line_flag, circle_flag, arc_flag])
        
        self._dataframe = dataframe
    
    @staticmethod
    def create_obbs(elements:Tensor, width:float, length_extension:float=0.0) -> Tuple[Tensor, Tensor]:
        
        F = _dataframe_field
        n_elements = elements.size(1)
        
        # Filter supported elements
        is_line = elements[F.line_flag] == 1
        is_circle = elements[F.circle_flag] == 1
        is_arc = elements[F.arc_flag] == 1
        
        filter = is_line | is_circle | is_arc
        
        obbs = torch.empty((n_elements, 4, 2), dtype=torch.float32, device='cuda')
        
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
        
        if is_circle.any():
            # === CIRCLE ELEMENTS ===
            circles = elements[:, is_circle]
            
            radius = circles[F.radius] + width
            
            min_x = circles[F.center_x] - radius
            min_y = circles[F.center_y] - radius
            max_x = circles[F.center_x] + radius
            max_y = circles[F.center_y] + radius
            
            corner1 = torch.stack([min_x, min_y], dim=1)
            corner2 = torch.stack([max_x, min_y], dim=1)
            corner3 = torch.stack([max_x, max_y], dim=1)
            corner4 = torch.stack([min_x, max_y], dim=1)
            
            obbs[is_circle] = torch.stack([corner1, corner2, corner3, corner4], dim=1)
        
        if is_arc.any():
            # === ARC ELEMENTS ===
            arcs = elements[:, is_arc]
            
            margin = width / 2
            
            u_x, u_y, n_x, n_y = arcs[F.u_x], arcs[F.u_y], arcs[F.n_x], arcs[F.n_y]
            arc_span, radius = arcs[F.arc_span], arcs[F.radius]
            
            dx_length = torch.where(arc_span < torch.pi, n_x * (radius * torch.sin(arc_span/2) + margin), n_x * (radius + margin))
            dy_length = torch.where(arc_span < torch.pi, n_y * (radius * torch.sin(arc_span/2) + margin), n_y * (radius + margin))
            
            dx_width = u_x * (radius * (1 - torch.cos(arc_span/2)) + margin)
            dy_width = u_y * (radius * (1 - torch.cos(arc_span/2)) + margin)
            
            dx_margin = u_x * margin
            dy_margin = u_y * margin
            
            corner1 = torch.stack([arcs[F.mid_x] - dx_length + dx_margin,
                                   arcs[F.mid_y] - dy_length + dy_margin], dim=1)
            
            corner2 = torch.stack([arcs[F.mid_x] + dx_length + dx_margin,
                                   arcs[F.mid_y] + dy_length + dy_margin], dim=1)
            
            corner3 = torch.stack([arcs[F.mid_x] + dx_length - dx_width,
                                   arcs[F.mid_y] + dy_length - dy_width], dim=1)
            
            corner4 = torch.stack([arcs[F.mid_x] - dx_length - dx_width,
                                   arcs[F.mid_y] - dy_length - dy_width], dim=1)
            
            obbs[is_arc] = torch.stack([corner1, corner2, corner3, corner4], dim=1)
        
        elements = elements[:, filter]
        obbs = obbs[filter]
        
        return elements, obbs # Shape obbs (n_elements, 4, 2)
    
    @staticmethod
    def get_overlaping_pairs(elements:Tensor, obbs:Tensor) -> Tuple[Tensor, Tensor]:
        
        F = _dataframe_field
        
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
    def get_overlap_ratios(lines_a:Tensor, lines_b:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        
        F = _dataframe_field
        
        # Gather line info
        start_xa, start_ya = lines_a[F.start_x], lines_a[F.start_y]
        u_xa, u_ya = lines_a[F.u_x], lines_a[F.u_y]
        length_a = lines_a[F.length]

        start_xb, start_yb = lines_b[F.start_x], lines_b[F.start_y]
        end_xb, end_yb = lines_b[F.end_x], lines_b[F.end_y]
        mid_xb, mid_yb = lines_b[F.mid_x], lines_b[F.mid_y]
        u_xb, u_yb = lines_b[F.u_x], lines_b[F.u_y]
        length_b = lines_b[F.length]
        
        # Project B's endpoints onto A
        t1 = (start_xb - start_xa) * u_xa + (start_yb - start_ya) * u_ya
        t2 = (end_xb - start_xa) * u_xa + (end_yb - start_ya) * u_ya

        # Get the projected range
        tmin, tmax = torch.minimum(t1, t2), torch.maximum(t1, t2)
        
        # Clip the projection range within [0, length_a]
        overlap_start = torch.clamp(tmin, min=0)
        overlap_end = torch.clamp(tmax, max=length_a)
        overlap_mid = (overlap_start + overlap_end) / 2
        
        # Overlap ratio
        overlap_length = torch.clamp(overlap_end - overlap_start, min=0)
        overlap_a_b = overlap_length / length_a
        overlap_b_a = overlap_length / length_b
        
        # Midpoint of projected overlap on line A
        overlap_mid_xa = start_xa + u_xa * overlap_mid
        overlap_mid_ya = start_ya + u_ya * overlap_mid
        
        # Perpendicular distance to line B
        dx = overlap_mid_xa - mid_xb
        dy = overlap_mid_ya - mid_yb
        
        distance = (dx * -u_yb + dy * u_xb).abs() # From overlaping segment midpoint on line_a to line_b
        
        return overlap_a_b, overlap_b_a, distance # Each: shape (n_pairs,)
    
    def _get_line_circle_intersections(lines:Tensor, circles:Tensor, margin:float):
        
        F = _dataframe_field
        
        dx_start = lines[F.start_x] - circles[F.center_x]
        dy_start = lines[F.start_y] - circles[F.center_y]
        dx_end = lines[F.end_x] - circles[F.center_x]
        dy_end = lines[F.end_y] - circles[F.center_y]
        dx_mid = lines[F.mid_x] - circles[F.center_x]
        dy_mid = lines[F.mid_y] - circles[F.center_y]
        
        distance_start = torch.sqrt(dx_start**2 + dy_start**2)
        distance_end = torch.sqrt(dx_end**2 + dy_end**2)
        distance_mid = torch.sqrt(dx_mid**2 + dy_mid**2)
        
        inner_radius = circles[F.radius] - margin
        outer_radius = circles[F.radius] + margin
        
        # Filter pairs in which the lines does't touch the circle perimeter
        remove= ~(((distance_start < inner_radius) & (distance_end < inner_radius)) | (
                (distance_start > outer_radius) & (distance_end > outer_radius) & (distance_mid > outer_radius)))
        
        circles, lines = circles[:, remove], lines[:, remove]
    
    
    
    
    
    @staticmethod
    def get_intersection_positions(elements_a:Tensor, elements_b:Tensor):
        
        F = _dataframe_field
        n_elements = elements_a.size(1)
        
        # Filter supported pairs
        is_line_a, is_line_b = elements_a[F.line_flag] == 1, elements_b[F.line_flag] == 1
        is_circle_a, is_circle_b = elements_a[F.circle_flag] == 1, elements_b[F.circle_flag] == 1
        is_arc_a, is_arc_b = elements_a[F.arc_flag] == 1, elements_b[F.arc_flag] == 1
        
        line_line_pair = is_line_a & is_line_b
        line_circle_pair, circle_line_pair = is_line_a & is_circle_b, is_circle_a & is_line_b
        
        filter = line_line_pair | (line_circle_pair | circle_line_pair)
        
        min_intersection_a = torch.empty(n_elements, dtype=torch.float32, device='cuda')
        max_intersection_a = torch.empty(n_elements, dtype=torch.float32, device='cuda')
        min_intersection_b = torch.empty(n_elements, dtype=torch.float32, device='cuda')
        max_intersection_b = torch.empty(n_elements, dtype=torch.float32, device='cuda')
        
        if line_line_pair.any():
            lines_a = elements_a[:, line_line_pair]
            lines_b = elements_b[:, line_line_pair]
            
            # Gather line info
            start_xa, start_ya = lines_a[F.start_x], lines_a[F.start_y]
            u_xa, u_ya = lines_a[F.u_x], lines_a[F.u_y]
            length_a = lines_a[F.length]

            start_xb, start_yb = lines_b[F.start_x], lines_b[F.start_y]
            u_xb, u_yb = lines_b[F.u_x], lines_b[F.u_y]
            length_b = lines_b[F.length]

            # Vector from B start to A start
            w_x = start_xa - start_xb
            w_y = start_ya - start_yb

            # Core dot products
            b = u_xa * u_xb + u_ya * u_yb
            d = u_xa * w_x + u_ya * w_y
            e = u_xb * w_x + u_yb * w_y

            denominator = 1 - b * b

            # Get closest point parameters along each line
            t = (b * e - d) / denominator  # Along A
            s = (e - b * d) / denominator  # Along B

            # Convert to relative position from center [-1, 1]
            min_intersection_a[line_line_pair] = max_intersection_a[line_line_pair] = ((t / length_a).clamp(0, 1)) * 2 - 1
            min_intersection_b[line_line_pair] = max_intersection_b[line_line_pair] = ((s / length_b).clamp(0, 1)) * 2 - 1
        
        if (line_circle_pair | circle_line_pair).any():
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return intersection_a, intersection_b
    
    
    
    line_obb_width=0.5
    parallel_angle_tolerance=0.01
    
    def DetectParallel(self, offset=25, angle_tolerance=None, colinear_threshold=0.5):
        F = _dataframe_field
        dataframe = self._dataframe
        angle_tolerance = (angle_tolerance or self.parallel_angle_tolerance) * torch.pi/180
        
        # Filter valid line indices
        is_line = dataframe[F.line_flag] == 1
        lines = dataframe[:, is_line]
        
        # Compute OBBs
        lines, obbs = self.create_obbs(elements=lines, width=offset, length_extension=self.line_obb_width) # Shape obbs (n_lines, 4, 2)
        
        # Get the pairs of overlapping obbs
        i, j = self.get_overlaping_pairs(lines, obbs)
        lines_a, lines_b = lines[:, i], lines[:, j]
        
        # Compute absolute angle difference in [0, pi]
        angle_difference = torch.abs(lines_a[F.angle] - lines_b[F.angle])
        angle_difference = torch.minimum(angle_difference, torch.pi - angle_difference)
        
        # Keep only pairs with angle difference below threshold
        parallel = angle_difference <= angle_tolerance
        lines_a, lines_b = lines_a[:, parallel], lines_b[:, parallel]
        
        # Compute overlap ratio and perpendicular distance
        overlap_a_b, overlap_b_a, distance = self.get_overlap_ratios(lines_a, lines_b)
        distance = torch.hstack([distance, distance]) # For both edges_i_j and edges_j_i
        
        # Create edges
        Att = _edge_attribute
        
        i, j = i[parallel], j[parallel]
        i, j = lines[F.original_index][i], lines[F.original_index][j]
        
        edge_pairs = torch.hstack([torch.vstack([i, j]), torch.vstack([j, i])])
        
        edges_i_j = edges_j_i = int(edge_pairs.size(1) / 2)
        
        attributes = torch.zeros((edges_i_j + edges_j_i, Att.count()), dtype=torch.float32, device='cuda')
        
        attributes[:, Att.parallel] = 1.0
        attributes[:, Att.colinear] = torch.where(distance < colinear_threshold, 1, 0)
        attributes[:, Att.perpendicular_distance] = distance
        attributes[:edges_i_j, Att.overlap_ratio] = overlap_a_b
        attributes[edges_j_i:, Att.overlap_ratio] = overlap_b_a
        
        return edge_pairs, attributes
     
    def DetectIntersection(self, obb_width=None, angle_tolerance=None):
        F = _dataframe_field
        dataframe = self._dataframe
        obb_width = obb_width or self.line_obb_width
        angle_tolerance = (angle_tolerance or self.parallel_angle_tolerance) * torch.pi/180
        
        # Compute OBBs
        elements, obbs = self.create_obbs(elements=dataframe, width=obb_width, length_extension=obb_width) # Shape obbs (n_lines, 4, 2)
        
        # Get the pairs of overlapping obbs
        i, j = self.get_overlaping_pairs(elements, obbs)
        elements_a, elements_b = elements[:, i], elements[:, j]
        
        # Compute absolute angle difference for line-line pairs in [0, pi]
        line_line_pair = (elements_a[F.line_flag] == 1) & (elements_b[F.line_flag] == 1)
        angle_difference = torch.where(line_line_pair, torch.abs(elements_a[F.angle] - elements_b[F.angle]), 0)
        angle_difference = torch.where(line_line_pair, torch.minimum(angle_difference, torch.pi - angle_difference), angle_difference)
        
        # Keep only pairs with angle difference above threshold
        oblique = torch.where(line_line_pair, angle_difference > angle_tolerance, True)
        elements_a, elements_b = elements_a[:, oblique], elements_b[:, oblique]
        
        # Compute intersection positions
        intersection_a, intersection_b = self.get_intersection_positions(elements_a, elements_b)
        
        
        
        
        
        
        
        
        
        
        # Compute absolute angle difference in [0, pi]
        angle_difference = torch.abs(lines_a[F.angle] - lines_b[F.angle])
        angle_difference = torch.minimum(angle_difference, torch.pi - angle_difference)
        
        # Keep only pairs with angle difference above threshold
        oblique = angle_difference > angle_tolerance
        lines_a, lines_b = lines_a[:, oblique], lines_b[:, oblique]
        
        # Compute intersection positions
        intersection_a, intersection_b = self.get_intersection_positions(lines_a, lines_b)
        
        # Create edges
        Att = _edge_attribute
        
        i, j = i[oblique], j[oblique]
        i, j = lines[F.original_index][i], lines[F.original_index][j]
        
        edge_pairs = torch.hstack([torch.vstack([i, j]), torch.vstack([j, i])])
        
        edges_i_j = edges_j_i = int(edge_pairs.size(1) / 2)
        
        attributes = torch.zeros((edges_i_j + edges_j_i, Att.count()), dtype=torch.float32, device='cuda')
        
        attributes[:, Att.oblique] = 1.0
        attributes[:edges_i_j, Att.intersection_position] = intersection_a
        attributes[edges_j_i:, Att.intersection_position] = intersection_b
        
        
        


import ezdxf

def CreateGraph(dxf_file):
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    
    entities = [entity for entity in modelSpace if entity.dxftype() in ('LINE', 'CIRCLE', 'ARC')]
    
    F = _dataframe_field
    dataframe = torch.zeros((F.count(), len(entities)), dtype=torch.float32, device='cuda')
    
    for i, entity in enumerate(entities):
        dataframe[F.original_index,i] = i
        entity_type = entity.dxftype()
        
        if entity_type == 'LINE':
            dataframe[F.line_flag,i] = 1
            dataframe[F.start_x,i], dataframe[F.start_y,i], _ = entity.dxf.start
            dataframe[F.end_x,i], dataframe[F.end_y,i], _ = entity.dxf.end
        
        elif entity_type == 'CIRCLE':
            dataframe[F.circle_flag,i] = 1
            dataframe[F.radius,i] = entity.dxf.radius
            dataframe[F.center_x,i], dataframe[F.center_y,i], _ = entity.dxf.center
        
        elif entity_type == 'ARC':
            dataframe[F.arc_flag,i] = 1
            dataframe[F.radius,i] = entity.dxf.radius
            dataframe[F.center_x,i], dataframe[F.center_y,i], _ = entity.dxf.center
            dataframe[F.start_angle,i] = entity.dxf.start_angle
            dataframe[F.end_angle,i] = entity.dxf.end_angle
    
    # === LINES ===
    is_line = dataframe[F.line_flag] == 1
    
    dataframe[F.mid_x] = torch.where(is_line, (dataframe[F.start_x] + dataframe[F.end_x]) / 2, dataframe[F.mid_x])
    dataframe[F.mid_y] = torch.where(is_line, (dataframe[F.start_y] + dataframe[F.end_y]) / 2, dataframe[F.mid_y])
    
    dx = torch.where(is_line, dataframe[F.end_x] - dataframe[F.start_x], 0)
    dy = torch.where(is_line, dataframe[F.end_y] - dataframe[F.start_y], 0)
    dataframe[F.length] = torch.where(is_line, torch.sqrt(dx**2 + dy**2), dataframe[F.length])
    
    is_point = is_line & (dataframe[F.length] <= 1e-3)
    
    is_line &= ~is_point
    
    dataframe[F.angle] = torch.where(is_line, torch.arctan2(dy, dx) % torch.pi, dataframe[F.angle])
    
    # Unit direction and unit perpendicular vectors
    dataframe[F.u_x] = torch.where(is_line, dx / dataframe[F.length], dataframe[F.u_x])
    dataframe[F.u_y] = torch.where(is_line, dy / dataframe[F.length], dataframe[F.u_y])
    dataframe[F.n_x] = torch.where(is_line, -dataframe[F.u_y], dataframe[F.n_x])
    dataframe[F.n_y] = torch.where(is_line, dataframe[F.u_x], dataframe[F.n_y])
    
    # === POINTS ===
    dataframe[F.point_flag] = torch.where(is_point, 1, dataframe[F.point_flag])
    dataframe[F.line_flag] = torch.where(is_point, 0, dataframe[F.line_flag])
    
    # === CIRCLES ===
    is_circle = dataframe[F.circle_flag] == 1
    
    dataframe[F.u_x] = torch.where(is_circle, 1, dataframe[F.u_x])
    dataframe[F.u_y] = torch.where(is_circle, 0, dataframe[F.u_y])
    dataframe[F.n_x] = torch.where(is_circle, 0, dataframe[F.n_x])
    dataframe[F.n_y] = torch.where(is_circle, 1, dataframe[F.n_y])
    
    dataframe[F.perimeter] = torch.where(is_circle, 2 * torch.pi * dataframe[F.radius], dataframe[F.perimeter])
    
    # === ARCS ===
    is_arc = dataframe[F.arc_flag] == 1
    
    dataframe[F.start_angle] = torch.where(is_arc, torch.deg2rad(dataframe[F.start_angle]), dataframe[F.start_angle])
    dataframe[F.end_angle] = torch.where(is_arc, torch.deg2rad(dataframe[F.end_angle]), dataframe[F.end_angle])
    
    dataframe[F.arc_span] = torch.where(is_arc, ((dataframe[F.end_angle] - dataframe[F.start_angle]) % (2*torch.pi)), 0)
    mid_angle = torch.where(is_arc, (dataframe[F.start_angle] + dataframe[F.arc_span] / 2) % (2*torch.pi), 0)
    
    dataframe[F.start_x] = torch.where(is_arc, dataframe[F.center_x] + dataframe[F.radius] * torch.cos(dataframe[F.start_angle]), dataframe[F.start_x])
    dataframe[F.start_y] = torch.where(is_arc, dataframe[F.center_y] + dataframe[F.radius] * torch.sin(dataframe[F.start_angle]), dataframe[F.start_y])
    dataframe[F.end_x] = torch.where(is_arc, dataframe[F.center_x] + dataframe[F.radius] * torch.cos(dataframe[F.end_angle]), dataframe[F.end_x])
    dataframe[F.end_y] = torch.where(is_arc, dataframe[F.center_y] + dataframe[F.radius] * torch.sin(dataframe[F.end_angle]), dataframe[F.end_y])
    dataframe[F.mid_x] = torch.where(is_arc, dataframe[F.center_x] + dataframe[F.radius] * torch.cos(mid_angle), dataframe[F.mid_x])
    dataframe[F.mid_y] = torch.where(is_arc, dataframe[F.center_y] + dataframe[F.radius] * torch.sin(mid_angle), dataframe[F.mid_y])
    
    dataframe[F.u_x] = torch.where(is_arc, (dataframe[F.mid_x] - dataframe[F.center_x]) / dataframe[F.radius], dataframe[F.u_x])
    dataframe[F.u_y] = torch.where(is_arc, (dataframe[F.mid_y] - dataframe[F.center_y]) / dataframe[F.radius], dataframe[F.u_y])
    dataframe[F.n_x] = torch.where(is_arc, -dataframe[F.u_y], dataframe[F.n_x])
    dataframe[F.n_y] = torch.where(is_arc, dataframe[F.u_x], dataframe[F.n_y])
    
    dataframe[F.perimeter] = torch.where(is_arc, dataframe[F.radius] * dataframe[F.arc_span], dataframe[F.perimeter])
    
    graph = Graph(dataframe)
    