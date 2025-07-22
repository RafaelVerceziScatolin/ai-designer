
from enum import IntEnum

class _dataframe_field(IntEnum):
    original_index = 0
    line_flag = 1
    circle_flag = 2
    arc_flag = 3
    start_x = 4
    start_y = 5
    end_x = 6
    end_y = 7
    mid_x = 8
    mid_y = 9
    length = perimeter = 10
    angle = 11
    u_x = 12
    u_y = 13
    n_x = 14
    n_y = 15
    center_x = 16
    center_y = 17
    radius = 18
    start_angle = 19
    end_angle = 20
    
    @classmethod
    def count(cls) -> int: return len(cls)

class _edge_attribute(IntEnum):
    parallel = 0
    colinear = 1
    perpendicular_distance = 2
    overlap_ratio = 3
    point_intersection = 4
    segment_intersection = 5
    angle_difference_sin = 6
    perimeter_intersection = 7
    
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
    def create_obbs(lines:Tensor, width:float, length_extension:float) -> Tensor:
        
        F = _dataframe_field
        
        # Half dimensions
        half_width = width / 2
        half_length = (lines[F.length] + length_extension) / 2
        
        # Compute displacements
        dx_length = lines[F.u_x] * half_length
        dy_length = lines[F.u_y] * half_length
        dx_width = lines[F.n_x] * half_width
        dy_width = lines[F.n_y] * half_width
        
        # Corners (4 per OBB)
        corner1_x = lines[F.mid_x] - dx_length - dx_width
        corner1_y = lines[F.mid_y] - dy_length - dy_width

        corner2_x = lines[F.mid_x] + dx_length - dx_width
        corner2_y = lines[F.mid_y] + dy_length - dy_width

        corner3_x = lines[F.mid_x] + dx_length + dx_width
        corner3_y = lines[F.mid_y] + dy_length + dy_width

        corner4_x = lines[F.mid_x] - dx_length + dx_width
        corner4_y = lines[F.mid_y] - dy_length + dy_width
        
        # Stack corners
        obbs = torch.stack([
            torch.stack([corner1_x, corner1_y], dim=1),
            torch.stack([corner2_x, corner2_y], dim=1),
            torch.stack([corner3_x, corner3_y], dim=1),
            torch.stack([corner4_x, corner4_y], dim=1)
        ], dim=1)

        return obbs # Shape (n_lines, 4, 2)
    
    @staticmethod
    def get_overlaping_pairs(lines:Tensor, obbs:Tensor) -> Tuple[Tensor, Tensor]:
        
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
        axes_i = torch.stack([lines[F.u_x, i], lines[F.u_y, i], lines[F.n_x, i], lines[F.n_y, i]], dim=1).reshape(-1,2,2)
        axes_j = torch.stack([lines[F.u_x, j], lines[F.u_y, j], lines[F.n_x, j], lines[F.n_y, j]], dim=1).reshape(-1,2,2)
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
    
    
    
    
    
    
    line_obb_width=0.5
    parallel_angle_tolerance=0.01
    
    def DetectParallel(self, offset=25, angle_tolerance=None, colinear_threshold=0.5):
        F = _dataframe_field
        dataframe = self._dataframe
        angle_tolerance = (angle_tolerance or self.parallel_angle_tolerance) * torch.pi/180
        
        # Filter valid line indices
        is_line = (dataframe[F.line_flag] == 1) & (dataframe[F.length] > 1e-4)
        lines = dataframe[:, is_line]
        
        # Compute OBBs
        obbs = self.create_obbs(lines, width=offset, length_extension=self.line_obb_width) # Shape (n_lines, 4, 2)
        
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
     
    def DetectIntersection (self, obb_width=0.5, angle_tolerance=None):
        F = _dataframe_field
        dataframe = self._dataframe
        angle_tolerance = (angle_tolerance or self.parallel_angle_tolerance) * torch.pi/180
        
        # Filter valid line indices
        is_line = (dataframe[F.line_flag] == 1) & (dataframe[F.length] > 1e-4)
        lines = dataframe[:, is_line]
        
        # Compute OBBs
        obbs = self.create_obbs(lines, width=obb_width, length_extension=True) # Shape (n_lines, 4, 2)
        
        # Get the pairs of overlapping obbs
        i, j = self.get_overlaping_pairs(lines, obbs)
        lines_a, lines_b = lines[:, i], lines[:, j]
        
        # Compute absolute angle difference in [0, pi]
        angle_difference = torch.abs(lines_a[F.angle] - lines_b[F.angle])
        angle_difference = torch.minimum(angle_difference, torch.pi - angle_difference)
        
        # Keep only pairs with angle difference above threshold
        oblique = angle_difference > angle_tolerance
        lines_a, lines_b = lines_a[:, oblique], lines_b[:, oblique]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


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
    
    # Filter lines
    is_line = (dataframe[F.line_flag] == 1)
    
    dataframe[F.mid_x] = torch.where(is_line, (dataframe[F.start_x] + dataframe[F.end_x]) / 2, dataframe[F.mid_x])
    dataframe[F.mid_y] = torch.where(is_line, (dataframe[F.start_y] + dataframe[F.end_y]) / 2, dataframe[F.mid_y])
    
    dx = torch.where(is_line, dataframe[F.end_x] - dataframe[F.start_x], 0)
    dy = torch.where(is_line, dataframe[F.end_y] - dataframe[F.start_y], 0)
    dataframe[F.length] = torch.where(is_line, torch.sqrt(dx**2 + dy**2), dataframe[F.length])
    
    is_line &= (dataframe[F.length] > 1e-4)
    
    dataframe[F.angle] = torch.where(is_line, torch.arctan2(dy, dx) % (torch.pi), dataframe[F.angle])
    
    # Unit direction and unit perpendicular vectors
    dataframe[F.u_x] = torch.where(is_line, dx / dataframe[F.length], dataframe[F.u_x])
    dataframe[F.u_y] = torch.where(is_line, dy / dataframe[F.length], dataframe[F.u_y])
    
    dataframe[F.n_x] = torch.where(is_line, -dataframe[F.u_y], dataframe[F.n_x])
    dataframe[F.n_y] = torch.where(is_line, dataframe[F.u_x], dataframe[F.n_y])
    
    # Compute circle perimeters
    is_circle = (dataframe[F.circle_flag] == 1)
    dataframe[F.perimeter] = torch.where(is_circle, 2 * torch.pi * dataframe[F.radius], dataframe[F.perimeter])
    
    # Compute arc lengths
    is_arc = (dataframe[F.arc_flag] == 1)
    arc_angle = torch.where(is_arc, ((dataframe[F.end_angle] - dataframe[F.start_angle]) % 360), 0)
    dataframe[F.perimeter] = torch.where(is_arc, dataframe[F.radius] * arc_angle * torch.pi/180, dataframe[F.perimeter])
    
    graph = Graph(dataframe)
    
    