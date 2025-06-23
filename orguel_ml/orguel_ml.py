
from enum import IntEnum

class _dataframe_field(IntEnum):
    index = 0
    start_x = 1
    start_y = 2
    end_x = 3
    end_y = 4
    length = 5
    angle = 6
    ux = 7
    uy = 8
    nx = 9
    ny = 10
    circle_flag = 11
    arc_flag = 12
    radius = 13
    start_angle = 14
    end_angle = 15
    
    @classmethod
    def count(cls) -> int: return len(cls)

class _edge_attribute(IntEnum):
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
    
    @classmethod
    def count(cls) -> int: return len(cls)

import torch
from torch import newaxis

class Graph:
    def __init__(self, dataframe:torch.Tensor):
        
        F = _dataframe_field
        
        start_x, start_y = dataframe[F.start_x], dataframe[F.start_y]
        end_x, end_y = dataframe[F.end_x], dataframe[F.end_y]
        length = dataframe[F.length]
        angle = dataframe[F.angle]
        
        # Normalize coordinates
        coordinates_x = torch.hstack([start_x, end_x])
        coordinates_y = torch.hstack([start_y, end_y])
        
        normalized_start_x = (start_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_start_y = (start_y - coordinates_y.mean()) / coordinates_y.std()
        normalized_end_x = (end_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_end_y = (end_y - coordinates_y.mean()) / coordinates_y.std()
        
        normalized_coordinates = torch.stack([normalized_start_x, normalized_start_y,
                                              normalized_end_x, normalized_end_y], dim=1) # shape (N, 4) for ML model
        
        # Normalize angles and lengths
        normalized_angle = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1) # shape (N, 2)
        normalized_length = (length / length.max()).reshape([-1, 1]) # shape (N, 1)
        
        # Flags
        circle_flag = dataframe[F.circle_flag].reshape([-1, 1]) # shape (N, 1)
        arc_flag = dataframe[F.arc_flag].reshape([-1, 1]) # shape (N, 1)
        
        # Node attributes
        self.nodeAttributes = torch.hstack([normalized_coordinates, normalized_angle,
                                            normalized_length, circle_flag, arc_flag])
        
        self._dataframe = dataframe
    
    @staticmethod
    def create_obb(lines:torch.Tensor, width:float, length_extension=True) -> torch.Tensor:
        
        F = _dataframe_field
        
        # Half dimensions
        half_width = width / 2
        half_length = (lines[F.length] / 2)
        if length_extension: half_length += half_width
        
        # Midpoints of the lines
        mid_x = (lines[F.start_x] + lines[F.end_x]) / 2
        mid_y = (lines[F.start_y] + lines[F.end_y]) / 2
        
        # Compute displacements
        dx_length = lines[F.ux] * half_length
        dy_length = lines[F.uy] * half_length
        dx_width = lines[F.nx] * half_width
        dy_width = lines[F.ny] * half_width
        
        # Corners (4 per OBB)
        corner1_x = mid_x - dx_length - dx_width
        corner1_y = mid_y - dy_length - dy_width

        corner2_x = mid_x + dx_length - dx_width
        corner2_y = mid_y + dy_length - dy_width

        corner3_x = mid_x + dx_length + dx_width
        corner3_y = mid_y + dy_length + dy_width

        corner4_x = mid_x - dx_length + dx_width
        corner4_y = mid_y - dy_length + dy_width
        
        # Stack corners
        obbs = torch.stack([
            torch.stack([corner1_x, corner1_y], dim=1),
            torch.stack([corner2_x, corner2_y], dim=1),
            torch.stack([corner3_x, corner3_y], dim=1),
            torch.stack([corner4_x, corner4_y], dim=1)
        ], dim=1)

        return obbs # Shape (n_lines, 4, 2)
    
    @staticmethod
    def check_overlap_sat(lines:torch.Tensor, obbs:torch.Tensor) -> torch.Tensor:
        
        F = _dataframe_field
        n_lines = obbs.shape[0]
        
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
        axes_i = torch.stack([lines[F.ux, i], lines[F.uy, i], lines[F.nx, i], lines[F.ny, i]], dim=1).reshape(-1,2,2)
        axes_j = torch.stack([lines[F.ux, j], lines[F.uy, j], lines[F.nx, j], lines[F.ny, j]], dim=1).reshape(-1,2,2)
        axes = torch.cat([axes_i, axes_j], dim=1) # shape (n_pairs, 4, 2)
        
        # Project corners onto axes
        projections_i = torch.einsum('nij,nkj->nik', axes, obbs_i) # Shape (n_pairs, 4, 4)
        projections_j = torch.einsum('nij,nkj->nik', axes, obbs_j)
        
        # Interval comparisons on each axis
        min_i = projections_i.min(dim=2).values
        max_i = projections_i.max(dim=2).values
        min_j = projections_j.min(dim=2).values
        max_j = projections_j.max(dim=2).values
        
        separating_axis = (max_i < min_j) | (max_j < min_i) # True if a separating axis exists
        overlap_flags = ~torch.any(separating_axis, dim=1) # True if overlap
        
        # Create overlap matrix
        obb_overlap_matrix = torch.zeros((n_lines, n_lines), dtype=torch.bool, device='cuda')
        obb_overlap_matrix[i, j] = overlap_flags
        obb_overlap_matrix[j, i] = overlap_flags # symmetric
        
        return obb_overlap_matrix
    
    @staticmethod
    def overlap_ratios(lines_a:torch.Tensor, lines_b:torch.Tensor) -> torch.Tensor:
        
        F = _dataframe_field
        
        # Gather line info
        start_xa, start_ya = lines_a[F.start_x], lines_a[F.start_y]
        end_xa, end_ya = lines_a[F.end_x], lines_a[F.end_y]
        ux_a, uy_a = lines_a[F.ux], lines_a[F.uy]
        length_a = lines_a[F.length]
        
        start_xb, start_yb = lines_b[F.start_x], lines_b[F.start_y]
        end_xb, end_yb = lines_b[F.end_x], lines_b[F.end_y]
        ux_b, uy_b = lines_b[F.ux], lines_b[F.uy]
        length_b = lines_b[F.length]
        
        # Project B's endpoints onto A
        tb1 = (start_xb - start_xa) * ux_a + (start_yb - start_ya) * uy_a
        tb2 = (end_xb - start_xa) * ux_a + (end_yb - start_ya) * uy_a
        
        # Get the projected range
        tmin, tmax = torch.minimum(tb1, tb2), torch.maximum(tb1, tb2)
        
        # Clip the projection range within [0, length_a]
        overlap_start_a = torch.clamp(tmin, min=0)
        overlap_end_a = torch.clamp(tmax, max=length_a)
        overlap_length_a = torch.clamp(overlap_end_a - overlap_start_a, min=0)
        overlap_a_b = overlap_length_a / length_a
        
        # Repeat for B
        ta1 = (start_xa - start_xb) * ux_b + (start_ya - start_yb) * uy_b
        ta2 = (end_xa - start_xb) * ux_b + (end_ya - start_yb) * uy_b
        
        smin, smax = torch.minimum(ta1, ta2), torch.maximum(ta1, ta2)
        
        overlap_start_b = torch.clamp(smin, min=0)
        overlap_end_b = torch.clamp(smax, max=length_b)
        overlap_length_b = torch.clamp(overlap_end_b - overlap_start_b, min=0)
        overlap_b_a = overlap_length_b / length_b
        
        return overlap_a_b, overlap_b_a # Each: shape (n_pairs,)
        
    
    
    
    
    
    
    
    
    
    
    def ParallelDetection(self, offset=25, angle_tolerance=0.01):
        F = _dataframe_field
        dataframe = self._dataframe
        angle_tolerance *= torch.pi/180
        
        # Filter valid line indices
        is_line = (dataframe[F.circle_flag] == 0) & (dataframe[F.arc_flag] == 0) & (dataframe[F.length] > 1e-4)
        lines = dataframe[:, is_line]
        lines[F.angle] = lines[F.angle] % torch.pi # collapse symmetrical directions
        
        # Compute OBBs
        obbs = self.create_obb(lines, width=offset, length_extension=False) # Shape (n_lines, 4, 2)
        
        # SAT check
        obb_overlap_matrix = self.check_overlap_sat(lines, obbs) # Shape (n_lines, n_lines)
        
        # Get the pairs of overlapping obbs (upper triangle, no self-pairs)
        i, j = torch.triu_indices(*obb_overlap_matrix.shape, offset=1, device='cuda')
        
        obb_overlap = obb_overlap_matrix[i, j] # True if overlap
        i, j = i[obb_overlap], j[obb_overlap]
        
        # Compute absolute angle difference in [0, pi]
        angle_difference = torch.abs(lines[F.angle, i] - lines[F.angle, j])
        angle_difference = torch.minimum(angle_difference, torch.pi - angle_difference)
        
        # Keep only pairs with angle difference below threshold
        parallel = angle_difference < angle_tolerance
        i, j = i[parallel], j[parallel]
        
        lines_a, lines_b = lines[:, i], lines[:, j]
        
        # Compute overlap ratio
        overlap_a_b, overlap_b_a = self.overlap_ratios(lines_a, lines_b)
        
        
        
        
        
        
        
        
        
        
        
        # Extract coordinates
        start_x = lines[start_x]
        start_y = lines[start_y]
        end_x = lines[end_x]
        end_y = lines[end_y]
        
        # Compute OBBs
        obbs = self.create_obb(start_x, start_y, end_x, end_y, width=offset, length_extension=False)
        
        # SAT check
        overlap = self.check_overlap_sat(obbs, obbs)
        
        
        
        
        
        
        
        
        
        offset_tolerance = 25
        angle_tolerance = 0.25 * torch.pi/180
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


import ezdxf

def CreateGraph(dxf_file):
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    
    entities = [entity for entity in modelSpace if entity.dxftype() in ('LINE', 'CIRCLE', 'ARC')]
    
    F = _dataframe_field
    dataframe = torch.zeros((F.count(), len(entities)), dtype=torch.float32, device='cuda')
    
    for i, entity in enumerate(entities):
        dataframe[F.index,i] = i
        entity_type = entity.dxftype()
        
        if entity_type == 'LINE':
            dataframe[F.start_x,i], dataframe[F.start_y,i], _ = entity.dxf.start
            dataframe[F.end_x,i], dataframe[F.end_y,i], _ = entity.dxf.end
            
        elif entity_type == 'CIRCLE':
            dataframe[F.circle_flag,i] = 1
            dataframe[F.radius,i] = entity.dxf.radius
            dataframe[F.start_x,i], dataframe[F.start_y,i], _ = entity.dxf.center
            
        elif entity_type == 'ARC':
            dataframe[F.arc_flag,i] = 1
            dataframe[F.radius,i] = entity.dxf.radius
            dataframe[F.start_x,i], dataframe[F.start_y,i], _ = entity.dxf.center
            dataframe[F.start_angle,i] = entity.dxf.start_angle
            dataframe[F.end_angle,i] = entity.dxf.end_angle
    
    # Filter lines
    is_line = (dataframe[F.circle_flag] == 0) & (dataframe[F.arc_flag] == 0) & (dataframe[F.length] > 1e-4)
    
    # Unit direction and unit perpendicular vectors
    dx = torch.where(is_line, dataframe[F.end_x] - dataframe[F.start_x], 0)
    dy = torch.where(is_line, dataframe[F.end_y] - dataframe[F.start_y], 0)
    
    dataframe[F.length] = torch.where(is_line, torch.sqrt(dx**2 + dy**2), dataframe[F.length])
    dataframe[F.angle] = torch.where(is_line, torch.arctan2(dy, dx) % (2*torch.pi), dataframe[F.angle])
    
    dataframe[F.ux] = torch.where(is_line, dx / dataframe[F.length], dataframe[F.ux])
    dataframe[F.uy] = torch.where(is_line, dy / dataframe[F.length], dataframe[F.uy])
    
    dataframe[F.nx] = torch.where(is_line, -dataframe[F.uy], dataframe[F.nx])
    dataframe[F.ny] = torch.where(is_line, dataframe[F.ux], dataframe[F.ny])
    
    # Compute circle perimeters
    is_circle = (dataframe[F.circle_flag] == 1)
    dataframe[F.length] = torch.where(is_circle, 2 * torch.pi * dataframe[F.radius], dataframe[F.length])
    
    # Compute arc lengths
    is_arc = (dataframe[F.arc_flag] == 1)
    arc_angle = torch.where(is_arc, ((dataframe[F.end_angle] - dataframe[F.start_angle]) % 360), 0)
    dataframe[F.length] = torch.where(is_arc, dataframe[F.radius] * arc_angle * torch.pi/180, dataframe[F.length])
    
    graph = Graph(dataframe)
    
    