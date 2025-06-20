
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
                                              normalized_end_x, normalized_end_y], axis=1) # shape (N, 4) for ML model
        
        # Normalize angles and lengths
        normalized_angle = torch.stack([torch.sin(angle), torch.cos(angle)], axis=1) # shape (N, 2)
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
        obb = torch.stack([
            torch.stack([corner1_x, corner1_y], axis=1),
            torch.stack([corner2_x, corner2_y], axis=1),
            torch.stack([corner3_x, corner3_y], axis=1),
            torch.stack([corner4_x, corner4_y], axis=1)
        ], axis=1)

        return obb # Shape (n_lines, 4, 2)
    
    @staticmethod
    def check_overlap_sat(obbs:torch.Tensor):
        
        n_lines = obbs.shape[0]
        
        # Compute AABBs (Axis-Aligned Bounding Boxes)
        min_x = obbs[..., 0].min(axis=1)[0]
        max_x = obbs[..., 0].max(axis=1)[0]
        min_y = obbs[..., 1].min(axis=1)[0]
        max_y = obbs[..., 1].max(axis=1)[0]
        
        # Computing AABB intersection matrix
        mask_x = (min_x[:, newaxis] <= max_x[newaxis, :]) & (min_x[newaxis, :] <= max_x[:, newaxis])
        mask_y = (min_y[:, newaxis] <= max_y[newaxis, :]) & (min_y[newaxis, :] <= max_y[:, newaxis])

        aabb_overlap = mask_x & mask_y
        
        pairs = torch.argwhere(aabb_overlap) # shape (n_pairs, 2)
        
        i, j = pairs[:, 0], pairs[:, 1]
        
        # Remove self-overlaps and lower triangle (e.g., keep only i < j)
        mask = ~((i == j) | (i > j))
        
        i, j = i[mask], j[mask]
        obbs_i, obbs_j = obbs[i], obbs[j]
    
    
    
    
    
    
    
    
    def ParallelDetection(self, offset=25):
        F = _dataframe_field
        dataframe = self._dataframe
        
        # Filter valid line indices
        is_line = (dataframe[F.circle_flag] == 0) & (dataframe[F.arc_flag] == 0)
        lines = dataframe[:, is_line]
        lines[F.angle] = lines[F.angle] % torch.pi # collapse symmetrical directions
        
        # Compute OBBs
        obbs = self.create_obb(lines, width=offset, length_extension=False)
        
        # SAT check
        overlap = self.check_overlap_sat(obbs)
        
        
        
        
        
        
        
        
        
        
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
    is_line = (dataframe[F.circle_flag] == 0) & (dataframe[F.arc_flag] == 0)
    
    # Unit direction and unit perpendicular vectors
    dx = torch.where(is_line, dataframe[F.end_x] - dataframe[F.start_x], 0)
    dy = torch.where(is_line, dataframe[F.end_y] - dataframe[F.start_y], 0)
    
    dataframe[F.length] = torch.where(is_line, torch.clamp(torch.sqrt(dx**2 + dy**2), min=1e-4), dataframe[F.length])
    
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
    
    