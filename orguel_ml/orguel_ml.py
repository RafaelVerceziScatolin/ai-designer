
from enum import IntEnum

class _dataframe_field(IntEnum):
    index = 0
    start_x = 1
    start_y = 2
    end_x = 3
    end_y = 4
    length = 5
    angle = 6
    offset = 7
    circle_flag = 8
    arc_flag = 9
    radius = 10
    start_angle = 11
    end_angle = 12
    
    @classmethod
    def count(cls) -> int: return len(cls)

# Suppress IDE warnings
index=start_x=start_y=end_x=end_y=length=angle=offset=\
circle_flag=arc_flag=radius=start_angle=end_angle=None

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

# Suppress IDE warnings
parallel=colinear=perpendicular_distance=overlap_ratio=point_intersection=segment_intersection=\
angle_difference=angle_difference_sin=angle_difference_cos=perimeter_intersection=None

import torch
from torch import newaxis

class Graph:
    def __init__(self, dataframe):
        self._dataframe = dataframe
        globals().update(_dataframe_field.__members__)
        
        start_x, start_y = dataframe[start_x], dataframe[start_y]
        end_x, end_y = dataframe[end_x], dataframe[end_y]
        length = dataframe[length]
        angle = dataframe[angle]
        
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
        circle_flag = dataframe[circle_flag].reshape([-1, 1]) # shape (N, 1)
        arc_flag = dataframe[arc_flag].reshape([-1, 1]) # shape (N, 1)
        
        # Node attributes
        self.nodeAttributes = torch.hstack([normalized_coordinates, normalized_angle,
                                            normalized_length, circle_flag, arc_flag])
    
    @staticmethod
    def create_obb(start_x, start_y, end_x, end_y, width, length_extension=True):
        # Compute line directions and lengths
        dx = end_x - start_x
        dy = end_y - start_y
        length = torch.sqrt(dx**2 + dy**2) + 1e-8 # sum 1e-8 to prevent division by zero

        # Unit direction vectors
        ux = dx / length
        uy = dy / length

        # Perpendicular vectors (unit)
        perp_x = -uy
        perp_y = ux

        # Half dimensions
        half_width = width / 2
        half_length = (length / 2)
        if length_extension: half_length += half_width

        # Midpoints of the lines
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        dx_length, dy_length = ux * half_length, uy * half_length
        dx_width, dy_width = perp_x * half_width, perp_y * half_width

        # Corners (4 per OBB)
        corner1_x = mid_x - dx_length - dx_width
        corner1_y = mid_y - dy_length - dy_width

        corner2_x = mid_x + dx_length - dx_width
        corner2_y = mid_y + dy_length - dy_width

        corner3_x = mid_x + dx_length + dx_width
        corner3_y = mid_y + dy_length + dy_width

        corner4_x = mid_x - dx_length + dx_width
        corner4_y = mid_y - dy_length + dy_width

        # Stack corners: shape (n_lines, 4, 2)
        obb = torch.stack([
            torch.stack([corner1_x, corner1_y], axis=1),
            torch.stack([corner2_x, corner2_y], axis=1),
            torch.stack([corner3_x, corner3_y], axis=1),
            torch.stack([corner4_x, corner4_y], axis=1)
        ], axis=1)

        return obb
    
    @staticmethod
    def check_overlap_sat(obbs):
        
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
    
    
    
    
    
    
    
    
    def ParallelDetection(self, max_offset=25):
        dataframe = self._dataframe.copy()
        globals().update(_dataframe_field.__members__)
        
        # Filter valid line indices
        mask = (dataframe[circle_flag] == 0) & (dataframe[arc_flag] == 0)
        lines = dataframe[:, mask]
        lines[angle] = lines[angle] % torch.pi # collapse symmetrical directions
        
        # Extract coordinates
        start_x = lines[start_x]
        start_y = lines[start_y]
        end_x = lines[end_x]
        end_y = lines[end_y]
        
        # Compute line midpoints and OBBs
        obbs = self.create_obb(start_x, start_y, end_x, end_y, width=max_offset, length_extension=False)
        
        # SAT check
        overlap = self.check_overlap_sat(obbs, obbs)
        
        
        
        
        
        
        
        
        
        offset_tolerance = 25
        angle_tolerance = 0.25 * torch.pi/180
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        








import ezdxf

def CreateGraph(dxf_file):
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    
    entities = [entity for entity in modelSpace if entity.dxftype() in ('LINE', 'CIRCLE', 'ARC')]
    
    globals().update(_dataframe_field.__members__)
    dataframe = torch.zeros((_dataframe_field.count(), len(entities)), dtype=torch.float32, device='cuda')
    
    for i, entity in enumerate(entities):
        dataframe[index,i] = i
        entity_type = entity.dxftype()
        
        if entity_type == 'LINE':
            sx, sy, _ = entity.dxf.start
            ex, ey, _ = entity.dxf.end
            dx, dy = ex - sx, ey - sy
            
            dataframe[start_x,i] = sx
            dataframe[start_y,i] = sy
            dataframe[end_x,i] = ex
            dataframe[end_y,i] = ey
            dataframe[length,i] = torch.hypot(dx, dy)
            dataframe[angle,i] = torch.arctan2(dy, dx) % (2*torch.pi)
            
        elif entity_type == 'CIRCLE':
            cx, cy, _ = entity.dxf.center
            r = entity.dxf.radius
            
            dataframe[circle_flag,i] = 1
            dataframe[start_x,i] = cx
            dataframe[start_y,i] = cy
            dataframe[length,i] = 2 * torch.pi * r
            dataframe[radius,i] = r
            
        elif entity_type == 'ARC':
            cx, cy, _ = entity.dxf.center
            r = entity.dxf.radius
            sa = entity.dxf.start_angle
            ea = entity.dxf.end_angle
            
            dataframe[arc_flag,i] = 1
            dataframe[start_x,i] = cx
            dataframe[start_y,i] = cy
            dataframe[length,i] = r * torch.radians((ea - sa) % 360)
            dataframe[radius,i] = r
            dataframe[start_angle,i] = sa
            dataframe[end_angle,i] = ea
    
    # Define anchor point
    anchor_x = dataframe[[start_x, end_x]].min() - 50000
    anchor_y = dataframe[[start_y, end_y]].min() - 100000
    
    # Compute offset
    sx, sy, ex, ey = dataframe[start_x], dataframe[start_y], dataframe[end_x], dataframe[end_y]
    
    # Set a safe length (to avoid division by zero)
    mask = (dataframe[circle_flag] == 0) & (dataframe[arc_flag] == 0) & (dataframe[length] > 1e-2)
    l = dataframe[length].where(mask, 1)
    
    # Perpendicular offset
    dataframe[offset] = torch.abs((sx - anchor_x) * (-(ey - sy) / l) + (sy - anchor_y) * ((ex - sx) / l))
    
    # Apply masking: invalid offsets get zero
    dataframe[offset] = dataframe[offset].where(mask, 0)
    
    graph = Graph(dataframe)
    
    