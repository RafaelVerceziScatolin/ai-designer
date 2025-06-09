
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
        
    def ParallelDetection(self):
        dataframe = self._dataframe.copy()
        globals().update(_dataframe_field.__members__)
        
        # Filter valid line indices
        mask = (dataframe[circle_flag] == 0) & (dataframe[arc_flag] == 0)
        lines = dataframe[:, mask]
        lines[angle] = lines[angle] % torch.pi # collapse symmetrical directions
        
        # bin settings
        bin_offset_size = 25
        bin_angle_size = 0.25 * torch.pi/180
        
        angle_bin_key = torch.floor(lines[angle] / bin_angle_size).to(torch.int32)
        offset_bin_key = torch.floor(lines[offset] / bin_offset_size).to(torch.int32)
        offset_bin_key -= offset_bin_key.min() # remove void
        
        bin_hash = angle_bin_key * (offset_bin_key.max() + 1) + offset_bin_key
        
        # Sort by bin
        sorted_hash, original_index = torch.sort(bin_hash)
        
        # Mask sorted hash to separate unique bins
        mask = torch.hstack([torch.tensor([True], device='cuda'), sorted_hash[1:]!=sorted_hash[:-1]])
        unique_bins = sorted_hash[mask]
        
        # Get the start and end indices for each bin
        bin_start = torch.where(mask)[0]
        bin_end = torch.hstack([bin_start[1:], torch.tensor([len(sorted_hash)], device='cuda')])
        
        # Split the hash key into angle and offset
        unique_angle_bin_key = unique_bins // (offset_bin_key.max() + 1)
        unique_offset_bin_key = unique_bins % (offset_bin_key.max() + 1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        








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
    anchor_x = dataframe[[start_x, end_x]].min() - 100000
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
    
    