
def _get_characters():
    # Set your character set
    character_set = {
                        "characters": {
                            "type": {
                                "latin_upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                "latin_lower": "abcdefghijklmnopqrstuvwxyz",
                                "numbers": "0123456789",
                                "basic_latin": "!\"#$%&'*+,-./:;=?@\\_`|~()[]{}<>",
                                "supplementary_latin": "º",
                                "special_upper": "ÀÁÂÃÇÉÊÍÓÔØÚ",
                                "special_lower": "àáâãçéêíóô÷øú"
                            },
                            "encoding": 0
                        },
                        "font": {
                            "type": {
                                "Arial": "arial.ttf",
                                "ROMANS": "romans.shx"
                            },
                            "encoding": 0
                        }      
                    }
    
    characters_list = ''.join(character_set["characters"]["type"].values())
    character_set["characters"]["encoding"] = {character: i for i, character in enumerate(characters_list)}
    character_set["font"]["encoding"] = {font: i for i, font in enumerate(character_set["font"]["type"].keys())}
    
    return character_set

character_set = _get_characters()
width_factors = (0.8, 1.0, 1.2)
rotations= [i * 15 for i in range(24)]
scales = [0.5*i for i in range(14, 60)] + [5*i for i in range(6, 21)]

import cupy
from torch.utils.dlpack import from_dlpack
from .orguel_ml import Graph as BaseGraph

class Graph(BaseGraph):
    _attributes = cupy.arange(9, dtype=cupy.int32)
    
    parallel = 0
    colinear = 1
    perpendicular_distance = 2
    overlap_ratio = 3
    point_intersection = 4
    segment_intersection = 5
    angle_difference = 6
    angle_difference_sin = 7
    angle_difference_cos = 8
    
    def __init__(self, dataframe, normalization_factor, edges_per_element=200):
        self._dataframe = dataframe
        
        character_height = dataframe['character_height']
        character_width = dataframe['character_width']
        character_insertion_x = dataframe['character_insertion_x']
        character_insertion_y = dataframe['character_insertion_y']
        start_x, start_y = dataframe['start_x'], dataframe['start_y']
        end_x, end_y = dataframe['end_x'], dataframe['end_y']
        length = dataframe['length']
        angle = dataframe['angle']
        
        # Normalize coordinates
        coordinates_x = cupy.concatenate([start_x.to_cupy(), end_x.to_cupy(), character_insertion_x.to_cupy()])
        coordinates_y = cupy.concatenate([start_y.to_cupy(), end_y.to_cupy(), character_insertion_y.to_cupy()])
        
        start_x_normalized = (start_x - coordinates_x.mean()) / coordinates_x.std()
        start_y_normalized = (start_y - coordinates_y.mean()) / coordinates_y.std()
        end_x_normalized = (end_x - coordinates_x.mean()) / coordinates_x.std()
        end_y_normalized = (end_y - coordinates_y.mean()) / coordinates_y.std()
        
        coordinates_normalized = cupy.stack([start_x_normalized.to_cupy(), start_y_normalized.to_cupy(),
                                            end_x_normalized.to_cupy(), end_y_normalized.to_cupy()], axis=1)
        
        # Normalize angles
        angles_normalized = cupy.stack([cupy.sin(angle.to_cupy()), cupy.cos(angle.to_cupy())], axis=1)
        
        # Normalize length and character dimensions
        if normalization_factor: normalizationFactor = normalization_factor
        else: normalizationFactor = cupy.max(cupy.stack([length.to_cupy(), character_height.to_cupy(), character_width.to_cupy()], axis=1))
        
        lengths_normalized = (length / normalizationFactor).to_cupy().reshape(-1, 1)
        character_heigth_normalized = (character_height / normalizationFactor)
        character_width_normalized = (character_width / normalizationFactor)
        
        characters = dataframe.groupby("character_id")
        
        insertion_x_normalized = (characters['character_insertion_x'].first().values - coordinates_x.mean()) / coordinates_x.std()
        insertion_y_normalized = (characters['character_insertion_y'].first().values - coordinates_y.mean()) / coordinates_y.std()
        
        self.classificationLabels = {
            "type": from_dlpack(cupy.asarray(characters['character_type'].first().values).toDlpack()),
            "font": from_dlpack(cupy.asarray(characters['character_font'].first().values).toDlpack())
        }
        
        self.regressionTargets = {
            "height": from_dlpack(cupy.asarray(character_heigth_normalized.groupby(dataframe['character_id']).first().values).toDlpack()),
            "width": from_dlpack(cupy.asarray(character_width_normalized.groupby(dataframe['character_id']).first().values).toDlpack()),
            "rotation": from_dlpack(cupy.asarray(characters['character_rotation'].first().values).toDlpack()),
            "insertion_x": from_dlpack(cupy.asarray(insertion_x_normalized).toDlpack()),
            "insertion_y": from_dlpack(cupy.asarray(insertion_y_normalized).toDlpack())
        }
        
        self.nodeAttributes = from_dlpack(cupy.hstack([coordinates_normalized, angles_normalized, lengths_normalized]).toDlpack())
        
        edge_capacity = len(dataframe) * edges_per_element
        
        self.size = 0
        self.edges = cupy.empty((2, edge_capacity), dtype=cupy.int32)
        self.edgeAttributes = cupy.empty((edge_capacity, len(self._attributes)), dtype=cupy.float32)

import math
import numpy
import ezdxf
import cudf
import torch
from torch_geometric.data import Data
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from .orguel_ml import LaplacianEigenvectors

def rotate_and_scale(x, y, angle, scale):
    angle_rad = math.radians(angle)
    x_aligned = x * scale * math.cos(angle_rad) - y * scale * math.sin(angle_rad)
    y_aligned = x * scale * math.sin(angle_rad) + y * scale * math.cos(angle_rad)
    return x_aligned, y_aligned

def _expand_quads(quads, epsilon=0.05):
    centroid = quads.mean(axis=1, keepdims=True)
    direction = quads - centroid  # vector from center to corner
    direction_normalized = cupy.linalg.norm(direction, axis=2, keepdims=True) + 1e-12  # avoid divide-by-zero
    unit = direction / direction_normalized
    quads_expanded = quads + unit * epsilon
    return quads_expanded

def find_character_lines(centroid_x, centroid_y, quads):
    
    # Expand quads slightly to catch edge points
    quads = _expand_quads(quads)
    
    cx, cy = centroid_x, centroid_y
    
    N = cx.shape[0]
    M = quads.shape[0]

    # Expand to shape (N, M, 1)
    cx = cx[:, None, None]
    cy = cy[:, None, None]

    # (M, 4, 2) → (1, M, 4, 2)
    vertices = quads[None, :, :, :]

    xv = vertices[:, :, :, 0]  # shape (1, M, 4)
    yv = vertices[:, :, :, 1]

    xv1 = cupy.roll(xv, -1, axis=2)
    yv1 = cupy.roll(yv, -1, axis=2)

    # Check if edge crosses ray (horizontal to right)
    condition1 = ((yv <= cy) & (yv1 > cy)) | ((yv > cy) & (yv1 <= cy))
    slope = (xv1 - xv) / (yv1 - yv + 1e-12)
    intersect_x = xv + slope * (cy - yv)
    condition2 = cx < intersect_x

    crossings = (condition1 & condition2).sum(axis=2) % 2 == 1  # Ray crossing parity rule

    matches = cupy.argwhere(crossings)
    return matches

def CreateGraph(dxf_file, character_positions, rotation=0, scale=1.0, normalization_factor=None):
    # Filter only rows for this dxf file
    characterPositions = character_positions[character_positions['file'] == dxf_file].copy()
    characterPositions = characterPositions.reset_index(drop=False)
    
    # Apply transform to bbox and insertion point
    for i, row in characterPositions.iterrows():
        bboxAligned = shapely_rotate(row['bbox'], angle=rotation, origin=(0, 0), use_radians=False)
        bboxAligned = shapely_scale(bboxAligned, xfact=scale, yfact=scale, origin=(0, 0))
        characterPositions.at[i, 'bbox'] = bboxAligned
        
        insertion_x, insertion_y = row['insertion']
        aligned_insertion_x, aligned_insertion_y = rotate_and_scale(insertion_x, insertion_y, rotation, scale)
        characterPositions.at[i, 'insertion'] = (aligned_insertion_x, aligned_insertion_y)
        
        characterPositions.at[i, 'height'] = row['height'] * scale
        characterPositions.at[i, 'width'] = row['width'] * scale
        characterPositions.at[i, 'rotation'] = (row['rotation'] + math.radians(rotation)) % (2 * math.pi)
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    lineCollector = [line for line in modelSpace if line.dxftype() == 'LINE']
    
    if not lineCollector: return
    
    count = len(lineCollector)
    
    character_id = numpy.zeros(count, dtype=numpy.int8)
    character_font = numpy.zeros(count, dtype=numpy.int8)
    character_type = numpy.zeros(count, dtype=numpy.int8)
    character_height = numpy.zeros(count, dtype=numpy.float32)
    character_width = numpy.zeros(count, dtype=numpy.float32)
    character_rotation = numpy.zeros(count, dtype=numpy.float32)
    character_insertion_x = numpy.zeros(count, dtype=numpy.float32)
    character_insertion_y = numpy.zeros(count, dtype=numpy.float32)
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
    
    for i, line in enumerate(lineCollector):
        layer[i] = line.dxf.layer
        
        sx, sy, _ = line.dxf.start
        ex, ey, _ = line.dxf.end
        
        sx, sy = rotate_and_scale(sx, sy, rotation, scale)
        ex, ey = rotate_and_scale(ex, ey, rotation, scale)
        
        dx, dy = ex - sx, ey - sy
        
        start_x[i], start_y[i] = sx, sy
        end_x[i], end_y[i] = ex, ey
        length[i] = numpy.hypot(dx, dy)
        angle[i] = numpy.arctan2(dy, dx) % (2*numpy.pi)
    
    dataframe = cudf.DataFrame(
        {
            "character_id": cupy.asarray(character_id),
            "character_font": cupy.asarray(character_font),
            "character_type": cupy.asarray(character_type),
            "character_height": cupy.asarray(character_height),
            "character_width": cupy.asarray(character_width),
            "character_rotation": cupy.asarray(character_rotation),
            "character_insertion_x": cupy.asarray(character_insertion_x),
            "character_insertion_y": cupy.asarray(character_insertion_y),
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
    
    # Extract line centroids
    centroid_x, centroid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
    centroid_x, centroid_y = cupy.asarray(centroid_x), cupy.asarray(centroid_y)
    
    # Build the CuPy stack *and* remember which DF row each quad is.
    quads = cupy.stack([ cupy.asarray(polygon.exterior.coords[:4], dtype=cupy.float32)
                            for polygon in characterPositions['bbox']])
    
    # Run the GPU matcher
    character_lines = find_character_lines(centroid_x, centroid_y, quads)
    
    # Use quad_indices to translate quad-position → DataFrame row
    for line, quad in character_lines:
        line, quad = int(line), int(quad)
        row = characterPositions.iloc[quad]
        character_id[line] = row.name # original character id
        character_font[line] = character_set['font']['encoding'].get(row['font'], 0)
        character_type[line] = character_set['characters']['encoding'].get(row['character'], 0)
        character_height[line] = row['height']
        character_width[line] = row['width']
        character_rotation[line]= row['rotation'] / (2 * math.pi)
        character_insertion_x[line] = row['insertion'][0]
        character_insertion_y[line] = row['insertion'][1]
    
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
    
    graph = Graph(dataframe, normalization_factor)
    
    graph.ParallelDetection(max_threshold=7.5)
    graph.IntersectionDetection()
    
    # 🐛 Debug block right here:
    print("[DEBUG] graph.edges type:", type(graph.edges))
    print("[DEBUG] graph.edges dtype:", graph.edges.dtype)
    print("[DEBUG] graph.edges shape:", graph.edges.shape)
    print("[DEBUG] graph.size:", graph.size, "type:", type(graph.size))

    valid = int(graph.size)
    print("[DEBUG] graph.edges[:, :valid].shape:", graph.edges[:, :valid].shape)
    
    graph.edges = torch.tensor(cupy.asnumpy(graph.edges[:, :graph.size]), dtype=torch.long).t().contiguous()
    graph.edgeAttributes = torch.tensor(cupy.asnumpy(graph.edgeAttributes[:graph.size]), dtype=torch.float)
    
    # Create PyG Data object with everything
    graph = Data(
        x=graph.nodeAttributes,
        edge_index=graph.edges,
        edge_attr=graph.edgeAttributes,
        y=graph.classificationLabels["type"],
        font=graph.classificationLabels["font"],
        height=graph.regressionTargets["height"],
        width=graph.regressionTargets["width"],
        rotation=graph.regressionTargets["rotation"],
        insertion_x=graph.regressionTargets["insertion_x"],
        insertion_y=graph.regressionTargets["insertion_y"],
        character_id=torch.tensor(cupy.asnumpy(dataframe["character_id"]), dtype=torch.long)
    )
    
    #graph: Data = LaplacianEigenvectors(graph, k=4)

    return graph

import os
from tqdm import tqdm
from multiprocessing import Pool

def CreateGraphWorker(args):
    print(f"[DEBUG] Calling CreateGraph with: {args[0]}")
    return CreateGraph(*args)

class CreateGraphDataset(torch.utils.data.Dataset):
    def __init__(self, arguments, chunksize=4):
        self.graphs = []
        #with Pool(processes=os.cpu_count()) as pool:
            #for graph in tqdm(pool.imap_unordered(CreateGraphWorker, arguments, chunksize=chunksize),
            #                  total=len(arguments),
            #                  desc="Creating graphs"):
            #    self.graphs.append(graph)
        for args in tqdm(arguments, desc="Creating graphs"): #DEBUG
            print(f"Processing: {args[0]}") #DEBUG
            self.graphs.append(CreateGraphWorker(args)) #DEBUG
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_scatter import scatter_mean

class OCRNetwork(nn.Module):
    def __init__(self, cluster_aware=False, transformer_heads=4, transformer_layers=2):
        super().__init__()
        
        n_characters = len(character_set["characters"]["encoding"])
        n_fonts = len(character_set["font"]["encoding"])
        
        # Define input and embedding dimensions
        edge_attributes = 7
        coordinates_dimensions = 7  # (start_x, start_y, end_x, end_y, sin(angle), cos(angle), length)
        embedding_dimensions = 64
        
        self.cluster_aware = cluster_aware
        
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
        
        self.cluster_embedding = nn.Embedding(118, 32)  # assuming max 118 clusters, change as needed
        
        # ------------- Output heads -------------
        self.character_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, n_characters)          # character logits
        )
        
        self.font_head = nn.Linear(32, n_fonts)
        self.regression_head = nn.Linear(32, 4)
        
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
        
        if self.cluster_aware:
            x += self.cluster_embedding(data.cluster_id)
            x = scatter_mean(x, data.cluster_id, dim=0)
        
        # Heads
        output = {
            "character_logits": self.character_head(x),
            "font_logits": self.font_head(x),
            "regression_values": self.regression_head(x)
        }
        
        return output

