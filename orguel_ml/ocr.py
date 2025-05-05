
def __get_characters():
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

character_set = __get_characters()
width_factors = (0.8, 1.0, 1.2)
rotations= [i * 15 for i in range(24)]
scales = [0.5*i for i in range(14, 60)] + [5*i for i in range(6, 21)]

from typing import List, Tuple
import cupy
from torch.utils.dlpack import from_dlpack
from .orguel_ml import Graph as BaseGraph

class Graph(BaseGraph):
    __edge_attributes = cupy.arange(9, dtype=cupy.int32)
    
    parallel = 0
    colinear = 1
    perpendicular_distance = 2
    overlap_ratio = 3
    point_intersection = 4
    segment_intersection = 5
    angle_difference = 6
    angle_difference_sin = 7
    angle_difference_cos = 8
    
    def __init__(self, dataframe, normalization_factor):
        self.__dataframe = dataframe
        
        character_height = dataframe['character_height']
        character_width = dataframe['character_width']
        character_insertion_x = dataframe['character_insertion_x']
        character_insertion_y = dataframe['character_insertion_y']
        start_x, start_y = dataframe['start_x'], dataframe['start_y']
        end_x, end_y = dataframe['end_x'], dataframe['end_y']
        length = dataframe['length']
        angle = dataframe['angle']
        
        # Normalize coordinates
        coordinates_x = cupy.concatenate(start_x.to_cupy(), end_x.to_cupy(), character_insertion_x.to_cupy())
        coordinates_y = cupy.concatenate(start_y.to_cupy(), end_y.to_cupy(), character_insertion_y.to_cupy())
        
        normalized_start_x = (start_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_start_y = (start_y - coordinates_y.mean()) / coordinates_y.std()
        normalized_end_x = (end_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_end_y = (end_y - coordinates_y.mean()) / coordinates_y.std()
        
        normalizedCoordinates = cupy.stack([normalized_start_x.to_cupy(), normalized_start_y.to_cupy(),
                                            normalized_end_x.to_cupy(), normalized_end_y.to_cupy()], axis=1)
        
        # Normalize angles
        normalizedAngles = cupy.stack([cupy.sin(angle.to_cupy()), cupy.cos(angle.to_cupy())], axis=1)
        
        # Normalize length and character dimensions
        if normalization_factor: normalizationFactor = normalization_factor
        else: normalizationFactor = cupy.max(cupy.stack([length.to_cupy(), character_height.to_cupy(), character_width.to_cupy()], axis=1))
        
        normalizedLengths = (length / normalizationFactor).to_cupy().reshape(-1, 1)
        normalizedCharacterHeigth = (character_height / normalizationFactor)
        normalizedCharacterWidth = (character_width / normalizationFactor)
        
        characters = dataframe.groupby("character_id")
        
        self.classificationLabels = {
            "type": from_dlpack(characters['character_type'].first().to_cupy().toDlpack()),
            "font": from_dlpack(characters['character_font'].first().to_cupy().toDlpack())
        }
        
        self.regressionTargets = {
            "height": from_dlpack(normalizedCharacterHeigth.groupby(dataframe['character_id']).first().to_cupy().toDlpack()),
            "width": from_dlpack(normalizedCharacterWidth.groupby(dataframe['character_id']).first().to_cupy().toDlpack()),
            "rotation": from_dlpack(characters['character_rotation'].first().to_cupy().toDlpack()),
            "insertion_x": from_dlpack(((characters['character_insertion_x'].first() - coordinates_x.mean()) / coordinates_x.std()).to_cupy().toDlpack()),
            "insertion_y": from_dlpack(((characters['character_insertion_y'].first() - coordinates_y.mean()) / coordinates_y.std()).to_cupy().toDlpack())
        }
        
        self.nodeAttributes = from_dlpack(cupy.hstack([normalizedCoordinates, normalizedAngles, normalizedLengths]).toDlpack())
        self.edges: List[Tuple[int, int]] = []
        self.edgeAttributes: List[List[float]] = []
           
from typing import Dict
import math
import cudf
import ezdxf
import torch
from torch_geometric.data import Data
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from .orguel_ml import LaplacianEigenvectors

def __rotate_and_scale(x, y, angle, scale):
    angle_rad = math.radians(angle)
    x_aligned = x * scale * math.cos(angle_rad) - y * scale * math.sin(angle_rad)
    y_aligned = x * scale * math.sin(angle_rad) + y * scale * math.cos(angle_rad)
    return x_aligned, y_aligned

def CreateGraph(dxf_file, character_positions, rotation=0, scale=1.0, normalization_factor=None):
    # Filter only rows for this dxf file
    characterPositions = character_positions[character_positions['file'] == dxf_file].copy()
    
    # Apply transform to bbox and insertion point
    for i, row in characterPositions.iterrows():
        bboxAligned = shapely_rotate(row['bbox'], angle=rotation, origin=(0, 0), use_radians=False)
        bboxAligned = shapely_scale(bboxAligned, xfact=scale, yfact=scale, origin=(0, 0))
        characterPositions.at[i, 'bbox'] = bboxAligned
        
        insertion_x, insertion_y = row['insertion']
        aligned_insertion_x, aligned_insertion_y = __rotate_and_scale(insertion_x, insertion_y, rotation, scale)
        characterPositions.at[i, 'insertion'] = (aligned_insertion_x, aligned_insertion_y)
        
        characterPositions.at[i, 'height'] = row['height'] * scale
        characterPositions.at[i, 'width'] = row['width'] * scale
        characterPositions.at[i, 'rotation'] = (row['rotation'] + math.radians(rotation)) % (2 * math.pi)

    # Prepare spatial index
    polygons = list(characterPositions['bbox'])
    space2D = STRtree(polygons)
    polygonToRow = {id(polygon): row for _, row in characterPositions.iterrows() for polygon in [row['bbox']]}
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    lineCollector = [line for line in modelSpace if line.dxftype() == 'LINE']

    dataframe: List[Dict] = []
    for line in lineCollector:
        start_x, start_y, _ = line.dxf.start
        end_x, end_y, _ = line.dxf.end
        
        # Apply rotation and scaling
        start_x, start_y = __rotate_and_scale(start_x, start_y, rotation, scale)
        end_x, end_y = __rotate_and_scale(end_x, end_y, rotation, scale)
        
        centroid = Point(((start_x + end_x) / 2, (start_y + end_y) / 2))

        length = math.hypot(end_x - start_x, end_y - start_y)
        angle = math.atan2(end_y - start_y, end_x - start_x) % (2*math.pi)

        label = {
            "character_id": 0,
            "character_font": 0,
            "character_type": 0,
            "character_height": 0,
            "character_width": 0,
            "character_rotation": 0,
            "character_insertion_x": 0,
            "character_insertion_y": 0,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "length": length,
            "angle": angle,
            "circle": 0,
            "arc": 0,
            "radius": 0,
            "start_angle": 0,
            "end_angle": 0
        }

        # Use spatial index to find potential matches
        for polygon in space2D.query(centroid):
            if polygon.contains(centroid):
                row = polygonToRow[id(polygon)]
                font = character_set['font']['encoding'].get(row['font'], 0)
                character = character_set['characters']['encoding'].get(row['character'], 0)

                label.update({
                    "character_id": row.name,
                    "character_font": font,
                    "character_type": character,
                    "character_height": row['height'],
                    "character_width": row['width'],
                    "character_rotation": row['rotation'] / (2 * math.pi),
                    "character_insertion_x": row['insertion'][0],
                    "character_insertion_y": row['insertion'][1]
                })
                break

        dataframe.append(label)

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
    
    graph = Graph(dataframe, normalization_factor)
    
    graph.ParallelDetection(max_threshold=7.5, bin_angle_size=numpy.radians(0.25))
    graph.IntersectionDetection()
    
    graph.classificationLabels = {
        "type": torch.tensor(graph.classificationLabels["type"], dtype=torch.long),
        "font": torch.tensor(graph.classificationLabels["font"], dtype=torch.long)
    }
    
    graph.regressionTargets = {
        "height": torch.tensor(graph.regressionTargets["height"], dtype=torch.float),
        "width": torch.tensor(graph.regressionTargets["width"], dtype=torch.float),
        "rotation": torch.tensor(graph.regressionTargets["rotation"], dtype=torch.float),
        "insertion_x": torch.tensor(graph.regressionTargets["insertion_x"], dtype=torch.float),
        "insertion_y": torch.tensor(graph.regressionTargets["insertion_y"], dtype=torch.float)
    }
    
    graph.nodeAttributes = torch.tensor(graph.nodeAttributes, dtype=torch.float)
    graph.edges = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
    graph.edgeAttributes = torch.tensor(graph.edgeAttributes, dtype=torch.float)
    
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
        character_id=torch.tensor(dataframe["character_id"].values, dtype=torch.long)
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

