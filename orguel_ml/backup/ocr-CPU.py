
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

import numpy
from typing import List, Tuple
from .orguel_ml import Graph as BaseGraph

class Graph(BaseGraph):
    __edge_attributes = {
                            "parallel": 0,
                            "colinear": 0,
                            "perpendicular_distance": 0,
                            "overlap_ratio": 0,
                            "point_intersection": 0,
                            "segment_intersection": 0,
                            "angle_difference": 0,
                            "angle_difference_sin": 0,
                            "angle_difference_cos": 0,
                        }
    
    def __init__(self, dataframe, normalization_factor):
        self.__dataframe = dataframe
        
        coordinates_x = dataframe[['start_x', 'end_x', 'character_insertion_x']].values.flatten()
        coordinates_y = dataframe[['start_y', 'end_y', 'character_insertion_y']].values.flatten()
        
        normalizedCoordinates = dataframe[['start_x', 'start_y', 'end_x', 'end_y']].copy()
        normalizedCoordinates[['start_x', 'end_x']] = (normalizedCoordinates[['start_x', 'end_x']] - coordinates_x.mean()) / coordinates_x.std()
        normalizedCoordinates[['start_y', 'end_y']] = (normalizedCoordinates[['start_y', 'end_y']] - coordinates_y.mean()) / coordinates_y.std()
        
        normalizedAngles = numpy.column_stack((numpy.sin(dataframe['angle']), numpy.cos(dataframe['angle'])))
        
        if normalization_factor: normalizationFactor = normalization_factor
        else: normalizationFactor = dataframe[['length', 'character_height', 'character_width']].values.max()
          
        normalizedLengths = dataframe[['length']] / normalizationFactor
        normalizedCharacterHeigth = dataframe[['character_height']] / normalizationFactor
        normalizedCharacterWidth = dataframe[['character_width']] / normalizationFactor
        
        characters = dataframe.groupby("character_id")
        
        self.classificationLabels = {
            "type": characters['character_type'].first().values,
            "font": characters['character_font'].first().values
        }
        
        self.regressionTargets = {
            "height": normalizedCharacterHeigth.groupby(dataframe['character_id']).first().values,
            "width": normalizedCharacterWidth.groupby(dataframe['character_id']).first().values,
            "rotation": characters['character_rotation'].first().values,
            "insertion_x": ((characters['character_insertion_x'].first() - coordinates_x.mean()) / coordinates_x.std()).values,
            "insertion_y": ((characters['character_insertion_y'].first() - coordinates_y.mean()) / coordinates_y.std()).values
        }
        
        self.nodesAttributes = numpy.hstack([normalizedCoordinates.values, normalizedAngles, normalizedLengths.values])
        self.edges: List[Tuple[int, int]] = []
        self.edgesAttributes: List[List[float]] = []

from typing import Dict
import math
import pandas
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

    dataframe = pandas.DataFrame(dataframe)
    
    # Define anchor point
    anchor_x = dataframe[["start_x", "end_x"]].values.flatten().min() - 100000
    anchor_y = dataframe[["start_y", "end_y"]].values.flatten().min() - 100000
    
    # Compute offset
    offsets: List[float] = []
    for i, row in dataframe.iterrows():
        if row["circle"] or row["arc"]: offsets.append(0.0); continue
        if row["length"] < 1e-12: offsets.append(0.0); continue
        start_x, start_y, end_x, end_y, length = row["start_x"], row["start_y"], row["end_x"], row["end_y"], row["length"]
        offset = abs((start_x - anchor_x) * (-(end_y - start_y) / length) + (start_y - anchor_y) * ((end_x - start_x) / length))
        offsets.append(offset)
    
    dataframe["offset"] = offsets
    
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
    
    graph.nodesAttributes = torch.tensor(graph.nodesAttributes, dtype=torch.float)
    graph.edges = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
    graph.edgesAttributes = torch.tensor(graph.edgesAttributes, dtype=torch.float)
    
    # Create PyG Data object with everything
    graph = Data(
        x=graph.nodesAttributes,
        edge_index=graph.edges,
        edge_attr=graph.edgesAttributes,
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

