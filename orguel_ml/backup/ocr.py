
def __get_characters():
    # Set your character set
    character_set = {
                        "characters": {
                            "type": {
                                "latin_upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                "latin_lower": "abcdefghijklmnopqrstuvwxyz",
                                "numbers": "0123456789",
                                "basic_latin": "!\"#$%&'*+,-./:;=?@\\_`|~()[]{}<>",
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
rotation_angles = [i * 22.5 for i in range(16)]
height_list = [8, 10, 12, 14, 18, 22.5, 27, 35, 80, 90]

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
                            "angle_difference": 0
                        }
    def __init__(self, dataframe):
        self.__dataframe = dataframe
        
        coordinates_x = dataframe[['start_x', 'end_x', 'cluster_insertion_x']]
        coordinates_y = dataframe[['start_y', 'end_y', 'cluster_insertion_y']]
        
        normalizedCoordinates = dataframe[['start_x', 'start_y', 'end_x', 'end_y']].copy()
        normalizedCoordinates[['start_x', 'end_x']] = (normalizedCoordinates[['start_x', 'end_x']] - coordinates_x.values.mean()) / coordinates_x.values.std()
        normalizedCoordinates[['start_y', 'end_y']] = (normalizedCoordinates[['start_y', 'end_y']] - coordinates_y.values.mean()) / coordinates_y.values.std()
        
        normalizedAngles = numpy.column_stack((numpy.sin(dataframe['angle']), numpy.cos(dataframe['angle'])))
        normalizedLengths = dataframe[['length']] / 176.53671467337412
        
        cluster = dataframe.groupby("cluster_id")
        
        self.classificationLabels = {
            "label": cluster['cluster_label'].first().values,
            "font": cluster['cluster_font'].first().values
        }
        self.regressionTargets = {
            "height": cluster['cluster_height'].first().values / 176.53671467337412,
            "rotation": cluster['cluster_rotation'].first().values,
            "insertion_x": ((cluster['cluster_insertion_x'].first() - coordinates_x.values.mean()) / coordinates_x.values.std()).values,
            "insertion_y": ((cluster['cluster_insertion_y'].first() - coordinates_y.values.mean()) / coordinates_y.values.std()).values
        }
        self.nodesAttributes = numpy.hstack([normalizedCoordinates.values, normalizedAngles, normalizedLengths.values])
        self.edges: List[Tuple[int, int]] = []
        self.edgesAttributes: List[List[float]] = []

from typing import Dict
import os
import math
import pandas
import ezdxf
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

def __rotate_around_origin(x, y, rotation):
    rotation_rad = math.radians(rotation)
    cos_a = math.cos(rotation_rad)
    sin_a = math.sin(rotation_rad)
    rx = x * cos_a - y * sin_a
    ry = x * sin_a + y * cos_a
    return rx, ry

def CreateGraph(dxf_file, spacing=500, cluster_radius=200, character_set=character_set):
    
    # Load DXF
    fileName = os.path.basename(dxf_file)
    fontName = fileName.split("_")[0]
    height = float(fileName.split("_height")[-1].split("_local")[0].split("_global")[0].replace(".dxf", "").replace("_", "."))
    rotation = float(fileName.split("_local")[-1].split("_global")[-1].replace(".dxf", "").replace("_", ".")) if 'local' in fileName or 'global' in fileName else 0
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    lineCollector = [line for line in modelSpace if line.dxftype() == 'LINE']
    
    # Extract centroids and full line geometry
    dataframe: List[Dict] = []
    centroids: List[Tuple[float, float]] = []
    for line in lineCollector:
        start_x, start_y, _ = line.dxf.start
        end_x, end_y, _ = line.dxf.end
        centroid = ((start_x + end_x) / 2, (start_y + end_y) / 2)
        
        length = math.hypot(end_x - start_x, end_y - start_y)
        angle = math.atan2(end_y - start_y, end_x - start_x) % math.pi
        
        nlen = math.hypot(start_y - end_y, end_x - start_x)
        offset = 0.0 if nlen < 1e-12 else abs(start_x * ((start_y - end_y) / nlen) + start_y * ((end_x - start_x) / nlen))
        
        centroids.append(centroid)
        dataframe.append(
            {
                "cluster_id": 0,
                "cluster_font": character_set["font"]["encoding"][fontName],
                "cluster_height": height,
                "cluster_rotation": rotation / 360,
                "cluster_label": 0,
                "cluster_insertion_x": 0,
                "cluster_insertion_y":0,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "length": length,
                "angle": angle,
                "offset": offset,
                "circle": 0,
                "arc": 0,
                "radius": 0,
                "start_angle": 0,
                "end_angle": 0
            }
        )

    # Build KDTree for spatial lookup
    space2D = cKDTree(centroids)
    
    # Cluster lines based on known insertion points
    id=0
    y=0
    for group, characters in character_set["characters"]["type"].items():
        x=50
        for character in characters:
            insertionPoint = (x, y)
            if "_global" in fileName: insertionPoint = __rotate_around_origin(x, y, rotation)
            
            clusterIndexes = space2D.query_ball_point(insertionPoint, cluster_radius)
            for i in clusterIndexes:
                dataframe[i]["cluster_id"] = id
                dataframe[i]["cluster_label"] = character_set["characters"]["encoding"][character]
                dataframe[i]["cluster_insertion_x"] = insertionPoint[0]
                dataframe[i]["cluster_insertion_y"] = insertionPoint[1]
            
            id += 1
            x += spacing
        y -= spacing
                
    dataframe = pandas.DataFrame(dataframe)
    
    graph = Graph(dataframe)
    graph.ParallelDetection()
    graph.IntersectionDetection()
    
    # Convert all tensors
    graph.classificationLabels = {
        "label": torch.tensor(graph.classificationLabels["label"], dtype=torch.long),
        "font": torch.tensor(graph.classificationLabels["font"], dtype=torch.long)
    }
    
    graph.regressionTargets = {
        "height": torch.tensor(graph.regressionTargets["height"], dtype=torch.float),
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
        y=graph.classificationLabels["label"],
        font=graph.classificationLabels["font"],
        height=graph.regressionTargets["height"],
        rotation=graph.regressionTargets["rotation"],
        insertion_x=graph.regressionTargets["insertion_x"],
        insertion_y=graph.regressionTargets["insertion_y"],
        cluster_id=torch.tensor(dataframe["cluster_id"].values, dtype=torch.long)
    )

    #graph: Data = LaplacianEigenvectors(graph)
    
    return graph

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