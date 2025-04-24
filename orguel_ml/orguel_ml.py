from typing import List, Tuple
import math
import numpy
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

class Graph:
    __edge_attributes = {
                            "parallel": 0,
                            "colinear": 0,
                            "perpendicular_distance": 0,
                            "overlap_ratio": 0,
                            "point_intersection": 0,
                            "segment_intersection": 0,
                            "angle_difference": 0,
                            "perimeter_intersection": 0
                        }
    def __init__(self, dataframe):
        self.__dataframe = dataframe
        
        coordinates_x = dataframe[['start_x', 'end_x']]
        coordinates_y = dataframe[['start_y', 'end_y']]
        normalizedCoordinates = dataframe[['start_x', 'start_y', 'end_x', 'end_y']].copy()
        normalizedCoordinates[['start_x', 'end_x']] = (coordinates_x - coordinates_x.values.mean()) / coordinates_x.values.std()
        normalizedCoordinates[['start_y', 'end_y']] = (coordinates_y - coordinates_y.values.mean()) / coordinates_y.values.std()
        
        normalizedAngles = numpy.column_stack((numpy.sin(dataframe['angle']), numpy.cos(dataframe['angle'])))
        normalizedLengths = (dataframe[['length']] - dataframe["length"].min()) / \
                             (dataframe["length"].max() - dataframe["length"].min())
        circleFlags = dataframe[['circle']].values
        arcFlags = dataframe[['arc']].values
        
        self.classificationLabels = dataframe['layer'].map({"beam": 0, "column": 1, "eave": 2,
                                                            "slab_hole": 3, "stair": 4, "section": 5,
                                                            "info": 6}).values if "layer" in dataframe.columns else None
        self.nodesAttributes = numpy.hstack([normalizedCoordinates.values, normalizedAngles, normalizedLengths.values, circleFlags, arcFlags])
        self.edges: List[Tuple[int, int]] = []
        self.edgesAttributes: List[List[float]] = []
    
    @staticmethod
    def __overlap_ratios(lineA, lineB):
        xA1, yA1, xA2, yA2, lengthA = lineA[['start_x', 'start_y', 'end_x', 'end_y', 'length']]
        xB1, yB1, xB2, yB2, lengthB = lineB[['start_x', 'start_y', 'end_x', 'end_y', 'length']]
        
        # Compute direction vector
        dxA, dyA = xA2 - xA1, yA2 - yA1
        dir_lengthA = math.sqrt(dxA**2 + dyA**2)
        
        # Check if the line has zero length
        if dir_lengthA < 1e-12: return 0.0, 0.0
        
        # Compute unit vector
        ndxA, ndyA = dxA / dir_lengthA, dyA / dir_lengthA  # Unit vector

        # Project B's endpoints onto A
        tB1 = ( (xB1 - xA1) * ndxA + (yB1 - yA1) * ndyA )
        tB2 = ( (xB2 - xA1) * ndxA + (yB2 - yA1) * ndyA )

        # Get the projected range
        tmin, tmax = sorted([tB1, tB2])
        
        # Clip the projection range within [0, lengthA]
        overlap_start = max(0, tmin)
        overlap_end = min(lengthA, tmax)
        overlap_len   = max(0, overlap_end - overlap_start)
        
        # fraction of A that’s overlapped by B
        overlapAinB = overlap_len / lengthA
        
        # Now do the same from B’s perspective
        dxB, dyB = xB2 - xB1, yB2 - yB1
        dir_lengthB = math.sqrt(dxB**2 + dyB**2)
        
        if dir_lengthB < 1e-12: return 0.0, 0.0
        
        ndxB, ndyB = dxB / dir_lengthB, dyB / dir_lengthB
        
        # Project A’s endpoints onto B
        tA1 = ((xA1 - xB1) * ndxB + (yA1 - yB1) * ndyB)
        tA2 = ((xA2 - xB1) * ndxB + (yA2 - yB1) * ndyB)
        smin, smax = sorted([tA1, tA2])
        
        overlap_start_B = max(0, smin)
        overlap_end_B   = min(lengthB, smax)
        overlap_len_B   = max(0, overlap_end_B - overlap_start_B)
        
        overlapBinA = overlap_len_B / lengthB
        
        return overlapAinB, overlapBinA
    
    @staticmethod
    def __point_intersection(intersection, endpoints, threshold):
        ix, iy = intersection
        for ex, ey in endpoints:
            distance = math.dist((ix, iy), (ex, ey))  # Euclidean distance
            if distance <= threshold:
                return True
        return False
    
    @staticmethod
    def __create_arc(center, radius, startAngle, endAngle, segments=16):
        points = []
        startAngle = math.radians(startAngle)
        endAngle   = math.radians(endAngle)

        arcRange = endAngle - startAngle
        if arcRange < 0:
            arcRange += 2*math.pi

        step = arcRange / segments
        for s in range(segments+1):
            angle = startAngle + s*step
            px = center.x + radius * math.cos(angle)
            py = center.y + radius * math.sin(angle)
            points.append((px, py))
        return points

    def ParallelDetection(self, max_threshold=50, colinear_threshold=0.5, min_overlap_ratio=0.2, angle_tolerance=numpy.radians(0.01)):
        # create the 2D space for parallel detection
        space2D = cKDTree(self.__dataframe[['angle', 'offset']].values)
        
        # Query for nearby parallel lines in (angle, offset) space
        parallelList: List[Tuple[int, int, float, float, float]] = []  # i, j, distance, overlapA, overlapB
        for i in range(len(self.__dataframe)):
            line_i = self.__dataframe.iloc[i]
            if line_i['circle'] or line_i['arc']: continue
            
            angle_i, offset_i = self.__dataframe[['angle', 'offset']].values[i]
            nearbyLines = space2D.query_ball_point([angle_i, offset_i], r=max_threshold)
            
            for j in nearbyLines:
                if i >= j: continue
                
                line_j = self.__dataframe.iloc[j]
                if line_j['circle'] or line_j['arc']: continue
                
                overlapA, overlapB = self.__overlap_ratios(line_i, line_j)
                if min(overlapA, overlapB) <= min_overlap_ratio: continue
                
                angle_j, offset_j = self.__dataframe[['angle', 'offset']].values[j]
                if abs(angle_i - angle_j) >= angle_tolerance: continue
                
                # perpendicular distance between the two parallel lines
                distance = abs(offset_j - offset_i) / numpy.cos(angle_i)
                
                parallelList.append((i, j, distance, overlapA, overlapB))
                    
        min_distance = min(distance[2] for distance in parallelList)
        max_distance = max(distance[2] for distance in parallelList)
        max_min_distance = max_distance - min_distance if max_distance != min_distance else 1
        
        # Second pass to assign normalized distances to edges
        for (i, j, distance, overlapA, overlapB) in parallelList:
            for (a, b, overlapRatio) in [(i, j, overlapA), (j, i, overlapB)]:
                edgeAttributes = self.__edge_attributes.copy()
                edgeAttributes["parallel"] = 1
                edgeAttributes["colinear"] = 1 if distance < colinear_threshold else 0
                edgeAttributes["perpendicular_distance"] = (distance - min_distance) / max_min_distance
                edgeAttributes["overlap_ratio"] = overlapRatio
                self.edges.append((a, b))
                self.edgesAttributes.append(list(edgeAttributes.values()))
    
    def IntersectionDetection(self, threshold=0.5, co_linear_tolerance=numpy.radians(0.01)):
        # Step 1: Separate lines vs circles
        geometryCircular: List[Tuple[int, LineString]] = []
        geometryLines: List[Tuple[int, LineString]] = []
        for i, row in self.__dataframe.iterrows():
            if row["circle"]:
                center = Point(row["start_x"], row["start_y"])
                radius = row["radius"]
                circle = center.buffer(radius).boundary
                geometryCircular.append((i, circle))
            elif row["arc"]:
                center = Point(row["start_x"], row["start_y"])
                radius = row["radius"]
                startAngle = row["start_angle"]
                endAngle = row["end_angle"]
                arcPoints = self.__create_arc(center, radius, startAngle, endAngle, segments=16)
                arc = LineString(arcPoints)
                geometryCircular.append((i, arc))      
            else:
                line = LineString([(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])])
                geometryLines.append((i, line))
        
        # Build STRtree for lines
        space2D = STRtree([geometry for _, geometry in geometryLines])
        
        # Step 2: First pass: Compute all angle differences
        intersectionsList: List[Tuple[int, int, Tuple[float, float], float]] = [] # (i, j, intersection, angleDifference)
        for i, lineA in geometryLines:
            nearbyLines = space2D.query(lineA)  # Fetch only nearby lines
            
            for k in nearbyLines:
                j, lineB = geometryLines[k]
                
                if i >= j: continue  # Avoid duplicate processing
                
                # Angle check (skip nearly identical lines)
                angle_i = self.__dataframe.iloc[i]["angle"]
                angle_j = self.__dataframe.iloc[j]["angle"]
                angleDifference = abs(angle_i - angle_j)
                angleDifference = min(angleDifference, math.pi - angleDifference)
                if angleDifference < co_linear_tolerance: continue
                
                # Step 3: Check actual intersection
                intersection = lineA.intersection(lineB)
                if intersection.is_empty:
                    if lineA.distance(lineB) < threshold:
                        pA, pB = nearest_points(lineA, lineB)
                        intersection_x, intersection_y = (pA.x + pB.x) / 2, (pA.y + pB.y) / 2
                        intersection = Point(intersection_x, intersection_y)
                    else: continue
                
                # Step 4: Ensure intersection is always stored as a point
                if intersection.geom_type == "Point": intersection = (intersection.x, intersection.y)
                elif intersection.geom_type == "MultiPoint":intersection = list(intersection.geoms)[0].coords[0]
                else: intersection = intersection.interpolate(0.5, normalized=True).coords[0]
                
                intersectionsList.append((i, j, intersection, angleDifference))
                
        # Normalize angle differences
        min_angle_difference = min(intersection[3] for intersection in intersectionsList)
        max_angle_difference = max(intersection[3] for intersection in intersectionsList)
        max_min_angle_difference = max_angle_difference - min_angle_difference if max_angle_difference != min_angle_difference else 1
        
        # Step 5: Second pass: Add edges with normalized angleDifference
        for i, j, intersection, angleDifference in intersectionsList:
            normalizedAngle = (angleDifference - min_angle_difference) / max_min_angle_difference
            for a, b in [(i, j), (j, i)]:
                row = self.__dataframe.iloc[a]
                rowPoints = [(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])]
                pointIntersection = self.__point_intersection(intersection, rowPoints, threshold)
                edgeAttributes = self.__edge_attributes.copy()
                edgeAttributes["point_intersection"] = 1 if pointIntersection else 0
                edgeAttributes["segment_intersection"] = 0 if pointIntersection else 1
                edgeAttributes["angle_difference"] = normalizedAngle
                self.edges.append((a, b))
                self.edgesAttributes.append(list(edgeAttributes.values()))
        
        # Step 5: circle-line perimeter intersections
        for i, circularElement in geometryCircular:
            nearbyLines = space2D.query(circularElement)
            
            for k in nearbyLines:
                j, line = geometryLines[k]
                
                # Check actual intersection
                intersection = circularElement.intersection(line)
                if intersection.is_empty:
                    if circularElement.distance(line) < threshold:
                        pA, pB = nearest_points(circularElement, line)
                        intersection_x, intersection_y = (pA.x + pB.x) / 2, (pA.y + pB.y) / 2
                        intersection = Point(intersection_x, intersection_y)
                    else: continue
                
                # Ensure intersection is always stored as a point
                if intersection.geom_type == "Point": intersection = (intersection.x, intersection.y)
                elif intersection.geom_type == "MultiPoint":intersection = list(intersection.geoms)[0].coords[0]
                else: intersection = intersection.interpolate(0.5, normalized=True).coords[0]
                
                row = self.__dataframe.iloc[j]
                rowPoints = [(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])] 
                pointIntersection = self.__point_intersection(intersection, rowPoints, threshold)
                
                edgeAttributes = self.__edge_attributes.copy()
                edgeAttributes["perimeter_intersection"] = 1
                self.edges.append((i, j))
                self.edgesAttributes.append(list(edgeAttributes.values()))
                
                edgeAttributes = self.__edge_attributes.copy()
                edgeAttributes["point_intersection"] = 1 if pointIntersection else 0
                edgeAttributes["segment_intersection"] = 0 if pointIntersection else 1
                self.edges.append((j, i))
                self.edgesAttributes.append(list(edgeAttributes.values()))
                         
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

def LaplacianEigenvectors(graph, k=2):
    try:
        # Step 1: Create scipy sparse Laplacian matrix
        L = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes).astype(float)
        L = csgraph.laplacian(L, normed=True)
        
        # Step 2: Compute the first k+1 eigenvectors (skip the trivial 0-th)
        eigenvals, eigenvecs = eigsh(L, k=k + 1, which='SM')
        eigenvecs = eigenvecs[:, 1:]  # Skip first trivial one
        
    except ArpackNoConvergence as e:
        print("[Warning] ARPACK did not fully converge. Using partial results.")
        if e.eigenvectors is not None and e.eigenvectors.shape[1] > k: eigenvecs = e.eigenvectors[:, 1:k + 1]
        else: eigenvecs = numpy.zeros((graph.num_nodes, k))
    
    except Exception as e:
        print(f"[Error] Laplacian eigenvector computation failed: {e}")
        eigenvecs = igenvecs = numpy.zeros((graph.num_nodes, k))
    
    # Step 3: Convert to torch and concatenate with node features
    eigenvecs = torch.from_numpy(eigenvecs).float()
    graph.x = torch.cat([graph.x, eigenvecs], dim=1)
    
    return graph

from typing import Dict
import ezdxf
import torch
import pandas
from torch_geometric.data import Data

def CreateGraph(dxf_file):
    dataframe: List[Dict] = [ ]
    
    doc = ezdxf.readfile(dxf_file)
    modelSpace = doc.modelspace()
    lineCollector = [line for line in modelSpace if line.dxftype() == 'LINE']
    circleCollector = [circle for circle in modelSpace if circle.dxftype() == 'CIRCLE']
    arcCollector = [arc for arc in modelSpace if arc.dxftype() == 'ARC']
    
    # create a dataframe with line information
    for line in lineCollector:
        start_x, start_y, _ = line.dxf.start
        end_x, end_y, _ = line.dxf.end
        layer = line.dxf.layer
        
        length = math.hypot(end_x - start_x, end_y - start_y)
        angle = math.atan2(end_y - start_y, end_x - start_x) % math.pi
        
        nlen = math.hypot(start_y - end_y, end_x - start_x)
        offset = 0.0 if nlen < 1e-12 else abs(start_x * ((start_y - end_y) / nlen) + start_y * ((end_x - start_x) / nlen))
        
        dataframe.append(
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "length": length,
                "angle": angle,
                "offset": offset,
                "layer": layer,
                "circle": 0,
                "arc": 0,
                "radius": 0,
                "start_angle": 0,
                "end_angle": 0
            }
        )
    
    for circle in circleCollector:
        center_x, center_y, _ = circle.dxf.center
        radius = circle.dxf.radius
        perimeter = 2 * math.pi * radius
        layer = circle.dxf.layer
        
        dataframe.append(
            {
                "start_x": center_x,
                "start_y": center_y,
                "end_x": center_x,
                "end_y": center_y,
                "length": perimeter,
                "angle": 0.0,
                "offset": 0.0,
                "layer": layer,
                "circle": 1,
                "arc": 0,
                "radius": radius,
                "start_angle": 0,
                "end_angle": 0
            }
        )
    
    for arc in arcCollector:
        center_x, center_y, _ = arc.dxf.center
        radius = arc.dxf.radius
        startAngle = arc.dxf.start_angle
        endAngle = arc.dxf.end_angle
        arcLength = radius * math.radians((endAngle - startAngle) % 360)
        layer = arc.dxf.layer
        
        dataframe.append(
            {
                "start_x": center_x,
                "start_y": center_y,
                "end_x": center_x,
                "end_y": center_y,
                "length": arcLength,
                "angle": 0.0,
                "offset": 0.0,
                "layer": layer,
                "circle": 0,
                "arc": 1,
                "radius": radius,
                "start_angle": startAngle,
                "end_angle": endAngle
            }
        )
    
    dataframe = pandas.DataFrame(dataframe)
    
    graph = Graph(dataframe)
    graph.ParallelDetection()
    graph.IntersectionDetection()
    
    graph.classificationLabels = torch.tensor(graph.classificationLabels, dtype=torch.long)
    graph.nodesAttributes = torch.tensor(graph.nodesAttributes, dtype=torch.float)
    graph.edges = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
    graph.edgesAttributes = torch.tensor(graph.edgesAttributes, dtype=torch.float)
    
    graph = Data(x=graph.nodesAttributes, edge_index=graph.edges, edge_attr=graph.edgesAttributes, y=graph.classificationLabels)
    #graph: Data = LaplacianEigenvectors(graph)
    
    return graph

import os
from tqdm import tqdm
from multiprocessing import Pool

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, dxf_files, CreateGraph=CreateGraph, chunksize=4):
        self.graphs = []
        with Pool(processes=os.cpu_count()) as pool:
            for graph in tqdm(pool.imap_unordered(CreateGraph, dxf_files, chunksize=chunksize), total=len(dxf_files), desc="Creating graphs"):
                self.graphs.append(graph)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphGPSNetwork(nn.Module):
    def __init__(self, edge_attributes=8, transformer_heads=4, transformer_layers=2):
        super().__init__()
        
        # Define input and embedding dimensions
        coordinates_dimensions = 9  # (start_x, start_y, end_x, end_y, sin(angle), cos(angle), length, circle, arc)
        embedding_dimensions = 64  # We map 4D coordinates into a 32D embedding
        
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
        
        # Fully connected layers for classification
        self.fully_connected = nn.Linear(32, 32)
        self.out = nn.Linear(32, 7) # classes: beam, column, eave, slab_hole, stair, section, info
        
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
        
        # Fully connected layers for classification
        x = F.relu(self.fully_connected(x))
        
        # Apply dropout before final layer
        x = F.dropout(x, p=0.4, training=self.training) # Dropout to prevent overfitting
        
        # Raw logits
        output = self.out(x)
        
        return output

from collections import Counter

def BalanceClassWeights(dataset, device="cpu", smoothing_factor=0.2, classification_labels=7):
    labels = [graph.y.tolist() for graph in dataset]
    labels_flattened = [label for sublist in labels for label in sublist]
    total_class = Counter(labels_flattened)
    total_samples = sum(total_class.values())
    
    class_weights = [(total_samples / (2 * total_class[i])) ** smoothing_factor for i in range(classification_labels)]
    
    return torch.tensor(class_weights, dtype=torch.float).to(device)
