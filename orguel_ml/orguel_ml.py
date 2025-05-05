from typing import List, Tuple
import math
import cupy
from torch.utils.dlpack import from_dlpack
from cupyx.scipy.spatial import cKDTree
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

class Graph:
    __edge_attributes = cupy.arange(10, dtype=cupy.int32)
    
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
    
    def __init__(self, dataframe):
        self.__dataframe = dataframe
        
        start_x, start_y = dataframe['start_x'], dataframe['start_y']
        end_x, end_y = dataframe['end_x'], dataframe['end_y']
        length = dataframe['length']
        angle = dataframe['angle']
        circle = dataframe['circle']
        arc = dataframe['arc']
        
        # Normalize coordinates
        coordinates_x = cupy.concatenate([start_x.to_cupy(), end_x.to_cupy()])
        coordinates_y = cupy.concatenate([start_y.to_cupy(), end_y.to_cupy()])
        
        normalized_start_x = (start_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_start_y = (start_y - coordinates_y.mean()) / coordinates_y.std()
        normalized_end_x = (end_x - coordinates_x.mean()) / coordinates_x.std()
        normalized_end_y = (end_y - coordinates_y.mean()) / coordinates_y.std()
        
        normalizedCoordinates = cupy.stack([normalized_start_x.to_cupy(), normalized_start_y.to_cupy(),
                                            normalized_end_x.to_cupy(), normalized_end_y.to_cupy()], axis=1)
        
        # Normalize angles and lengths
        normalizedAngles = cupy.stack([cupy.sin(angle.to_cupy()), cupy.cos(angle.to_cupy())], axis=1)
        normalizedLengths = (length / length.max()).to_cupy().reshape(-1, 1)
        
        # Flags
        circleFlags = circle.to_cupy().reshape(-1, 1)
        arcFlags = arc.to_cupy().reshape(-1, 1)
        
        # Classification labels
        self.classificationLabels = (dataframe['layer']
            .map({"beam": 0, "column": 1, "eave": 2, "slab_hole": 3, "stair": 4, "section": 5, "info": 6})
            .to_cupy() if "layer" in dataframe.columns else None)
        
        # Node attributes
        self.nodeAttributes = from_dlpack(cupy.hstack([
            normalizedCoordinates,
            normalizedAngles,
            normalizedLengths,
            circleFlags,
            arcFlags
        ]).toDlpack())
        
        self.edges: List[Tuple[int, int]] = []
        self.edgeAttributes: List[List[float]] = []
           
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
    def __is_point_intersection(intersection, endpoints, threshold):
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

    @staticmethod
    def __create_bins(dataframe, angle_size, offset_size):
        # Bin keys for each line
        dataframe['angle_bin'] = cupy.floor((dataframe['angle'] % cupy.pi) / angle_size).astype(cupy.int32)
        dataframe['offset_bin'] = cupy.floor(dataframe['offset'] / offset_size).astype(cupy.int32)
        return dataframe.groupby(['angle_bin', 'offset_bin'])
        
    def ParallelDetection(self, max_threshold=50, colinear_threshold=0.5, min_overlap_ratio=0.2,
                          angle_tolerance=cupy.radians(0.01), bin_angle_size=cupy.radians(0.25)):
        dataframe = self.__dataframe.copy()
        dataframe['angle'] = dataframe['angle'] % cupy.pi  # collapse symmetrical directions
        
        bins = self.__create_bins(dataframe, bin_angle_size, max_threshold)
        
        # Include neighboring bins for robustness
        for (angle, offset), _ in bins:
            neighboringBins = [(angle + da, offset + do) for da in (-1, 0, 1) for do in (-1, 0, 1)]
            neighboringBinsIndexes = [bins.groups[bin] for bin in neighboringBins if bin in bins.groups]
            
            lineIndexes = cupy.concatenate([i.to_cupy() for i in neighboringBinsIndexes])
            
            if len(lineIndexes) < 2: continue
            
            subset = dataframe.take(lineIndexes).reset_index(drop=True)
            
            coordinates = cupy.stack([subset['angle'].to_cupy(), subset['offset'].to_cupy()], axis=1)
            space2D = cKDTree(coordinates)
            
            # Query for nearby parallel lines in (angle, offset) space
            for i in range(len(subset)):
                if subset['circle'].iloc[i] or subset['arc'].iloc[i]: continue
                
                angle_i, offset_i = subset['angle'].iloc[i], subset['offset'].iloc[i]
                nearbyLines = space2D.query_ball_point([angle_i, offset_i], r=max_threshold)
                
                for j in nearbyLines:
                    if i >= j: continue
                    if subset['circle'].iloc[j] or subset['arc'].iloc[j]: continue
                    
                    line_i, line_j = subset.iloc[i], subset.iloc[j]
                    overlapA, overlapB = self.__overlap_ratios(line_i, line_j)
                    if min(overlapA, overlapB) <= min_overlap_ratio: continue
                    
                    angle_j = subset['angle'].iloc[j]
                    if cupy.abs(angle_i - angle_j) >= angle_tolerance: continue
                    
                    # perpendicular distance between the two parallel lines
                    distance = cupy.abs(subset['offset'].iloc[j] - offset_i) / cupy.cos(angle_i)
                    
                    # Assign normalized distances to edges
                    for (a, b, overlapRatio) in [(i, j, overlapA), (j, i, overlapB)]:
                        edgeAttributes = cupy.zeros(len(self.__edge_attributes), dtype=cupy.float32)
                        edgeAttributes[self.parallel] = 1
                        edgeAttributes[self.colinear] = 1 if distance < colinear_threshold else 0
                        edgeAttributes[self.perpendicular_distance] = distance / dataframe["length"].max()
                        edgeAttributes[self.overlap_ratio] = overlapRatio
                        self.edges.append((lineIndexes[a], lineIndexes[b]))
                        self.edgeAttributes.append(list(edgeAttributes.values()))
                                   
    def IntersectionDetection(self, threshold=0.5, co_linear_tolerance=cupy.radians(0.01)):
        dataframe = self.__dataframe.copy()
        
        lines = dataframe[(dataframe['circle'] == 0) & (dataframe['arc'] == 0)].reset_index(drop=True)
        
        
        
        
        
        
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
        
        # Step 2: Compute all angle differences
        for i, lineA in geometryLines:
            nearbyLines = space2D.query(lineA)  # Fetch only nearby lines
            
            for k in nearbyLines:
                j, lineB = geometryLines[k]
                
                if i >= j: continue  # Avoid duplicate processing
                
                # Angle check (skip nearly identical lines)
                angle_i = self.__dataframe.iloc[i]["angle"]
                angle_j = self.__dataframe.iloc[j]["angle"]
                minAngleDifference = (angle_j - angle_i) % math.pi
                minAngleDifference = min(minAngleDifference, math.pi - minAngleDifference)
                if minAngleDifference < co_linear_tolerance: continue
                
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
                
                angleDifference = (angle_j - angle_i) % (2 * math.pi)
                
                # Step 5: Add edges with normalized angleDifference
                for a, b in [(i, j), (j, i)]:
                    row = self.__dataframe.iloc[a]
                    rowPoints = [(row["start_x"], row["start_y"]), (row["end_x"], row["end_y"])]
                    isPointIntersection = self.__is_point_intersection(intersection, rowPoints, threshold)
                    edgeAttributes = self.__edge_attributes.copy()
                    edgeAttributes["point_intersection"] = 1 if isPointIntersection else 0
                    edgeAttributes["segment_intersection"] = 0 if isPointIntersection else 1
                    edgeAttributes["angle_difference"] = minAngleDifference / (math.pi / 2)
                    edgeAttributes["angle_difference_sin"] = math.sin(angleDifference)
                    edgeAttributes["angle_difference_cos"] = math.cos(angleDifference)
                    self.edges.append((a, b))
                    self.edgeAttributes.append(list(edgeAttributes.values()))
        
        # Step 6: circle-line perimeter intersections
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
                isPointIntersection = self.__is_point_intersection(intersection, rowPoints, threshold)
                
                edgeAttributes = self.__edge_attributes.copy()
                edgeAttributes["perimeter_intersection"] = 1
                self.edges.append((i, j))
                self.edgeAttributes.append(list(edgeAttributes.values()))
                
                edgeAttributes = self.__edge_attributes.copy()
                edgeAttributes["point_intersection"] = 1 if isPointIntersection else 0
                edgeAttributes["segment_intersection"] = 0 if isPointIntersection else 1
                self.edges.append((j, i))
                self.edgeAttributes.append(list(edgeAttributes.values()))
                         
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
import cudf
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
        angle = math.atan2(end_y - start_y, end_x - start_x) % (2*math.pi)
        
        dataframe.append(
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "length": length,
                "angle": angle,
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
                "layer": layer,
                "circle": 0,
                "arc": 1,
                "radius": radius,
                "start_angle": startAngle,
                "end_angle": endAngle
            }
        )
    
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
        
    graph = Graph(dataframe)
    graph.ParallelDetection()
    graph.IntersectionDetection()
    
    graph.classificationLabels = torch.tensor(graph.classificationLabels, dtype=torch.long)
    graph.nodeAttributes = torch.tensor(graph.nodeAttributes, dtype=torch.float)
    graph.edges = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
    graph.edgeAttributes = torch.tensor(graph.edgeAttributes, dtype=torch.float)
    
    graph = Data(x=graph.nodeAttributes, edge_index=graph.edges, edge_attr=graph.edgeAttributes, y=graph.classificationLabels)
    
    #graph: Data = LaplacianEigenvectors(graph)
    return graph

import os
from tqdm import tqdm
from multiprocessing import Pool

class CreateGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dxf_files, chunksize=4):
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


"""
import cupy
import cudf
import cuspatial

class Graph:
    def IntersectionDetection(self, threshold=0.5, co_linear_tolerance=cupy.radians(0.01)):
        df = self.__dataframe.copy()
        is_line = (df['circle'] == 0) & (df['arc'] == 0)
        df_lines = df[is_line].reset_index(drop=True)

        if len(df_lines) < 2:
            return

        # Create cuSpatial-compatible linestrings
        offsets = cupy.arange(0, len(df_lines) * 2 + 1, 2, dtype=cupy.int32)
        points_x = cupy.empty(len(df_lines) * 2, dtype=cupy.float32)
        points_y = cupy.empty(len(df_lines) * 2, dtype=cupy.float32)

        points_x[::2] = df_lines['start_x'].to_cupy()
        points_y[::2] = df_lines['start_y'].to_cupy()
        points_x[1::2] = df_lines['end_x'].to_cupy()
        points_y[1::2] = df_lines['end_y'].to_cupy()

        # Build linestring geometry
        geometry = cuspatial.GeometryColumn.from_lines(
            cuspatial.make_linestring(offsets, points_x, points_y)
        )

        # Compute pairwise intersections
        result = cuspatial.linestring_intersection(geometry, geometry)
        i_indices = result["lhs_index"].to_cupy()
        j_indices = result["rhs_index"].to_cupy()
        intersections = result["point"].copy()

        for idx in range(len(i_indices)):
            i, j = int(i_indices[idx]), int(j_indices[idx])
            if i >= j:
                continue

            angle_i = df_lines['angle'].iloc[i]
            angle_j = df_lines['angle'].iloc[j]
            minAngleDifference = (angle_j - angle_i) % cupy.pi
            minAngleDifference = min(minAngleDifference, cupy.pi - minAngleDifference)
            if minAngleDifference < co_linear_tolerance:
                continue

            intersection = intersections.iloc[idx]
            intersection_point = (intersection.x, intersection.y)
            angleDifference = (angle_j - angle_i) % (2 * cupy.pi)

            for a, b in [(i, j), (j, i)]:
                row = df_lines.iloc[a]
                rowPoints = [(row['start_x'], row['start_y']), (row['end_x'], row['end_y'])]
                isPoint = self.__is_point_intersection(intersection_point, rowPoints, threshold)
                attr = cupy.zeros(len(self.__edge_attributes), dtype=cupy.float32)
                attr[self.point_intersection] = 1 if isPoint else 0
                attr[self.segment_intersection] = 0 if isPoint else 1
                attr[self.angle_difference] = minAngleDifference / (cupy.pi / 2)
                attr[self.angle_difference_sin] = cupy.sin(angleDifference)
                attr[self.angle_difference_cos] = cupy.cos(angleDifference)
                self.edges.append((int(df_lines.index[a]), int(df_lines.index[b])))
                self.edgeAttributes.append(list(attr.tolist()))


"""