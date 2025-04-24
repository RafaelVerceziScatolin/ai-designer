from typing import List, Tuple
import sys
import math
import torch
import pandas
import numpy
from torch_geometric.data import Data
sys.path.append("D:\orguel_ml_library")
from orguel_ml import Graph as BaseGraph

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
        
        coordinates_x = dataframe[['start_x', 'end_x']]
        coordinates_y = dataframe[['start_y', 'end_y']]
        
        normalizedCoordinates = dataframe[['start_x', 'start_y', 'end_x', 'end_y']].copy()
        normalizedCoordinates[['start_x', 'end_x']] = (normalizedCoordinates[['start_x', 'end_x']] - coordinates_x.values.mean()) / coordinates_x.values.std()
        normalizedCoordinates[['start_y', 'end_y']] = (normalizedCoordinates[['start_y', 'end_y']] - coordinates_y.values.mean()) / coordinates_y.values.std()
        
        normalizedAngles = numpy.column_stack((numpy.sin(dataframe['angle']), numpy.cos(dataframe['angle'])))
        normalizedLengths = dataframe[['length']] / 176.53671467337412
        
        self.nodesAttributes = numpy.hstack([normalizedCoordinates.values, normalizedAngles, normalizedLengths.values])
        self.edges: List[Tuple[int, int]] = []
        self.edgesAttributes: List[List[float]] = []

def CreateGraph(dataframe):
    # Calculate length, angle, and offset
    def Length(row):
        return ((row["end_x"] - row["start_x"])**2 + (row["end_y"] - row["start_y"])**2) ** 0.5
    
    def Angle(row):
        return math.atan2(row["end_y"] - row["start_y"], row["end_x"] - row["start_x"]) % math.pi
    
    def Offset(row):
        nlen = math.hypot(row["start_y"] - row["end_y"], row["end_x"] - row["start_x"])
        if nlen < 1e-12: return 0.0
        return abs(row["start_x"] * ((row["start_y"] - row["end_y"]) / nlen) +
                   row["start_y"] * ((row["end_x"] - row["start_x"]) / nlen))
    
    dataframe["length"] = dataframe.apply(Length, axis=1)
    dataframe["angle"] = dataframe.apply(Angle, axis=1)
    dataframe["offset"] = dataframe.apply(Offset, axis=1)
    
    graph = Graph(dataframe)
    graph.ParallelDetection()
    graph.IntersectionDetection()
    
    graph.nodesAttributes = torch.tensor(graph.nodesAttributes, dtype=torch.float)
    graph.edges = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
    graph.edgesAttributes = torch.tensor(graph.edgesAttributes, dtype=torch.float)
    
    return Data(x=graph.nodesAttributes, edge_index=graph.edges, edge_attr=graph.edgesAttributes, y=None)

import sys
from orguel_ml.ocr import OCRNetwork

def main():

    # ✅ Step 1: Receive CSV string from AutoCAD
    csv_file = "C:\\Users\\Rafael\\Desktop\\text_recognition\\database.csv"
    
    dataframe = pandas.read_csv(csv_file, header=None)
    dataframe.columns = ['handle', 'start_x', 'start_y', 'end_x', 'end_y']

    # ✅ Step 2: Convert CSV string into a graph
    graph: Data = CreateGraph(dataframe)  # Convert AutoCAD data into a graph
    
    # ✅ Step 3: Load trained GNN model
    model = OCRNetwork()
    model.load_state_dict(torch.load(r"C:\Users\Rafael\Desktop\text_recognition\OCRNetwork.pt", map_location=torch.device('cpu')))
    model.eval()
    
    # ✅ Step 4: Perform Inference
    with torch.no_grad():
        output = model(graph)  # Run through the model
        predictions = output.argmax(dim=1).tolist()  # Convert logits to class indices
    
    # ✅ Step 5: Convert predictions into a structured LISP-friendly list
    lisp_list = "(" + " ".join([f'("{h}" {p})' for h, p in zip(dataframe["handle"], predictions)]) + ")"
    
    # ✅ Step 6: Save it as a properly formatted LISP list
    with open("C:\\Users\\Rafael\\Desktop\\text_recognition\\temp.txt", "w", encoding="utf-8") as f: f.write(lisp_list)

if __name__ == "__main__": main()
