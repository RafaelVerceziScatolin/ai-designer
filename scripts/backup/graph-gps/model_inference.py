import sys
import math
import torch
import pandas
from torch_geometric.data import Data
sys.path.append("D:\orguel_ml_library")
from orguel_ml import Graph

def CreateGraph(dataframe):
    # Calculate length, angle, and offset
    def Length(row):
        if row["circle"]: return 2 * math.pi * row["radius"]
        elif row["arc"]: return row["radius"] * (math.radians((row["end_angle"] - row["start_angle"]) % 360))
        return ((row["end_x"] - row["start_x"])**2 + (row["end_y"] - row["start_y"])**2) ** 0.5
    
    def Angle(row):
        if row["circle"] or row["arc"]: return 0.0
        return math.atan2(row["end_y"] - row["start_y"], row["end_x"] - row["start_x"]) % math.pi
    
    def Offset(row):
        if row["circle"] or row["arc"]: return 0.0
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
from orguel_ml import GraphGPSNetwork

def main():

    # ✅ Step 1: Receive CSV string from AutoCAD
    csv_file = "C:\\Users\\Rafael\\Desktop\\autocad_ml\\database.csv"
    
    dataframe = pandas.read_csv(csv_file, header=None)
    dataframe.columns = ['handle', 'start_x', 'start_y', 'end_x', 'end_y', 'circle', 'arc', 'radius', 'start_angle', 'end_angle']

    # ✅ Step 2: Convert CSV string into a graph
    graph: Data = CreateGraph(dataframe)  # Convert AutoCAD data into a graph
    
    # ✅ Step 3: Load trained GNN model
    model = GraphGPSNetwork()
    #model.load_state_dict(torch.load(r"C:\Users\Rafael\Desktop\autocad_ml\GraphGPSNetwork.pt"))
    model.load_state_dict(torch.load(r"C:\Users\Rafael\Desktop\autocad_ml\GraphGPSNetwork.pt", map_location=torch.device('cpu')))
    model.eval()
    
    # ✅ Step 4: Perform Inference
    with torch.no_grad():
        output = model(graph)  # Run through the model
        predictions = output.argmax(dim=1).tolist()  # Convert logits to class indices
    
    # ✅ Step 5: Convert predictions into a structured LISP-friendly list
    lisp_list = "(" + " ".join([f'("{h}" {p})' for h, p in zip(dataframe["handle"], predictions)]) + ")"
    
    # ✅ Step 6: Save it as a properly formatted LISP list
    with open("C:\\Users\\Rafael\\Desktop\\autocad_ml\\temp.txt", "w", encoding="utf-8") as f: f.write(lisp_list)

if __name__ == "__main__": main()
