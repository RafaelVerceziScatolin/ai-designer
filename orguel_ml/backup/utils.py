import os
import math
import copy
import ezdxf
from ezdxf.math import Matrix44

def __rotate_file(arguments):
    file, rotation_angles, dxf_folder = arguments
    
    fileName = os.path.splitext(os.path.basename(file))[0]
    doc = ezdxf.readfile(file)
    
    for angle in rotation_angles:
        copyDoc = copy.deepcopy(doc)
        
        # Rotate all entities in modelspace around origin (0, 0, 0)
        angleRad = math.radians(angle)
        m = Matrix44.z_rotate(angleRad)
        
        # Apply transformation to all entities
        modelSpace = copyDoc.modelspace()
        for entity in modelSpace: entity.transform(m)
        
        suffix = f"_global{angle}" if angle != 0 else ""
        suffix = suffix.replace('.', '_')        
        outputPath = os.path.join(dxf_folder, f"{fileName}{suffix}.dxf")
        copyDoc.saveas(outputPath)
        print(f"Saved: {outputPath}")

from tqdm.notebook import tqdm
from multiprocessing import Pool

def RotateAroundOrigin(dxf_files, rotation_angles, dxf_folder):
    if 0 in rotation_angles: rotation_angles.remove(0)
    tasks = [(file, rotation_angles, dxf_folder) for file in dxf_files]
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(__rotate_file, tasks), total=len(tasks), desc="Rotating DXFs"))
