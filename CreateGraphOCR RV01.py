










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
        
        coordinates_x = dataframe[['start_x', 'end_x', 'character_insertion_x']]
        coordinates_y = dataframe[['start_y', 'end_y', 'character_insertion_y']]
        
        normalizedCoordinates = dataframe[['start_x', 'start_y', 'end_x', 'end_y']].copy()
        normalizedCoordinates[['start_x', 'end_x']] = (normalizedCoordinates[['start_x', 'end_x']] - coordinates_x.values.mean()) / coordinates_x.values.std()
        normalizedCoordinates[['start_y', 'end_y']] = (normalizedCoordinates[['start_y', 'end_y']] - coordinates_y.values.mean()) / coordinates_y.values.std()
        
        normalizedAngles = numpy.column_stack((numpy.sin(dataframe['angle']), numpy.cos(dataframe['angle'])))
        
        if dataframe['character_id'].nunique() == 1 and dataframe['character_id'].iloc[0] != 0:
            normalizedLengths = dataframe[['length']] / 176.53671467337412
            normalizedCharacterHeigth = dataframe[['character_height']] / 176.53671467337412
            normalizedCharacterWidth = dataframe[['character_width']] / 176.53671467337412
        else:
            lengths = dataframe[['length', 'character_height', 'character_width']].copy().values
            if lengths.max() - lengths.min() > 1e-6:
                normalizedLengths = (dataframe[['length']] - lengths.min()) / (lengths.max() - lengths.min())
                normalizedCharacterHeigth = (dataframe[['character_height']] - lengths.min()) / (lengths.max() - lengths.min())
                normalizedCharacterWidth = (dataframe[['character_width']] - lengths.min()) / (lengths.max() - lengths.min())
            
            

        
        
        
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














def __rotate_and_scale(x, y, angle, scale):
    angle_rad = math.radians(angle)
    x_aligned = x * scale * math.cos(angle_rad) - y * scale * math.sin(angle_rad)
    y_aligned = y * scale * math.sin(angle_rad) + y * scale * math.cos(angle_rad)
    return x_aligned, y_aligned

def CreateGraph(dxf_file, character_positions, rotation=0, scale=1.0):
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
        angle = math.atan2(end_y - start_y, end_x - start_x) % math.pi

        nlen = math.hypot(start_y - end_y, end_x - start_x)
        offset = 0.0 if nlen < 1e-12 else abs(start_x * ((start_y - end_y) / nlen) + start_y * ((end_x - start_x) / nlen))

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
            "offset": offset,
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
    
    graph = Graph(dataframe)
    
    
    
    
    
    return dataframe
