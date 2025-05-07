import cupy
import cudf
import cuspatial
from cupyx.scipy.spatial import cKDTree

class Graph:
    def __is_point_intersection(self, px, py, x1, y1, x2, y2, threshold):
        d1 = cupy.hypot(px - x1, py - y1)
        d2 = cupy.hypot(px - x2, py - y2)
        return (d1 <= threshold) | (d2 <= threshold)

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

        start_x = df_lines['start_x'].to_cupy()
        start_y = df_lines['start_y'].to_cupy()
        end_x = df_lines['end_x'].to_cupy()
        end_y = df_lines['end_y'].to_cupy()

        points_x[::2] = start_x
        points_y[::2] = start_y
        points_x[1::2] = end_x
        points_y[1::2] = end_y

        geometry = cuspatial.GeometryColumn.from_lines(
            cuspatial.make_linestring(offsets, points_x, points_y)
        )

        result = cuspatial.linestring_intersection(geometry, geometry)
        i_indices = result["lhs_index"].to_cupy()
        j_indices = result["rhs_index"].to_cupy()
        intersections_x = result["x"].to_cupy()
        intersections_y = result["y"].to_cupy()

        angles = df_lines['angle'].to_cupy()

        for idx in range(len(i_indices)):
            i, j = int(i_indices[idx]), int(j_indices[idx])
            if i >= j:
                continue

            angle_i = angles[i]
            angle_j = angles[j]
            minAngleDifference = (angle_j - angle_i) % cupy.pi
            minAngleDifference = min(minAngleDifference, cupy.pi - minAngleDifference)
            if minAngleDifference < co_linear_tolerance:
                continue

            px, py = intersections_x[idx], intersections_y[idx]

            angleDifference = (angle_j - angle_i) % (2 * cupy.pi)

            for a, b in [(i, j), (j, i)]:
                x1, y1 = start_x[a], start_y[a]
                x2, y2 = end_x[a], end_y[a]
                isPoint = self.__is_point_intersection(px, py, x1, y1, x2, y2, threshold)

                attr = cupy.zeros(len(self.__edge_attributes), dtype=cupy.float32)
                attr[self.point_intersection] = 1 if isPoint else 0
                attr[self.segment_intersection] = 0 if isPoint else 1
                attr[self.angle_difference] = minAngleDifference / (cupy.pi / 2)
                attr[self.angle_difference_sin] = cupy.sin(angleDifference)
                attr[self.angle_difference_cos] = cupy.cos(angleDifference)

                self.edges.append((int(df_lines.index[a]), int(df_lines.index[b])))
                self.edgeAttributes.append(list(attr.tolist()))

        # Endpoint-to-segment proximity intersection check
        endpoints_x = cupy.concatenate([start_x, end_x])
        endpoints_y = cupy.concatenate([start_y, end_y])
        endpoints_idx = cupy.arange(len(endpoints_x))

        segment_centers_x = (start_x + end_x) / 2
        segment_centers_y = (start_y + end_y) / 2

        tree = cKDTree(cupy.stack([segment_centers_x, segment_centers_y], axis=1))

        for k in range(len(endpoints_x)):
            ex, ey = endpoints_x[k], endpoints_y[k]
            neighbors = tree.query_ball_point([ex, ey], r=threshold)

            for seg in neighbors:
                # Avoid self-linking
                if k // 2 == seg:
                    continue

                # Midpoint between endpoint and segment center
                mx = (ex + segment_centers_x[seg]) / 2
                my = (ey + segment_centers_y[seg]) / 2

                angle_i = angles[k // 2]
                angle_j = angles[seg]
                angleDifference = (angle_j - angle_i) % (2 * cupy.pi)
                minAngleDifference = min(angleDifference, cupy.pi - angleDifference)

                for a, b in [(k // 2, seg), (seg, k // 2)]:
                    attr = cupy.zeros(len(self.__edge_attributes), dtype=cupy.float32)
                    attr[self.point_intersection] = 1
                    attr[self.segment_intersection] = 0
                    attr[self.angle_difference] = minAngleDifference / (cupy.pi / 2)
                    attr[self.angle_difference_sin] = cupy.sin(angleDifference)
                    attr[self.angle_difference_cos] = cupy.cos(angleDifference)

                    self.edges.append((int(df_lines.index[a]), int(df_lines.index[b])))
                    self.edgeAttributes.append(list(attr.tolist()))
