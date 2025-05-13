# GPU version of IntersectionDetection using OBBs and CuPy
create_obb hilbert_sort create_bins_spatial check_overlap_sat segment_intersection endpoint_threshold_check


def GPU_IntersectionDetection(self, threshold=0.5, co_linear_tolerance=cupy.radians(0.01)):
    dataframe = self._dataframe.copy()

    # Step 1: Filter lines and circular elements
    isLine = (dataframe['circle'] == 0) & (dataframe['arc'] == 0)
    lines = dataframe[isLine].reset_index(drop=True)
    circular_elements = dataframe[~isLine].reset_index(drop=True)

    # Extract coordinates
    angle = lines['angle'].to_cupy()
    start_x = lines['start_x'].to_cupy()
    start_y = lines['start_y'].to_cupy()
    end_x = lines['end_x'].to_cupy()
    end_y = lines['end_y'].to_cupy()

    # Compute line midpoints and OBBs
    obbs = self.create_obb(start_x, start_y, end_x, end_y, width=threshold)
    obb_centroids = obbs.mean(axis=1)
    mid_x = obb_centroids[:, 0]
    mid_y = obb_centroids[:, 1]

    # Hilbert sort and dynamic binning
    hilbert_order = self.hilbert_sort(mid_x, mid_y)
    avg_lines_per_bin = 150
    num_bins = max(1, int(len(lines) / avg_lines_per_bin))
    bins_matrix, bin_counts = self.create_bins_spatial(hilbert_order, num_bins=num_bins)

    # Define neighbor depth based on percentage of total bins, with minimum of 10
    depth_percentage = 0.1
    neighbor_depth = max(10, int(num_bins * depth_percentage))

    # Step 2: Check pairwise overlaps within bins and neighbor bins (forward only)
    for bin_i in range(num_bins):
        for offset in range(neighbor_depth + 1):
            bin_j = bin_i + offset
            if bin_j >= num_bins:
                continue

            indices_i = bins_matrix[bin_i]
            indices_j = bins_matrix[bin_j]

            indices_i = indices_i[indices_i != -1]
            indices_j = indices_j[indices_j != -1]

            if indices_i.shape[0] == 0 or indices_j.shape[0] == 0:
                continue

            A_idx, B_idx = cupy.meshgrid(indices_i, indices_j, indexing='ij')
            A_idx = A_idx.flatten()
            B_idx = B_idx.flatten()

            valid_mask = A_idx < B_idx
            A_idx = A_idx[valid_mask]
            B_idx = B_idx[valid_mask]

            if A_idx.shape[0] == 0:
                continue

            obb_a = obbs[A_idx]
            obb_b = obbs[B_idx]

            overlap_mask = self.check_overlap_sat(obb_a, obb_b)
            A_idx = A_idx[overlap_mask]
            B_idx = B_idx[overlap_mask]

            if A_idx.shape[0] == 0:
                continue

            p1 = cupy.stack([start_x[A_idx], start_y[A_idx]], axis=1)
            p2 = cupy.stack([end_x[A_idx], end_y[A_idx]], axis=1)
            q1 = cupy.stack([start_x[B_idx], start_y[B_idx]], axis=1)
            q2 = cupy.stack([end_x[B_idx], end_y[B_idx]], axis=1)

            intersections, intersect_mask = self.segment_intersection(p1, p2, q1, q2)

            # Determine if fallback was used and if fallback returned valid midpoint
            fallback_ij = self.endpoint_threshold_check(p1, p2, q1, q2, threshold) #########################
            fallback_ji = self.endpoint_threshold_check(q1, q2, p1, p2, threshold) #########################

            fallback_valid_ij = ~cupy.isnan(fallback_ij).any(axis=1) #########################
            fallback_valid_ji = ~cupy.isnan(fallback_ji).any(axis=1) #########################
            
            angle_i = angle[A_idx]
            angle_j = angle[B_idx]
            angle_diff = (angle_j - angle_i) % (2 * cupy.pi)
            angle_diff_min = cupy.minimum(angle_diff, 2 * cupy.pi - angle_diff)

            angle_diff_mod = angle_diff_min % cupy.pi
            colinear_mask = angle_diff_mod >= co_linear_tolerance
            A_idx = A_idx[colinear_mask]
            B_idx = B_idx[colinear_mask]

            angle_diff = angle_diff[colinear_mask] 
            angle_diff_min = angle_diff_min[colinear_mask]
            intersect_mask = intersect_mask[colinear_mask] #########################
            fallback_valid_ij = fallback_valid_ij[colinear_mask] #########################
            fallback_valid_ji = fallback_valid_ji[colinear_mask] #########################

            normalized_angle_diff = angle_diff_min / (cupy.pi / 2)

            i_nodes = cupy.array(A_idx, dtype=cupy.int32)
            j_nodes = cupy.array(B_idx, dtype=cupy.int32)

            edges_ij = cupy.stack([i_nodes, j_nodes], axis=0)
            edges_ji = cupy.stack([j_nodes, i_nodes], axis=0)
            edges_all = cupy.concatenate([edges_ij, edges_ji], axis=1)

            num_edges = i_nodes.shape[0]

            point_ij = fallback_valid_ij | (intersect_mask & ~fallback_valid_ij) #########################
            point_ji = fallback_valid_ji | (intersect_mask & ~fallback_valid_ji) #########################

            segment_ij = ~point_ij #########################
            segment_ji = ~point_ji #########################

            attrs_ij = cupy.zeros((num_edges, 10), dtype=cupy.float32) #########################
            attrs_ij[:, self.point_intersection] = point_ij.astype(cupy.float32) #########################
            attrs_ij[:, self.segment_intersection] = segment_ij.astype(cupy.float32) #########################
            attrs_ij[:, self.angle_difference] = normalized_angle_diff #########################
            attrs_ij[:, self.angle_difference_sin] = cupy.sin(angle_diff) #########################
            attrs_ij[:, self.angle_difference_cos] = cupy.cos(angle_diff) #########################

            attrs_ji = cupy.zeros((num_edges, 10), dtype=cupy.float32) #########################
            attrs_ji[:, self.point_intersection] = point_ji.astype(cupy.float32) #########################
            attrs_ji[:, self.segment_intersection] = segment_ji.astype(cupy.float32) #########################
            attrs_ji[:, self.angle_difference] = normalized_angle_diff #########################
            attrs_ji[:, self.angle_difference_sin] = cupy.sin(angle_diff) #########################
            attrs_ji[:, self.angle_difference_cos] = cupy.cos(angle_diff) #########################

            attrs_all = cupy.concatenate([attrs_ij, attrs_ji], axis=0) #########################

            if self._size + 2 * num_edges > self._capacity:
                raise RuntimeError("Edge buffer overflow: increase preallocated edge size.")

            start = self._size
            end = start + edges_all.shape[1]
            self.edges[:, start:end] = edges_all
            self.edgeAttributes[start:end, :] = attrs_all
            self._size = end







