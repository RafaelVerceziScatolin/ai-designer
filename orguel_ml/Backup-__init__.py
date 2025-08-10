
F = _dataframe_field

start_x, start_y = dataframe[F.start_x], dataframe[F.start_y]
end_x, end_y = dataframe[F.end_x], dataframe[F.end_y]
length = dataframe[F.length]
angle = dataframe[F.angle]

# Normalize coordinates
coordinates_x = torch.hstack([start_x, end_x])
coordinates_y = torch.hstack([start_y, end_y])
coordinates_x_mean, coordinates_x_std = coordinates_x.mean(), coordinates_x.std()
coordinates_y_mean, coordinates_y_std = coordinates_y.mean(), coordinates_y.std()

normalized_start_x = (start_x - coordinates_x_mean) / coordinates_x_std
normalized_start_y = (start_y - coordinates_y_mean) / coordinates_y_std
normalized_end_x = (end_x - coordinates_x_mean) / coordinates_x_std
normalized_end_y = (end_y - coordinates_y_mean) / coordinates_y_std

normalized_coordinates = torch.stack([normalized_start_x, normalized_start_y,
                                        normalized_end_x, normalized_end_y], dim=1) # shape (N, 4) for ML model

# Normalize angles and lengths
normalized_angle = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1) # shape (N, 2)
normalized_length = (length / length.max()).reshape([-1, 1]) # shape (N, 1)

# Flags
line_flag = dataframe[F.line_flag].reshape([-1, 1]) # shape (N, 1)
circle_flag = dataframe[F.circle_flag].reshape([-1, 1]) # shape (N, 1)
arc_flag = dataframe[F.arc_flag].reshape([-1, 1]) # shape (N, 1)

# Node attributes
self.nodeAttributes = torch.hstack([normalized_coordinates, normalized_angle,
                                    normalized_length, line_flag, circle_flag, arc_flag])

self._dataframe = dataframe