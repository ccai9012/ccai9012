import pandas as pd
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import os

def visualize_video(
    input_csv,
    input_video,
    output_video,
    grid_size=20,
    heatmap_alpha=0.3,
    blur_sigma=15,
    trajectory_color=(0, 255, 0),
    trajectory_width=1,
    max_trajectory_length=1000,
    min_move_distance=5,
    max_connect_distance=50,
    show_window=False,
    save_output=True
):
    """Visualize pedestrian trajectories and heatmap from tracking CSV and video input"""

    # ========== Initialization ==========
    df = pd.read_csv(input_csv)
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    trajectories = {}
    grid_rows = height // grid_size
    grid_cols = width // grid_size
    cumulative_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint32)

    if save_output:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        output_writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
    else:
        output_writer = None

    def get_bottom_center(x1, y1, w, h):
        return (int(x1 + w / 2), int(y1 + h))

    def update_trajectories(frame_data):
        current_points = {}
        for _, row in frame_data.iterrows():
            track_id = row['id']
            x1, y1, w, h = row['x1'], row['y1'], row['x2']-row['x1'], row['y2']-row['y1']
            point = get_bottom_center(x1, y1, w, h)
            current_points[track_id] = point

            if track_id not in trajectories:
                trajectories[track_id] = []

            trajectories[track_id] = trajectories[track_id][-max_trajectory_length:]

            if len(trajectories[track_id]) == 0 or \
                np.linalg.norm(np.array(point) - np.array(trajectories[track_id][-1])) > min_move_distance:
                trajectories[track_id].append(point)

        return current_points

    def draw_trajectories(frame):
        for track_id, points in trajectories.items():
            for i in range(1, len(points)):
                if np.linalg.norm(np.array(points[i]) - np.array(points[i - 1])) <= max_connect_distance:
                    cv2.line(frame, points[i - 1], points[i], trajectory_color, trajectory_width)

    def update_cumulative_grid(points_dict):
        current_grid = np.zeros_like(cumulative_grid)
        for point in points_dict.values():
            x, y = point
            col = min(max(0, x // grid_size), grid_cols - 1)
            row = min(max(0, y // grid_size), grid_rows - 1)
            current_grid[row, col] += 1
        return current_grid

    def create_heatmap(grid):
        enhanced = np.power(grid.astype(float), 0.6)
        enlarged = cv2.resize(enhanced, (width, height), interpolation=cv2.INTER_NEAREST)
        blurred = gaussian_filter(enlarged, sigma=blur_sigma)
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(np.uint8(normalized), cv2.COLORMAP_JET)

    # ========== Processing Loop ==========
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_data = df[df['frame'] == current_frame]

        current_points = update_trajectories(frame_data)
        draw_trajectories(frame)
        cumulative_grid += update_cumulative_grid(current_points)
        heatmap = create_heatmap(cumulative_grid)

        blended = cv2.addWeighted(frame, 1 - heatmap_alpha, heatmap, heatmap_alpha, 0)

        if save_output:
            output_writer.write(blended)

        if show_window:
            # Show in Jupyter Notebook
            from IPython.display import display, clear_output
            import matplotlib.pyplot as plt
        
            rgb_frame = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 9))
            plt.imshow(rgb_frame)
            plt.axis('off')
            clear_output(wait=True)
            display(plt.gcf())
            plt.clf()
        
            # Optionally support 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if output_writer:
        output_writer.release()
    if show_window:
        cv2.destroyAllWindows()
