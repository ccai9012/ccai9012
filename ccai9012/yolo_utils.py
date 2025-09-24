"""
YOLO Utilities Module
====================

This module provides utilities for object detection, tracking, and visualization using YOLO models.
It offers a streamlined interface for processing videos with YOLO-based detection and tracking,
along with tools for visualizing detection results as trajectories and heatmaps.

The module is designed to simplify video analysis tasks for pedestrian tracking and crowd movement
analysis, with support for real-time visualization in Jupyter notebooks and comprehensive result export.

Main components:
- Video processing: Functions for processing videos with YOLO models and object tracking
- Trajectory visualization: Tools for visualizing tracked object paths and movement patterns
- Heatmap generation: Functionality for creating density heatmaps from tracked object positions
- Result export: Methods for saving processed videos and detection/tracking data

Usage:
    ### Load a pre-trained YOLO model
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    ### Detect and track objects in a video
    results_df = detect_and_track(
        video_path="input_video.mp4",
        model=model,
        output_dir="results"
    )

    ### Visualize trajectories and heatmap
    visualize_video(
        input_csv="results/input_video_results.csv",
        input_video="input_video.mp4",
        output_video="results/visualization.mp4"
    )
"""

import pandas as pd
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from PIL import Image
from IPython.display import display, clear_output


def detect_and_track(
        video_path,
        model,
        output_dir,
        save_video=True,
        show_in_notebook=True,
        tracker_config="bytetrack.yaml",
        classes_to_track=[0]
):
    """
    Detect and track objects in a video using a YOLO model with tracking (e.g., ByteTrack).

    Args:
        video_path (str): Path to the input video file.
        model: YOLO model object with .track() method.
        output_dir (str): Directory to save output video and CSV.
        save_video (bool): Whether to save annotated output video.
        show_in_notebook (bool): Whether to display frames in a notebook environment.
        tracker_config (str): Tracker config file (e.g., "bytetrack.yaml").
        classes_to_track (list[int]): List of class IDs to track (default [0] for person).

    Returns:
        pd.DataFrame: DataFrame of detection results.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_output.mp4")

    out_writer = None
    all_detections = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tracking
        results = model.track(
            frame,
            persist=True,
            classes=classes_to_track,
            tracker=tracker_config
        )
        annotated_frame = results[0].plot()

        # Initialize VideoWriter
        if save_video and out_writer is None:
            h, w = annotated_frame.shape[:2]
            out_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Extract detection info
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id
        ids = ids.int().tolist() if ids is not None else [-1] * len(bboxes)

        for obj_id, bbox in zip(ids, bboxes):
            x1, y1, x2, y2 = bbox
            all_detections.append({
                "frame": frame_index,
                "id": int(obj_id),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })

        # Notebook display
        if show_in_notebook:
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            clear_output(wait=True)
            display(Image.fromarray(img_rgb))

        # Save frame
        if save_video:
            out_writer.write(annotated_frame)

        frame_index += 1

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    df = pd.DataFrame(all_detections)
    csv_path = os.path.join(output_dir, f"{video_name}_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"Tracking complete. Results saved to {output_dir}")
    return df

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
    """
    Visualize pedestrian trajectories and heatmap from tracking CSV and video input.

    This function combines the original video with overlays showing tracked object trajectories
    and a density heatmap based on object positions. It's particularly useful for analyzing
    pedestrian movement patterns, crowd flow, and identifying high-traffic areas.

    Args:
        input_csv (str): Path to the CSV file containing tracking data, typically
                        generated by the detect_and_track function.
        input_video (str): Path to the original input video file.
        output_video (str): Path where the visualization video will be saved.
        grid_size (int): Size of grid cells in pixels for heatmap generation.
                        Smaller values create more detailed heatmaps but may be slower.
        heatmap_alpha (float): Opacity of the heatmap overlay (0.0 to 1.0).
        blur_sigma (int): Gaussian blur sigma for smoothing the heatmap.
                        Higher values create a more diffuse, smoother heatmap.
        trajectory_color (tuple): BGR color for trajectory lines, as (B,G,R).
        trajectory_width (int): Width of trajectory lines in pixels.
        max_trajectory_length (int): Maximum number of points to keep in each trajectory.
                                    Limits memory usage for long videos.
        min_move_distance (int): Minimum distance in pixels that an object must move
                                to register as a new trajectory point.
        max_connect_distance (int): Maximum allowed distance in pixels between consecutive
                                  trajectory points. Prevents connecting broken tracks.
        show_window (bool): Whether to display the visualization in a Jupyter notebook
                          during processing.
        save_output (bool): Whether to save the visualization as a video file.

    Returns:
        None
    """

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
            x1, y1, w, h = row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1']
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
