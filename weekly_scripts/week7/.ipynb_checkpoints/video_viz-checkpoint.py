import pandas as pd
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ========== 参数配置 ==========
INPUT_CSV = 'runs/track/crowdhuman_yolov5m_osnet_x0_5_MSMT1748/nuclear_long_3.csv'
INPUT_VIDEO = 'data/video_viz/nuclear_long_3.mp4'
OUTPUT_VIDEO = 'data/video_viz/viz/nuclear_long_3_viz.mp4'

# 轨迹参数   _
TRAJECTORY_COLOR = (0, 255, 0)  # 轨迹线颜色
TRAJECTORY_WIDTH = 1  # 轨迹线宽
MAX_TRAJECTORY_LENGTH = 1000  # 最大轨迹长度(帧数)
MIN_MOVE_DISTANCE = 5  # 最小移动距离(像素)
MAX_CONNECT_DISTANCE = 50  # 最大允许连线距离（像素）

# 热力图参数
GRID_SIZE = 20  # 密度统计网格大小(像素)
HEATMAP_ALPHA = 0.3  # 热力图透明度
DECAY_RATE = 1  # 热力衰减系数
BLUR_SIGMA = 15  # 高斯模糊强度

# ========== 初始化 ==========
df = pd.read_csv(INPUT_CSV)
cap = cv2.VideoCapture(INPUT_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建轨迹存储字典 {id: [points]}
trajectories = {}

# 初始化密度网格
grid_rows = height // GRID_SIZE
grid_cols = width // GRID_SIZE
cumulative_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint32)

# 视频写入对象
output_video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (width, height)
)


def get_bottom_center(x1, y1, w, h):
    """计算检测框下边沿中点坐标（地面投影点）"""
    return (int(x1 + w / 2), int(y1 + h))


def update_trajectories(frame_data):
    """更新所有目标的轨迹"""
    # 临时存储当前帧的所有底部中点
    current_points = {}

    for _, row in frame_data.iterrows():
        track_id = row['ID']
        x1, y1, w, h = row['x1'], row['y1'], row['w'], row['h']
        point = get_bottom_center(x1, y1, w, h)
        current_points[track_id] = point

        # 初始化或更新轨迹
        if track_id not in trajectories:
            trajectories[track_id] = []

        # 只保留最近MAX_TRAJECTORY_LENGTH个点
        trajectories[track_id] = trajectories[track_id][-MAX_TRAJECTORY_LENGTH:]

        # 添加新点（如果移动距离足够）
        if len(trajectories[track_id]) == 0:
            trajectories[track_id].append(point)
        else:
            last_point = trajectories[track_id][-1]
            if np.linalg.norm(np.array(point) - np.array(last_point)) > MIN_MOVE_DISTANCE:
                trajectories[track_id].append(point)

    return current_points


def draw_trajectories(frame):
    """绘制所有轨迹（带智能连线逻辑）"""
    for track_id, points in trajectories.items():
        # 绘制轨迹线（智能连线）
        for i in range(1, len(points)):
            # 计算两点距离
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
            # 只有距离小于阈值时才连线
            if distance <= MAX_CONNECT_DISTANCE:
                cv2.line(frame, points[i - 1], points[i],
                         TRAJECTORY_COLOR, TRAJECTORY_WIDTH)

        # # 标记最新位置
        # if points:
        #     cv2.circle(frame, points[-1], 5, (0, 0, 255), -1)  # 红色终点标记


def update_cumulative_grid(points_dict):
    """更新累积网格（无衰减）"""
    current_grid = np.zeros_like(cumulative_grid)
    for point in points_dict.values():  # 遍历字典中的所有点
        x, y = point
        col = min(max(0, x // GRID_SIZE), grid_cols-1)
        row = min(max(0, y // GRID_SIZE), grid_rows-1)
        current_grid[row, col] += 1
    return current_grid


def create_heatmap(grid):
    """生成热力图可视化"""
    # 非线性增强（gamma校正，使低值区更明显）
    enhanced = np.power(grid.astype(float), 0.6)

    # 放大到原图尺寸
    enlarged = cv2.resize(enhanced, (width, height), interpolation=cv2.INTER_NEAREST)

    # 高斯模糊
    blurred = gaussian_filter(enlarged, sigma=BLUR_SIGMA)

    # 归一化并应用颜色
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(np.uint8(normalized), cv2.COLORMAP_JET)

# ========== 主处理循环 ==========
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_data = df[df['frame'] == current_frame]

    # 1. 更新并绘制轨迹
    current_points = update_trajectories(frame_data)
    draw_trajectories(frame)

    # === 累积热力图更新 ===
    cumulative_grid += update_cumulative_grid(current_points)  # 直接累加
    heatmap = create_heatmap(cumulative_grid)

    # 3. 叠加显示
    blended = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA,
                              heatmap, HEATMAP_ALPHA, 0)

    # # 可选：绘制网格线
    # if GRID_SIZE <= 30:  # 只在网格较小时显示
    #     for i in range(0, width, GRID_SIZE):
    #         cv2.line(blended, (i, 0), (i, height), (50, 50, 50), 1)
    #     for j in range(0, height, GRID_SIZE):
    #         cv2.line(blended, (0, j), (width, j), (50, 50, 50), 1)

    # 输出
    output_video.write(blended)
    cv2.imshow('Trajectories & Footprint Heatmap', blended)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()