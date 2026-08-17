# CV Models (Segmentation, Detection, Tracking)

**Related API:** [`ccai9012.yolo_utils`](../api/ccai9012/yolo_utils.html) · [`ccai9012.svi_utils`](../api/ccai9012/svi_utils.html) 

### Overview
**Category:** Perception & Prediction from Visual Data

**Modular Components:**
- Object Detection/Tracking with YOLO
- Semantic Segmentation Model
- Trajectory Extraction
- Visualization

### Use Cases
- What factors influence walking behavior?
- How does visual cleanliness (graffiti, trash, lighting) relate to perceived safety?
- Can we predict CO2 emission using the SVIs?

### Code Examples

#### Pedestrian Behavior Analysis in Public Spaces
**Content:**
- Detect pedestrians using YOLO
- Track movement using DeepSORT
- Analyze flow, dwell time, and walkability

**Dataset:**
- Webcam data
- Source: https://www.skylinewebcams.com/en.html

**Required Packages:** YOLOv5, OpenCV, DeepSORT, numpy, matplotlib

<p align="center">
  <img src="../figs/SCR-20251218-lzkz.jpeg" width="600"><br>
  <em>Identify pedestrian location and generate footprint heatmap with tracking.</em>
</p>

#### SVI-Based Housing Price Prediction
**Content:**
- Use subjective perception scores (e.g., cleanliness, greenery) on SVI
- Combine CV scoring with regression to predict housing price
- Visual quality → real estate value linkage

**Datasets:**
- Google Street View Imagery (SVI) from Google Map API
- California housing price dataset from sklearn.datasets

**Required Packages:** OpenCV, scikit-learn, pandas, matplotlib, PyTorch

<p align="center">
  <img src="../figs/SCR-20251218-mawu.jpeg" width="600"><br>
  <em>SVI-based housing price estimation. Nouriani, A., Lemke, L., 2022. Vision-based housing price estimation using interior, exterior & satellite images. Intelligent Systems with Applications 14, 200081. https://doi.org/10.1016/j.iswa.2022.200081.</em>
</p>
