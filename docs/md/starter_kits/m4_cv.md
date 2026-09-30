# CV Models (Segmentation, Detection, Tracking)

**Related API:** [`ccai9012.yolo_utils`](../api/ccai9012/yolo_utils.html) · [`ccai9012.svi_utils`](../api/ccai9012/svi_utils.html) 

### Overview
**Category:** Perception & Prediction from Visual Data

The module assumes basic Python, pandas, image/video handling, and the
scikit-learn workflow from earlier Starter Kits. Both notebooks default to
bounded local assets. Network acquisition and model downloads are explicit
opt-in branches, so a fresh lesson run can first verify paths, joins, schemas,
and output locations offline.

**Modular Components:**

- **Image to perception**
  - Object detection and tracking with YOLO
  - Semantic segmentation model
- **Perception to measures**
  - Trajectory extraction
  - Housing-price feature construction
- **Measures to interpretation**
  - Frame, map, and trajectory visualisation
  - Regression or spatial summary

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
- A short local video clip when available (`video/closed_test_15.mp4` or
  another user-provided clip)
- Optional external source: [SkylineWebcams](https://www.skylinewebcams.com/en.html);
  download and licensing must be checked by the user before placing a clip in
  the local `video/` directory

**Required Packages:** Ultralytics YOLO11, OpenCV, pandas, numpy, matplotlib,
SciPy (for the optional rendered-video helper)

The notebook records one row per detected person per frame:
`frame`, persistent `id`, and bounding-box coordinates `x1`, `y1`, `x2`, `y2`.
Detection answers “what is in this frame”; tracking associates detections over
time. Confidence is the detector's score, while IoU-based association and
ByteTrack determine whether an ID persists. Occlusion, camera motion, missed
detections, and a perspective-distorted bottom-centre point limit any direct
interpretation as pedestrian flow or walkability.

The expected model is the tracked `yolo11m.pt` file (YOLO11 medium). The
notebook loads it from the Starter Kit directory and stops with a helpful
message if it is absent; it does not silently replace it with a newer YOLO
generation. Video, CSV, heatmap, and rendered-clip outputs belong under the
ignored local `output/` directory.

<p align="center">
  <img src="../figs/yolo_tracking_storyboard.svg" alt="Storyboard from per-frame pedestrian detection through persistent track IDs, trajectory table and heatmap, to a local rendered clip." width="100%"><br>
  <em>Frame-to-trajectory storyboard. Every downstream visualisation depends on the detection table.</em>
</p>

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

The default path uses a deterministic sample from the 151 tracked SVI images
and the matching California housing rows. Because the source CSV can contain
several rows at the same rounded coordinate, the notebook aggregates `Target`
to a coordinate-level mean before joining the image manifest. It then runs the
same held-out split and MAE metric for three columns: tabular-only, image-only,
and combined features. The offline image branch uses deterministic colour and
edge statistics; cached/approved ResNet weights are an explicit opt-in. A
Google Maps key is only requested when the acquisition flag is enabled.

<p align="center">
  <img src="../figs/svi_housing_ablation.svg" alt="Three parallel SVI housing prediction columns: tabular-only, image-only, and combined features, using the same validated rows and held-out MAE." width="100%"><br>
  <em>Modality ablation with a shared manifest, target, split, and metric.</em>
</p>

The result is a predictive comparison under this sampling and feature setup;
it does not identify a causal visual-quality effect or a fair property-value
assessment. Image filenames, coordinates, and targets must remain aligned at
every step.

<p align="center">
  <img src="../figs/SCR-20251218-mawu.jpeg" width="600"><br>
  <em>SVI-based housing price estimation. Nouriani, A., Lemke, L., 2022. Vision-based housing price estimation using interior, exterior & satellite images. Intelligent Systems with Applications 14, 200081. https://doi.org/10.1016/j.iswa.2022.200081.</em>
</p>
