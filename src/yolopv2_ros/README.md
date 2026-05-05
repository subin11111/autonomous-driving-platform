yolopv2_ros (Multi-model perception fusion)
=============================================

Overview
--------

This package integrates **multiple perception models** to provide comprehensive autonomous driving perception:

1. **YOLOPv2**: Vehicle detection + Drivable area segmentation + Lane line segmentation
2. **Ultralytics YOLO**: Pedestrian detection (person class)
3. **Ultralytics YOLO**: Traffic light detection (traffic light class, bounding box only)
4. **Fusion Visualizer**: Combines all perception results into a single debug image and fused detection list

The package is designed as a modular perception stack where each detection component is independent, yet can be fused together for comprehensive scene understanding.

### Key Components

- **YOLOPv2 Node**: Detects vehicles, drivable areas, and lane lines from `/camera/image_raw`. Independent outputs.
- **Pedestrian Detector**: Ultralytics YOLO pretrained model detecting COCO class `person`. Independent outputs.
- **Traffic Light Detector**: Ultralytics YOLO pretrained model detecting COCO class `traffic light` (box only, no state classification). Independent outputs.
- **Fusion Visualizer**: Subscribes to all detector outputs and publishes a combined debug image and fused detection list.

Important notes
---------------

- This package does NOT perform any training or fine-tuning.
- YOLOPv2 remains active and intact (vehicle, drivable, lane functionality preserved).
- All detectors run independently and publish their own outputs.
- Fusion is additive and non-destructive; individual detector outputs are retained.
- The fusion visualizer only visualizes and combines; it performs no inference.

Topics
------

Input:
- `/camera/image_raw` : `sensor_msgs/msg/Image`

YOLOPv2 outputs:
- `/yolopv2/result_image` : `sensor_msgs/msg/Image`
- `/yolopv2/drivable_mask` : `sensor_msgs/msg/Image` (mono8)
- `/yolopv2/lane_mask` : `sensor_msgs/msg/Image` (mono8)
- `/yolopv2/vehicle_detections` : `vision_msgs/msg/Detection2DArray`

Pedestrian node outputs:
- `/yolo/person/detections` : `vision_msgs/msg/Detection2DArray`
- `/yolo/person/result_image` : `sensor_msgs/msg/Image`

Traffic light node outputs:
- `/yolo/traffic_light/detections` : `vision_msgs/msg/Detection2DArray`
- `/yolo/traffic_light/result_image` : `sensor_msgs/msg/Image`

Fusion visualizer outputs:
- `/perception/fused_debug_image` : `sensor_msgs/msg/Image` (combined visualization)
- `/perception/fused_detections` : `vision_msgs/msg/Detection2DArray` (fused detection list)

Detection2DArray class_id policy
-------------------------------

Each detector outputs Detection2DArray with normalized class_id:

- **YOLOPv2 vehicle**: Numeric COCO class id (or mapped to name if yolopv2_class_id_map is provided)
- **Pedestrian detector**: `"person"`
- **Traffic light detector**: `"traffic light"`

The fused_detections combines all three and normalizes class_ids for downstream algorithms.

Installation
------------

Install system packages and ROS dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-vision-msgs \
  python3-opencv \
  python3-numpy
```

Install Python pip dependencies (recommended in your Python environment):

```bash
python3 -m pip install --user ultralytics opencv-python numpy
```

Build inside a ROS2 workspace
----------------------------

```bash
# put this package under src/ in your ROS2 workspace
cd ~/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --packages-select yolopv2_ros
source install/setup.bash
```

Usage
-----

### Full perception pipeline (recommended)

This launch file starts YOLOPv2, pedestrian_detector, traffic_light_detector, and fusion_visualizer together:

```bash
ros2 launch yolopv2_ros full_perception.launch.py \
  input_image_topic:=/camera/image_raw \
  yolopv2_weights:=/path/to/yolopv2.pt \
  yolopv2_device:=0 \
  yolo_model_name:=yolov8n.pt \
  yolo_device:=0
```

### Run individual components

YOLOPv2 only:
```bash
ros2 run yolopv2_ros perception_inference \
  --ros-input-topic /camera/image_raw \
  --ros-output-topic /yolopv2/result_image \
  --ros-drivable-mask-topic /yolopv2/drivable_mask \
  --ros-lane-mask-topic /yolopv2/lane_mask \
  --ros-detections-topic /yolopv2/vehicle_detections \
  --weights /path/to/yolopv2.pt \
  --device 0
```

Pedestrian detector only:
```bash
ros2 run yolopv2_ros pedestrian_detector
```

Traffic light detector only:
```bash
ros2 run yolopv2_ros traffic_light_detector
```

Fusion visualizer only:
```bash
ros2 run yolopv2_ros fusion_visualizer
```

Both YOLO detectors (without YOLOPv2):
```bash
ros2 launch yolopv2_ros two_detectors.launch.py
```

Parameters
----------

- `model_name` (string): Ultralytics pretrained model filename (default: `yolov8n.pt`). You can set values such as `yolov8n.pt`, `yolov8s.pt`, etc.
- `device` (string): `cpu` or device id (e.g. `0`).
- `confidence_threshold` (double): default `0.25`.
- `iou_threshold` (double): default `0.45`.
- `publish_result_image` (bool): default `true`.

License
-------

TODO: License declaration
Full Perception Pipeline Architecture
-------------------------------------

### System diagram

```
/camera/image_raw
  ├─> [YOLOPv2 Node] ──> /yolopv2/vehicle_detections
  │                   ──> /yolopv2/drivable_mask
  │                   ──> /yolopv2/lane_mask
  │                   ──> /yolopv2/result_image
  │
  ├─> [Pedestrian Detector] ──> /yolo/person/detections
  │                         ──> /yolo/person/result_image
  │
  ├─> [Traffic Light Detector] ──> /yolo/traffic_light/detections
  │                            ──> /yolo/traffic_light/result_image
  │
  └─> [Fusion Visualizer] ──> /perception/fused_debug_image
                          ──> /perception/fused_detections
```

### Resolution requirements

**IMPORTANT**: Image resolution consistency is critical for accurate fusion.

- YOLOPv2 internally resizes input to 1280x720 for inference.
- All detectors (pedestrian, traffic light) must operate on the same resolution.
- Bounding box coordinates and masks must align.

**Recommendation**: Publish `/camera/image_raw` at **1280x720** resolution.

If your camera publishes a different resolution:
1. Add a resize node before perception
2. Ensure all detectors receive the resized stream
3. The fusion visualizer will attempt to resize masks, but bounding box coordinates cannot be automatically adjusted.

### Mask information

- Drivable area mask and lane line mask are NOT included in Detection2DArray.
- They are published as Image topics: `/yolopv2/drivable_mask`, `/yolopv2/lane_mask` (mono8).
- The fused debug image overlays these masks on top of detection bboxes.
- For downstream processing: use Image topics directly if mask segmentation is needed.

### Traffic light note

- The traffic light detector only outputs bounding boxes.
- Red/yellow/green state classification is NOT performed in this node.
- Future extension: crop detections and apply HSV-based or classifier-based state detection.

### Fusion visualizer parameters

Key parameters for fine-tuning fusion output:

- `drivable_alpha`: Transparency of drivable area overlay (default: 0.35)
- `lane_alpha`: Transparency of lane line overlay (default: 0.75)
- `max_stale_time_sec`: Max age of messages before marking as stale (default: 0.5 sec)
- `min_vehicle_confidence`: Vehicle bbox confidence threshold (default: 0.25)
- `min_person_confidence`: Person bbox confidence threshold (default: 0.25)
- `min_traffic_light_confidence`: Traffic light bbox confidence threshold (default: 0.25)
- `yolopv2_class_id_map`: Optional dict to map YOLOPv2 class ids to names (default: empty)

### YOLOPv2 class id mapping

YOLOPv2's class ids depend on the model weights and training data. To properly label vehicle types (car, truck, bus, etc.), provide a mapping:

```python
yolopv2_class_id_map = {
    "0": "car",
    "1": "truck",
    "2": "bus",
}
```

Without mapping, vehicle detections are labeled as `yolopv2_cls_{id}`.

Verifying the model
-------------------

To determine the class ids in your YOLOPv2 weight file:

1. Check the yaml file associated with your weight file (e.g., `yolopv2_coco.yaml`)
2. Look for the `names:` section listing all class names
3. Note the numeric indices (0, 1, 2, ...) corresponding to each class name

Result verification
-------------------

### Check that all topics are publishing

```bash
ros2 topic list | grep -E 'yolopv2|yolo|perception'
```

Expected output should include:
- `/yolopv2/vehicle_detections`
- `/yolopv2/drivable_mask`, `/yolopv2/lane_mask`
- `/yolo/person/detections`
- `/yolo/traffic_light/detections`
- `/perception/fused_debug_image`, `/perception/fused_detections`

### Monitor message rates

```bash
ros2 topic hz /perception/fused_debug_image
ros2 topic hz /perception/fused_detections
```

### Visualize images

```bash
ros2 run rqt_image_view rqt_image_view
```

Then select topics:
- `/perception/fused_debug_image` (main output)
- `/yolopv2/result_image` (YOLOPv2 only)
- `/yolo/person/result_image` (pedestrian only)
- `/yolo/traffic_light/result_image` (traffic light only)

### Inspect detections

```bash
ros2 topic echo /perception/fused_detections
ros2 topic echo /yolopv2/vehicle_detections
ros2 topic echo /yolo/person/detections
ros2 topic echo /yolo/traffic_light/detections
```

Troubleshooting
---------------

### Issue: Bounding boxes are misaligned with masks

**Cause**: Different input resolutions for YOLOPv2 vs Ultralytics detectors.

**Solution**: Standardize input resolution to 1280x720.

### Issue: Fusion visualizer not publishing /perception/fused_debug_image

**Cause**: Missing input topics or messages are too stale.

**Solution**:
1. Check all detector nodes are running
2. Verify input image topic exists: `ros2 topic echo /camera/image_raw`
3. Increase `max_stale_time_sec` parameter if network latency is high

### Issue: YOLOPv2 node fails with "weights file not found"

**Cause**: Invalid `--weights` path.

**Solution**: Provide absolute path to yolopv2.pt:
```bash
--weights /full/path/to/yolopv2.pt
```

### Issue: Ultralytics YOLO model download fails

**Cause**: ultralytics package not installed or network unavailable.

**Solution**:
```bash
pip install --user ultralytics
# Pre-download model
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Issue: Memory/VRAM exhaustion

**Cause**: Running too many detectors on limited hardware, or using large models.

**Solution**:
1. Use smaller models: `yolov8n.pt` instead of `yolov8x.pt`
2. Use CPU inference: `device:=cpu`
3. Run detectors on separate machines
4. Reduce input resolution

Future extensions
-----------------

- Traffic light state classification (red/yellow/green)
- Pedestrian signal intent (walking, stopped)
- Crosswalk detection and tracking
- Vehicle tracking across frames
- Multi-object tracking (MOT)
- Risk assessment for autonomous vehicle decisions
- Integrated risk event publisher

Verified Executables
--------------------

After building, verify that all executables are properly registered:

```bash
cd ~/autonomous-driving-platform
colcon build --packages-select yolopv2_ros
source install/setup.bash
ros2 pkg executables yolopv2_ros
```

Expected output:

```
yolopv2_ros image_resize
yolopv2_ros perception_inference
yolopv2_ros perception_inference_node
yolopv2_ros pedestrian_detector
yolopv2_ros traffic_light_detector
yolopv2_ros fusion_visualizer
yolopv2_ros video_to_topic
yolopv2_ros masked_ray_ground_projection
```

Note: `perception_inference` and `perception_inference_node` point to the same executable for compatibility.

Unified Resolution (1280x720)
-----------------------------

### Why unified resolution?

All perception nodes (YOLOPv2, pedestrian_detector, traffic_light_detector, fusion_visualizer) must operate on the **same input resolution** to ensure:

- **YOLOPv2 bounding boxes** are aligned with **pedestrian/traffic light bounding boxes**
- **YOLOPv2 segmentation masks** (drivable, lane) overlay correctly on **fused debug image**
- **Fusion visualizer** can accurately combine all results

YOLOPv2 internally resizes images to 1280x720 for inference, so all detections are referenced to this resolution.

### Resolution pipeline

```
/camera/image_raw (arbitrary resolution, e.g., 1920x1080)
    ↓
   [image_resize_node]
    ↓
/camera/image_1280x720 (unified resolution)
    ↓
    ├─→ [yolopv2_node]
    ├─→ [pedestrian_detector_node]
    ├─→ [traffic_light_detector_node]
    └─→ [fusion_visualizer_node]
```

Run with local video file
-------------------------

Input video:
- /home/subin/test.mp4

Pre-check (ensure files exist):

```bash
ls -l /home/subin/test.mp4
ls -l /home/subin/YOLOPv2/data/weights/yolopv2.pt
ls -l /home/subin/YOLOPv2/utils/utils.py
```

Full run (recommended):

```bash
cd /home/subin/autonomous-driving-platform
source install/setup.bash
ros2 launch yolopv2_ros full_perception.launch.py \
  use_video_source:=true \
  video_path:=/home/subin/test.mp4 \
  video_output_topic:=/camera/image_raw \
  raw_image_topic:=/camera/image_raw \
  resized_image_topic:=/camera/image_1280x720 \
  resize_width:=1280 resize_height:=720 \
  yolopv2_root:=/home/subin/YOLOPv2 \
  yolopv2_weights:=/home/subin/YOLOPv2/data/weights/yolopv2.pt \
  yolopv2_device:=0 \
  yolo_model_name:=yolov8n.pt \
  yolo_device:=0
```

CPU example:

```bash
ros2 launch yolopv2_ros full_perception.launch.py \
  use_video_source:=true \
  video_path:=/home/subin/test.mp4 \
  video_output_topic:=/camera/image_raw \
  raw_image_topic:=/camera/image_raw \
  resized_image_topic:=/camera/image_1280x720 \
  resize_width:=1280 resize_height:=720 \
  yolopv2_root:=/home/subin/YOLOPv2 \
  yolopv2_weights:=/home/subin/YOLOPv2/data/weights/yolopv2.pt \
  yolopv2_device:=cpu \
  yolo_model_name:=yolov8n.pt \
  yolo_device:=cpu
```

Check topics:

```bash
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 topic hz /camera/image_1280x720
ros2 topic hz /perception/fused_debug_image
```

Resolution check:

```bash
ros2 topic echo /camera/image_1280x720 --once | grep -E "height|width|encoding|frame_id"
```

Expected:

```
height: 720
width: 1280
encoding: bgr8
frame_id: camera
```

Visualize:

```bash
ros2 run rqt_image_view rqt_image_view
```

Recommended topic order to inspect:
1. /camera/image_raw
2. /camera/image_1280x720
3. /yolo/person/result_image
4. /yolo/traffic_light/result_image
5. /yolopv2/result_image
6. /perception/fused_debug_image

If problems occur:
1. Ensure /home/subin/test.mp4 exists
2. Ensure /home/subin/YOLOPv2/data/weights/yolopv2.pt exists
3. Ensure /home/subin/YOLOPv2/utils/utils.py exists
4. Check raw publish: `ros2 topic hz /camera/image_raw`
5. Check resized publish: `ros2 topic hz /camera/image_1280x720`
6. Check fused publish: `ros2 topic hz /perception/fused_debug_image`


### Verify resized image

Check that `/camera/image_1280x720` has the correct dimensions:

```bash
ros2 topic echo /camera/image_1280x720 --once

# Look for:
# height: 720
# width: 1280
```

Full Perception Launch Command
------------------------------

```bash
ros2 launch yolopv2_ros full_perception.launch.py \
  raw_image_topic:=/camera/image_raw \
  resized_image_topic:=/camera/image_1280x720 \
  resize_width:=1280 \
  resize_height:=720 \
  yolopv2_weights:=/path/to/yolopv2.pt \
  yolopv2_device:=0 \
  yolo_model_name:=yolov8n.pt \
  yolo_device:=0
```

### CPU-based example

```bash
ros2 launch yolopv2_ros full_perception.launch.py \
  raw_image_topic:=/camera/image_raw \
  resized_image_topic:=/camera/image_1280x720 \
  resize_width:=1280 \
  resize_height:=720 \
  yolopv2_weights:=/path/to/yolopv2.pt \
  yolopv2_device:=cpu \
  yolo_model_name:=yolov8n.pt \
  yolo_device:=cpu
```

### Custom resolution example

If you need a different resolution (e.g., 640x480):

```bash
ros2 launch yolopv2_ros full_perception.launch.py \
  raw_image_topic:=/camera/image_raw \
  resized_image_topic:=/camera/image_640x480 \
  resize_width:=640 \
  resize_height:=480 \
  yolopv2_weights:=/path/to/yolopv2.pt \
  yolopv2_device:=0 \
  yolo_model_name:=yolov8n.pt \
  yolo_device:=0
```

Required Topics
---------------

After launching full_perception, these topics must be present:

```bash
ros2 topic list
```

Expected topics:

```
/camera/image_raw
/camera/image_1280x720
/yolopv2/result_image
/yolopv2/drivable_mask
/yolopv2/lane_mask
/yolopv2/vehicle_detections
/yolo/person/detections
/yolo/person/result_image
/yolo/traffic_light/detections
/yolo/traffic_light/result_image
/perception/fused_debug_image
/perception/fused_detections
```

Visualizing Fused Results
-------------------------

Open rqt_image_view to see all perception outputs:

```bash
ros2 run rqt_image_view rqt_image_view
```

**Priority viewing order**:

1. `/perception/fused_debug_image` ← Main combined output
2. `/yolopv2/result_image` ← YOLOPv2 only
3. `/yolo/person/result_image` ← Pedestrian only
4. `/yolo/traffic_light/result_image` ← Traffic light only
5. `/camera/image_1280x720` ← Input to all detectors

Troubleshooting
---------------

### Issue: Bounding boxes misaligned with segmentation masks

**Symptoms**:
- Vehicle bboxes don't match drivable area
- Pedestrian bboxes appear at wrong locations
- Lane overlay doesn't align with road

**Causes**:
- Detectors operating on different resolutions
- `/camera/image_1280x720` not created or has wrong size

**Solution**:
```bash
# Verify resized image exists and has correct dimensions
ros2 topic echo /camera/image_1280x720 --once
# Check:  height: 720, width: 1280

# Verify all nodes use resized_image_topic
ros2 topic hz /camera/image_1280x720
ros2 topic hz /yolopv2/vehicle_detections
ros2 topic hz /yolo/person/detections
ros2 topic hz /yolo/traffic_light/detections

# If mismatch, ensure full_perception.launch uses consistent resized_image_topic
```

### Issue: Image resize node not starting

**Symptom**:
- image_resize executable not found

**Solution**:
```bash
colcon build --packages-select yolopv2_ros
source install/setup.bash
ros2 pkg executables yolopv2_ros | grep image_resize
```

### Issue: Perception node claims input topic doesn't exist

**Symptom**:
- "Could not create subscription to /camera/image_raw: topic not found"

**Solution**:
- Ensure camera driver or simulation (e.g., Gazebo) is publishing `/camera/image_raw`
- Or use a rosbag or video replay node to publish

### Issue: Fusion visualizer not publishing /perception/fused_debug_image

**Symptom**:
- fused_debug_image topic exists but no data

**Causes**:
- Input messages (masks, detections) are missing or stale
- image_topic parameter mismatch

**Solution**:
```bash
# Check all input topics are publishing
ros2 topic hz /camera/image_1280x720
ros2 topic hz /yolopv2/drivable_mask
ros2 topic hz /yolopv2/lane_mask
ros2 topic hz /yolopv2/vehicle_detections
ros2 topic hz /yolo/person/detections
ros2 topic hz /yolo/traffic_light/detections

# Check fusion visualizer parameters
ros2 param list /fusion_visualizer_node
ros2 param get /fusion_visualizer_node image_topic
```

### Issue: Out of memory / GPU full

**Solutions**:
1. Use smaller YOLO model: `yolo_model_name:=yolov8n.pt` (vs yolov8x)
2. Use CPU inference: `yolo_device:=cpu`
3. Reduce resize resolution: `resize_width:=640 resize_height:=480`
4. Run detectors on separate machines