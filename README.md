# opencv-vehicle-detection
````markdown
# OpenCV Vehicle Detection and Counting

基于 Python 和 OpenCV 实现的车辆检测、目标跟踪与车辆计数项目。

本项目主要用于学习计算机视觉中的视频处理、运动目标检测、目标跟踪以及基于穿线的车辆计数。

---

## 1. 项目简介

本项目使用固定摄像头拍摄的道路视频作为输入，通过 OpenCV 对视频进行逐帧处理。

首先利用 MOG2 背景建模算法提取运动目标，然后通过图像滤波、二值化和形态学处理优化前景区域，再利用轮廓检测获取车辆目标。

在此基础上，通过车辆中心点之间的距离进行简单的目标匹配，为车辆分配唯一 ID，并根据车辆中心点是否穿过指定计数线实现车辆数量统计。

---

## 2. 项目功能

- 视频读取与逐帧处理
- Gaussian Blur 高斯滤波
- MOG2 背景建模
- 前景提取与二值化
- 形态学开运算、闭运算
- 图像膨胀
- Contour 轮廓检测
- Bounding Box 目标框检测
- 车辆中心点计算
- 基于欧氏距离的目标匹配
- 简单多目标 ID 跟踪
- 车辆运动方向判断
- 基于穿线的车辆计数
- 实时显示检测结果

---

## 3. 技术栈

- Python
- OpenCV
- NumPy
- MOG2 Background Subtraction
- Morphological Operations
- Contour Detection
- Centroid-based Tracking

---

## 4. 系统处理流程

```text
Video Input
     │
     ▼
Gaussian Blur
     │
     ▼
MOG2 Background Subtraction
     │
     ▼
Threshold
     │
     ▼
Morphological Processing
     │
     ▼
Contour Detection
     │
     ▼
Bounding Box
     │
     ▼
Centroid Calculation
     │
     ▼
Nearest-neighbor Tracking
     │
     ▼
ID Assignment
     │
     ▼
Line Crossing Detection
     │
     ▼
Vehicle Counting
````

---

## 5. 核心算法

### 5.1 MOG2 背景建模

使用 OpenCV 提供的 MOG2 背景减除算法，将视频中的静态背景与运动目标进行分离。

```python
bgsubmog = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=60,
    detectShadows=True
)
```

---

### 5.2 图像预处理

首先对视频帧进行高斯滤波，降低噪声。

随后通过 Threshold、开运算、闭运算和膨胀等操作优化前景区域。

```text
Gaussian Blur
      ↓
Threshold
      ↓
Opening
      ↓
Closing
      ↓
Dilation
```

---

### 5.3 车辆检测

通过 `findContours()` 获取前景区域轮廓，并使用 `boundingRect()` 获取目标的外接矩形。

同时通过目标宽度、高度以及宽高比过滤较小或异常的目标。

---

### 5.4 目标跟踪

计算每个车辆检测框的中心点：

```python
cx = x + w // 2
cy = y + h // 2
```

然后使用欧氏距离寻找当前帧与上一帧中距离最近的车辆，从而实现简单的目标匹配和 ID 跟踪。

```text
上一帧                 当前帧

ID 0 ●  ───────────→  ● ID 0
ID 1 ●  ───────────→  ● ID 1
ID 2 ●  ───────────→  ● ID 2
```

---

### 5.5 车辆计数

设置一条水平计数线：

```python
LINE_Y = 600
```

记录车辆中心点在连续帧中的位置变化。

当车辆中心点从计数线一侧移动到另一侧时，认为车辆完成一次穿线，并将车辆数量加一。

同时使用 `counted` 状态避免同一个车辆 ID 被重复计数。

---

## 6. 项目运行效果

运行程序后可以实时显示：

* 车辆检测框
* 车辆中心点
* 车辆 ID
* 车辆运动方向
* 车辆计数结果
* 前景 Mask

示例：

```text
┌─────────────────────────────────────┐
│                                     │
│     ┌──────────┐                    │
│     │   CAR    │                    │
│     │          │                    │
│     └──────────┘                    │
│          ●                          │
│        ID: 3                        │
│                                     │
│────────────────── Counting Line─────│
│                                     │
│  Car Count: 5                       │
└─────────────────────────────────────┘
```

---

## 7. 项目结构

目前项目采用单文件结构：

```text
opencv-vehicle-detection/
│
├── vehicle_detection.py
├── README.md
├── requirements.txt
│
└── screenshots/
    └── result.png
```

---

## 8. 环境要求

Python 3.x

安装依赖：

```bash
pip install opencv-python numpy
```

或者：

```bash
pip install -r requirements.txt
```

---

## 9. 运行项目

修改 `vehicle_detection.py` 中的视频路径：

```python
VIDEO_PATH = r"C:\path\to\your\video.mp4"
```

然后运行：

```bash
python vehicle_detection.py
```

按 `ESC` 可以退出程序。

---

## 10. 参数调整

项目中的部分参数可以根据不同视频进行调整。

```python
MIN_WIDTH = 50
MIN_HEIGHT = 50

MAX_DISTANCE = 100
MAX_MISSING = 8

LINE_Y = 600
OFFSET = 15
```

### MIN_WIDTH / MIN_HEIGHT

用于过滤过小的检测目标。

### MAX_DISTANCE

用于控制当前帧检测目标与历史目标进行 ID 匹配时允许的最大距离。

### MAX_MISSING

允许目标连续丢失检测的最大帧数。

### LINE_Y

设置车辆计数线的位置。

### OFFSET

用于减少车辆中心点抖动对穿线判断造成的影响。

---

## 11. 项目学习内容

通过本项目学习并实践了以下计算机视觉基础知识：

* OpenCV 视频处理
* NumPy 图像数据处理
* 图像滤波
* 二值图像处理
* 形态学操作
* 背景建模
* 运动目标检测
* 轮廓检测
* 目标框提取
* 目标中心点计算
* 多目标跟踪基础
* ID 管理
* 轨迹分析
* 穿线计数

---

## 12. Future Improvements

后续计划进一步学习深度学习目标检测，并对当前项目进行升级：

```text
MOG2 + Contour
      ↓
YOLO Object Detection
      ↓
Multi-object Tracking
      ↓
Vehicle Counting
```

计划进一步尝试：

* YOLO 目标检测
* ByteTrack / DeepSORT
* 目标速度估计
* 更稳定的车辆计数
* 实时摄像头检测
* ROS 2 视觉节点



另外，**你现在最好加一张实际运行截图**到 `screenshots/result.png`，这个比 README 里写很多文字更有用。
```
