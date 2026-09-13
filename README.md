
# 车辆检测、跟踪与计数项目

本项目面向固定道路监控视频，实现车辆检测、目标跟踪、运动方向判断与区域计数，并对两种不同方案进行对比：

1. 传统计算机视觉方案：
   **MOG2 + 形态学处理 + 轮廓检测 + 质心跟踪**

2. 深度学习方案：
   **YOLO11 + ByteTrack**

项目重点不是单纯实现“车辆计数”，而是通过同一视频、同一计数区域和同一评价指标，对传统计算机视觉方案与现代目标检测/多目标跟踪方案进行实验对比。

---

## 1. 项目功能

### 传统计算机视觉版本

传统方案主要包含：

- 高斯滤波
- MOG2 背景建模
- 二值化
- 形态学开运算与闭运算
- 膨胀处理
- 轮廓检测
- Bounding Box 提取
- 基于中心点距离的最近邻匹配
- 简单 Track ID 管理
- 车辆运动方向判断
- Counting Zone 区域穿越计数
- 处理 FPS 统计
- Ground Truth 对比

传统方案本质上检测的是“运动前景”，无法直接识别目标属于 car、truck、bus 还是 motorcycle。

---

### YOLO + ByteTrack 版本

深度学习版本主要包含：

- YOLO11n 车辆检测
- COCO 车辆类别筛选
- ByteTrack 多目标跟踪
- 稳定 Track ID
- 轨迹历史记录
- 车辆运动方向判断
- Counting Zone 区域穿越计数
- 防重复计数
- 多帧类别投票
- 分类车辆数量统计
- 实时处理 FPS 统计
- Ground Truth 对比
- 结果视频保存

支持的车辆类别：

```text
car
truck
bus
motorcycle
````

---

# 2. 整体处理流程

## 2.1 传统 CV 流程

```text
视频输入
  ↓
Gaussian Blur
  ↓
MOG2 背景减除
  ↓
Threshold 二值化
  ↓
形态学处理
  ↓
Contour Detection
  ↓
Bounding Box
  ↓
Centroid Matching
  ↓
Track ID
  ↓
方向判断
  ↓
Counting Zone
  ↓
车辆计数
```

---

## 2.2 YOLO + ByteTrack 流程

```text
视频输入
  ↓
YOLO11 目标检测
  ↓
车辆类别过滤
  ↓
ByteTrack
  ↓
Track ID
  ↓
轨迹历史
  ↓
方向判断
  ↓
Counting Zone
  ↓
车辆计数
```

---

# 3. Counting Zone 计数区域

相比只使用一条计数线，本项目使用一个计数区域进行车辆穿越判断。

当前测试视频采用：

```python
ZONE_TOP = 300
ZONE_BOTTOM = 400
```

图像区域可以理解为：

```text
        ABOVE

-------------------- y = 300

        ZONE

-------------------- y = 400

        BELOW
```

---

## 向上运动

当车辆轨迹满足：

```text
BELOW
  ↓
ZONE
  ↓
ABOVE
```

则判定：

```text
UP + 1
```

---

## 向下运动

当车辆轨迹满足：

```text
ABOVE
  ↓
ZONE
  ↓
BELOW
```

则判定：

```text
DOWN + 1
```

每个 Track ID 只允许计数一次，从而避免目标在计数区域附近发生位置抖动时重复计数。

---

# 4. 多帧类别投票

YOLO 对同一个目标在不同帧中的类别预测可能存在一定波动。

例如同一辆车可能出现：

```text
Frame 1: truck
Frame 2: car
Frame 3: car
Frame 4: truck
Frame 5: car
```

如果只使用“穿过计数区域那一帧”的类别作为最终类别，容易受到单帧误分类影响。

因此本项目对每个 Track ID 保存其整个跟踪过程中的类别结果，并使用多数投票得到最终类别。

例如：

```text
ID 26

car   : 60
truck : 5
```

则最终类别为：

```text
car
```

这样可以降低单帧分类波动对最终统计结果的影响。

---

# 5. 测试视频

测试视频参数：

```text
分辨率：1280 × 720
总帧数：984
原视频 FPS：29.97
```

人工统计 Ground Truth：

```text
52 辆车
```

两种方案均使用：

* 同一个视频
* 同一个 Counting Zone
* 同一个 Ground Truth
* 同样的计数规则

以保证实验结果具有可比性。

---

# 6. 实验结果

| 指标                        | MOG2 + Centroid | YOLO11n + ByteTrack |
| ------------------------- | --------------: | ------------------: |
| Ground Truth              |              52 |                  52 |
| 预测车辆数量                    |              47 |                  52 |
| Count Error               |               5 |                   0 |
| Counting Accuracy         |          90.38% |             100.00% |
| Average Processing FPS    |      131.72 FPS |           26.58 FPS |
| Detection + Tracking Time |               - |       36.2 ms/frame |
| 车辆类别识别                    |             不支持 |                  支持 |
| Track ID                  |         简单中心点匹配 |           ByteTrack |
| 轨迹可视化                     |             不支持 |                  支持 |

---

# 7. 实验结果分析

## 7.1 MOG2 + Centroid

传统方案处理速度非常快：

```text
131.72 FPS
```

远高于原视频：

```text
29.97 FPS
```

说明传统背景减除方法在固定摄像头、场景相对简单的情况下具有很高的计算效率。

但最终结果为：

```text
47 / 52
```

存在 5 辆车漏计。

可能原因包括：

* 小目标前景面积不足，被尺寸阈值过滤
* 两辆相邻车辆前景区域发生粘连
* 遮挡导致轮廓发生明显变化
* 某些帧前景 Mask 不稳定
* Centroid Tracker 匹配失败
* Track ID 在复杂情况下中断

因此传统方案虽然速度较快，但鲁棒性有限。

---

## 7.2 YOLO11 + ByteTrack

YOLO + ByteTrack 最终结果为：

```text
52 / 52
```

当前测试视频下：

```text
Count Error: 0
Counting Accuracy: 100.00%
```

平均处理速度：

```text
26.58 FPS
```

平均 Detection + Tracking 时间：

```text
36.2 ms/frame
```

虽然处理速度低于传统方案，但已经接近原视频 29.97 FPS，能够满足接近实时的处理需求。

相比传统 MOG2 方案，YOLO + ByteTrack 主要优势包括：

* 直接基于目标语义进行车辆检测
* 对背景变化依赖更小
* 能够区分车辆类别
* Track ID 更稳定
* 对短暂遮挡和检测波动更鲁棒
* 支持轨迹可视化和进一步行为分析

---

# 8. 类别投票结果示例

部分 Track ID 的投票结果如下：

```text
ID:26
Final: car
Votes:
car   : 60
truck : 5
```

```text
ID:260
Final: car
Votes:
car   : 56
truck : 8
```

```text
ID:295
Final: car
Votes:
car   : 53
truck : 3
```

可以看到，个别帧中 YOLO 会产生 truck / car 分类波动。

通过 Track 级多帧投票，可以降低单帧误分类带来的影响。

需要说明的是：

当前项目只人工标注了“车辆总数 Ground Truth”，并没有对每一辆车辆的类别单独进行人工标注。

因此：

```text
100% Accuracy
```

指的是：

```text
车辆计数准确率
```

而不是：

```text
车辆分类准确率
```

---

# 9. 项目结构

```text
opencv-vehicle-detection/
│
├── cars.py
│   └── MOG2 + Centroid Tracking 传统方案
│
├── yolo_vehicle_detection.py
│   └── YOLO11 + ByteTrack 方案
│
├── xzg_875610.mp4
│   └── 测试视频
│
├── yolo11n.pt
│   └── YOLO11n 模型权重
│
└── README.md
```

程序运行后还可以生成：

```text
mog2_centroid_result.mp4
yolo_bytetrack_result.mp4
```

用于查看最终检测、跟踪和计数效果。

---

# 10. 环境依赖

主要依赖：

```text
Python
OpenCV
Ultralytics
PyTorch
ByteTrack
NumPy
```

安装：

```bash
pip install ultralytics opencv-python numpy
```

---

# 11. 运行方式

运行传统 CV 版本：

```bash
python cars.py
```

运行 YOLO + ByteTrack 版本：

```bash
python yolo_vehicle_detection.py
```

运行过程中按：

```text
ESC
```

退出程序。

---

# 12. 主要参数

## YOLO 置信度

```python
CONF_THRESHOLD = 0.35
```

---

## Counting Zone

```python
ZONE_TOP = 300
ZONE_BOTTOM = 400
```

---

## Centroid Tracker

```python
MAX_DISTANCE = 100
MAX_MISSING = 8
```

---

## MOG2

```python
HISTORY = 500
VAR_THRESHOLD = 60
```

---

# 13. 项目局限性

当前项目仍存在以下限制。

### 1. 测试数据较少

当前实验仅使用一段道路视频。

因此当前得到的：

```text
100% Counting Accuracy
```

只能说明 YOLO + ByteTrack 在当前测试视频中实现了零计数误差，不能代表在所有交通场景中都能达到 100%。

---

### 2. Counting Zone 需要人工设置

当前：

```python
ZONE_TOP = 300
ZONE_BOTTOM = 400
```

是根据当前摄像机视角和视频分辨率人工调整得到的。

如果更换摄像头、视频分辨率或道路视角，可能需要重新调整。

---

### 3. 未进行自定义模型训练

当前 YOLO11n 使用的是 COCO 预训练模型，没有针对当前道路场景进行额外 Fine-tuning。

---

### 4. 分类准确率未单独评估

虽然系统实现了：

```text
car / truck / bus / motorcycle
```

分类统计以及多帧类别投票，但当前没有人工车辆类别 Ground Truth，因此无法给出严格的分类准确率。

---

### 5. ByteTrack 仍可能发生 ID Switch

如果目标长时间被遮挡、检测完全丢失，或者多个目标严重重叠，ByteTrack 仍然可能重新分配 Track ID。

在更复杂的视频中可能造成重复计数或漏计。

---

### 6. MOG2 依赖固定摄像头

传统方案高度依赖：

```text
静态摄像机
+
相对稳定的背景
```

如果摄像机本身发生运动，背景减除效果会明显下降。

---

# 14. 后续可以继续改进

可以进一步扩展：

* 多段交通视频测试
* 不同天气与光照条件测试
* 车辆类别 Ground Truth 标注
* 车辆速度估计
* 分车道车辆计数
* ROI 区域过滤
* 单应性变换
* YOLO 自定义数据集微调
* MOT 跟踪指标评估
* 摄像头实时输入
* 自动 Counting Zone 设计
* 车辆轨迹分析

---

# 15. 总结

本项目分别实现了传统计算机视觉和深度学习两种车辆检测与计数方案。

传统方案：

```text
MOG2 + Centroid Tracking
```

具有较高计算效率，在当前测试环境下达到：

```text
131.72 FPS
```

但存在一定漏计情况。

YOLO 方案：

```text
YOLO11n + ByteTrack
```

虽然计算开销更高，但在当前 1280×720、984 帧测试视频中实现：

```text
52 / 52
```

车辆正确计数，并保持约：

```text
26.58 FPS
```

的整体处理速度。

实验体现了传统计算机视觉与深度学习目标检测方案之间较典型的权衡：

```text
传统 CV
→ 速度快、计算量低
→ 场景适应性和鲁棒性较弱

YOLO + ByteTrack
→ 计算量更高
→ 检测、跟踪和扩展能力更强
```

该项目主要用于学习并实践：

```text
OpenCV
目标检测
多目标跟踪
车辆计数
YOLO
ByteTrack
传统 CV 与深度学习方案对比
```

