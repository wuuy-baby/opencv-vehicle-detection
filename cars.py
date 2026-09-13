import cv2
import numpy as np
import math
import time


# ============================================================
# 1. 参数配置
# ============================================================

VIDEO_PATH = "xzg_875610.mp4"

# ------------------------------------------------------------
# MOG2 参数
# ------------------------------------------------------------

HISTORY = 500

VAR_THRESHOLD = 60


# ------------------------------------------------------------
# 轮廓尺寸过滤
# ------------------------------------------------------------

MIN_WIDTH = 50

MIN_HEIGHT = 50


# ------------------------------------------------------------
# Centroid Tracker 参数
# ------------------------------------------------------------

# 当前检测中心点和历史车辆中心点允许的最大距离
MAX_DISTANCE = 100

# 一辆车最多允许连续丢失多少帧
MAX_MISSING = 8


# ------------------------------------------------------------
# Counting Zone
#
# 为了和 YOLO 版本公平比较，
# 使用完全相同的计数区域
# ------------------------------------------------------------

ZONE_TOP = 300

ZONE_BOTTOM = 400


# ------------------------------------------------------------
# 播放速度
# ------------------------------------------------------------

WAIT_TIME = 30


# ------------------------------------------------------------
# Ground Truth
#
# 你人工数完真实车辆数量后填写
#
# 例如：
#
# GROUND_TRUTH_COUNT = 36
#
# 如果暂时不知道：
# ------------------------------------------------------------

GROUND_TRUTH_COUNT = None


# ------------------------------------------------------------
# 是否保存结果视频
# ------------------------------------------------------------

SAVE_OUTPUT_VIDEO = True

OUTPUT_VIDEO_PATH = "mog2_centroid_result.mp4"


# ============================================================
# 2. Tracker 数据
#
# 每一辆车保存：
#
# center
# previous
# bbox
# missing
# counted
# direction
# last_region
# entry_region
# ============================================================

cars = {}


# 下一个新车辆 ID
next_id = 0


# ============================================================
# 3. 统计
# ============================================================

vehicle_count = 0

up_count = 0

down_count = 0


# ============================================================
# 4. 性能统计
# ============================================================

total_frames = 0

total_processing_time = 0.0


# ============================================================
# 5. 工具函数
# ============================================================

def get_center(x, y, w, h):

    cx = x + w // 2

    cy = y + h // 2

    return cx, cy


def distance(p1, p2):

    return math.sqrt(

        (p1[0] - p2[0]) ** 2

        +

        (p1[1] - p2[1]) ** 2

    )


# ============================================================
# 6. 判断车辆位于哪个区域
# ============================================================

def get_region(cy):

    if cy < ZONE_TOP:

        return "ABOVE"

    elif cy > ZONE_BOTTOM:

        return "BELOW"

    else:

        return "ZONE"


# ============================================================
# 7. 打开视频
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():

    print("Error opening video stream or file")

    exit()


# ============================================================
# 8. 获取原视频参数
# ============================================================

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

video_frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("====================================")
print("Input Video Information")
print("====================================")

print(
    "Resolution:",
    frame_width,
    "x",
    frame_height
)

print(
    "Original FPS:",
    video_fps
)

print(
    "Total Frames:",
    video_frame_count
)

print()


# ============================================================
# 9. 输出视频
# ============================================================

writer = None


if SAVE_OUTPUT_VIDEO:

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(

        OUTPUT_VIDEO_PATH,

        fourcc,

        video_fps,

        (
            frame_width,
            frame_height
        )
    )


# ============================================================
# 10. 创建 MOG2
# ============================================================

bgsubmog = cv2.createBackgroundSubtractorMOG2(

    history=HISTORY,

    varThreshold=VAR_THRESHOLD,

    detectShadows=True
)


# ============================================================
# 11. 形态学 Kernel
# ============================================================

kernel = cv2.getStructuringElement(

    cv2.MORPH_RECT,

    (5, 5)
)


# ============================================================
# 12. 主循环
# ============================================================

while True:


    # --------------------------------------------------------
    # 12.1 读取视频帧
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        break


    total_frames += 1


    # --------------------------------------------------------
    # 开始记录这一帧完整处理时间
    # --------------------------------------------------------

    frame_start_time = time.perf_counter()


    # ========================================================
    # 13. Gaussian Blur
    # ========================================================

    blur = cv2.GaussianBlur(

        frame,

        (5, 5),

        0
    )


    # ========================================================
    # 14. MOG2 Background Subtraction
    # ========================================================

    mask = bgsubmog.apply(
        blur
    )


    # ========================================================
    # 15. Threshold
    #
    # 去除 MOG2 阴影区域
    # ========================================================

    _, mask = cv2.threshold(

        mask,

        200,

        255,

        cv2.THRESH_BINARY
    )


    # ========================================================
    # 16. Opening
    # ========================================================

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel,

        iterations=1
    )


    # ========================================================
    # 17. Closing
    # ========================================================

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=2
    )


    # ========================================================
    # 18. Dilation
    # ========================================================

    mask = cv2.dilate(

        mask,

        kernel,

        iterations=1
    )


    # ========================================================
    # 19. Contour Detection
    # ========================================================

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    detections = []


    # ========================================================
    # 20. 轮廓转 Bounding Box
    # ========================================================

    for cnt in contours:


        x, y, w, h = cv2.boundingRect(
            cnt
        )


        # ----------------------------------------------------
        # 太小的前景目标不要
        # ----------------------------------------------------

        if (
            w < MIN_WIDTH

            or

            h < MIN_HEIGHT
        ):

            continue


        # ----------------------------------------------------
        # 过滤异常宽高比
        # ----------------------------------------------------

        ratio = w / float(h)


        if (
            ratio > 5

            or

            ratio < 0.2
        ):

            continue


        cx, cy = get_center(

            x,

            y,

            w,

            h
        )


        detections.append({

            "center": (
                cx,
                cy
            ),

            "bbox": (
                x,
                y,
                w,
                h
            )
        })


    # ========================================================
    # 21. 所有历史 Track missing +1
    # ========================================================

    for car_id in cars:

        cars[car_id]["missing"] += 1


    used_ids = set()


    # ========================================================
    # 22. Centroid Nearest-Neighbor Tracking
    # ========================================================

    for detection in detections:


        center = detection["center"]


        best_id = None

        best_dist = MAX_DISTANCE


        # ----------------------------------------------------
        # 找离当前 Detection 最近的历史 Track
        # ----------------------------------------------------

        for car_id, car in cars.items():


            if car_id in used_ids:

                continue


            d = distance(

                center,

                car["center"]
            )


            if d < best_dist:

                best_dist = d

                best_id = car_id


        # ====================================================
        # 23. 匹配到旧 Track
        # ====================================================

        if best_id is not None:


            car = cars[best_id]


            previous = car["center"]

            previous_y = previous[1]

            current_y = center[1]


            # ------------------------------------------------
            # 方向判断
            # ------------------------------------------------

            dy = (
                current_y
                -
                previous_y
            )


            if dy > 2:

                car["direction"] = "DOWN"


            elif dy < -2:

                car["direction"] = "UP"


            # =================================================
            # 24. 当前所在 Counting Region
            # =================================================

            current_region = get_region(
                current_y
            )

            last_region = car[
                "last_region"
            ]


            # =================================================
            # 25. 第一次拥有区域状态
            # =================================================

            if last_region is None:


                car[
                    "last_region"
                ] = current_region


                if current_region in (
                    "ABOVE",
                    "BELOW"
                ):

                    car[
                        "entry_region"
                    ] = current_region


            # =================================================
            # 26. 已经有区域历史
            # =================================================

            else:


                # --------------------------------------------
                # ABOVE → ZONE
                # --------------------------------------------

                if (
                    last_region == "ABOVE"

                    and

                    current_region == "ZONE"
                ):

                    car[
                        "entry_region"
                    ] = "ABOVE"


                # --------------------------------------------
                # BELOW → ZONE
                # --------------------------------------------

                elif (
                    last_region == "BELOW"

                    and

                    current_region == "ZONE"
                ):

                    car[
                        "entry_region"
                    ] = "BELOW"


                # ============================================
                # 27. ABOVE → ZONE → BELOW
                #
                # DOWN
                # ============================================

                elif (
                    last_region == "ZONE"

                    and

                    current_region == "BELOW"
                ):


                    if (
                        car[
                            "entry_region"
                        ] == "ABOVE"

                        and

                        not car[
                            "counted"
                        ]
                    ):


                        vehicle_count += 1

                        down_count += 1

                        car[
                            "counted"
                        ] = True


                        print(

                            "Vehicle ID:",

                            best_id,

                            "crossed DOWN",

                            "Total:",

                            vehicle_count
                        )


                # ============================================
                # 28. BELOW → ZONE → ABOVE
                #
                # UP
                # ============================================

                elif (
                    last_region == "ZONE"

                    and

                    current_region == "ABOVE"
                ):


                    if (
                        car[
                            "entry_region"
                        ] == "BELOW"

                        and

                        not car[
                            "counted"
                        ]
                    ):


                        vehicle_count += 1

                        up_count += 1

                        car[
                            "counted"
                        ] = True


                        print(

                            "Vehicle ID:",

                            best_id,

                            "crossed UP",

                            "Total:",

                            vehicle_count
                        )


                # ============================================
                # 29. 快速移动兜底
                #
                # ABOVE → BELOW
                # ============================================

                elif (
                    last_region == "ABOVE"

                    and

                    current_region == "BELOW"
                ):


                    if not car[
                        "counted"
                    ]:


                        vehicle_count += 1

                        down_count += 1

                        car[
                            "counted"
                        ] = True


                        print(

                            "Vehicle ID:",

                            best_id,

                            "jumped DOWN",

                            "Total:",

                            vehicle_count
                        )


                # ============================================
                # BELOW → ABOVE
                # ============================================

                elif (
                    last_region == "BELOW"

                    and

                    current_region == "ABOVE"
                ):


                    if not car[
                        "counted"
                    ]:


                        vehicle_count += 1

                        up_count += 1

                        car[
                            "counted"
                        ] = True


                        print(

                            "Vehicle ID:",

                            best_id,

                            "jumped UP",

                            "Total:",

                            vehicle_count
                        )


                # --------------------------------------------
                # 更新区域状态
                # --------------------------------------------

                car[
                    "last_region"
                ] = current_region


            # =================================================
            # 30. 更新 Track
            # =================================================

            car[
                "previous"
            ] = previous


            car[
                "center"
            ] = center


            car[
                "bbox"
            ] = detection[
                "bbox"
            ]


            car[
                "missing"
            ] = 0


            used_ids.add(
                best_id
            )


        # ====================================================
        # 31. 新 Track
        # ====================================================

        else:


            current_region = get_region(
                center[1]
            )


            entry_region = None


            if current_region in (
                "ABOVE",
                "BELOW"
            ):

                entry_region = (
                    current_region
                )


            cars[next_id] = {

                "center":
                    center,

                "previous":
                    center,

                "bbox":
                    detection[
                        "bbox"
                    ],

                "missing":
                    0,

                "counted":
                    False,

                "direction":
                    "UNKNOWN",

                "last_region":
                    current_region,

                "entry_region":
                    entry_region
            }


            used_ids.add(
                next_id
            )


            next_id += 1


    # ========================================================
    # 32. 删除长期丢失 Track
    # ========================================================

    delete_ids = []


    for car_id, car in cars.items():


        if (
            car[
                "missing"
            ]
            >
            MAX_MISSING
        ):

            delete_ids.append(
                car_id
            )


    for car_id in delete_ids:

        del cars[
            car_id
        ]


    # ========================================================
    # 33. 画 Counting Zone
    # ========================================================

    cv2.line(

        frame,

        (
            0,
            ZONE_TOP
        ),

        (
            frame.shape[1],
            ZONE_TOP
        ),

        (
            0,
            255,
            255
        ),

        2
    )


    cv2.line(

        frame,

        (
            0,
            ZONE_BOTTOM
        ),

        (
            frame.shape[1],
            ZONE_BOTTOM
        ),

        (
            0,
            255,
            255
        ),

        2
    )


    # ========================================================
    # 34. 画 Track
    # ========================================================

    for car_id, car in cars.items():


        x, y, w, h = (
            car[
                "bbox"
            ]
        )


        cx, cy = (
            car[
                "center"
            ]
        )


        # ----------------------------------------------------
        # Bounding Box
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (
                x,
                y
            ),

            (
                x + w,
                y + h
            ),

            (
                0,
                0,
                255
            ),

            2
        )


        # ----------------------------------------------------
        # 中心点
        # ----------------------------------------------------

        cv2.circle(

            frame,

            (
                cx,
                cy
            ),

            5,

            (
                0,
                255,
                0
            ),

            -1
        )


        # ----------------------------------------------------
        # ID + Direction
        # ----------------------------------------------------

        label = (

            f"ID:{car_id} "
            f"{car['direction']}"
        )


        cv2.putText(

            frame,

            label,

            (
                x,
                max(
                    20,
                    y - 10
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                0,
                255,
                0
            ),

            2
        )


    # ========================================================
    # 35. 当前帧处理时间
    # ========================================================

    frame_end_time = time.perf_counter()


    frame_processing_time = (

        frame_end_time
        -
        frame_start_time
    )


    total_processing_time += (
        frame_processing_time
    )


    # ========================================================
    # 36. 实时 FPS
    # ========================================================

    if frame_processing_time > 0:

        current_fps = (

            1.0
            /
            frame_processing_time
        )

    else:

        current_fps = 0.0


    # ========================================================
    # 37. 显示统计
    # ========================================================

    cv2.putText(

        frame,

        f"Total: {vehicle_count}",

        (
            30,
            50
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.2,

        (
            0,
            0,
            255
        ),

        3
    )


    cv2.putText(

        frame,

        f"UP: {up_count}",

        (
            30,
            95
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            255,
            255,
            0
        ),

        2
    )


    cv2.putText(

        frame,

        f"DOWN: {down_count}",

        (
            30,
            130
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            255,
            255,
            0
        ),

        2
    )


    cv2.putText(

        frame,

        f"Processing FPS: {current_fps:.1f}",

        (
            30,
            170
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            255,
            255,
            255
        ),

        2
    )


    # ========================================================
    # 38. 保存结果视频
    # ========================================================

    if writer is not None:

        writer.write(
            frame
        )


    # ========================================================
    # 39. 显示
    # ========================================================

    cv2.imshow(

        "MOG2 + Centroid Evaluation",

        frame
    )


    # 如果你还想看 mask，
    # 可以取消下面注释

    # cv2.imshow(
    #     "Foreground Mask",
    #     mask
    # )


    key = cv2.waitKey(
        WAIT_TIME
    ) & 0xFF


    if key == 27:

        break


# ============================================================
# 40. 释放资源
# ============================================================

cap.release()


if writer is not None:

    writer.release()


cv2.destroyAllWindows()


# ============================================================
# 41. 平均 Processing FPS
# ============================================================

average_processing_fps = 0.0


if total_processing_time > 0:

    average_processing_fps = (

        total_frames
        /
        total_processing_time
    )


# ============================================================
# 42. Count Error / Accuracy
# ============================================================

count_error = None

count_accuracy = None


if (
    GROUND_TRUTH_COUNT is not None

    and

    GROUND_TRUTH_COUNT > 0
):


    count_error = abs(

        vehicle_count

        -

        GROUND_TRUTH_COUNT
    )


    count_accuracy = max(

        0.0,

        1.0
        -
        (
            count_error

            /

            GROUND_TRUTH_COUNT
        )

    ) * 100


# ============================================================
# 43. 最终实验结果
# ============================================================

print()

print(
    "========================================"
)

print(
    "MOG2 + Centroid Evaluation"
)

print(
    "========================================"
)


print(
    "Frames Processed:",
    total_frames
)


print(
    "Average Processing FPS:",
    round(
        average_processing_fps,
        2
    )
)


print()


print(
    "Predicted Total:",
    vehicle_count
)


print(
    "UP:",
    up_count
)


print(
    "DOWN:",
    down_count
)


# ============================================================
# Ground Truth
# ============================================================

if GROUND_TRUTH_COUNT is not None:


    print()


    print(
        "Ground Truth Count:",
        GROUND_TRUTH_COUNT
    )


    print(
        "Count Error:",
        count_error
    )


    print(
        "Counting Accuracy:",
        f"{count_accuracy:.2f}%"
    )


# ============================================================
# 输出视频
# ============================================================

if SAVE_OUTPUT_VIDEO:


    print()


    print(
        "Result Video:",
        OUTPUT_VIDEO_PATH
    )


print(
    "========================================"
)