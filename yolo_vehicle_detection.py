import cv2
import time

from collections import defaultdict, deque, Counter
from ultralytics import YOLO


# ============================================================
# 1. 参数配置
# ============================================================

VIDEO_PATH = "xzg_875610.mp4"

MODEL_PATH = "yolo11n.pt"

CONF_THRESHOLD = 0.35

WAIT_TIME = 30

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}

TRACK_HISTORY_LENGTH = 30

MOTION_THRESHOLD = 2


# ============================================================
# 2. Counting Zone
# ============================================================

ZONE_TOP = 300
ZONE_BOTTOM = 400


# ============================================================
# 3. Ground Truth
#
# 如果你确认真实数量是 52：
#
# GROUND_TRUTH_COUNT = 52
# ============================================================

GROUND_TRUTH_COUNT = 52


# ============================================================
# 4. 输出视频
# ============================================================

SAVE_OUTPUT_VIDEO = True

OUTPUT_VIDEO_PATH = "yolo_bytetrack_result.mp4"


# ============================================================
# 5. 加载 YOLO
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# 6. Track History
#
# 保存每辆车最近的中心点
# ============================================================

track_history = defaultdict(
    lambda: deque(maxlen=TRACK_HISTORY_LENGTH)
)


# ============================================================
# 7. Track State
#
# 用于 Counting Zone 状态判断
# ============================================================

track_states = defaultdict(
    lambda: {
        "last_region": None,
        "entry_region": None
    }
)


# ============================================================
# 8. 类别投票
#
# 示例：
#
# class_votes[18]
#
# Counter({
#     "truck": 20,
#     "car": 3
# })
#
# 最终：
#
# ID 18 → truck
# ============================================================

class_votes = defaultdict(Counter)


# ============================================================
# 9. 已经完成计数的 Track ID
# ============================================================

counted_ids = set()


# ============================================================
# 10. 总数 / 方向统计
# ============================================================

vehicle_count = 0

up_count = 0

down_count = 0


# ============================================================
# 11. 性能统计
# ============================================================

total_frames = 0

total_processing_time = 0.0

total_model_time = 0.0


# ============================================================
# 12. 判断中心点所在区域
# ============================================================

def get_region(cy):

    if cy < ZONE_TOP:
        return "ABOVE"

    elif cy > ZONE_BOTTOM:
        return "BELOW"

    else:
        return "ZONE"


# ============================================================
# 13. 获得某一个 Track 的稳定类别
#
# 使用多数投票
# ============================================================

def get_stable_class(track_id):

    votes = class_votes[track_id]

    if not votes:
        return "unknown"

    stable_class = votes.most_common(1)[0][0]

    return stable_class


# ============================================================
# 14. 根据当前所有 counted ID
#     重新计算类别统计
#
# 这样即使车辆穿线以后，
# 后续帧提供了更多类别信息，
# 最终统计也可以继续修正。
# ============================================================

def get_class_counts():

    counts = defaultdict(int)

    for track_id in counted_ids:

        stable_class = get_stable_class(track_id)

        if stable_class in VEHICLE_CLASSES:

            counts[stable_class] += 1

    return counts


# ============================================================
# 15. 打开视频
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():

    print("Error opening video stream or file")

    exit()


# ============================================================
# 16. 获取视频信息
# ============================================================

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

video_frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
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
# 17. 输出视频
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
# 18. 主循环
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:
        break


    total_frames += 1


    # ========================================================
    # 当前帧总处理时间
    # ========================================================

    frame_start_time = time.perf_counter()


    # ========================================================
    # 19. YOLO + ByteTrack
    # ========================================================

    model_start_time = time.perf_counter()


    results = model.track(
        source=frame,
        conf=CONF_THRESHOLD,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )


    model_end_time = time.perf_counter()


    total_model_time += (
        model_end_time
        -
        model_start_time
    )


    result = results[0]


    # ========================================================
    # 20. 遍历所有检测目标
    # ========================================================

    for box in result.boxes:


        # ----------------------------------------------------
        # 必须有 Track ID
        # ----------------------------------------------------

        if box.id is None:
            continue


        track_id = int(
            box.id[0]
        )


        # ----------------------------------------------------
        # 当前帧类别
        # ----------------------------------------------------

        cls_id = int(
            box.cls[0]
        )

        current_class = model.names[
            cls_id
        ]


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            box.conf[0]
        )


        # ----------------------------------------------------
        # 只处理车辆
        # ----------------------------------------------------

        if current_class not in VEHICLE_CLASSES:
            continue


        # ====================================================
        # 21. 类别投票
        #
        # 每看到一次当前 Track，
        # 就给对应类别投一票
        # ====================================================

        class_votes[
            track_id
        ][
            current_class
        ] += 1


        # ====================================================
        # 22. 当前稳定类别
        # ====================================================

        stable_class = get_stable_class(
            track_id
        )


        # ====================================================
        # 23. Bounding Box
        # ====================================================

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().tolist()
        )


        # ====================================================
        # 24. 中心点
        # ====================================================

        cx = (
            x1 + x2
        ) // 2

        cy = (
            y1 + y2
        ) // 2


        # ====================================================
        # 25. 保存轨迹
        # ====================================================

        points = track_history[
            track_id
        ]

        points.append(
            (
                cx,
                cy
            )
        )


        # ====================================================
        # 26. 方向判断
        # ====================================================

        direction = "UNKNOWN"


        if len(points) >= 5:

            old_y = points[-5][1]

            current_y = points[-1][1]

            dy = (
                current_y
                -
                old_y
            )


            if dy > MOTION_THRESHOLD:

                direction = "DOWN"


            elif dy < -MOTION_THRESHOLD:

                direction = "UP"


        elif len(points) >= 2:

            old_y = points[-2][1]

            current_y = points[-1][1]

            dy = (
                current_y
                -
                old_y
            )


            if dy > MOTION_THRESHOLD:

                direction = "DOWN"


            elif dy < -MOTION_THRESHOLD:

                direction = "UP"


        # ====================================================
        # 27. Counting Zone 状态
        # ====================================================

        current_region = get_region(
            cy
        )


        state = track_states[
            track_id
        ]


        last_region = state[
            "last_region"
        ]


        # ====================================================
        # 28. 第一次看到 Track
        # ====================================================

        if last_region is None:

            state[
                "last_region"
            ] = current_region


            if current_region in (
                "ABOVE",
                "BELOW"
            ):

                state[
                    "entry_region"
                ] = current_region


        # ====================================================
        # 29. 已经存在历史状态
        # ====================================================

        else:


            # ------------------------------------------------
            # ABOVE → ZONE
            # ------------------------------------------------

            if (
                last_region == "ABOVE"
                and
                current_region == "ZONE"
            ):

                state[
                    "entry_region"
                ] = "ABOVE"


            # ------------------------------------------------
            # BELOW → ZONE
            # ------------------------------------------------

            elif (
                last_region == "BELOW"
                and
                current_region == "ZONE"
            ):

                state[
                    "entry_region"
                ] = "BELOW"


            # =================================================
            # 30. ABOVE → ZONE → BELOW
            #
            # DOWN
            # =================================================

            elif (
                last_region == "ZONE"
                and
                current_region == "BELOW"
            ):

                if (
                    state["entry_region"] == "ABOVE"
                    and
                    track_id not in counted_ids
                ):

                    vehicle_count += 1

                    down_count += 1

                    counted_ids.add(
                        track_id
                    )


                    print(
                        f"ID:{track_id} "
                        f"{stable_class} "
                        f"crossed DOWN | "
                        f"Total:{vehicle_count}"
                    )


            # =================================================
            # 31. BELOW → ZONE → ABOVE
            #
            # UP
            # =================================================

            elif (
                last_region == "ZONE"
                and
                current_region == "ABOVE"
            ):

                if (
                    state["entry_region"] == "BELOW"
                    and
                    track_id not in counted_ids
                ):

                    vehicle_count += 1

                    up_count += 1

                    counted_ids.add(
                        track_id
                    )


                    print(
                        f"ID:{track_id} "
                        f"{stable_class} "
                        f"crossed UP | "
                        f"Total:{vehicle_count}"
                    )


            # =================================================
            # 32. ABOVE → BELOW
            #
            # 快速移动兜底
            # =================================================

            elif (
                last_region == "ABOVE"
                and
                current_region == "BELOW"
            ):

                if track_id not in counted_ids:

                    vehicle_count += 1

                    down_count += 1

                    counted_ids.add(
                        track_id
                    )


                    print(
                        f"ID:{track_id} "
                        f"{stable_class} "
                        f"jumped DOWN | "
                        f"Total:{vehicle_count}"
                    )


            # =================================================
            # 33. BELOW → ABOVE
            #
            # 快速移动兜底
            # =================================================

            elif (
                last_region == "BELOW"
                and
                current_region == "ABOVE"
            ):

                if track_id not in counted_ids:

                    vehicle_count += 1

                    up_count += 1

                    counted_ids.add(
                        track_id
                    )


                    print(
                        f"ID:{track_id} "
                        f"{stable_class} "
                        f"jumped UP | "
                        f"Total:{vehicle_count}"
                    )


            # ------------------------------------------------
            # 更新区域状态
            # ------------------------------------------------

            state[
                "last_region"
            ] = current_region


        # ====================================================
        # 34. Bounding Box
        # ====================================================

        cv2.rectangle(
            frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (
                0,
                255,
                0
            ),
            2
        )


        # ====================================================
        # 35. 中心点
        # ====================================================

        cv2.circle(
            frame,
            (
                cx,
                cy
            ),
            5,
            (
                0,
                0,
                255
            ),
            -1
        )


        # ====================================================
        # 36. 标签
        #
        # 注意：
        # 显示的是 stable_class，
        # 而不是当前单帧 current_class
        # ====================================================

        label = (
            f"ID:{track_id} "
            f"{stable_class} "
            f"{confidence:.2f} "
            f"{direction}"
        )


        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                0
            ),
            2
        )


        # ====================================================
        # 37. 绘制轨迹
        # ====================================================

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (
                    255,
                    0,
                    255
                ),
                2
            )


    # ========================================================
    # 38. Counting Zone
    # ========================================================

    cv2.line(
        frame,
        (
            0,
            ZONE_TOP
        ),
        (
            frame_width,
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
            frame_width,
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
    # 39. 当前帧处理时间
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


    if frame_processing_time > 0:

        current_fps = (
            1.0
            /
            frame_processing_time
        )

    else:

        current_fps = 0.0


    # ========================================================
    # 40. 根据当前 Track 投票结果实时计算类别数量
    # ========================================================

    live_class_count = get_class_counts()


    # ========================================================
    # 41. 显示统计信息
    # ========================================================

    cv2.putText(
        frame,
        f"Total: {vehicle_count}",
        (
            30,
            50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
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
            90
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
            125
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
            165
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


    text_y = 205


    for name in (
        "car",
        "truck",
        "bus",
        "motorcycle"
    ):

        cv2.putText(
            frame,
            f"{name}: {live_class_count[name]}",
            (
                30,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2
        )

        text_y += 30


    # ========================================================
    # 42. 保存结果视频
    # ========================================================

    if writer is not None:

        writer.write(
            frame
        )


    # ========================================================
    # 43. 显示
    # ========================================================

    cv2.imshow(
        "YOLO + ByteTrack Evaluation",
        frame
    )


    key = cv2.waitKey(
        WAIT_TIME
    ) & 0xFF


    if key == 27:

        break


# ============================================================
# 44. 释放资源
# ============================================================

cap.release()


if writer is not None:

    writer.release()


cv2.destroyAllWindows()


# ============================================================
# 45. 最终类别统计
#
# 使用整段视频的所有 Track 类别投票
# ============================================================

final_class_count = get_class_counts()


# ============================================================
# 46. 平均 Processing FPS
# ============================================================

average_processing_fps = 0.0

average_model_ms = 0.0


if total_processing_time > 0:

    average_processing_fps = (
        total_frames
        /
        total_processing_time
    )


if total_frames > 0:

    average_model_ms = (
        total_model_time
        /
        total_frames
        *
        1000
    )


# ============================================================
# 47. Counting Accuracy
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
# 48. 最终结果
# ============================================================

print()

print(
    "========================================"
)

print(
    "YOLO + ByteTrack Evaluation"
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


print(
    "Average Detection + Tracking Time:",
    round(
        average_model_ms,
        2
    ),
    "ms"
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


print()


print(
    "Car:",
    final_class_count["car"]
)


print(
    "Truck:",
    final_class_count["truck"]
)


print(
    "Bus:",
    final_class_count["bus"]
)


print(
    "Motorcycle:",
    final_class_count["motorcycle"]
)


# ============================================================
# 49. Ground Truth
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
# 50. 输出每一个被计数车辆的投票结果
#
# 这是现在最值得观察的地方
# ============================================================

print()

print(
    "========================================"
)

print(
    "Vehicle Class Voting Details"
)

print(
    "========================================"
)


for track_id in sorted(
    counted_ids
):

    stable_class = get_stable_class(
        track_id
    )

    votes = class_votes[
        track_id
    ]


    print(
        f"ID:{track_id} "
        f"Final:{stable_class} "
        f"Votes:{dict(votes)}"
    )


if SAVE_OUTPUT_VIDEO:

    print()

    print(
        "Result Video:",
        OUTPUT_VIDEO_PATH
    )


print(
    "========================================"
)