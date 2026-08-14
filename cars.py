import cv2
import numpy as np
import math


# ============================================================
# 参数
# ============================================================

VIDEO_PATH = r"C:\Users\wuuy2\Downloads\xzg_875610.mp4"

# 车辆最小尺寸
MIN_WIDTH = 50
MIN_HEIGHT = 50

# MOG2
HISTORY = 500
VAR_THRESHOLD = 60

# 车辆跟踪最大距离
MAX_DISTANCE = 100

# 车辆最多允许丢失多少帧
MAX_MISSING = 8

# ============================================================
# 计数线
# ============================================================

LINE_Y = 600

# 允许车辆中心点在计数线附近有一点抖动
OFFSET = 15

# ============================================================
# 播放速度
# ============================================================

WAIT_TIME = 30


# ============================================================
# cars
#
# 每一辆车保存：
#
# center       当前中心点
# previous     上一帧中心点
# bbox         矩形框
# missing      连续丢失帧数
# counted      是否已经计数
# direction    移动方向
# ============================================================

cars = {}

next_id = 0

car_count = 0


# ============================================================
# 中心点
# ============================================================

def get_center(x, y, w, h):

    cx = x + w // 2
    cy = y + h // 2

    return cx, cy


# ============================================================
# 两点距离
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# 打开视频
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("视频打开失败")

    exit()


# ============================================================
# MOG2
# ============================================================

bgsubmog = cv2.createBackgroundSubtractorMOG2(

    history=HISTORY,

    varThreshold=VAR_THRESHOLD,

    detectShadows=True
)


# ============================================================
# 形态学 Kernel
# ============================================================

kernel = cv2.getStructuringElement(

    cv2.MORPH_RECT,

    (5, 5)
)


# ============================================================
# 主循环
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        break


    # ========================================================
    # 1. 高斯滤波
    # ========================================================

    blur = cv2.GaussianBlur(

        frame,

        (5, 5),

        0
    )


    # ========================================================
    # 2. MOG2
    # ========================================================

    mask = bgsubmog.apply(blur)


    # ========================================================
    # 3. 去除阴影
    # ========================================================

    _, mask = cv2.threshold(

        mask,

        200,

        255,

        cv2.THRESH_BINARY
    )


    # ========================================================
    # 4. 开运算
    # ========================================================

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel,

        iterations=1
    )


    # ========================================================
    # 5. 闭运算
    # ========================================================

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=2
    )


    # ========================================================
    # 6. 膨胀
    # ========================================================

    mask = cv2.dilate(

        mask,

        kernel,

        iterations=1
    )


    # ========================================================
    # 7. 找轮廓
    # ========================================================

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    detections = []


    # ========================================================
    # 8. 筛选车辆
    # ========================================================

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)


        # 太小的目标不要

        if w < MIN_WIDTH or h < MIN_HEIGHT:

            continue


        # 过滤特别细长的目标

        ratio = w / float(h)

        if ratio > 5 or ratio < 0.2:

            continue


        cx, cy = get_center(

            x,
            y,
            w,
            h
        )


        detections.append({

            "center": (cx, cy),

            "bbox": (x, y, w, h)
        })


    # ========================================================
    # 9. 已有车辆先增加 missing
    # ========================================================

    for car_id in cars:

        cars[car_id]["missing"] += 1


    used_ids = set()


    # ========================================================
    # 10. 最近邻匹配
    # ========================================================

    for detection in detections:

        center = detection["center"]

        best_id = None

        best_dist = MAX_DISTANCE


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
        # 找到旧车辆
        # ====================================================

        if best_id is not None:

            car = cars[best_id]


            # 保存上一帧位置

            previous = car["center"]

            previous_y = previous[1]

            current_y = center[1]


            # ==================================================
            # 判断运动方向
            # ==================================================

            dy = current_y - previous_y


            if dy > 2:

                car["direction"] = "DOWN"

            elif dy < -2:

                car["direction"] = "UP"


            # ==================================================
            # 核心计数逻辑
            #
            # 从计数线一侧移动到另一侧
            #
            # 上 -> 下
            # 或
            # 下 -> 上
            # ==================================================

            if not car["counted"]:

                # ------------------------------------------
                # 上往下
                # ------------------------------------------

                crossed_down = (

                    previous_y < LINE_Y - OFFSET

                    and

                    current_y >= LINE_Y + OFFSET
                )


                # ------------------------------------------
                # 下往上
                # ------------------------------------------

                crossed_up = (

                    previous_y > LINE_Y + OFFSET

                    and

                    current_y <= LINE_Y - OFFSET
                )


                # ------------------------------------------
                # 由于视频可能一帧移动很快，
                # 如果直接从线左边跳到右边，
                # 也允许直接判断。
                # ------------------------------------------

                simple_down = (

                    previous_y < LINE_Y

                    and

                    current_y >= LINE_Y
                )


                simple_up = (

                    previous_y > LINE_Y

                    and

                    current_y <= LINE_Y
                )


                if crossed_down or crossed_up:

                    car_count += 1

                    car["counted"] = True

                    print(
                        "车辆 ID:",
                        best_id,
                        "通过计数线",
                        "当前数量:",
                        car_count
                    )


                elif simple_down or simple_up:

                    car_count += 1

                    car["counted"] = True

                    print(
                        "车辆 ID:",
                        best_id,
                        "通过计数线",
                        "当前数量:",
                        car_count
                    )


            # ==================================================
            # 更新车辆位置
            # ==================================================

            car["previous"] = previous

            car["center"] = center

            car["bbox"] = detection["bbox"]

            car["missing"] = 0


            used_ids.add(best_id)


        # ====================================================
        # 没有找到对应车辆
        # 创建新 ID
        # ====================================================

        else:

            cars[next_id] = {

                "center": center,

                "previous": center,

                "bbox": detection["bbox"],

                "missing": 0,

                "counted": False,

                "direction": "UNKNOWN"
            }

            next_id += 1


    # ========================================================
    # 11. 删除消失太久的车辆
    # ========================================================

    delete_ids = []


    for car_id, car in cars.items():

        if car["missing"] > MAX_MISSING:

            delete_ids.append(car_id)


    for car_id in delete_ids:

        del cars[car_id]


    # ========================================================
    # 12. 画计数线
    # ========================================================

    cv2.line(

        frame,

        (0, LINE_Y),

        (frame.shape[1], LINE_Y),

        (255, 255, 0),

        3
    )


    # ========================================================
    # 13. 画计数范围
    # ========================================================

    cv2.line(

        frame,

        (0, LINE_Y - OFFSET),

        (frame.shape[1], LINE_Y - OFFSET),

        (0, 255, 255),

        1
    )


    cv2.line(

        frame,

        (0, LINE_Y + OFFSET),

        (frame.shape[1], LINE_Y + OFFSET),

        (0, 255, 255),

        1
    )


    # ========================================================
    # 14. 绘制车辆
    # ========================================================

    for car_id, car in cars.items():

        x, y, w, h = car["bbox"]

        cx, cy = car["center"]


        # 车辆框

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 0, 255),

            2
        )


        # 中心点

        cv2.circle(

            frame,

            (cx, cy),

            5,

            (0, 255, 0),

            -1
        )


        # ID

        cv2.putText(

            frame,

            "ID:" + str(car_id),

            (x, y - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        # 方向

        cv2.putText(

            frame,

            car["direction"],

            (x, y + h + 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 0),

            2
        )


    # ========================================================
    # 15. 显示 Car Count
    # ========================================================

    cv2.putText(

        frame,

        "Car Count: " + str(car_count),

        (30, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        2,

        (0, 0, 255),

        4
    )


    # ========================================================
    # 16. 显示视频
    # ========================================================

    cv2.imshow(

        "Vehicle Detection",

        frame
    )



    # ========================================================
    # 18. 播放速度
    # ========================================================

    key = cv2.waitKey(WAIT_TIME) & 0xFF


    if key == 27:

        break


# ============================================================
# 结束
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("最终车辆数量:", car_count)