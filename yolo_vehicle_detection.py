import math
import time

import cv2
from ultralytics import YOLO


# ============================================================
# 1. Basic configuration
# ============================================================

# Use the video already stored in this repository.
VIDEO_PATH = "xzg_875610.mp4"

# COCO-pretrained YOLO11 model.
# n = nano: fast and suitable for the first real-time baseline.
MODEL_PATH = "yolo11n.pt"

# YOLO inference parameters.
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.70
IMGSZ = 640

# Vehicle categories kept from the COCO dataset.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Centroid-tracker parameters inherited from the original project.
MAX_DISTANCE = 100
MAX_MISSING = 8

# Counting-line parameters inherited from the original project.
LINE_Y = 600
OFFSET = 15

# Video display speed.
WAIT_TIME = 1


# ============================================================
# 2. Tracking state
# ============================================================

# Each tracked vehicle stores:
# center       current centroid
# previous     previous centroid
# bbox         (x, y, w, h)
# missing      consecutive missing frames
# counted      whether it has crossed the counting line
# direction    UP / DOWN / UNKNOWN
# class_name   current YOLO class name
# confidence   current YOLO confidence
cars = {}
next_id = 0
car_count = 0


# ============================================================
# 3. Helper functions
# ============================================================

def get_center(x, y, w, h):
    """Return the center point of an xywh bounding box."""
    return x + w // 2, y + h // 2


def distance(p1, p2):
    """Euclidean distance between two center points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# 4. Load YOLO once
# ============================================================

# Important: the model is loaded once, not once per video frame.
model = YOLO(MODEL_PATH)


# ============================================================
# 5. Open video
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")


# ============================================================
# 6. Main loop
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_start = time.perf_counter()

    # ========================================================
    # 6.1 YOLO detection
    #
    # This replaces the old pipeline:
    # GaussianBlur -> MOG2 -> Threshold -> Morphology
    # -> findContours -> boundingRect
    # ========================================================

    result = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMGSZ,
        verbose=False,
    )[0]

    detections = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        class_name = result.names[cls_id]
        confidence = float(box.conf.item())

        # Keep vehicle-related COCO classes only.
        if class_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().tolist(),
        )

        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        cx, cy = get_center(x, y, w, h)

        detections.append({
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "class_name": class_name,
            "confidence": confidence,
        })

    # ========================================================
    # 6.2 Existing centroid tracking logic
    # ========================================================

    # Every existing track is considered missing until matched
    # by a detection in the current frame.
    for car_id in cars:
        cars[car_id]["missing"] += 1

    used_ids = set()

    for detection in detections:
        center = detection["center"]

        best_id = None
        best_dist = MAX_DISTANCE

        # Nearest-neighbour matching inherited from the original
        # OpenCV project. ByteTrack will replace this in phase 2.
        for car_id, car in cars.items():
            if car_id in used_ids:
                continue

            d = distance(center, car["center"])

            if d < best_dist:
                best_dist = d
                best_id = car_id

        # ----------------------------------------------------
        # Existing track matched
        # ----------------------------------------------------
        if best_id is not None:
            car = cars[best_id]

            previous = car["center"]
            previous_y = previous[1]
            current_y = center[1]

            dy = current_y - previous_y

            if dy > 2:
                car["direction"] = "DOWN"
            elif dy < -2:
                car["direction"] = "UP"

            # Count once when the track crosses LINE_Y.
            if not car["counted"]:
                crossed_down = (
                    previous_y < LINE_Y - OFFSET
                    and current_y >= LINE_Y + OFFSET
                )

                crossed_up = (
                    previous_y > LINE_Y + OFFSET
                    and current_y <= LINE_Y - OFFSET
                )

                simple_down = (
                    previous_y < LINE_Y
                    and current_y >= LINE_Y
                )

                simple_up = (
                    previous_y > LINE_Y
                    and current_y <= LINE_Y
                )

                if crossed_down or crossed_up or simple_down or simple_up:
                    car_count += 1
                    car["counted"] = True

                    print(
                        f"Vehicle ID {best_id} crossed the line | "
                        f"class={detection['class_name']} | "
                        f"total={car_count}"
                    )

            car["previous"] = previous
            car["center"] = center
            car["bbox"] = detection["bbox"]
            car["missing"] = 0
            car["class_name"] = detection["class_name"]
            car["confidence"] = detection["confidence"]

            used_ids.add(best_id)

        # ----------------------------------------------------
        # New track
        # ----------------------------------------------------
        else:
            cars[next_id] = {
                "center": center,
                "previous": center,
                "bbox": detection["bbox"],
                "missing": 0,
                "counted": False,
                "direction": "UNKNOWN",
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
            }

            used_ids.add(next_id)
            next_id += 1

    # Remove tracks that have disappeared for too long.
    delete_ids = [
        car_id
        for car_id, car in cars.items()
        if car["missing"] > MAX_MISSING
    ]

    for car_id in delete_ids:
        del cars[car_id]

    # ========================================================
    # 6.3 Visualization
    # ========================================================

    # Main counting line.
    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (255, 255, 0),
        3,
    )

    # Counting tolerance region.
    cv2.line(
        frame,
        (0, LINE_Y - OFFSET),
        (frame.shape[1], LINE_Y - OFFSET),
        (0, 255, 255),
        1,
    )

    cv2.line(
        frame,
        (0, LINE_Y + OFFSET),
        (frame.shape[1], LINE_Y + OFFSET),
        (0, 255, 255),
        1,
    )

    # Draw active vehicle tracks.
    for car_id, car in cars.items():
        x, y, w, h = car["bbox"]
        cx, cy = car["center"]

        class_name = car["class_name"]
        confidence = car["confidence"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2,
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"ID:{car_id} {class_name} {confidence:.2f}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            car["direction"],
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    # FPS is useful later when comparing MOG2 and YOLO.
    elapsed = time.perf_counter() - frame_start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    cv2.putText(
        frame,
        f"Vehicle Count: {car_count}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2,
    )

    cv2.imshow("YOLO Vehicle Detection + Centroid Tracking", frame)

    key = cv2.waitKey(WAIT_TIME) & 0xFF

    if key == 27:  # ESC
        break


# ============================================================
# 7. Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Final vehicle count:", car_count)
