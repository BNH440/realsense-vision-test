import pyrealsense2 as rs
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Object detection model
model = YOLO("yolov8n.pt", verbose=False)

cv2.namedWindow("Color Image", cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow("Depth Image", cv2.WINDOW_AUTOSIZE)

pipeline = rs.pipeline()
config = rs.config()

wrapper = rs.pipeline_wrapper(pipeline)
profile = config.resolve(wrapper)

colorizer = rs.colorizer()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

print(f"Depth scale:{depth_scale}")

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

tracker = sv.ByteTrack()
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    ### SUM (very computationally intensive)
    sum = 0.00
    for y in range(480):
        for x in range(640):
            dist = depth_frame.get_distance(x, y)
            sum += dist

    avg = sum / (640 * 480)
    print(f"Avg Distance: {avg}")
    ###

    # Object detection
    result = model(color_image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"{result.names[class_id]} {str(round(confidence, 2))} D:{str(round(depth_frame.get_distance(int(((xyxy[2] - xyxy[0])/2)+xyxy[0]), int(((xyxy[3] - xyxy[1])/2)+xyxy[1])) * 100))}cm"
        for class_id, confidence, xyxy in zip(
            detections.class_id, detections.confidence, detections.xyxy
        )
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=color_image.copy(), detections=detections
    )
    labeled_annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    cv2.imshow("Color Image", labeled_annotated_image)
    # cv2.imshow("Depth Image", depth_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
