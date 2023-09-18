import cv2
import argparse
import random

from supervision import Color
from ultralytics import YOLO
import supervision as sv
import time
import threading


person_counter = 0  # İlk kişi için ID sayacı
person_id_map = {}  # {Tracker ID: Person ID} çiftlerini tutacak bir sözlük


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alperen Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[720, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def process_camera_frames(cap, model, box_annatator):
    global person_counter, person_id_map
    for result in model.track(source=0, show=False, stream=True):
        frame = result.orig_img

        # ret, frame = cap.read()
        # result = model(frame)[0]
        # frame = result.orig_img

        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        custom_labels = []

        for xyxy, mask, confidence, class_id, tracker_id in detections:
            class_name = model.model.names[class_id]

            if class_name == "person":
                if tracker_id not in person_id_map:
                    #print(tracker_id)
                    person_counter += 1
                    person_id = person_counter
                    person_id_map[tracker_id] = person_id
                else:
                    person_id = person_id_map[tracker_id]
                custom_label = f"id:{person_id} {class_name} {confidence:.2f}"
            else:
                if tracker_id in person_id_map:
                    person_id = person_id_map[tracker_id]
                    #custom_label = f"id:{person_id}.{class_name} {confidence:.2f}"
                    custom_label = f"{class_name} {confidence:.2f}"
                else:
                    #custom_label = f"{class_name} {confidence:.2f}"
                    custom_label = f"id:{person_id}.{class_name} {confidence:.2f}"

            custom_labels.append(custom_label)

        frame = box_annatator.annotate(scene=frame, detections=detections, labels=custom_labels)
        cv2.imshow("Object_Detect", frame)

        if cv2.waitKey(30) == 27:
            break


def main():
    args = parse_arguments()
    frame_genislik, frame_yükseklik = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_genislik)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_yükseklik)

    model = YOLO("ppe50_1.pt")

    box_annatator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5,
        text_color=Color.white()
    )

    thread = threading.Thread(target=process_camera_frames, args=(cap, model, box_annatator))
    thread.start()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    thread.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()