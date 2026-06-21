import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2


def main():
    model_path = "/home/lm/herbal_project/best_ncnn_model"
    print(f"Loading NCNN model: {model_path}")
    model = YOLO(model_path, task='detect')

    print("Initializing Picamera2 for IMX219...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    print("Inference started. Press 'Q' to quit.")
    prev_time = 0

    while True:
        try:
            frame_raw = picam2.capture_array()
            frame = frame_raw

        except Exception as e:
            print(f"Failed to grab frame: {e}")
            break

        results = model.predict(source=frame, save=False, show=False, conf=0.3)
        annotated_frame = results[0].plot()

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Herbal Medicine Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()