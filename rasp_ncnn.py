import cv2
import time
from ultralytics import YOLO


def main():
    model_path = "/home/lm/best_ncnn_model"
    print(f"loading ncnn_model: {model_path}")

    model = YOLO(model_path, task='detect')

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Start inference implementation.....press Q to quit")

    prev_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("can't get the frame")
            break


        results = model.predict(source=frame, save=False, show=False, conf=0.5)

        annotated_frame = results[0].plot()

        current_time = time.time()
        # 补齐了右括号
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Herbal Medicine Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()