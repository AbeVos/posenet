import numpy as np
import cv2
import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 5)

frames = []

recording = False

while True:
    #print(cap.get(cv2.CAP_PROP_FPS))
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if recording:
        frames.append(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
        cv2.putText(frame, "Press Q to save and quit recording", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0))
    else:
        cv2.putText(frame, "Press Enter to start recording", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0))

    cv2.putText(frame, "Press Escape to quit without saving", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0))
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 10:
        recording = True
    elif key == ord('q') or len(frames) >= 300:

        if recording:
            path = "data/raw/"
            filename = "raw_" + str(datetime.datetime.now())
            
            print("Saving " + str(len(frames)) + " frames to " + path + filename)
            
            frames = np.asarray(frames, dtype=type(float))

            np.save(path + filename, frames)
            print("Saved " + str(frames.shape[0]) + " frames")

        break

cv2.destroyAllWindows()
cap.release()
