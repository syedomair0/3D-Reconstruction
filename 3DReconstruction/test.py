from cv2 import cv2
import numpy as np
import cv2_helpers


cap = cv2.VideoCapture('pictures/test2.mp4')
cap2 = cv2.VideoCapture('pictures/test2.mp4')

fps = round(cv2_helpers.get_fps(cap))

def next_frame(video):
    ret2, frame2 = video.read()
    if ret2 == True:
        curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 2)
        return frame2

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        img1 = cv2_helpers.get_blackandwhite(cv2_helpers.get_resized(frame,30))
        cv2.imshow("image 1", img1)
        img2 = next_frame(cap2)
        img2 = cv2_helpers.get_blackandwhite(cv2_helpers.get_resized(img2,30))
        cv2.imshow("image 2",img2)
        print(cap.get(cv2.CAP_PROP_POS_FRAMES), cap2.get(cv2.CAP_PROP_POS_FRAMES))
        disparity_map = cv2_helpers.get_disparity(img2,img1)
        cv2.imshow("disparity", disparity_map)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
 
cap.release()
cv2.destroyAllWindows()

