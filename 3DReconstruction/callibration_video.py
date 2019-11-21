from cv2 import cv2
import numpy as np
import cv2_helpers

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cap = cv2.VideoCapture('pictures/ChessboardPattern/11.mp4')
nx, ny = 8,6
current_frame = 0
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
fps = cv2_helpers.get_fps(cap)
num_frames = cv2_helpers.get_frames(cap)
frames_with_corners = 0

def skip_frame(video):
    ret2,frame2 = video.read()
    if ret2:
        curr_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        if curr_frame <= num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 0.5*fps)
            return frame2

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = skip_frame(cap)    
        r_frame = cv2_helpers.get_resized(frame,40)
        r_frame_bw = cv2_helpers.get_resized_and_bw(frame,40)
        cv2.imshow("something",r_frame)
        found_corners,corners = cv2.findChessboardCorners(r_frame_bw, (nx,ny), None)
        if found_corners:
            frames_with_corners += 1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(r_frame_bw,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            new_img = cv2.drawChessboardCorners(r_frame,(nx,ny), corners, ret) 
            cv2.imshow("test",new_img)
            if frames_with_corners == 16:
                break

cap.release()
cv2.destroyAllWindows()


print("callibrating the given video...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, r_frame_bw.shape[::-1],None,None)

print("ret: ",ret)
print("mtx: ",mtx)
print("dist: ",dist)
print("rvecs: ",rvecs)
print("tvecs: ",tvecs)
