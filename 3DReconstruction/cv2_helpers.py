import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

def get_resized(image, percent):
    width = int(image.shape[1] * percent/ 100)
    height = int(image.shape[0] * percent/ 100)
    dim = (width, height)
    resized_image = cv2.resize(image,dim,interpolation =cv2.INTER_AREA )
    return resized_image

def get_blackandwhite(image):
    bw_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return bw_image

##________May not workkk!!!_____________
def get_colored(image):
    color_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    return color_image
#________________________________________
def get_resized_and_bw(frame,percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    resized_image = cv2.resize(frame,dim,interpolation =cv2.INTER_AREA)
    bw_frame = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    return bw_frame


def get_disparity(left_image,right_image):
    window_size = 3                     
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,             
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.2
    #visual_multiplier = 1.0
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(right_image,left_image)  
    dispr = right_matcher.compute(left_image, right_image) 
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, left_image, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg

def get_frames(video):
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return frames

def get_fps(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps

def get_height_width(video):
    ret, frame = video.read()
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return (height,width)

def create_flann_picture(self,left_image,right_image):
    orb = cv2.ORB_create(nfeatures=2000)
    left_keypoint, left_descriptors = orb.detectAndCompute(left_image,None)
    right_keypoints, right_descriptors = orb.detectAndCompute(right_image,None)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 1) #2
    
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(left_descriptors,right_descriptors,k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = cv2.DrawMatchesFlags_DEFAULT)
    final_image = cv2.drawMatchesKnn(left_image,left_keypoint,right_image,right_keypoints,matches,None,**draw_params)
    cv2.imshow("something",final_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def fast_detection(image):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(image,None)
    fast.setNonmaxSuppression(0)
    image_with_keypoints = cv2.drawKeypoints(image, kp, None, color=(255,0,0))
    return image_with_keypoints

def get_orb(image):
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints_orb, descriptors = orb.detectAndCompute(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints_orb,None)
    return image_with_keypoints

def compute_disparity(left_image,right_image):
    stereo = cv2.StereoBM(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_image,right_image)
    return disparity

def get_epipolar_lines(img1,img2):
    pass

def get_callibration_stuff():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cap = cv2.VideoCapture('pictures/ChessboardPattern/11.mp4')
    nx, ny = 8,6
    objp = np.zeros((8*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    fps = get_fps(cap)
    num_frames = get_frames(cap)
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
            r_frame_bw = get_resized_and_bw(frame,40)
            found_corners,corners = cv2.findChessboardCorners(r_frame_bw, (nx,ny), None)
            if found_corners:
                frames_with_corners += 1
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(r_frame_bw,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                if frames_with_corners == 16:
                    break

    cap.release()
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, r_frame_bw.shape[::-1],None,None)
    return ret,mtx,dist,rvecs,tvecs

    