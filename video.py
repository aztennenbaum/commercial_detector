import numpy as np
import cv2
import collections
import sys

#file to read for capture
cap = cv2.VideoCapture(sys.argv[1])

#reads the most recent score from the score window. if its above the threshold, we are not in commercial
def exited_commercial(sw,threshold = 0.2):
    return sw[-1]>threshold

# when the logo shrinks or slides, there are a couple frames where we get a partial match
# look at the last 5 matches and check for a sharp transition before flagging as a commercial,
# to avoid false positives
# NOTE: for this to work, we must process every frame
def entered_commercial(sw,threshold = 0.9975):
    if(len(sw)<5):
        return False
    else:
        return (sw[0]+sw[1])/np.sqrt(2*(sw[0]**2+sw[1]**2+sw[-2]**2+sw[-1]**2)+np.finfo(float).eps) >threshold

# Display image & allow user to make a rectangular selection
def select_roi(img):
    def select_roi_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param[0] = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            param[0].append((x, y))
        elif len(param[0]) == 1:
            clone = param[1].copy()
            cv2.rectangle(clone, param[0][0], (x, y), (0, 0, 255), 2)
            cv2.imshow("image", clone)
    
    print("Click and drag to select region to use for commercial detection")
    print("Press any key when finished")
    cv2.namedWindow("image")
    param = [[],img] #convert to image boundary
    cv2.setMouseCallback("image", select_roi_cb, param)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sx=param[0][0][0] #start x
    sy=param[0][0][1] #start y
    ex=param[0][1][0] #end x
    ey=param[0][1][1] #end y
    return (sx,sy,ex,ey)

def edge_detector(img):
    x = np.asarray(cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)).flatten()
    return x/np.sqrt((x**2).sum()+np.finfo(x.dtype).eps)

ret,img = cap.read()
sx, sy, ex, ey = select_roi(img)
mask=edge_detector(img[sy : ey, sx : ex])
score_window = collections.deque(maxlen=5)

isCommercial = False
frame=0
print("Press 'q' to quit")
while(cap.isOpened()):
    ret,img = cap.read()
    score_window.append(np.dot(edge_detector(img[sy : ey, sx : ex]),mask))
    
    if exited_commercial(score_window):
        isCommercial = False
    elif entered_commercial(score_window):
        isCommercial = True
    if frame%30==0:
        print("Frame={}, score={}, isCommercial={}".format(frame,score_window[-1],isCommercial))
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame+=1

cap.release()
cv2.destroyAllWindows()
