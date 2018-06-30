import cv2
import numpy as np

# 分類器ディレクトリ(以下から取得)
# https://github.com/opencv/opencv/blob/master/data/haarcascades/
# https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades/

def resize_add_space(inputim):
    tmp = inputim[:,:]
    height, width = inputim.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit)/2)
    fin = int((size + limit) / 2)
    new_img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
        new_img[start:fin, :] = tmp
    ret_im = cv2.resize(new_img,(256,256))
    return ret_im
# End def

cascade_path = "models/haarcascade_frontalface_default.xml"

# 他のモデルファイル(参考)
#cascade_path = "./models/haarcascade_frontalface_alt.xml"
#cascade_path = "./models/haarcascade_frontalface_alt2.xml"
#cascade_path = "./models/haarcascade_frontalface_alt_tree.xml"
#cascade_path = "./models/haarcascade_profileface.xml"
#cascade_path = "./models/haarcascade_mcs_nose.xml"

#camear capurture
cap = cv2.VideoCapture(0)

#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

while True:
    ret, im = cap.read()
    im_re = resize_add_space(im)
    im_org = im_re.copy()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

    facerect = cascade.detectMultiScale(im_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30,30))   
    color = (255, 255, 255) #白
    cv2.imshow('CamCap', im_org)
    # cv2.moveWindow('CamCap', 0, 0)
    # 検出した場合
    if len(facerect) > 0:
        #検出した顔を囲む矩形の作成
        for rect in facerect:
            cv2.rectangle(im_re, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        cv2.imshow('RGB2GBR',im_re)
        # cv2.moveWindow('RGB2GBR', 640, 0)        
    else:
        print("not found faces")
    key =cv2.waitKey(10)
    if key == 27:
        break
    # End if
# End while