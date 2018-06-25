import cv2

# 分類器ディレクトリ(以下から取得)
# https://github.com/opencv/opencv/blob/master/data/haarcascades/
# https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades/

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
    im_org = im.copy()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

    facerect = cascade.detectMultiScale(im_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))   
    color = (255, 255, 255) #白
    # 検出した場合
    if len(facerect) > 0:
        #検出した顔を囲む矩形の作成
        for rect in facerect:
            cv2.rectangle(im, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        cv2.imshow('camera capture', im_org)
        cv2.imshow('RGB2GBR',im)
        cv2.moveWindow('CamCap', 0, 0)

    else:
        print("not found faces")
        cv2.moveWindow('Face_rect', 640, 0)
    key =cv2.waitKey(10)
    if key == 27:
        break
    # End if
# End while