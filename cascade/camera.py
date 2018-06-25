import cv2

cap = cv2.VideoCapture(0)
ret, img = cap.read()
# カラーとグレースケールで場合分け
if len(img.shape) == 3:
    height, width, channels = img.shape[:3]
else:
    height, width = img.shape[:2]
    channels = 1
# 取得結果（幅，高さ，チャンネル数，depth）を表示
print("width: " + str(width))
print("height: " + str(height))
print("channels: " + str(channels))
print("dtype: " + str(img.dtype))

while True:
    ret, im = cap.read()
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('camera capture', im)
    cv2.imshow('RGB2GBR',im_rgb)
    cv2.moveWindow('camera capture', 0, 0)
    cv2.moveWindow('RGB2GBR', 640, 0)
    key = cv2.waitKey(10)
    if key == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()