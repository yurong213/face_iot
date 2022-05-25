import cv2
import os
import numpy as np
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列
pathfile = os.listdir("caltech_faces/")
pathfile = [x for x in pathfile if '.' not in x]

for i in range(len(pathfile)):
    print(i)
    path = pathfile[i]
    all_file = os.listdir("caltech_faces/" + path)
    for j in range(len(all_file)):
        img = cv2.imread(f'caltech_faces/{path}/{all_file[j]}')  # 依序開啟每一張蔡英文的照片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
        img_np = np.array(gray, 'uint8')               # 轉換成指定編碼的 numpy 陣列
        face = detector.detectMultiScale(gray)        # 擷取人臉區域
        for(x, y, w, h) in face:
            faces.append(img_np[y:y+h, x:x+w])         # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(i+1)                             # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1


print('camera...')                                # 提示啟用相機
cap = cv2.VideoCapture(2)                         # 啟用相機
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()                         # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray, 'uint8')               # 轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray)        # 擷取人臉區域
    for(x, y, w, h) in face:
        faces.append(img_np[y:y+h, x:x+w])         # 記錄自己人臉的位置和大小內像素的數值
        ids.append(82)                             # 記錄自己人臉對應的 id，只能是整數，都是 1 表示川普的 id
    cv2.imshow('oxxostudio', img)                      # 顯示攝影機畫面
    if cv2.waitKey(100) == ord('q'):              # 每一毫秒更新一次，直到按下 q 結束
        break

print('training...')                              # 提示開始訓練
recog.train(faces, np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為 face.yml
print('ok!')
cap.release()
cv2.destroyAllWindows()