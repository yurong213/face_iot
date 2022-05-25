# import cv2
#
# cap = cv2.VideoCapture(2)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
#
# while(True):
#     # 擷取影像
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#
#     # 彩色轉灰階
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 顯示圖片
#     cv2.imshow('live', frame)
#     #cv2.imshow('live', gray)
#
#     # 按下 q 鍵離開迴圈
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# # 釋放該攝影機裝置
# cap.release()
# cv2.destroyAllWindows()
import os
pathfile = os.listdir("caltech_faces/")
pathfile = [x for x in pathfile if '.' not in x]
name = {}
for i in range(len(pathfile)):
    path = pathfile[i]
    all_file = os.listdir("caltech_faces/" + path)
    name[str(i+1)] = path

print(name)