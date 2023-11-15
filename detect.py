import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('VID2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0

tracker = Tracker()

cy1 = 222
cy2 = 368
offset = 6

# Thêm một biến flag để kiểm soát trạng thái dừng lại
stop_flag = False

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    listcar = []
    listtruck = []
    listmotorcycle = []
             
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            listcar.append([x1, y1, x2, y2])
        if 'truck' in c:
            listtruck.append([x1, y1, x2, y2])
        if 'motorcycle' in c:
            listmotorcycle.append([x1, y1, x2, y2])
    
    bbox_car= tracker.update(listcar)
    for bbox in bbox_car:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.putText(frame, "car", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        #cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    bbox_truck= tracker.update(listtruck)
    for bbox in bbox_truck:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.putText(frame, "truck", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        #cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)  

    bbox_motorcycle= tracker.update(listmotorcycle)
    for bbox in bbox_motorcycle:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.putText(frame, "motorcycle", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1) 

    cv2.line(frame, (100, 250), (800, 440), (255, 255, 255), 1)
    cv2.line(frame, (320, 230), (850, 350), (255, 255, 255), 1)
    cv2.imshow("RGB", frame)

    # Kiểm tra xem flag có được đặt thành True không
    if stop_flag:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nếu phím Q được nhấn
        stop_flag = True

cap.release()
cv2.destroyAllWindows()
