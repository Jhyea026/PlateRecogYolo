import cv2
from ultralytics import YOLO
import easyocr as ocr

model = YOLO('datasets/thebest.pt')
index = 0
cap = cv2.VideoCapture('video/video4.mp4')
reader = ocr.Reader(['en'])

if not cap.isOpened():
    raise Exception('Video indisponivel')

while True:
    sucess, frame = cap.read()
    if sucess:

        results = model(frame, conf=0.25)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        for result in results:
                for i in range(len(result.boxes)):
                    box = result.boxes.xywh[i]
                    x0 = int(box[0] - box[2] / 2) 
                    y0 = int(box[1] - box[3] / 2)
                    x1 = int(box[0] + box[2] / 2)
                    y1 = int(box[1] + box[3] / 2)
                    region = frame[(y0):(y1), (x0):(x1)]

                    #Redimensiona tamanho da janela que mostra a placa
                    region = cv2.resize(region, (region.shape[1] * 5, region.shape[0] * 5), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    _ , thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

                    #Ler os textos escritos nas placas
                    plate = reader.readtext(thresh, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                    # cv2.putText(img, str(plate), (x0,y1 + 50),  cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)          
                    # cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 5)
                    # cv2.putText(img, label_text, (x0,y0 - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
                    cv2.imshow('Placas', region)
                    cv2.imshow('Placas-thresh', thresh)
                    index += 1

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        key = cv2.waitKey(45)
        if key == ord ('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
