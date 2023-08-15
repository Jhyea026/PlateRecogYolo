import os
from ultralytics import YOLO
import cv2
import easyocr as ocr


if __name__ == '__main__':

    index = 0
    model = YOLO("datasets/best.pt")
    reader = ocr.Reader(['en'])

    with os.scandir('test') as test_folder:
        for img_path in test_folder:

            img = cv2.imread(img_path.path)
            

            results = model.predict(img, conf=0.6)
            for result in results:
                for i in range(len(result.boxes)):
                    box = result.boxes.xywh[i]
                    x0 = int(box[0] - box[2] / 2) 
                    y0 = int(box[1] - box[3] / 2)
                    x1 = int(box[0] + box[2] / 2)
                    y1 = int(box[1] + box[3] / 2)

                    label_text = f"Placa: {round(result.boxes.conf[i].item(), 2) * 100} %"

                    region = img[(y0):(y1), (x0):(x1)]

                    # Processamento IMagens

                    region = cv2.resize(region, (region.shape[1] * 3, region.shape[0] * 3), interpolation=cv2.INTER_AREA)
                    # kernel = np.array([[0,-1,0],
                    #                    [-1,6,-1],
                    #                    [0,-1,0]])
                    
                    # sharp = cv2.filter2D(region, -1, kernel)
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    # _ , thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
                    
                    
                    cv2.imwrite(f"outputs/crop_plate_{index}.jpg", gray)
                    
                    index += 1
                    plate = reader.readtext(gray, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                    cv2.putText(img, str(plate), (x0,y1 + 50),  cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)          
                    cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 5)
                    cv2.putText(img, label_text, (x0,y0 - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)

            cv2.imwrite(f"outputs/imagem_{index}.jpg", img)
            
