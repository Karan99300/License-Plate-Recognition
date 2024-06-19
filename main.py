from ultralytics import YOLO
import cv2
from utils import *

# regular pre-trained yolov8 model for car recognition
# coco_model = YOLO('yolov8n.pt')
coco_model = YOLO('yolov8s.pt')
# yolov8 model trained to detect number plates
np_model = YOLO('train4/weights/best.pt')

video_path='istockphoto-1425716386-640_adpp_is.mp4'
video=cv2.VideoCapture(video_path)

results={}

ret=True
frame_number =-1

# all vehicle class IDs from the COCO dataset (car, motorbike,bus, truck) https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
vehicles = [2,3,5,7]
vehicle_bounding_boxes = []

while ret:
    frame_number+=1
    ret,frame=video.read()
    
    if ret and frame_number<10:
        results[frame_number]={}
        detections=coco_model.track(frame,persist=True)[0]
        
        #Car Detection
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            # I am only interested in class IDs that belong to vehicles
            if int(class_id) in vehicles and score > 0.5:
                vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
                
                #License Plate detection
                for bbox in vehicle_bounding_boxes:
                    #print(bbox)
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    license_plates=np_model(roi)[0]
                    
                    for license_plate in license_plates.boxes.data.tolist():
                        plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                        #print(license_plate, 'track_id: ' + str(bbox[4]))
                        plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                        #cv2.imwrite(str(track_id) + '.jpg', plate)
                        # de-colorize
                        plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
                        #posterize
                        #_, plate_threshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        #cv2.imwrite(str(track_id) + '_gray.jpg', plate_gray)
                        #cv2.imwrite(str(track_id) + '_thresh.jpg', plate_threshold)            
                        
                        #OCR License plate reader
                        np_text,np_score=read_license_plate(plate_gray)
                        print(np_text)
                        print(np_score)
                        
                        if np_text is not None:
                            results[frame_number][track_id]={
                                'car':{
                                    'bbox':[x1,y1,x2,y2],
                                    'bbox_score':score
                                },
                                'license_plate':{
                                    'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                    'bbox_score': plate_score,
                                    'number': np_text,
                                    'text_score': np_score
                                }
                            }
                            
write_csv(results, 'results.csv')            
video.release()