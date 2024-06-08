from ultralytics import YOLO
import cv2
from sort.sort import *
from utils import *

#load two models to detect vehicle and then detect license plates
vehicle_detector =YOLO("yolov8n.pt")
license_plate_detector =YOLO("train4/weights/best.pt")

#stock video to check performance
cap=cv2.VideoCapture('istockphoto-1425716386-640_adpp_is.mp4')

#class indices of vehicle
#car,motorbike,bus and truck
vehicles=[2,3,5,7]

results={}

#read the frames of the video
frame_number= -1
ret=True

vehicle_tracker=Sort()

while ret:
    frame_number+=1
    ret,frame=cap.read() 
    if ret and frame_number<10:
        results[frame_number]={}
        detections=vehicle_detector(frame)[0]
        vehicle_detections=[]
        for detection in detections.boxes.data.tolist():
            #detection shape --> x1,y1,x2,y2,conf_score,class
            x1,y1,x2,y2,conf_score_car,class_index=detection
            if int(class_index) in vehicles:
                vehicle_detections.append([x1,y1,x2,y2,conf_score_car])
                
        #vehicle tracking
        track_vehicles=vehicle_tracker.update(np.asarray(vehicle_detections))
        
        #license plate detection
        license_plates=license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist(): 
            x1,y1,x2,y2,conf_score_license,class_index=license_plate
            
            #assign license plate to car
            x1_car,y1_car,x2_car,y2_car,car_id=get_car(license_plate,track_vehicles)
            
            #crop license plate
            license_plate_crop=frame[int(y1):int(y2),int(x1):int(x2),:]
            
            #process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_threshed = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            
            #read license plate
            license_plate_text,license_plate_text_score=read_license_plate(license_plate_crop_threshed)
            
            if license_plate_text is not None:
                results[frame_number][car_id]={
                    'car':{'bbox':[x1_car,y1_car,x2_car,y2_car]},
                    'license_plate':{'bbox':[x1,y1,x2,y2],
                                     'text':license_plate_text,
                                     'bbox_score':conf_score_license,
                                     'text_score':license_plate_text_score}
                }
            
#update results
write_csv(results,'sample_result.csv')

            