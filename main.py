from ultralytics import YOLO
import cv2
from sort.sort import *

#load two models to detect vehicle and then detect license plates
vehicle_detector =YOLO("yolov8n.pt")
license_plate_detector =YOLO("train4/weights/best.pt")

#stock video to check performance
cap=cv2.VideoCapture('istockphoto-1425716386-640_adpp_is.mp4')

#class indices of vehicle
#car,motorbike,bus and truck
vehicles=[2,3,5,7]

#read the frames of the video
frame_number= -1
ret=True

vehicle_tracker=Sort()

while ret:
    frame_number+=1
    ret,frame=cap.read() 
    if ret and frame_number<10:
        detections=vehicle_detector(frame)[0]
        vehicle_detections=[]
        for detection in detections.boxes.data.tolist():
            #detection shape --> x1,y1,x2,y2,conf_score,class
            x1,y1,x2,y2,conf_score,class_index=detection
            if int(class_index) in vehicles:
                vehicle_detections.append([x1,y1,x2,y2,conf_score])
                
        #vehicle tracking
        track_vehicles=vehicle_tracker.update(np.asarray(vehicle_detections))
        
        #license plate detection
        license_plates=license_plate_detector(frame)[0]
        for license_plate in license_plates.data.tolist(): 
            x1,y1,x2,y2,conf_score,class_index=license_plate
            
            #assign license plate to car
            
            #crop license plate
            
            #process license plate
            
            #read license plate
            
            #update results
                
            