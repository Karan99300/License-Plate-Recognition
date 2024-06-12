import easyocr
import re

reader=easyocr.Reader(["en"],gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'S': '5',
    'B': '8',
    'G': '6',
    'Z': '2'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '5': 'S',
    '8': 'B',
    '6': 'G',
    '2': 'Z'
}

def license_plate_format(text):
    """
    Check if license plate is in accordance with indian license plate format

    Args:
        text (str): License plate text

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    
    if re.match(pattern, text):
        return True
    else:
        return False
    
def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    formatted_plate = ''
    for idx, char in enumerate(text):
        if idx == 2 or idx == 3 or idx == 6 or idx == 7:  # Digits
            if char in dict_char_to_int:
                formatted_plate += dict_char_to_int[char]
            else:
                formatted_plate += char
        else:  # Letters
            if char in dict_int_to_char:
                formatted_plate += dict_int_to_char[char]
            else:
                formatted_plate += char

    return formatted_plate


def write_csv(results,output_path):
    with open(output_path,'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        
        for frame_number in results.keys():
            for car_id in results[frame_number].keys():
                print(results[frame_number][car_id])
                if 'car' in results[frame_number][car_id].keys() and \
                'license_plate' in results[frame_number][car_id].keys() and \
                'text' in results[frame_number][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_number,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][car_id]['car']['bbox'][0],
                                                                results[frame_number][car_id]['car']['bbox'][1],
                                                                results[frame_number][car_id]['car']['bbox'][2],
                                                                results[frame_number][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][car_id]['license_plate']['bbox'][0],
                                                                results[frame_number][car_id]['license_plate']['bbox'][1],
                                                                results[frame_number][car_id]['license_plate']['bbox'][2],
                                                                results[frame_number][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_number][car_id]['license_plate']['bbox_score'],
                                                            results[frame_number][car_id]['license_plate']['text'],
                                                            results[frame_number][car_id]['license_plate']['text_score'])
                            )
        f.close()


def get_car(license_plate,tracked_vehicles):
    """
    Get the vehicle coordinates and ID based on license plate coordinates

    Args:
        license_plate(tuple) : Tuple of the format (x1,y1,x2,y2,score,class_index)
        vehicle_tracker(list) : List of vehicle ids and coordinates

    Returns:
        tuple : return vehicle coordinates (x1_car,y1_car,x2_car,y2_car) and ID number
    """
    x1,y1,x2,y2,score,class_index =license_plate
    
    found_license=False
    for j in range(len(tracked_vehicles)):
        x1_car,y1_car,x2_car,y2_car,car_id=tracked_vehicles[j]
        
        if x1>x1_car and y1>y1_car and x2<x2_car and y2_car:
            car_index=j
            found_license=True
            break
        
    if found_license:
        return tracked_vehicles[car_index]            
    
    return -1,-1,-1,-1,-1

def read_license_plate(license_plate_crop_threshed):
    """
    Read the license plate

    Args:
        license_plate_crop_threshed (Image): Image of license plate

    Returns:
        tuple: returns license plate text and confidence score
    """
    detections=reader.readtext(license_plate_crop_threshed)
    
    for detection in detections:
        bbox,text,score=detection
        
        text=text.upper().replace(' ','')
        
        if license_plate_format(text):
            return format_license(text),score
        
    return None,None