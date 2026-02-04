import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file - ONLY the highest scoring frame per car_id.
    Analyzes all frames but outputs only the best detection for each vehicle.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    # First, collect all detections per car_id and find the highest scoring one
    best_detections = {}  # {car_id: {'frame_nmr': x, 'data': {...}, 'combined_score': y}}
    
    total_frames_analyzed = 0
    total_detections = 0
    
    for frame_nmr in results.keys():
        total_frames_analyzed += 1
        for car_id in results[frame_nmr].keys():
            if 'car' in results[frame_nmr][car_id].keys() and \
               'license_plate' in results[frame_nmr][car_id].keys() and \
               'text' in results[frame_nmr][car_id]['license_plate'].keys():
                
                total_detections += 1
                
                # Calculate combined score (average of bbox_score and text_score)
                bbox_score = results[frame_nmr][car_id]['license_plate']['bbox_score']
                text_score = results[frame_nmr][car_id]['license_plate']['text_score']
                combined_score = (bbox_score + text_score) / 2
                
                # Check if this is a better detection for this car_id
                if car_id not in best_detections or combined_score > best_detections[car_id]['combined_score']:
                    best_detections[car_id] = {
                        'frame_nmr': frame_nmr,
                        'data': results[frame_nmr][car_id],
                        'combined_score': combined_score
                    }
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames analyzed: {total_frames_analyzed}")
    print(f"Total license plate detections: {total_detections}")
    print(f"Unique vehicles detected: {len(best_detections)}")
    print(f"{'='*60}\n")
    
    # Try to open the file, if permission denied, use alternative filename
    try:
        f = open(output_path, 'w')
    except PermissionError:
        # File is locked, try alternative filename
        import os
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_new{ext}"
        print(f"Original file is locked. Saving to: {output_path}")
        f = open(output_path, 'w')
    
    with f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        # Write only the best detection for each car_id
        for car_id in sorted(best_detections.keys()):
            detection = best_detections[car_id]
            frame_nmr = detection['frame_nmr']
            data = detection['data']
            
            print(f"Car ID {car_id}: Best frame #{frame_nmr} | "
                  f"License: {data['license_plate']['text']} | "
                  f"BBox Score: {data['license_plate']['bbox_score']:.3f} | "
                  f"Text Score: {data['license_plate']['text_score']:.3f} | "
                  f"Combined: {detection['combined_score']:.3f}")
            
            f.write('{},{},{},{},{},{},{}\n'.format(
                frame_nmr,
                car_id,
                '[{} {} {} {}]'.format(
                    data['car']['bbox'][0],
                    data['car']['bbox'][1],
                    data['car']['bbox'][2],
                    data['car']['bbox'][3]),
                '[{} {} {} {}]'.format(
                    data['license_plate']['bbox'][0],
                    data['license_plate']['bbox'][1],
                    data['license_plate']['bbox'][2],
                    data['license_plate']['bbox'][3]),
                data['license_plate']['bbox_score'],
                data['license_plate']['text'],
                data['license_plate']['text_score'])
            )
        f.close()
    
    print(f"\nResults saved to: {output_path}")
    print(f"Only {len(best_detections)} best detections written (1 per vehicle)")


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
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
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # First try strict format validation
        if license_complies_format(text):
            return format_license(text), score
    
    # If no strict format match, return the best detection with alphanumeric characters only
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        # Keep only alphanumeric characters
        text = ''.join(c for c in text if c.isalnum())
        if len(text) >= 4 and score > 0.3:  # At least 4 characters and reasonable confidence
            return text, score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
