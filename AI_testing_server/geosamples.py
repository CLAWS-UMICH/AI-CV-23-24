from ultralytics import YOLO
import cv2 as cv
import numpy as np

# Load a model
model = YOLO('./geosample_models/best.pt')  # Load a pretrained model

# Image filename
image_file = "sam.JPG"

def find_rocks(img):
    # Predict using the model
    results = model.predict(img)

    # Assuming the first result is what we need
    result = results[0]
    
    # Accessing the original image
    img = result.orig_img

    # Filter rocks based on the confidence and size of their bounding box
    rock_boxes = [box for box in result.boxes if result.names[box.cls[0].item()] == 'detect' and box.conf.item() >= 0.95]
    rock_boxes.sort(key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]), reverse=True)
    print(f"Total rocks detected with >= 90% accuracy: {len(rock_boxes)}")

    # Select top three largest rocks with >= 90% accuracy
    top_3_rocks = rock_boxes[:3]

    # Draw rectangle around each of the top three rocks
    for box in top_3_rocks:
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]

        img = cv.rectangle(img, (cords[0], cords[1]), (cords[2], cords[3]), (255, 0, 0), 20)
        confidence = round(100 * box.conf.item(), 2)

        # Set text properties
        text = f"{confidence}%"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # White
        thickness = 2
        cv.putText(img, text, (cords[0], cords[1] - 10), font, font_scale, color, thickness)

    # Save the image with rectangles
    new_filename = f"rock_output.jpg"
    # Save the image with rectangles
    cv.imwrite(new_filename, img)

    if not top_3_rocks:
        rock_com = [0, 0]
    else:        
        cords = top_3_rocks[0].xyxy[0].tolist()        
        cords = [round(x) for x in cords]        
        rock_com = [(cords[0]+cords[2])//2, (cords[1]+cords[3])//2]    
    return [rock_com], "This is a cup"
