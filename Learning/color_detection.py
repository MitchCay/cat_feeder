import cv2
from util import get_limits
from PIL import Image # Pillow

"""
Note on how this will work:
The HSV color space is defined by a Hue, Saturation and Value (See HSV_Color_Cylinder.png for reference).
By defining a range of values for HUE we can detect the desired color, in this case yellow. 
"""

yellow = [0, 255, 255] # yellow in BGR colorspace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=yellow)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # MY IMPLEMENTATION
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for cnt in contours:
    #     if cv2.contourArea(cnt) > 200:
    #         x1, y1, w, h = cv2.boundingRect(cnt)
    #         cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

    # TUTORIAL IMPLEMENTATION
    mask_ = Image.fromarray(mask) # converting from OpenCV representation to Pillow representation
    
    bbox = mask_.getbbox() # gets bounding box

    # bbox is either None or (x1, y1, x2, y2)
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()