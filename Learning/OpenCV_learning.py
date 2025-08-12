import os
import cv2
import numpy as np

"""
As I take notes I am noticing a need for some helper functions so lets write 
those here.

Also reading in the "bird.jpg" img a lot so lets just do it once at the top.
"""

img = cv2.imread(os.path.join('.','data','bird.jpg'))

def show_images(images: dict, scale: float = 1, waitkey: int = 0):
    if scale != 1:
        scaled_images = rescale_images(images.values(), scale)
        images.update(zip(images.keys(), scaled_images))
    
    for win_name, matrix in images.items():
        cv2.imshow(win_name, matrix)

    cv2.waitKey(waitkey)

def rescale_images(images: list, scale: float):
    scaled_images = []
    for image in images:
        scaled_images.append(
            cv2.resize(
                image,
                (int(image.shape[1]/scale**-1), int(image.shape[0]/scale**-1))
            )
        )

    return scaled_images
"""
IO
"""

"""
IMPORTING VISUALIZING AND EXPORTING IMAGES
"""

# # read image
# image_path = os.path.join('.', 'data', 'bird.jpg')

# img = cv2.imread(image_path)

# # write image
# cv2.imwrite(os.path.join('.', 'data', 'bird_out.jpg'), img)

# # visualize image
# cv2.imshow('image', img)

# cv2.waitKey(0) # 0 => wait indefinitely while greater numbers specify milliseconds


"""
IMPORTING AND VISUALIZING VIDEOS
"""

# # read video
# video_path = os.path.join('.', 'data', 'monkey.mp4')

# video = cv2.VideoCapture(video_path)

# # visualize video
# ret = True
# while ret:
#     ret, frame = video.read() 
#     # ret defines if the frame is read successfully. So while we still have
#     # frames to read ret will return True

#     frame_resize = cv2.resize(frame, (800, 800))

#     if ret:
#         cv2.imshow('frame', frame_resize)

#         cv2.waitKey(1) # display each frame for one millisecond?

# video.release()
# cv2.destroyAllWindows()


"""
READING AND VISUALIZING A WEBCAM
"""

# # read webcam
# webcam = cv2.VideoCapture(0) # different integers specify different webcam addresses

# # visualize webcam
# while True:
#     ret, frame = webcam.read()

#     if ret:
#         cv2.imshow('frame', frame)
#         if (
#             cv2.waitKey(1) 
#             & 0xFF == ord('q') # if user presses letter 'q' we break 
#         ): 
#             break

# webcam.release()
# cv2.destroyAllWindows()

"""
BASIC OPERATIONS
"""

"""
IMAGE RESIZING
"""

# img = cv2.imread(os.path.join('.','data','bird.jpg'))
# resized_img = cv2.resize(img, 
#                          (int(img.shape[1]/2), # shape returns (height, width, channels) but resize takes (x,y)
#                           int(img.shape[0]/2))
#                         )

# print(f'Original Shape: {img.shape}\nResized Shape: {resized_img.shape}')

# cv2.imshow('img', img)
# cv2.imshow('resized_img', resized_img)

# cv2.waitKey(0)

"""
CROPPING
"""

# img = cv2.imread(os.path.join('.','data','bird.jpg'))

# # cropping is just selecting intervals of numpy array we want
# print(img.shape)
# cropped_img = img[140:810, 470:750]

# cv2.imshow('img', img)
# cv2.imshow('cropped_img', cropped_img)
# cv2.waitKey(0)

"""
COLORSPACES
"""

# # converting to a different colorspace
# img_rgb = cv2.cvtColor(
#     img, 
#     cv2.COLOR_BGR2RGB # BGR colorspace -> RGB colorspace
#     )
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # Just for fun in understanding
# random_pixel = np.random.randint(0,len(img)+1)
# print(
#     '\033[4mBGR colorspace compared to RGB and GRAY for one (random) pixel:\033[0m\n'
#     f'GBR: {img[0][random_pixel]}\n'
#     f'RGB: {img_rgb[0][random_pixel]}\n'
#     f'GRAY: {img_gray[0][random_pixel]}\n' # Notice how the grayscale conversion reduced data down to one value
#     f'HSV: {img_hsv[0][random_pixel]}'
#     )

# show_images({'img': img,
#              'img_rgb': img_rgb,
#              'img_gray': img_gray,
#              'img_hsv': img_hsv}, 0.6)

"""
BLURRING
"""

# k_size = 15 # the neighborhood of each pixel with which to average for the blur
# img_blur = cv2.blur(img, (k_size, k_size))

# img_gaussian_blur = cv2.GaussianBlur(
#     img,
#     (k_size, k_size),
#     5
# )

# img_median_blur = cv2.medianBlur(img, k_size) # always takes a square

# show_images({
#     'img': img,
#     'img_blur': img_blur,
#     'img_gaussian_blur': img_gaussian_blur,
#     'img_median_blur': img_median_blur
#     }, 0.6)

"""
A POPULAR USE CASE OF BLUR IS TO REMOVE NOISE FROM AN IMAGE.

Lets try and use the above blurring to see what happens to a noisy image.

(This sums up pretty much all of my classwork from my Applied Mathematics course
on image deblurring. Either tech has come a long way or OpenCV is much more 
user friendly than MatLAB)
"""

# img_salt_and_pepper = cv2.imread(os.path.join('.','data','fabio_salt_and_pepper.png'))

# k_size = 5 # the neighborhood of each pixel with which to average for the blur
# img_blur = cv2.blur(img_salt_and_pepper, (k_size, k_size))

# img_gaussian_blur = cv2.GaussianBlur(
#     img_salt_and_pepper,
#     (k_size, k_size),
#     5
# )

# img_median_blur = cv2.medianBlur(img_salt_and_pepper, k_size) # always takes a square

# show_images({
#     'img_salt_and_pepper': img_salt_and_pepper,
#     'img_blur': img_blur,
#     'img_gaussian_blur': img_gaussian_blur,
#     'img_median_blur': img_median_blur
#     })


"""
THRESHOLDING

The Image I am using and the result of this is a bad example but a usecase for
this would be to have the threshold (and blurring) applied in such a way that 
we are clearly identifying an object. In the course I am studying from they used
a brown bear on a green field and the result of the threshold was to (mostly)
capture the bear in 255 and the grass all went to 0.

Why was my picture a bad example though? is it because the grayscale value of
the blue and the green were very similar? perhaps a different colorspace would
have allowed us to see better separation. 
"""

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# _, img_threshold = cv2.threshold(
#     img_gray,
#     115, # the value threshold at which we either go 0 or 255
#     255, # we do have to specify the max value
#     cv2.THRESH_BINARY # have to specify the type of thresholding we are performing
# )

# # can also remove some of the noise from the image
# img_blur = cv2.blur(img_threshold, (7,7))
# _, blurred_threshold = cv2.threshold(img_blur, 115, 255, cv2.THRESH_BINARY)

# # Using an adaptive threshold - I think the idea of this is that it goes chunk by
# # chunk and applies a threshold?
# adaptive_threshold = cv2.adaptiveThreshold(
#     img_gray,
#     255, # still have to specify max value
#     cv2.ADAPTIVE_THRESH_MEAN_C, # have to specify an addaptive threshold method
#     cv2.THRESH_BINARY,
#     25,
#     6
# )

# show_images({
#     'img': img,
#     'img_gray': img_gray,
#     'img_threshold': img_threshold,
#     'blurred_threshold': blurred_threshold,
#     'adaptive_threshold': adaptive_threshold
# }, 0.6)

"""
EDGE DETECTION
"""

# img_edge = cv2.Canny(img, 300, 400)

# # Good time to bring up dilation. Just makes our lines thicker (binary only?)
# img_dilate = cv2.dilate(
#     img_edge, 
#     np.ones((3,3), dtype=np.int8) # numpy array determines thickness. smaller matrix equals thinner
# )

# # erode is the opposite of dilate
# img_erode = cv2.erode(
#     img_dilate, 
#     np.ones((3,3), dtype=np.int8) # numpy array determines thickness. smaller matrix equals thinner
# )

# show_images({
#     'img': img,
#     'img_edge': img_edge,
#     'img_dilate': img_dilate,
#     'img_errode': img_erode
# }, 0.6)

"""
DRAWING

*Picture is a 900x900 image
"""

# # line
# cv2.line(
#     img, # image to draw on
#     (100, 200), # starting x,y coordinate
#     (420, 69), # ending x,y coordinate
#     (255, 0, 0), # BGR color value of the line
#     3, # thickness
# )

# # rectangle
# cv2.rectangle(
#     img, # image to draw on
#     (200, 800), # starting x,y coordinate
#     (40, 690), # ending x,y coordinate
#     (0, 255, 0), # BGR color value of the rectangle
#     -1 # thickness -> specifying -1 causes a shape to fill
# )

# # circle
# cv2.circle(
#     img, # image to draw on
#     (800, 550), # center x,y coordinate
#     200, # radius
#     (0, 0, 255), # BGR color value of the circle
#     -1 # thickness -> specifying -1 causes a shape to fill
# )

# # text
# cv2.putText(
#     img,
#     'CAW!',
#     (700, 600),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     2,
#     (0, 0, 0),
#     4
# )

# cv2.imshow('img', img)
# cv2.waitKey(0)

"""
CONTOURS
"""

# img = cv2.imread(os.path.join('.','data','black and white birds.jpg'))
# print(f'original image shape: {img.shape}')

# _, img_thresh = cv2.threshold(
#     img, 127, 255, 
#     # When working with contours the object to detect needs to be white
#     cv2.THRESH_BINARY_INV
# )
# print(f'threshold image shape: {img_thresh.shape}')

# # Countours need the image to be 8UC1. What this means is uint8 (img.dtype -> uint8) and single channel (img.shape -> (h, w)). As is we will error because we have 
# # a 8UC3 image (img.shape -> (h, w, 3)). Thresholding does not reduce the number of channels so before we take the threshold we need to reduce to 1 channel (grayscale)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(f'gray image shape: {img_gray.shape}')

# _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
# print(f'new threshold shape: {img_thresh.shape}')

# # countours is a list of the different isolated white objects
# contours, heirarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # we can see a lot of noise in the found contours. These are all the small values of countour area which are obviously not large enough to be an object.
# for cnt in contours:
#     if cv2.contourArea(cnt) > 200:

#         # DRAWING CONTOURS

#         # cv2.drawContours(
#         #     img, # image to draw on
#         #     cnt, # the contour(s) we are drawing
#         #     -1, # contourIdx? just make it -1 but unsure what this is
#         #     (0, 255, 0), # color to draw the line in BGR
#         #     2 # thickness of the line
#         # )

#         # FINDING BOUNDING BOX OF OBJECTS

#         x1, y1, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

# show_images({
#     'img': img,
#     'img_thresh': img_thresh, 
# }, 0.4)