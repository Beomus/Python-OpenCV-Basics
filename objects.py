import imutils
import cv2
import os

save_folder = "modified/"
if not os.path.isdir(save_folder):
	os.makedirs(save_folder)

file = 'random-objects.png'
image = cv2.imread(file)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyWindow("Image")

# convert the image to grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_img", gray_img)
cv2.waitKey(0)
cv2.imwrite(f'{save_folder}/gray_img.png', gray_img)
cv2.destroyWindow("gray_img")

# finding the outline of objects aka edge detections
edge = cv2.Canny(gray_img, 50, 100)
cv2.imshow("Outline", edge)
cv2.waitKey(0)
cv2.imwrite(f'{save_folder}/edge.png', edge)
cv2.destroyWindow("Outline")

"""
Thresholding an image:
- Convert all pixel values less than 225 to 255 (to white)
- Convert all pixel values more than 225 to 0 (to black)
"""
threshold_img = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV)[1]
# this returns a tuple with 2 values, the first is the scale and the second one is the actual image
# print(type(threshold_img))
# print(len(threshold_img))
cv2.imshow("Threshold", threshold_img)
cv2.waitKey(0)
cv2.imwrite(f'{save_folder}/threshold_img.png', threshold_img)
cv2.destroyWindow("Threshold")


# finding the contours (outlines) of the object in the front of the threshold image
contours = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
output = image.copy()

# looping over each contour
for contour in contours:
	# drawing the contour
	cv2.drawContours(output, [contour], -1, (255, 0, 0), 2)
	cv2.imshow("contour", output)
	cv2.waitKey(0)
cv2.imwrite(f'{save_folder}/contours.png', output)
cv2.destroyWindow("contour")

"""
Erosion and dilations are used to reduce noise in the process
of thresholding and image
"""
erosion = threshold_img.copy()
erosion = cv2.erode(erosion, None, iterations = 3)
cv2.imshow("Erosion", erosion)
cv2.waitKey(0)
cv2.imwrite(f'{save_folder}/eroded.png', erosion)
cv2.destroyWindow("Erosion")


dilated = threshold_img.copy()
dilated = cv2.dilate(dilated, None, iterations = 3)
cv2.imshow("Dilation", dilated)
cv2.waitKey(0)
cv2.imwrite(f'{save_folder}/dilated.png', dilated)
cv2.destroyWindow("Dilation")

cv2.destroyAllWindows()