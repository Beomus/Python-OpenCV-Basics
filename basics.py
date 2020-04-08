import cv2
import imutils

# load the image
file = 'smile.jpg'
image = cv2.imread(file)
# this is a tuple so we must use parenthesis
(h, w ,d) = image.shape
print(f'height: {h}, width: {w}, depth: {d}')


# display the image
cv2.imshow(f"{file}", image)
cv2.waitKey(0)

# cropping a random part of the image
cropped_img = image[400:600, 400:600]
cv2.imshow("cropped_img",cropped_img)
cv2.waitKey(0)


# resizing the image, ignoring the aspect ratio
resized_img = cv2.resize(image, (400, 400))
cv2.imshow("resized_img", resized_img)
cv2.waitKey(0)


# resizing while maintaining the aspect ratio
new_dim = (400, int(h * (400 / w)))
new_resized_img = cv2.resize(image, new_dim)
cv2.imshow("new_resized_img", new_resized_img)
cv2.waitKey(0)

# resizing using imutils
resized_imutils = imutils.resize(image, width = 400)
cv2.imshow('resized_imutils', resized_imutils)
cv2.waitKey(0)

cv2.destroyAllWindows()