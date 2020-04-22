from PIL import Image
import cv2

plate = cv2.imread("plate-trimmed.jpg")
gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
cv2.imwrite("gray.jpg", gray)
cv2.imwrite("thresh.jpg", thresh[1])
