import cv2
import numpy as np
import imutils

class Shape:
    def __init__(self):
        pass

    def detect(self, c):
        shape = 'unidentified'
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        num_vertices = len(approximation)
        
        if num_vertices == 3:
            shape = "triangle"
        elif num_vertices == 4:
            (x, y, w, h) = cv2.boundingRect(approximation)
            ratio = w / float(h)
            shape = "square" if ratio >= 0.95 and ratio <= 1.05 else "rectangle"
        elif num_vertices == 5:
            shape = "pentagon"
        elif num_vertices >= 10:
            if self.is_star(c, num_vertices):
                shape = "star"
            else:
                shape = "circle"
        else:
            shape = "circle"
        return shape

    def is_star(self, contour, num_vertices):
        if num_vertices % 2 == 0:
            angles = self.calculate_angles(contour)
            if len(angles) > 0 and all(a < 180 for a in angles):
                return True
        return False

    def calculate_angles(self, contour):
        angles = []
        for i in range(len(contour)):
            pt1 = contour[i - 1][0]
            pt2 = contour[i][0]
            pt3 = contour[(i + 1) % len(contour)][0]
            angle = self.angle_between_three_points(pt1, pt2, pt3)
            angles.append(angle)
        return angles

    def angle_between_three_points(self, pt1, pt2, pt3):
        v1 = pt1 - pt2
        v2 = pt3 - pt2
        angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        return np.degrees(angle)

# Path to image
path = "../problems/test_cases/shapes_and_colors.jpg"
# path = "../problems/test_cases/images.png"
# path = "../problems/test_cases/images2.png"

# Load and resize image
image = cv2.imread(path)
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# Convert the resized image to grayscale, blur it slightly, and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Initialize shape detector
sd = Shape()

for c in cnts:
    # Compute the center of the contour in the resized image
    M = cv2.moments(c)
    cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
    cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
    
    # Detect the shape of the contour
    shape = sd.detect(c)
    
    # Adjust the contour coordinates back to the original image size
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    
    # Draw the contours and the name of the shape on the original image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
