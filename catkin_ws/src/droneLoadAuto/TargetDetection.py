import cv2
import numpy as np
import time
from pytessy import pytessy
from PIL import Image
from sklearn.decomposition import PCA
import pickle
import collections
import math

class TargetDetection: #class for detection of specified area

    target_size = (160, 90)
    min_area = 1500

    def get_polygone_points(self, compute_img, nb_polygone_min, nb_polygone_max, order=False, epsilon=0.03): #returns the outlines of the object captured by the camera 
    #compute_img : frame captured by camera
    #nb_polygone_min, nb_polygone_max : defines the range of polygones to implement
    #epsilon : coefficient used to reduce the number of points of the outlines
        contours, h = cv2.findContours(cv2.cvtColor(compute_img, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_areas = [0]
        rects = {}
        ret = None
        for c in contours:
            area = cv2.contourArea(c) #stores the number of points which composed the outlines
            if area > self.min_area:
                p = cv2.arcLength(c, True)
                polygone = cv2.approxPolyDP(c, epsilon * p, True) #creates a polygone with 0.03*p number of points of the outlines
                if len(polygone) >= nb_polygone_min and len(polygone) <= nb_polygone_max and cv2.isContourConvex(polygone):#area > np.min(max_areas)
                    if max_areas[0] == 0:
                        max_areas = []
                    rects[int(area)] = np.array(list(map(lambda e: e[0], polygone)), dtype=np.float32)
                    max_areas = np.sort(np.append(max_areas, area)) #adds the value of corresponding area in max_areas tab and sorts it
                    if len(max_areas) > 5:
                        max_areas = max_areas[1:] #deletes the first element of max_areas if it has more than 5 elements 
        print(len(rects))
        if len(rects) > 0:
            ret = []
            if order:
                max_areas = sorted(max_areas)
            for max_area in max_areas:
                ret.append(rects[int(max_area)])
                #cv2.drawContours(compute_img, [np.array([[e] for e in rects[int(max_area)]], dtype=int)], 0, (255, 0, 0), 2)
        #cv2.imshow("get_polygone_points", compute_img)
        return ret

    def compute_img_for_detection(self, compute_img, blur, erode, dilate, mask_low, mask_high):
        #compute_img += 1
        if blur:
            compute_img = cv2.GaussianBlur(compute_img, (blur, blur), 0) #filters the computed image by convolving each point with a Gaussian Kernal
        mask = cv2.inRange(compute_img, mask_low, mask_high) #defines the range of colors to better detect the corresponding object
        compute_img = cv2.bitwise_and(compute_img, compute_img, mask=mask) #compars the computed image and itself filtered
        compute_img = cv2.cvtColor(compute_img, cv2.COLOR_HSV2BGR) #converts the computed image from HSV (seen by camera) to BGR (processed by algo)
        compute_img = cv2.erode(compute_img, None, iterations=erode) #erodes away the boundaries of foreground object to diminish the features of an image.
        compute_img = cv2.dilate(compute_img, None, iterations=dilate) #increases the white region in the image to accentuate features
        return compute_img

    def getCenter(self, rect): #rect : order of the points corresponding to the outlines of the object
        return (((rect[1] - rect[3]) / 2) + rect[3]).astype(int)

    def getZoomedRect(self, rect):
        center = self.getCenter(rect)
        return np.array(((rect-center)*self.quick_scan_frame_coeff)+center, dtype=int)

    def cutFromContour(self, img, rect, size):
        target = np.float32([[0, 0], [0, size[1]], [size[0], size[1]], [size[0], 0]]) #constructs the set of destination points
        transform_matrix = cv2.getPerspectiveTransform(np.float32(rect), target)
        return cv2.warpPerspective(img, transform_matrix, size)

    def getLevelContour(self, rect):
        maxi = np.max(rect, 0)
        mini = np.min(rect, 0)
        self.target_size = (int(maxi[0]-mini[0]), int(maxi[1]-mini[1]))
        return np.array([[mini[0], mini[1]], [mini[0], maxi[1]], [maxi[0], maxi[1]], [maxi[0], mini[1]]])

    def extract_target(self, img, rect):
        if cv2.norm(rect[0] - rect[3]) < cv2.norm(rect[0] - rect[1]):
            rect = np.roll(rect, 2)
        return self.cutFromContour(img, rect, (160, 90))

class TaxiDetection(TargetDetection): #class for detection of panel with text "TAXI"

    last_text_scan = 0
    time_between_text_scan = 3
    quick_scan_frame_coeff = 1.4
    counter_fail_detection = 0
    counter_fail_detection_limit = 3
    last_seen_area = None
    target_color = 90
    modelPCA = pickle.load(open("modelPCA", "rb"))
    ii = 0

    def extract_target(self, img, rect):
        if cv2.norm(rect[0] - rect[3]) < cv2.norm(rect[0] - rect[1]):
            rect = np.roll(rect, 2)
        return self.cutFromContour(img, rect, (160, 90))

    def ocrWith180Rot(self, target, ocr):
        if ocr(target, "TAXI"):
            self.last_text_scan = time.time()
            return True
        target = np.rot90(target, 2)
        if ocr(target, "TAXI"):
            self.last_text_scan = time.time()
            return True
        return False

    def ocr(self, target, word):
        h, s, v = cv2.split(target)
        ret, target = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)
        similar_character = ['!', '/', '\\', '|', '[', ']']
        detect_word = pytessy.PyTessy().read(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)).tobytes(), 160, 90, 3)
        score = 0
        size = 4
        if detect_word is None: detect_word = ""
        if len(detect_word) <= size-1: size = len(detect_word)
        detect_word = list(detect_word.upper())
        for i in range(size):
            if detect_word[i] in similar_character:
                detect_word[i] = 'I'
            if word[i] == detect_word[i]:
                score += 1
        return score > 2

    def PCA_ocr(self, img, word):
        #cv2.imshow("target_ocr_taxi", img)
        h, s, v = cv2.split(img)
        img = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 19)

        #cv2.imshow("target_ocr_taxi_bin", img)
        img = (img/255.0) - self.modelPCA["avg"]
        search_pca = self.modelPCA["PCA"].transform([np.reshape(cv2.resize(img, (160, 90)), (-1, 14400))[0]])[0]/np.max(self.modelPCA["eigen"])
        i = 0
        for vector in self.modelPCA["vectors"]:
            dist = np.linalg.norm(search_pca - vector)
            if dist < 9.75:
                i += 1
        if i > 3:
            return True
        return False

    def get_target(self, img): #returns the positions of the corresponding object if detected in the image 
        computed_img = self.compute_img_for_detection(img, 5, 5, 5, np.array([self.target_color-10, 60, 60]), np.array([self.target_color+10, 255, 255]))

        if self.last_seen_area is not None:
            computed_img = self.cutFromContour(computed_img, self.last_seen_area, tuple(np.array(self.quick_scan_frame_coeff*np.array(self.target_size), dtype=int))) #defines the area to find the object around target size
        rect_list = self.get_polygone_points(computed_img, 4, 4)
        if rect_list is not None:
            for rect in rect_list:
                if self.last_seen_area is not None:
                    rect = rect + self.last_seen_area[0]
                target = self.extract_target(img, rect)
                text = "TAXI" #returns the name of object
                if self.last_seen_area is None:
                    if self.ocrWith180Rot(target, self.PCA_ocr) is False:
                        continue

                self.last_seen_area = self.getZoomedRect(self.getLevelContour(rect))
                self.counter_fail_detection = 0
                return target, self.getCenter(rect), text
        if self.counter_fail_detection > self.counter_fail_detection_limit:
            self.last_seen_area = None
        self.counter_fail_detection += 1
        return None, (0, 0)

class JeuxDetection(TargetDetection): #class for detection of panel with text "JEUX"

    last_text_scan = 0
    time_between_text_scan = 3
    quick_scan_frame_coeff = 1.6
    counter_fail_detection = 0
    counter_fail_detection_limit = 3
    last_seen_area = None
    target_color = 90
    modelPCA = pickle.load(open("modelPCA_Jeux", "rb"))
    ii = 0

    def compute_img_for_detection(self, compute_img, blur):
        mask = cv2.dilate(cv2.inRange(compute_img, np.array([0, 0, 0]), np.array([255, 255, 200])), None, iterations=5)
        compute_img = cv2.GaussianBlur(compute_img, (blur, blur), 0)
        compute_img = cv2.Canny(compute_img, 50, 100) #detects the edges of the image
        cv2.imshow("JeuxDetection_cannny", compute_img)
        #compute_img = cv2.adaptiveThreshold(cv2.cvtColor(cv2.cvtColor(compute_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 30)
        compute_img = cv2.bitwise_and(compute_img, compute_img, mask=mask)
        compute_img = cv2.dilate(compute_img, np.ones((2, 2), np.uint8), iterations=3)
        cv2.imshow("JeuxDetection", compute_img)
        return cv2.cvtColor(compute_img, cv2.COLOR_GRAY2BGR)

    def ocrWith180Rot(self, target, ocr):
        if ocr(target, "JEUX"):
            self.last_text_scan = time.time()
            return True
        target = np.rot90(target, 2)
        if ocr(target, "JEUX"):
            self.last_text_scan = time.time()
            return True
        return False

    def ocr(self, target, word):
        h, s, v = cv2.split(target)
        ret, target = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)
        similar_character = ['!', '/', '\\', '|', '[', ']']
        detect_word = pytessy.PyTessy().read(Image.fromarray(cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)).tobytes(), 160, 90, 3)
        score = 0
        size = 4
        if detect_word is None: detect_word = ""
        if len(detect_word) <= size-1: size = len(detect_word)
        detect_word = list(detect_word.upper())
        for i in range(size):
            if detect_word[i] in similar_character:
                detect_word[i] = 'I'
            if word[i] == detect_word[i]:
                score += 1
        return score > 2

    def PCA_ocr(self, img, word):
        h, s, v = cv2.split(img)
        img = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 19)
        img = (img/255.0) - self.modelPCA["avg"]
        search_pca = self.modelPCA["PCA"].transform([np.reshape(cv2.resize(img, (160, 90)), (-1, 14400))[0]])[0]/np.max(self.modelPCA["eigen"])
        i = 0
        for vector in self.modelPCA["vectors"]:
            dist = np.linalg.norm(search_pca - vector)
            if dist < 11.5:
                i += 1
        if i > 3:
            return True
        return False

    def get_target(self, img):
        computed_img = self.compute_img_for_detection(img, 3)

        if self.last_seen_area is not None:
            computed_img = self.cutFromContour(computed_img, self.last_seen_area, tuple(np.array(self.quick_scan_frame_coeff*np.array(self.target_size), dtype=int)))
        rect_list = self.get_polygone_points(computed_img, 4, 4)
        cv2.imshow("jeux", computed_img)
        if rect_list is not None:
            for rect in rect_list:
                if self.last_seen_area is not None:
                    rect = rect + self.last_seen_area[0]
                target = self.extract_target(img, rect)
                text = "JEUX"
                if self.last_seen_area is None:
                    if self.ocrWith180Rot(target, self.PCA_ocr) is False:
                        continue

                self.last_seen_area = self.getZoomedRect(self.getLevelContour(rect))
                self.counter_fail_detection = 0
                return target, self.getCenter(rect), text
        if self.counter_fail_detection > self.counter_fail_detection_limit:
            self.last_seen_area = None
        self.counter_fail_detection += 1
        return None, (0, 0)

#class WindowDetection(TargetDetection):

    #quick_scan_frame_coeff = 2.5
    #counter_fail_detection = 0
    #counter_fail_detection_limit = 3
    #last_seen_area = None
    #min_area = 15000

    #def get_target(self, img):
        #img = cv2.bitwise_not(img)
        #img -= 1
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #cv2.imshow("window_", cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
        #computed_img = self.compute_img_for_detection(img, 0, 2, 27, np.array([0, 0, 240]), np.array([180, 15, 255]))
        #if self.last_seen_area is not None:
            #computed_img = self.cutFromContour(computed_img, self.last_seen_area, tuple(np.array(self.quick_scan_frame_coeff * np.array(self.target_size), dtype=int)))
        #rect_list = self.get_polygone_points(computed_img, 4, 4, True, 0.05)
        #cv2.imshow("window", computed_img)
        #if rect_list is not None:
            #for rect in rect_list:
                #if self.check_square(rect) > 40: continue

                #if self.last_seen_area is not None:
                    #rect = rect + self.last_seen_area[0]
                #target = self.extract_target(img, rect)

                #self.last_seen_area = self.getZoomedRect(self.getLevelContour(rect))
                #self.counter_fail_detection = 0
                #return target, self.getCenter(rect)
        #if self.counter_fail_detection > self.counter_fail_detection_limit:
            #self.last_seen_area = None
        #self.counter_fail_detection += 1
        #return None, (0, 0)

    #def check_square(self, rect):
        #side_len = np.zeros(4)
        #for i in range(4):
            #side_len[i] = math.sqrt(pow(rect[i-1][0]-rect[i][0], 2) + pow(rect[i-1][1]-rect[i][1], 2))
        #return np.std(side_len)

class StadeDetection(TargetDetection): #class for detection of panel with text "STADE"

    quick_scan_frame_coeff = 2
    last_seen_area = None
    target_color_red = 115
    target_color_green = 65
    min_area = 1000

    def get_target(self, img):
        computed_img = self.compute_img_for_detection(img, 5, 6, 6, np.array([self.target_color_green-30, 21, 23]), np.array([self.target_color_green+30, 145, 225]))
        sphere = self.get_polygone_points(computed_img, 3, 8)
        if sphere is not None:
            sphere = sphere[0]
            sphere = self.getZoomedRect(self.getLevelContour(sphere))
            cut_img = self.cutFromContour(img, sphere, (320, 180))
            computed_img = self.compute_img_for_detection(cut_img, 5, 1, 3, np.array([self.target_color_red-15, 35, 85]), np.array([self.target_color_red+15, 235, 255]))
            text = "STADE"
            red_sphere = self.get_polygone_points(computed_img, 6, 11)
            if red_sphere is not None:
                return computed_img, self.getCenter(sphere), text
        return None, (0, 0)



cap = cv2.VideoCapture('testDetectionPanels.mp4') #stores the video feed and sends it into an outpout file named 'testDetectionPanels.mp4'
#cap = cv2.VideoCapture('testDetectionWindow-2.mp4')
#cap = cv2.VideoCapture('testDetectionMarqueur.avi')

hasFrames, img_from_cam = cap.read()
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (img_from_cam.shape[1], img_from_cam.shape[0]))


taxiDetection = TaxiDetection()
stadeDetection = StadeDetection()
jeuxDetection = JeuxDetection()
#windowDetection = WindowDetection()
i = 10
j = 0.5
while True:
    start_time = time.time()
    hasFrames, img_from_cam = cap.read()
    if not hasFrames:
        break
    #img_from_cam = cap.read()[1]
    t, c, s = stadeDetection.get_target(cv2.cvtColor(img_from_cam, cv2.COLOR_RGB2HSV))
    t2, c2, s2 = taxiDetection.get_target(cv2.cvtColor(img_from_cam, cv2.COLOR_RGB2HSV))
    t3, c3, s3 = jeuxDetection.get_target(cv2.cvtColor(img_from_cam, cv2.COLOR_RGB2HSV))
    #t4, c4, s4 = windowDetection.get_target(img_from_cam)

    if i == 10:
        fps = str(round((1/j)*10, 1)) #stores the number of frames received per second 
        i = 0
        j = 0
    i += 1
    cv2.putText(img_from_cam, fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1) #puts corresponding text based on object captured
    cv2.putText(img_from_cam, s, (c[0]-9, c[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
    cv2.putText(img_from_cam, s2, (c2[0]-9, c2[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
    cv2.putText(img_from_cam, s3, (c3[0]-9, c3[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    #cv2.putText(img_from_cam, "x-WINDOW", (c4[0]-9, c4[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4) 

    cv2.imshow("VIDEO_FEED", img_from_cam) #displays a window with name "VIDEO_FEED" and content of img_from_cam

    key = cv2.waitKey(30)
    j = j+time.time()-start_time
    if key == ord('a'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
