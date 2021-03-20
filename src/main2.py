import sys
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet
from pathlib import Path
################################################################################
def main():
    darknet_ros = DarknetROS()

    input = "/Users/timothy.hudson/Documents/Projects/RoadMap/HighwayEx.mp4"
    cap = cv2.VideoCapture(input)
    #cap = cv2.VideoCapture(0) # Webcam

    while True:
    	# read the next frame from the file
        (ret, frame) = cap.read()

    	# if the frame was not grabbed, then we have reached the end
    	# of the stream
        if ret is False:
            break

        #darknet_ros.process_frame(frame)

        window_name = "Darknet"
        #cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
        #cv2.moveWindow(window_name, 0,0)
        #cv2.resizeWindow(window_name, 640, 480)
        #cv2.resizeWindow(window_name, 1280,1024)
        #cv2.resizeWindow(window_name, 1280,800)
        cv2.imshow(window_name, frame)

        '''# construct a blob from the input frame and then perform a forward
    	# pass of the YOLO object detector, giving us our bounding boxes
    	# and associated probabilities
    	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    		swapRB=True, crop=False)
    	net.setInput(blob)
    	start = time.time()
    	layerOutputs = net.forward(ln)
    	end = time.time()

    	# initialize our lists of detected bounding boxes, confidences,
    	# and class IDs, respectively
    	boxes = []
    	confidences = []
    	classIDs = []

    	# loop over each of the layer outputs
    	for output in layerOutputs:
    		# loop over each of the detections
    		for detection in output:
    			# extract the class ID and confidence (i.e., probability)
    			# of the current object detection
    			scores = detection[5:]
    			classID = np.argmax(scores)
    			confidence = scores[classID]

    			# filter out weak predictions by ensuring the detected
    			# probability is greater than the minimum probability
    			if confidence > CONFIDENCE:
    				# scale the bounding box coordinates back relative to
    				# the size of the image, keeping in mind that YOLO
    				# actually returns the center (x, y)-coordinates of
    				# the bounding box followed by the boxes' width and
    				# height
    				box = detection[0:4] * np.array([W, H, W, H])
    				(centerX, centerY, width, height) = box.astype("int")

    				# use the center (x, y)-coordinates to derive the top
    				# and and left corner of the bounding box
    				x = int(centerX - (width / 2))
    				y = int(centerY - (height / 2))

    				# update our list of bounding box coordinates,
    				# confidences, and class IDs
    				boxes.append([x, y, int(width), int(height)])
    				confidences.append(float(confidence))
    				classIDs.append(classID)

    	# apply non-maxima suppression to suppress weak, overlapping
    	# bounding boxes
    	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
    		THRESHOLD)

    	# ensure at least one detection exists
    	if len(idxs) > 0:
    		# loop over the indexes we are keeping
    		for i in idxs.flatten():
    			# extract the bounding box coordinates
    			(x, y) = (boxes[i][0], boxes[i][1])
    			(w, h) = (boxes[i][2], boxes[i][3])

    			# draw a bounding box rectangle and label on the frame
    			color = [int(c) for c in COLORS[classIDs[i]]]
    			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
    				confidences[i])
    			cv2.putText(frame, text, (x, y - 5),
    				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    	# check if the video writer is None
    	if writer is None:
    		# initialize our video writer
    		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    		writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
    			(frame.shape[1], frame.shape[0]), True)



    	# write the output frame to disk
    	writer.write(frame)
    	# show the output frame
    	cv2.imshow("Frame", cv2.resize(frame, (800, 600)))
    	key = cv2.waitKey(1) & 0xFF
    	#print ("key", key)
    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break

    	# update the FPS counter
    	fps.update()

    # stop the timer and display FPS information
    fps.stop() '''

    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if darknet_ros.videoFlag and darknet_ros.initFlag:
        darknet_ros.video.release()
    cv2.destroyAllWindows()

class DarknetROS:
    def __init__(self):

        #self. = str(Path(__file__).parent)+    # Change to relative path file?
        #self.weights = "/Users/timothy.hudson/Documents/Projects/RoadMap/cfg/yolov4.weights" #yolo weights path
        self.weights = "/Users/timothy.hudson/Documents/Projects/RoadMap/cfg/yolov4-tiny.weights" #yolo weights path

        #self.config_file = "/darknet/cfg/yolov4.cfg" #path to config file
        #self.config_file = "/Users/timothy.hudson/Documents/Projects/yolov4/cfg/yolov4.cfg" #path to config file
        self.config_file = "/Users/timothy.hudson/Documents/Projects/yolov4/cfg/yolov4-tiny.cfg" #path to config file

        self.data_file = "/Users/timothy.hudson/Documents/Projects/RoadMap/cfg/coco.data" #path to data file
        #self.data_file = str(Path(__file__).parent.parent)+'/cfg/coco.data'    # Change to relative path file?

        self.network, self.class_names, self.class_colours = darknet.load_network(self.config_file,self.data_file,self.weights)

        self.class_colours = {"person": (255, 255, 0), "car": (0, 0, 255)}  # Set bounding box colours of person (yellow) and cell phone (blue)
        self.allowed_classes = ["person", "car"]     # Selected object classes to display

        self.initFlag = False

        self.threshold = 0.2 # Confidence threshold (default 0.2)
        self.displayFlag = True  # Display out during processing
        self.videoFlag = False      # Save output as video
        self.output_file = '/Users/timothy.hudson/Documents/Projects/RoadMap/output.avi'

        self.frame_id = 1

    def process_frame(self, frame):

        if self.initFlag is False:
            self.height, self.width, self.channels = frame.shape
            print('INITIALISED')
            print('VIDEO PROPERTIES:')
            print('     WIDTH:   ', self.width)
            print('     HEIGHT:  ', self.height)
            print('     CHANNELS:', self.channels)
            if self.videoFlag:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video = cv2.VideoWriter(self.output_file, fourcc, 30, (self.width, self.height))
            self.initFlag = True

        #print("STAMP: "+str(image.header.stamp.secs)+"."+str(image.header.stamp.nsecs))
        #print("SEQ:",image.header.seq)
        print("FRAME:",self.frame_id)
        self.frame_id += 1
        #print("TIME:",rospy.Time.now())

        # Process Image
        prev_time = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = self.image_detection(frame)


        # Iterate through each detection and create BoundingBox message
        for class_name, confidence, bbox in detections:

            xmin, ymax, xmax, ymin = darknet.bbox2points(bbox)
            confidence = confidence
            class_name = class_name
            id = -1

            if self.displayFlag:
                colour = self.class_colours[class_name]
                left, top, right, bottom = darknet.bbox2points(bbox)
                cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
                cv2.putText(frame, str(class_name)+" ["+str(round(float(confidence),1))+'%]',(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,colour, 2)


        print("DETECTION COUNT:", len(detections))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = round((1/(time.time() - prev_time)),1)
        #fps = int(1/(time.time() - self.prevTime))
        #self.prevTime = time.time()
        print("FPS:",fps)
        if self.displayFlag:
            window_name = "Darknet"
            cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
            #cv2.moveWindow(window_name, 0,0)
            cv2.resizeWindow(window_name, 640, 480)
            #cv2.resizeWindow(window_name, 1280,1024)
            #cv2.resizeWindow(window_name, 1280,800)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        print("test")

        if self.videoFlag:
            self.video.write(frame);

    def image_detection(self, image):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        width = darknet.network_width(self.network)
        height = darknet.network_height(self.network)
        darknet_image = darknet.make_image(width, height, 3)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.threshold)
        detections = self.filter_classes(detections)       # Filters out unwanted classes
        detections = self.convert2relative(image_resized, detections)
        return detections

    # Function to remove unwanted class detections
    def filter_classes(self, classes):
        filtered = []
        for obj in classes:
            if obj[0] in self.allowed_classes:
                filtered.append(obj)
        return filtered

    def convert2relative(self, image, detections):
        #YOLO format use relative coordinates for annotation
        converted_detections = []
        for class_name, confidence, bbox in detections:
            x, y, w, h = bbox
            width, height, _ = image.shape
            xp = x/width*self.width
            yp = y/height*self.height
            wp = w/width*self.width
            hp = h/height*self.height
            bboxp = xp, yp, wp, hp
            converted_detections.append((class_name, confidence, bboxp))
        return converted_detections

if __name__ == "__main__":
    main()
