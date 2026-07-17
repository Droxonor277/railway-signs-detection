
# Importing all necessary libraries
import cv2
import os
import sys

print(sys.argv)

# Read the video from specified path
cam = cv2.VideoCapture(str(sys.argv[1]))#load first argv as source
fps = int(sys.argv[2])

try:
    
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')

# frame
currentframe = 0
hr = 0
min = 0
sec = 0
while(True):
    
    # reading from frame
    ret,frame = cam.read()
    #print(ret)
    if ret:
        if(currentframe==fps):
            currentframe=0
            sec+=1
        if(sec == 60):
            sec = 0
            min += 1
            print(hr,":",min)
        if(min == 60):
            min = 0
            hr += 1
        
        # if video is still left continue creating images
        name = '\\data\\frame' +str(hr)+'_'+str(min)+'_'+str(sec)+'_'+ str(currentframe) + '.jpg'
        
        #print ('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()

