### importing required libraries
import torch
import cv2
import time
import os


### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]


            if text_d == 'mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background

                
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'nomask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1) ## for text label background

                
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
            ## print(row[4], type(row[4]),int(row[4]), len(text_d))

    return frame




### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('C:\\Users\\PC-I\\Desktop\\yolov5_deploy-main\\yolov5_deploy-main\\yolov5_deploy\\yolov5-master', 'custom', source ='local', path='last.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format


    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        #while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            #cv2.imshow("img_only", frame)

            #if cv2.waitKey(5) & 0xFF == ord('q'):
                #print(f"[INFO] Exiting. . . ")
        path = 'C:\\Users\\PC-I\\Desktop\\yolov5_deploy-main\\yolov5_deploy-main\\yolov5_deploy\\static\\results'
        justName = img_path.split("/")
        justName.reverse()
        
        cv2.imwrite(os.path.join(path , justName[0]), frame) #here



    elif vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            #codec = cv2.VideoWriter_fourcc(*'MJPG') ##(*'XVID') #mp4v
            out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        #cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret :
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame,classes = classes)
                
                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
            else:
                break
        
        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


#main(vid_path="Los Angeles mask mandate could return as COVID cases rise.mp4",vid_out="facemask_result.mp4") ### for custom video
#main(vid_path=0,vid_out="webcam_facemask_result.mp4") #### for webcam
#main(img_path="zidane.jpg") ## for image
            

