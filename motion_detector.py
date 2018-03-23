import cv2, pandas
from datetime import datetime

first_frame=None #inititak frame to setup as environment
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

#Select the recording device, here 0 means webcam
video=cv2.VideoCapture(0)

while True:
    check, frame = video.read() #start reading the video frames
    status=0
    #Calculate the gray version of the frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Apply gaussian blurr to improve accracy
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue
    #set a delta frame that is first frame for reference
    delta_frame=cv2.absdiff(first_frame,gray)
    #set a threshold frame
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
    #Find a contour and find area that cnotour represents
    (_,cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1
#put a rectangle around detected objects
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

#show frames
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)
#wait for input to termainate the frames here its 'q'
    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)
#release camera and close the frame windows
video.release()
cv2.destroyAllWindows()
#write down the data in dataframe
for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)
#save data to csv file
df.to_csv("S:\Git\Webcam motion det\Times.csv")




#Visualiztion
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource

df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")


cds=ColumnDataSource(df)
#Graph of start and end of activity time
p=figure(x_axis_type='datetime',height=100, width=500, responsive=True,title="Motion Graph")
p.yaxis.minor_tick_line_color=None
p.ygrid[0].ticker.desired_num_ticks=1
#add hovertools
hover=HoverTool(tooltips=[("Start","@Start_string"),("End","@End_string")])
p.add_tools(hover)

q=p.quad(left="Start",right="End",bottom=0,top=1,color="green",source=cds)
#display the activity graph in we browser
output_file("Graph1.html")
show(p)
