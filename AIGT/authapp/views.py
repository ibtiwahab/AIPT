import datetime
from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from authapp.models import Contact,Trainer,Enrollment,Gallery,Attendance,Blog
from django.shortcuts import render
import cv2
import mediapipe as md
from django.http import StreamingHttpResponse
from django.template import Context, Template
from django.shortcuts import render
from .form import BlogForm
from django.contrib.auth.decorators import login_required
from .models import BicepsHistory, Blog, PushupHistory, ShoulderpressHistory, SquatHistory, TricepsHistory
from .form import ExerciseForm
from .models import Attendance
import random
# Create your views here.
def Home(request):
    return render(request,"index.html")

def gallery(request):
    posts=Gallery.objects.all()
    context={"posts":posts}
    return render(request,"gallery.html",context)


def attendance(request):
    if not request.user.is_authenticated:
        messages.warning(request,"Please Login and Try Again")
        return redirect('/login')
    SelectTrainer=Trainer.objects.all()
    context={"SelectTrainer":SelectTrainer}
    if request.method=="POST":
        phonenumber=request.POST.get('PhoneNumber')
        Login=request.POST.get('logintime')
        Logout=request.POST.get('loginout')
        SelectWorkout=request.POST.get('workout')
        
        query=Attendance(phonenumber=phonenumber,Login=Login,Logout=Logout,SelectWorkout=SelectWorkout)
        query.save()
        messages.warning(request,"Attendace Applied Success")
        return redirect('/attendance')
    return render(request,"attendance.html",context)

def profile(request):
    if not request.user.is_authenticated:
        messages.warning(request,"Please Login and Try Again")
        return redirect('/login')
    user_phone=request.user
    posts=Enrollment.objects.filter(PhoneNumber=user_phone)
    attendance=Attendance.objects.filter(phonenumber=user_phone)
    print(posts)
    context={"posts":posts,"attendance":attendance}
    return render(request,"profile.html",context)


def signup(request):
    if request.method=="POST":
        username=request.POST.get('usernumber')
        email=request.POST.get('email')
        pass1=request.POST.get('pass1')
        pass2=request.POST.get('pass2')
      
        if len(username)>10 or len(username)<10:
            messages.info(request,"Phone Number Must be 10 Digits")
            return redirect('/signup')

        if pass1!=pass2:
            messages.info(request,"Password is not Matching")
            return redirect('/signup')
       
        try:
            if User.objects.get(username=username):
                messages.warning(request,"Phone Number is Taken")
                return redirect('/signup')
           
        except Exception as identifier:
            pass
        
        
        try:
            if User.objects.get(email=email):
                messages.warning(request,"Email is Taken")
                return redirect('/signup')
           
        except Exception as identifier:
            pass
        
        
        
        myuser=User.objects.create_user(username,email,pass1)
        myuser.save()
        messages.success(request,"User is Created Please Login")
        return redirect('/login')
        
        
    return render(request,"signup.html")




def handlelogin(request):
    if request.method=="POST":        
        username=request.POST.get('usernumber')
        pass1=request.POST.get('pass1')
        myuser=authenticate(username=username,password=pass1)
        if myuser is not None:
            login(request,myuser)
            messages.success(request,"Login Successful")
            return redirect('/')
        else:
            messages.error(request,"Invalid Credentials")
            return redirect('/login')
            
        
    return render(request,"handlelogin.html")


def handleLogout(request):
    logout(request)
    messages.success(request,"Logout Success")    
    return redirect('/login')

def contact(request):
    if request.method=="POST":
        name=request.POST.get('fullname')
        email=request.POST.get('email')
        number=request.POST.get('num')
        desc=request.POST.get('desc')
        myquery=Contact(name=name,email=email,phonenumber=number,description=desc)
        myquery.save()       
        messages.info(request,"Thanks for Contacting us we will get back you soon")
        return redirect('/contact')
        
    return render(request,"contact.html")


def enroll(request):
    if not request.user.is_authenticated:
        messages.warning(request,"Please Login and Try Again")
        return redirect('/login')

    
   
    if request.method=="POST":
        FullName=request.POST.get('FullName')
        email=request.POST.get('email')
        gender=request.POST.get('gender')
        PhoneNumber=request.POST.get('PhoneNumber')
        DOB=request.POST.get('DOB')
        
        address=request.POST.get('address')
        query=Enrollment(FullName=FullName,Email=email,Gender=gender,PhoneNumber=PhoneNumber,DOB=DOB,Address=address)
        query.save()
        messages.success(request,"Thanks For Enrollment")
        return redirect('/join')



    return render(request,"enroll.html")



from django.shortcuts import render
import cv2
import mediapipe as md
from django.http import StreamingHttpResponse
from django.template import Context, Template

md_drawing=md.solutions.drawing_utils
md_drawing_styles=md.solutions.drawing_styles
md_pose=md.solutions.pose


#PUSH-UP EXERCISE
#PUSH-UP EXERCISE
#PUSH-UP EXERCISE
import datetime

def pushup_counter(request):
    import cv2
    import imutils
    import mediapipe as mp
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pushup_count = 0
    position = None
    User = request.user    
    # Get the Enrollment object for the current user
    enrollment = Enrollment.objects.get(PhoneNumber=User)
    if request.method == 'GET':
        # It's a new biceps exercise session, reset the biceps_count
        enrollment.biceps_count = 0
        enrollment.save()
    
    # Initialize the biceps_count for the user
    pushup_count = enrollment.pushup_count
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.7) as pose:
        while cap.isOpened():
            success, image=cap.read()

            if not success:
                print("empty camera")
                break

            #image = imutils.resize(image, width=500)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            result = pose.process(image)
            lmList = []

            if result.pose_landmarks:
                # Draws the landmarks' points and connects them
                mp_draw.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                for id, im in enumerate(result.pose_landmarks.landmark):
                    # Finding the length and width of the video input
                    h, w, _ = image.shape

                    # Finding the exact coordinates of the body points
                    X, Y = int(im.x * w), int(im.y * h)
                    lmList.append([id, X, Y])

                # Checking whether there are any identified landmarks
                if len(lmList) != 0:
                    # Condition that identifies the down position
                    if lmList[12][2] >= lmList[14][2] and lmList[11][2] >= lmList[13][2]:
                        position = "down"

                    # Condition that identifies the up position
                    if lmList[12][2] <= lmList[14][2] and lmList[11][2] <= lmList[13][2] and position == "down":
                        position = "up"
                        pushup_count += 1
                        enrollment.pushup_count = pushup_count
                        enrollment.save()  
                        # Create a new BicepsHistory object
                        pushup_history = PushupHistory.objects.filter(enrollment=enrollment, timestamp__date=datetime.datetime.now().date()).first()
                        if pushup_history:
                            # A BicepsHistory object already exists, update the existing record
                            pushup_history.pushup_count = pushup_count
                            pushup_history.save()
                        else:
                            # Create a new BicepsHistory object
                            pushup_history = PushupHistory(
                                enrollment=enrollment,
                                pushup_count=pushup_count,
                                timestamp=datetime.datetime.now()
                            )
                            pushup_history.save()
                #print(count)
            blue_color = (255, 0, 0)  # Blue color in BGR format
            text_position = (10, 30)  # Position of the pushup count text

            # Draw a blue box on the image
            cv2.rectangle(image, (10, 10), (150, 60), blue_color, -1)

            # Display the pushup count on the image
            pushup_text = "{}".format(pushup_count)
            cv2.putText(image, pushup_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame=cv2.imencode('.jpg', cv2.flip(image,1))[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #cv2.imshow("Push-up counter", cv2.flip(image, 1))
            key = cv2.waitKey(1)

            # Program terminates when q is pressed
            if key == ord('q'):
                break

        cap.release()

def index(request):
    return render(request, 'index.html')
def video_feed(request):
    return StreamingHttpResponse(pushup_counter(request),content_type='multipart/x-mixed-replace; boundary=frame')

from cvzone.PoseModule import PoseDetector
import math
import numpy as np
from django.shortcuts import render
import cv2
from django.template import Context, Template

#SQAUT EXERCISE
#SQAUT EXERCISE
#SQUAT EXERCISE 
def squat_counter(request):
    import cv2
    from cv2 import destroyAllWindows
    import mediapipe as mp
    import numpy as np
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    User = request.user    
    # Get the Enrollment object for the current user
    enrollment = Enrollment.objects.get(PhoneNumber=User)
    if request.method == 'GET':
        # It's a new biceps exercise session, reset the biceps_count
        enrollment.biceps_count = 0
        enrollment.save()
    
    # Initialize the biceps_count for the user
    squat_count = enrollment.squat_count
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle
    # webcam input

    cap = cv2.VideoCapture(0)
    # Curl counter variables
    squat_count = 0 
    stage = None

    """width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)"""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
            
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
                
                angle_hip = calculate_angle(shoulder, hip, knee)
                hip_angle = 180-angle_hip
                knee_angle = 180-angle_knee
                
                # Visualize angle
                #cv2.putText(image, str(angle), 
                            #tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    #)
                        
                    
                #cv2.putText(image, str(angle_knee), 
                            #tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                    #)
                
                #cv2.putText(image, str(angle_hip), 
                            #tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    #)
                
                # Curl counter logic
                if angle_knee > 169:
                    stage = "UP"
                if angle_knee <= 90 and stage =='UP':
                    stage="DOWN"
                    squat_count +=1
                    enrollment.squat_count = squat_count
                    enrollment.save()  
                    squat_history = SquatHistory.objects.filter(enrollment=enrollment, timestamp__date=datetime.datetime.now().date()).first()
                    if squat_history:
                        # A squat_history object already exists, update the existing record
                        squat_history.squat_count = squat_count
                        squat_history.save()
                    else:
                        # Create a new BicepsHistory object
                        squat_history = SquatHistory(
                            enrollment=enrollment,
                            squat_count=squat_count,
                            timestamp=datetime.datetime.now()
                        )
                        squat_history.save()
            except:
                pass
            
            # Render squat counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(squat_count), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            #cv2.putText(image, 'STAGE', (65,12), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            #cv2.putText(image, stage, 
                        #(60,60), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=4) 
                                    )               
            frame=cv2.imencode('.jpg', cv2.flip(image,1))[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #out.write(image)
            #cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        #out.release()
        cv2.destroyAllWindows()
        
    #destroyAllWindows()


def index(request):
    return render(request, 'index.html')
def squat_feed(request):
    return StreamingHttpResponse(squat_counter(request),content_type='multipart/x-mixed-replace; boundary=frame')

def shoulder_press(request):
    User = request.user    
    # Get the Enrollment object for the current user
    enrollment = Enrollment.objects.get(PhoneNumber=User)
    if request.method == 'GET':
        # It's a new biceps exercise session, reset the biceps_count
        enrollment.shoulder_count = 0
        enrollment.save()
    
    # Initialize the biceps_count for the user
    shoulder_count = enrollment.shoulder_count
    cap = cv2.VideoCapture(0)
    detector = PoseDetector(detectionCon=0.7,trackCon=0.7)



    # Creating Angle finder class
    class angleFinder:
        def __init__(self,lmlist,p1,p2,p3,p4,p5,p6,drawPoints):
            self.lmlist = lmlist
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.p4 = p4
            self.p5 = p5
            self.p6 = p6
            self.drawPoints = drawPoints
        #    finding angles

        def angle(self):
            if len(self.lmlist) != 0:
                point1 = self.lmlist[self.p1]
                point2 = self.lmlist[self.p2]
                point3 = self.lmlist[self.p3]
                point4 = self.lmlist[self.p4]
                point5 = self.lmlist[self.p5]
                point6 = self.lmlist[self.p6]

                x1,y1 = point1[1:-1]
                x2, y2 = point2[1:-1]
                x3, y3 = point3[1:-1]
                x4, y4 = point4[1:-1]
                x5, y5 = point5[1:-1]
                x6, y6 = point6[1:-1]

                # calculating angle for left and right hands
                leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                            math.atan2(y1 - y2, x1 - x2))

                rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) -
                                            math.atan2(y4 - y5, x4 - x5))

                leftHandAngle = int(np.interp(leftHandAngle, [-170, 180], [100, 0]))
                rightHandAngle = int(np.interp(rightHandAngle, [-50, 20], [100, 0]))

                # drawing circles and lines on selected points
                if self.drawPoints == True:
                    cv2.circle(img, (x1, y1), 10, (245,117,66), 5)
                    cv2.circle(img, (x1, y1), 15, (245,117,66), 6)
                    cv2.circle(img, (x2, y2), 10, (245,117,66), 5)
                    cv2.circle(img, (x2, y2), 15, (245,117,66), 6)
                    cv2.circle(img, (x3, y3), 10, (245,117,66), 5)
                    cv2.circle(img, (x3, y3), 15, (245,117,66), 6)
                    cv2.circle(img, (x4, y4), 10, (245,117,66), 5)
                    cv2.circle(img, (x4, y4), 15, (245,117,66), 6)
                    cv2.circle(img, (x5, y5), 10, (245,117,66), 5)
                    cv2.circle(img, (x5, y5), 15, (245,117,66), 6)
                    cv2.circle(img, (x6, y6), 10, (245,117,66), 5)
                    cv2.circle(img, (x6, y6), 15, (245,117,66), 6)

                    cv2.line(img,(x1,y1),(x2,y2),(245,66,230),4)
                    cv2.line(img, (x2, y2), (x3, y3), (245,66,230), 4)
                    cv2.line(img, (x4, y4), (x5, y5), (245,66,230), 4)
                    cv2.line(img, (x5, y5), (x6, y6), (245,66,230), 4)
                    cv2.line(img, (x1, y1), (x4, y4), (245,66,230), 4)

                return list([leftHandAngle,rightHandAngle])

    # defining some variables
    shoulder_count = 0
    direction = 0
    #left, right = hands[0:]
    while True:
        ret, img = cap.read()
        img = cv2.resize(img,(640,480))

        detector.findPose(img,draw=0)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=0,draw=False)

        angle1 = angleFinder(lmList,11,13,15,12,14,16,drawPoints=True)
        hands = angle1.angle()
        left, right = hands[0:]

        # Counting number of shoulder ups
        if left >= 90 and right >= 90:
            if direction == 0:
                direction = 1
        if left <= 70 and right <= 70:
            if direction == 1:
                shoulder_count += 1
                direction = 0
                #Update and save the biceps_count in the enrollment object
                enrollment.shoulder_count = shoulder_count
                enrollment.save()  
                shoulder_history = ShoulderpressHistory.objects.filter(enrollment=enrollment, timestamp__date=datetime.datetime.now().date()).first()
                if shoulder_history:
                    # A BicepsHistory object already exists, update the existing record
                    shoulder_history.shoulder_count = shoulder_count
                    shoulder_history.save()
                else:
                    # Create a new BicepsHistory object
                    shoulder_history = ShoulderpressHistory(
                        enrollment=enrollment,
                        biceps_count=shoulder_count,
                        timestamp=datetime.datetime.now()
                    )
                    shoulder_history.save()

        
        #putting scores on the screen
        height, width, _ = img.shape
        cv2.putText(img, str(int(shoulder_count)), (width - 115, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6, cv2.LINE_AA)
        # print(counter)

        # Converting values for rectangles
        leftval = np.interp(left,[0,100],[400,200])
        rightval = np.interp(right, [0, 100], [400, 200])

        # For color changing
        value_left = np.interp(left, [0, 100], [0, 100])
        value_right = np.interp(right,  [0, 100], [0, 100])

        # Drawing right rectangle and putting text
        #cv2.putText(img,'R',(24,195),cv2.FONT_HERSHEY_DUPLEX,1,(255, 0, 0),5)
        #cv2.rectangle(img,(8,200),(50,400),(0,255,0),5)
        #cv2.rectangle(img, (8, int(rightval)), (50, 400), (255, 0, 0), -1)


        # Drawing right rectangle and putting text
        #cv2.putText(img, 'L', (604,195),cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0),5)
        #cv2.rectangle(img, (582, 200), (632, 400), (0, 255, 0), 5)
        #cv2.rectangle(img, (582, int(leftval)), (632, 400), (255, 0, 0), -1)

        #if value_left > 70:
            #cv2.rectangle(img, (582, int(leftval)), (632, 400), (0, 0, 255), -1)

        #if value_right > 70:
            #cv2.rectangle(img, (8, int(rightval)), (50, 400), (0, 0, 255), -1)

        frame=cv2.imencode('.jpg', cv2.flip(img,1))[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        #cv2.imshow("Image",img)
        cv2.waitKey(1)
    
def index(request):
    return render(request, 'index.html')
def shoulder_feed(request):
    return StreamingHttpResponse(shoulder_press(request),content_type='multipart/x-mixed-replace; boundary=frame')
import mediapipe as mp
def biceps(request):
    #import dependecies 
    mp_drawing = mp.solutions.drawing_utils
    #mp_pose is importing pose estimation model
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    biceps_count = 0 
    stage = None
    User = request.user
    
    # Get the Enrollment object for the current user
    enrollment = Enrollment.objects.get(PhoneNumber=User)
    if request.method == 'GET':
        # It's a new biceps exercise session, reset the biceps_count
        enrollment.biceps_count = 0
        enrollment.save()
    
    # Initialize the biceps_count for the user
    biceps_count = enrollment.biceps_count
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle 
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                
                # Calculate angle
                angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
                # Visualize angle
                
                
                # Curl counter logic
                if angle1 > 160 and angle2 > 160:
                    stage = "down"
                if angle1 < 30 and angle2 < 30 and stage =='down':
                    stage="up"
                    biceps_count +=1
                    biceps_history = BicepsHistory.objects.filter(enrollment=enrollment, timestamp__date=datetime.datetime.now().date()).first()
                    if biceps_history:
                        # A BicepsHistory object already exists, update the existing record
                        biceps_history.biceps_count = biceps_count
                        biceps_history.save()
                    else:
                        # Create a new BicepsHistory object
                        biceps_history = BicepsHistory(
                            enrollment=enrollment,
                            biceps_count=biceps_count,
                            timestamp=datetime.datetime.now()
                        )
                        biceps_history.save()
                    
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(biceps_count),(10,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            #cv2.putText(image, 'STAGE', (65,12), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            #cv2.putText(image, stage, 
            #            (60,60), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=4) 
                                    )               
            frame=cv2.imencode('.jpg', cv2.flip(image,1))[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def index(request):
    return render(request, 'index.html')
def biceps_feed(request):
    return StreamingHttpResponse(biceps(request),content_type='multipart/x-mixed-replace; boundary=frame')

def triceps(request):
    import cv2
    import mediapipe as mp

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    cap = cv2.VideoCapture(0)
    side_count=0
    User = request.user    
    # Get the Enrollment object for the current user
    enrollment = Enrollment.objects.get(PhoneNumber=User)
    if request.method == 'GET':
        # It's a new biceps exercise session, reset the biceps_count
        enrollment.side_count = 0
        enrollment.save()
    
    # Initialize the biceps_count for the user
    side_count = enrollment.side_count
    up = False
    while True:
        success, img = cap.read()
        #img = cv2.resize(img,(1280,720))
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                mpDraw.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=4), 
                                mpDraw.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=4) 
                              )
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            if 14 in points and 12 in points:
                if not up and points[14][1] + 30 < points[12][1] and points[13][1] +30 < points[11][1]:
                    #print("UP")
                    up = True
                    side_count += 1
                    enrollment.side_count = side_count
                    enrollment.save()  
                    triceps_history = TricepsHistory.objects.filter(enrollment=enrollment, timestamp__date=datetime.datetime.now().date()).first()
                    if triceps_history:
                        # A BicepsHistory object already exists, update the existing record
                        triceps_history.side_count = side_count
                        triceps_history.save()
                    else:
                        # Create a new BicepsHistory object
                        triceps_history = TricepsHistory(
                            enrollment=enrollment,
                            side_count=side_count,
                            timestamp=datetime.datetime.now()
                        )
                        triceps_history.save()
                elif points[14][1] > points[12][1] and points[13][1] > points[11][1]:
                    #print("Down")
                    up = False
            cv2.rectangle(img, (0,0), (225,73), (245,117,16), -1)    
            cv2.putText(img, str(side_count), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        #cv2.imshow("img",img)
        frame=cv2.imencode('.jpg', cv2.flip(img,1))[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cv2.waitKey(1)
def index(request):
    return render(request, 'index.html')
def triceps_feed(request):
    return StreamingHttpResponse(triceps(request),content_type='multipart/x-mixed-replace; boundary=frame')

def exercise_suggestion(request):
    exercises = ["biceps", "triceps", "squats", "sit ups", "chin ups"]
    
    if request.method == 'POST':
        form = ExerciseForm(request.POST)
        if form.is_valid():
            # Retrieve the form data
            name = form.cleaned_data['name']
            age = form.cleaned_data['age']
            weight = form.cleaned_data['weight']
            height = form.cleaned_data['height']
            bmi = form.cleaned_data['bmi']

            # Generate random exercise suggestions
            random_exercises = random.sample(exercises, 5)

            # Pass the exercise suggestions to the template
            context = {
                'name': name,
                'random_exercises': random_exercises,
            }
            return render(request, 'exercise_suggestion.html', context)
    else:
        form = ExerciseForm()
    
    context = {'form': form}
    return render(request, 'exercise_form.html', context)


@login_required
def create_blog(request):
    if request.method == 'POST':
        form = BlogForm(request.POST, request.FILES)
        if form.is_valid():
            blog = form.save(commit=False)
            
            blog.save()
            return redirect('blog_list')
    else:
        form = BlogForm()
    return render(request, 'create_blog.html', {'form': form})


def blog_list(request):
    blogs = Blog.objects.all().order_by('-published_at')
    return render(request, 'blog_list.html', {'blogs': blogs})

from django.db.models import Max

from django.db.models import Max

def biceps_table(request):
    # Get the highest biceps_count for the signed-in user per day
    user = request.user
    highest_biceps_counts = BicepsHistory.objects.filter(enrollment__PhoneNumber=user).values('enrollment__PhoneNumber', 'timestamp__date').annotate(max_biceps_count=Max('biceps_count'))
    
    # Retrieve the BicepsHistory objects with the highest biceps_count for the signed-in user per day
    biceps_history = BicepsHistory.objects.filter(
        enrollment__PhoneNumber=user,
        timestamp__date__in=[obj['timestamp__date'] for obj in highest_biceps_counts],
        biceps_count__in=[obj['max_biceps_count'] for obj in highest_biceps_counts]
    )
    
    context = {
        'biceps_history': biceps_history
    }
    
    return render(request, 'biceps_table.html', context)

from django.db.models import Max

def shoulder_table(request):
    user = request.user

    # Get the highest shoulder count for each date for the current user
    highest_shoulder_counts = ShoulderpressHistory.objects.filter(enrollment__PhoneNumber=user).values('timestamp__date').annotate(max_shoulder_count=Max('shoulder_count'))

    # Retrieve the ShoulderpressHistory objects with the highest shoulder count for each date for the current user
    shoulder_history = ShoulderpressHistory.objects.filter(
        enrollment__PhoneNumber=user,
        timestamp__date__in=[obj['timestamp__date'] for obj in highest_shoulder_counts],
        shoulder_count__in=[obj['max_shoulder_count'] for obj in highest_shoulder_counts]
    )

    context = {
        'shoulder_history': shoulder_history
    }

    return render(request, 'shoulder_table.html', context)


def triceps_table(request):
    user=request.user
    # Get the highest biceps_count for each date
    highest_side_counts = TricepsHistory.objects.filter(enrollment__PhoneNumber=user).values('timestamp__date').annotate(max_side_count=Max('side_count'))
    
    # Retrieve the BicepsHistory objects with the highest biceps_count for each date
    triceps_history = TricepsHistory.objects.filter(
        enrollment__PhoneNumber=user,
        timestamp__date__in=[obj['timestamp__date'] for obj in highest_side_counts],
        side_count__in=[obj['max_side_count'] for obj in highest_side_counts]
    )
    
    context = {
        'triceps_history': triceps_history
    }
    
    return render(request, 'triceps_table.html', context)
def squat_table(request):
    user=request.user
    # Get the highest biceps_count for each date
    highest_squat_counts = SquatHistory.objects.filter(enrollment__PhoneNumber=user).values('timestamp__date').annotate(max_squat_count=Max('squat_count'))
    
    # Retrieve the BicepsHistory objects with the highest biceps_count for each date
    squat_history = SquatHistory.objects.filter(
        enrollment__PhoneNumber=user,
        timestamp__date__in=[obj['timestamp__date'] for obj in highest_squat_counts],
        squat_count__in=[obj['max_squat_count'] for obj in highest_squat_counts]
    )
    
    context = {
        'squat_history': squat_history
    }
    
    return render(request, 'squat_table.html', context)

def pushup_table(request):
    user=request.user
    # Get the highest biceps_count for each date
    highest_pushup_counts = PushupHistory.objects.filter(enrollment__PhoneNumber=user).values('timestamp__date').annotate(max_pushup_count=Max('pushup_count'))
    
    # Retrieve the BicepsHistory objects with the highest biceps_count for each date
    pushup_history = PushupHistory.objects.filter(
        enrollment__PhoneNumber=user,
        timestamp__date__in=[obj['timestamp__date'] for obj in highest_pushup_counts],
        pushup_count__in=[obj['max_pushup_count'] for obj in highest_pushup_counts]
    )
    
    context = {
        'pushup_history': pushup_history
    }
    
    return render(request, 'pushup_table.html', context)

def biceps_inst(request):
    return render(request, 'biceps_inst.html')

def triceps_inst(request):
    return render(request, 'triceps_inst.html')

def squats_inst(request):
    return render(request, 'squats_inst.html')

def pushups_inst(request):
    return render(request, 'pushups_inst.html')



def shoulderpresss_inst(request):
    return render(request, 'shoulderpresss_inst.html')


