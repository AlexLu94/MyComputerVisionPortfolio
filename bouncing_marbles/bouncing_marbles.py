import cv2       as cv
import mediapipe as mp
import pymunk
import random

class ball:
    def __init__(self, position, velocity, mass, radius, color):
        self.radius   = radius
        self.color    = color

        self.physics_inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.physics_body    = pymunk.Body(mass, self.physics_inertia)

        self.physics_body.position = position
        self.physics_body.velocity = velocity

        self.physics_shape = pymunk.Circle(self.physics_body, radius, (0, 0))
        self.physics_shape.elasticity = 0.95
        self.physics_shape.friction = 0.9
        space.add(self.physics_body, self.physics_shape)
    
    def cv_draw(self, frame):
        x = int(self.physics_body.position[0])
        y = int(self.physics_body.position[1])
        cv.circle(frame, (x, y), self.radius, self.color, -1)

    def is_inside(self, point):
        return ((point[0]-self.physics_body.position[0])**2 + (point[1]-self.physics_body.position[1])**2)**0.5 < self.radius

    def change_color(self, color):
        self.color = color

    def translate(self, delta_position, new_velocity):
        self.physics_body.position = (self.physics_body.position[0] + delta_position[0], self.physics_body.position[1] + delta_position[1])
        self.physics_body.velocity = new_velocity

TRIGGER_FINGER_DISTANCE = 40

BALL_RADIUS  = 100
BALL_MASS    = 1

NUM_BALLS    = 3

MAX_VELOCITY = 5

COLOR_REST  = (200, 140, 140)
COLOR_TOUCH = (140, 140, 200)
TRANSPARENCY = 0.2

SIMULATION_TIME_STEP = 1

# Init hand detection
mp_hands   = mp.solutions.hands

# Init video capture from webcam
video_capt = cv.VideoCapture(0)
if not video_capt.isOpened():
    print("Cannot open camera. Exiting.")
    exit()
retv, frame = video_capt.read()
if not retv:
    print("Cannot get frame. Exiting.")
    exit()
size_x = frame.shape[1]
size_y = frame.shape[0]

# Init pymonk as physics engine
space = pymunk.Space()
space.gravity = (0.0, 0.0)

# Define the borders, balls are going to bounce here
border_limit_body = space.static_body
for a, b in [((0, 0), (size_x, 0)), ((0, 0), (0, size_y)), ((0, size_y), (size_x, size_y)), ((size_x, 0), (size_x, size_y))]:
    line_body = pymunk.Segment(border_limit_body, a, b, 0.0)
    line_body.elasticity = 0.95
    line_body.friction   = 0.9
    space.add(line_body)

# Position of the two fingers on screen
finger1 = [0, 0]
finger2 = [size_x, size_y]
prev_touching_position = [-1, -1]
position_delta         = [0, 0]
finger_velocity = [0, 0]

# Randomly generate the balls
balls = []
for _ in range(NUM_BALLS):
    balls.append(
        ball((random.randint(BALL_RADIUS, size_x-BALL_RADIUS), random.randint(BALL_RADIUS, size_y-BALL_RADIUS)),
            (random.randrange(-MAX_VELOCITY, MAX_VELOCITY), random.randrange(-MAX_VELOCITY, MAX_VELOCITY)),
            BALL_MASS, BALL_RADIUS, COLOR_REST))

currently_touched = -1

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        # Capture new frame
        retv, frame = video_capt.read()
        if not retv:
            print("Cannot get frame. Exiting.")
            exit()
        frame = cv.flip(frame, 1) 
        original_frame = frame.copy()

        # Process hand position
        touching = False
        results  = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            finger1 = [int(results.multi_hand_landmarks[0].landmark[4].x*size_x), int(results.multi_hand_landmarks[0].landmark[4].y*size_y)]
            finger2 = [int(results.multi_hand_landmarks[0].landmark[8].x*size_x), int(results.multi_hand_landmarks[0].landmark[8].y*size_y)]
            finger_distance = (abs(finger2[0]-finger1[0])**2+abs(finger2[1]-finger1[1])**2)**0.5
            #cv.circle(frame, (finger1[0], finger1[1]), TRIGGER_FINGER_DISTANCE//2, (200, 0,0), -1)
            #cv.circle(frame, (finger2[0], finger2[1]), TRIGGER_FINGER_DISTANCE//2, (0, 200,0), -1)
            if finger_distance<=TRIGGER_FINGER_DISTANCE:
                touching            = True
                touching_position   = [(finger1[0]+finger2[0])//2, (finger1[1]+finger2[1])//2]
                if prev_touching_position != [-1, -1]:
                    position_delta[0] = touching_position[0]-prev_touching_position[0]
                    position_delta[1] = touching_position[1]-prev_touching_position[1]
                    finger_velocity[0] = position_delta[0]/SIMULATION_TIME_STEP
                    finger_velocity[1] = position_delta[1]/SIMULATION_TIME_STEP
                prev_touching_position = touching_position
            else:
                prev_touching_position = [-1, -1]
                position_delta         = [0, 0]
        
        for i in range(len(balls)):
            if touching and balls[i].is_inside(touching_position):
                currently_touched = i
                balls[i].change_color(COLOR_TOUCH)
                balls[i].translate(position_delta, [0*position_delta[0]/SIMULATION_TIME_STEP, 0*position_delta[1]/SIMULATION_TIME_STEP])
            else:
                balls[i].change_color(COLOR_REST)
            balls[i].cv_draw(frame)
        if touching == False and currently_touched > -1:
            balls[currently_touched].translate([0, 0], finger_velocity)
            currently_touched = -1
                
        
        # Show frame
        cv.imshow('Stream', cv.addWeighted(original_frame, TRANSPARENCY, frame, 1 - TRANSPARENCY, 0) )

        # Process input
        if cv.waitKey(1) == ord('q'):
            break
        
        space.step(SIMULATION_TIME_STEP)
 
# When everything done, release the capture
video_capt.release()
cv.destroyAllWindows()
