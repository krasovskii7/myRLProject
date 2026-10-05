# Import standard libraries
from pathlib import Path
import time
import math

# Import mujoco and displat
from IPython.display import clear_output
import mujoco
import mujoco.viewer

# Set MJCF
MJCF_PATH = Path(__file__).resolve().parents[1] / "DummyRobot" / "dummy-robot.xml"
MOTOR_SPEED_STEP = 0.1
MOTOR_SPEED_LIMIT = 2.0
PRINT_EVERY = 50

# Actuators
LEFT_MOTOR = "left_motor"
RIGHT_MOTOR = "right_motor"

# Sensors
IMU_ACCEL = "imu_accel"
IMU_GYRO = "imu_gyro"
IMU_ORIENTATION = "imu_orientation"

LEFT_WHEEL_POS = "left_wheel_pos"
RIGHT_WHEEL_POS = "right_wheel_pos"
LEFT_WHEEL_VEL = "left_wheel_vel"
RIGHT_WHEEL_VEL = "right_wheel_vel"

# Keys code
KEY_BACKSPACE = 259
KEY_UP = 265
KEY_DOWN = 264

# Motor speed
ctrl = {"left": 0.0, "right": 0.0}

def clamp(x):
    '''Limit motor speed and direction to a minimum and maximum'''
    return max(-MOTOR_SPEED_LIMIT, min(MOTOR_SPEED_LIMIT, x))

def key_callback(keycode):
    '''Handle a key press'''

    # MuJoCo resets simulation with backspace
    if keycode == KEY_BACKSPACE:
        ctrl["left"] = ctrl["right"] = 0

    # Assign motor speeds for each up and down arrow button
    if keycode == KEY_UP:
        ctrl["left"] = clamp(ctrl["left"] + MOTOR_SPEED_STEP)
        ctrl["right"] = clamp(ctrl["right"] + MOTOR_SPEED_STEP)

    if keycode == KEY_DOWN:
            ctrl["left"] = clamp(ctrl["left"] - MOTOR_SPEED_STEP)
            ctrl["right"] = clamp(ctrl["right"] - MOTOR_SPEED_STEP)

# Load model into MuJoCo
model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))

# Use model to get the simulation state
data = mujoco.MjData(model)

# Get ID of motors from MJCF
left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_MOTOR)
right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_MOTOR)

# Print IDs
print(f"Left Motor ID: {left_motor_id}")
print(f"Right Motor ID: {right_motor_id}")

# Filter and PID parameters
ALPHA = 0.99
KP = 7.0
KD = 0.5

# Tip threshold
TIP_THRESHOLD = 20  # degrees

# Resets simulation data to defaults
mujoco.mj_resetData(model, data)

# Launch MuJoCo
steps = 0
pitch = 0.0
tipped = False
prev_time = 0.0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # Define free-look camera (control with mouse), looking at robot's back right
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0, 0.05]
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -25

    # Loop
    while viewer.is_running():
        step_start = time.time()

        # Detect MuJoCo reset and reset variables
        if data.time < prev_time:
            tipped = False
            mujoco.mj_forward(model, data)
            accel_x, _, accel_z = data.sensor(IMU_ACCEL).data
            pitch = -1*math.atan2(accel_x, accel_z)
        prev_time = data.time

        # Read sensors

        accel_x, accel_y, accel_z = data.sensor(IMU_ACCEL).data
        pitch_rate = data.sensor(IMU_GYRO).data[1]

        # Accelerometer derived pitch
        accel_pitch = -1*math.atan2(accel_x, accel_z)

        # Complementary filter for estimating pitch
        pitch = ALPHA * (pitch + (pitch_rate * model.opt.timestep)) #+ (1-ALPHA)*accel_pitch

        if abs(pitch) > TIP_THRESHOLD:
            tipped = True
        
        if tipped:
            motor_spd_target = 0.0
        else:
            motor_spd_target = clamp((KP*pitch) + (KD*pitch_rate) + 0.001)
            
        # Write latest motor data to sim
        data.ctrl[left_motor_id] = motor_spd_target
        data.ctrl[right_motor_id] = motor_spd_target

        # Advance simulation by one step
        mujoco.mj_step(model, data)

        # Render simulation state
        viewer.sync()

        # print sensor data
        # Clear output before printing so we do not flood it
        clear_output(wait=True)
        print(f"Accel:          {accel_pitch}")
        print(f"Gyro:           {pitch_rate}")
        print(f"Pitch:          {pitch}")
                
        # Make sure the loop iteration time matches the MJCF timestep time (2 ms)
        # I.E. slow the simulation down so we can see the robot behave in real time

        slack = model.opt.timestep - (time.time() - step_start)
        if slack > 0:
           time.sleep(slack)
                        


