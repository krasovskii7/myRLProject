# Import standard libraries
from pathlib import Path
import time

# Import mujoco and displat
from IPython.display import clear_output
import mujoco
import mujoco.viewer

# Set MJCF
MJCF_PATH = Path(__file__).resolve().parents[1] / "StairClimbingRobot" / "robot.xml"
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

LEFT_ARM_POS = "left_arm_pos"
RIGHT_ARM_POS = "right_arm_pos"
LEFT_ARM_VEL = "left_arm_vel"
RIGHT_ARM_VEL = "right_arm_vel"

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

# Resets simulation data to defaults
mujoco.mj_resetData(model, data)

# Launch MuJoCo
steps = 0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # Define free-look camera (control with mouse), looking at robot's back right
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0, 0.05]
    viewer.cam.distance = 3.8
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -25

    # Loop
    while viewer.is_running():
        step_start = time.time()

        # Write latest motor data to sim
        data.ctrl[left_motor_id] = ctrl["left"]
        data.ctrl[right_motor_id] = ctrl["right"]

        # Advance simulation by one step
        mujoco.mj_step(model, data)

        # Render simulation state
        viewer.sync()

        # print sensor data
        steps+=1

        if steps % PRINT_EVERY == 0:
            accel = data.sensor(IMU_ACCEL).data
            gyro = data.sensor(IMU_GYRO).data
            orientation = data.sensor(IMU_ORIENTATION).data
            left_pos = data.sensor(LEFT_ARM_POS).data
            right_pos = data.sensor(RIGHT_ARM_POS).data
            left_vel = data.sensor(LEFT_ARM_VEL).data
            right_vel = data.sensor(RIGHT_ARM_VEL).data

            # Clear output before printing so we do not flood it
            clear_output(wait=True)
            print(f"Accel:          {accel}")
            print(f"Gyro:           {gyro}")
            print(f"Orientation:    {orientation}")
            print(f"Left Position:  {left_pos}")
            print(f"Right Position: {right_pos}")
            print(f"Left Velocity:  {left_pos}")
            print(f"Right Velocity: {right_pos}")

            # Make sure the loop iteration time matches the MJCF timestep time (2 ms)
            # I.E. slow the simulation down so we can see the robot behave in real time

            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)
