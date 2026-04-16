import math
import numpy as np
from controller import Robot, DistanceSensor, Motor, Compass, GPS
from controller import Supervisor
from starter_controller import StudentController


MIN_VELOCITY = -6.25
MAX_VELOCITY = 6.25
MAX_BOXES = 10


# Define the robot class
class TurtleBotController:
    def __init__(self):
        # Initialize the robot and get basic time step
        self.ego_id = "ROBOT_TWO"
        if self.ego_id == "ROBOT_ONE":
            self.opponent_id = "ROBOT_TWO"
        else:
            self.opponent_id = "ROBOT_ONE"
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Create the Supervisor instance
        # self.supervisor = Supervisor()
        self.ego_robot_node = self.robot.getFromDef(self.ego_id)
        self.opponent_robot_node = self.robot.getFromDef(self.opponent_id)

        # Get motors (assuming left and right motors are available)
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")

        # Set motor velocity to maximum (adjust as needed)
        self.left_motor.setPosition(float("inf"))  # Infinity means continuous rotation
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Initialize Lidar (Laser Distance Sensor)
        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.time_step)

        # Initialize Compass (for orientation)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.time_step)

        # Sensor and control noise
        self._lidar_noise = 0.15
        self._detection_range = 1.0
        self._control_noise_pct = 0.05
        self._heading_noise = 0 * (math.pi / 180)
        self._odometry_noise = 0.01

        # Odometry variables
        pose = self.provide_pose()
        self.prev_position = pose[:2]
        self.prev_rotation = None

        self.student_controller = StudentController()

    def provide_compass(self):
        values = self.compass.getValues()
        heading = math.atan2(values[0], values[1])
        return heading

    def provide_lidar(self):
        """Get lidar range image and lower detect range below default 3.5."""
        lidar_image = self.lidar.getRangeImage()
        for i in range(len(lidar_image)):
            lidar_image[i] = (
                float("inf")
                if lidar_image[i] >= self._detection_range
                else lidar_image[i]
            )
        noise = np.random.normal(0, self._lidar_noise, size=len(lidar_image))
        lidar_image += noise
        return lidar_image

    def clip_control(self, control):
        """Non-linear behavior in controls."""
        if abs(control) < 0.05:
            control = 0.0
        control = min(max(control, MIN_VELOCITY), MAX_VELOCITY)
        return control

    def provide_pose(self):
        position = self.ego_robot_node.getField("translation").getSFVec3f()[:2]
        # rotation = self.ego_robot_node.getField("rotation").getSFVec3f()[3]
        rotation = self.provide_compass()
        return position + [rotation]

    def is_in_fov(self, observer_pos, observer_angle_rad, fov_rad, target_pos):
        """
        Check if the target_pos is within the field of view of the observer.

        Parameters:
        - observer_pos: (x, y) tuple
        - observer_angle_deg: direction the observer is facing, in degrees (0 = +x, 90 = +y)
        - fov_rad: total field of view angle (e.g., 60 means 30° left and right)
        - target_pos: (x, y) tuple

        Returns:
        - True if target is within the FOV cone, else False
        """

        # Vector from observer to target
        dx = target_pos[0] - observer_pos[0]
        dy = target_pos[1] - observer_pos[1]

        # Angle from observer to target
        angle_to_target = math.atan2(dy, dx)

        # Normalize angle difference
        angle_diff = angle_to_target - observer_angle_rad
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Wrap to [-180, 180]

        # Check if within half the FOV
        return abs(angle_diff) <= fov_rad / 2

    def get_polar_obs(self, ego_pose, ego_angle, point):
        dx = point[0] - ego_pose[0]
        dy = point[1] - ego_pose[1]
        dist = np.sqrt(dx**2 + dy**2)

        angle_to_target = math.atan2(dy, dx)
        angle = -1 * (ego_angle - angle_to_target)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi

        return dist, angle

    def get_obs(self, point):
        ego_pose = self.provide_pose()
        if self.is_in_fov(ego_pose[:2], ego_pose[2], math.pi/2, point):
            true_polar_coords = self.get_polar_obs(ego_pose[:2], ego_pose[2], point)
            dist_noise = 0  # np.random.normal(0, 0.02 * true_polar_coords[0])
            angle_noise = 0 # np.random.normal(0, 0.01 * true_polar_coords[0])
            obs = (true_polar_coords[0] + dist_noise, true_polar_coords[1] + angle_noise)
            return obs
        return None

    def provide_ball_observation(self):
        ball = self.robot.getFromDef("BALL")
        position = ball.getField("translation").getSFVec3f()[:2]
        return self.get_obs(position)

    def provide_goal_observations(self):

        goals = [(4.5, 0), (-4.5, 0)]
        observations = []
        for goal in goals:
            obs = self.get_obs(goal)
            if obs is not None:
                observations.append(obs)
        return observations

    def provide_center_observation(self):
        center = (0, 0)
        obs = self.get_obs(center)
        return obs

    def provide_cross_observations(self):
        crosses = [(3.25, 0), (-3.25, 0)]
        observations = []
        for cross in crosses:
            obs = self.get_obs(cross)
            if obs is not None:
                observations.append(obs)
        return observations

    def provide_corner_observations(self):
        corners = [(-4.5, 3), (-4.5, -3), (4.5, 3), (4.5, -3)]
        observations = []
        for corner in corners:
            obs = self.get_obs(corner)
            if obs is not None:
                observations.append(obs)
        return observations

    def provide_opponent_observation(self):
        if self.opponent_robot_node is not None:
            position = self.opponent_robot_node.getField("translation").getSFVec3f()[:2]
            return self.get_obs(position)
        return None

    def provide_odometry(self):
        """Compute a noisy odometry estimate."""
        pose = self.provide_pose()
        rotation = pose[2]
        position = pose[:2]

        if self.prev_rotation is None:
            self.prev_rotation = rotation

        delta_forward = (position[0] - self.prev_position[0]) ** 2 + (position[1] - self.prev_position[1]) ** 2
        delta_forward = np.sqrt(delta_forward)

        delta_rotation = rotation - self.prev_rotation
        if abs(delta_rotation) > np.pi:  # Impossible so indicates angle boundary
            if rotation < 0:
                delta_rotation = (2 * np.pi + rotation) - self.prev_rotation
            else:
                delta_rotation = rotation - (2*np.pi + self.prev_rotation)

        self.prev_position = position
        self.prev_rotation = rotation

        odom_vector = np.array([delta_forward, delta_rotation])
        noise = np.random.normal(0, self._odometry_noise, size=2)
        return odom_vector + noise

    def run(self):
        """
        The main loop that controls the robot's behavior.
        """

        print("Starting run loop for %s" % self.ego_id)
        while self.robot.step(self.time_step) != -1:
            # Pack sensor values for student controller
            sensors = {
                "ball": self.provide_ball_observation(),  # ball observation or None
                "goal": self.provide_goal_observations(),  # list of goal sightings
                "center_circle": self.provide_center_observation(),
                "penalty_cross": self.provide_cross_observations(),
                "corners": self.provide_corner_observations(),
                "opponent": self.provide_opponent_observation(),
                "odometry": self.provide_odometry()
            }

            # Get control values from student controller
            controls = self.student_controller.step(sensors)
            lwhl = controls.get("left_motor", 0.0)
            rwhl = controls.get("right_motor", 0.0)

            # Apply noise to control and clip to remain in bounds
            lwhl += np.random.normal(0, self._control_noise_pct * abs(lwhl))
            rwhl += np.random.normal(0, self._control_noise_pct * abs(rwhl))
            lwhl = self.clip_control(lwhl)
            rwhl = self.clip_control(rwhl)

            # Set control
            self.left_motor.setVelocity(lwhl)
            self.right_motor.setVelocity(rwhl)


# Create a controller instance and run it
controller = TurtleBotController()

controller.run()
