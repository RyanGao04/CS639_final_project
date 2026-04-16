"""student_controller controller."""

import math
import numpy as np


class StudentController:
    def __init__(self):
        pass

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict: dict, contains control for "left_motor" and "right_motor"
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # TODO: add your controllers here.
        control_dict["left_motor"] = 6.5
        control_dict["right_motor"] = 6.5

        return control_dict
