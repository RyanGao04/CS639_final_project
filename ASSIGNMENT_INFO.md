# CS639 Final Project Assignment Info

Source: `/Users/gaoruiyu/Desktop/Learning_Area/CS639/final_project/S26_Final_Project.pdf`

This file is a project-local text transcript and working reference for the assignment.

## Operational Summary

- Main task: implement a TurtleBot controller that can score a goal on an empty field with no opponents.
- Primary file to implement: `final_project/controllers/robot_one_controller/starter_controller.py`
- Do not modify provided files other than `starter_controller.py`, or the instructor's tests may fail.
- Local exception for this workspace: wrapper/controller harness files may be modified temporarily for local testing, debugging, or ground-truth instrumentation, as long as those changes do not affect the final submitted deliverable or the final submission state.
- The controller must work with noisy, incomplete sensor readings and unresolved data association.
- The robot must find the ball, move to it, and push it into the correct goal.
- Submission: submit `starter_controller.py` on Gradescope.

## Local Debugging Note

- This project may use temporary edits to wrapper scripts, such as `robot_one_controller.py`, to expose ground-truth state or other debugging data during local development.
- Treat those edits as development-only instrumentation unless the user explicitly says otherwise.
- Before final submission or final validation, ensure the submission-compatible state still respects the assignment rule that only `starter_controller.py` is submitted and relied upon.

## Transcript

### CS639: Autonomous Robotics: Final Project

Josiah Hanna

April 2026

### 1 Overview

Your final project is to implement a robot controller for a TurtleBot to successfully play soccer. Whereas the programming assignments have focused on individual components of a robot architecture, in your final project, you will have to consider the integration of different components. Specifically, your controller must:

1. Take noisy and incomplete sensor readings, localize your robot, and estimate the state of dynamic objects in the world.
2. Control your robot to navigate to a target position.
3. And perform fine-grained control to manipulate a dynamic object (a soccer ball).
4. (Potential Extra Credit) Control in the face of adversarial perturbations.

There is no single prescribed approach. Some suggestions are made in the "Hints" section below.

### 1.1 Note on Collaboration

You are free to discuss algorithmic details of implementations with each other, i.e., pseudocode. However, your final implementation must be your work alone - sharing code will result in a zero on the assignment.

### 1.2 AI Policy

You are free to use AI coding assistants on this assignment, however, the final submitted code will be viewed no differently than if you had written it yourself.

### 2 Getting Started

Your initial steps are:

1. Open Webots.
2. Download the provided project directory (from Canvas) and open it in Webots. To open it, go to `File >> Open World`, and then select `<assignment_root>/worlds/soccer_solo.wbt` where `assignment_root` is the top-level directory in the project directory.
3. You should see a TurtleBot on a soccer field in front of a ball. If you run the simulation, the robot will move forward until it hits a wall at the end of the field.

### 3 Main Task

Your main task is to implement a controller that enables your robot to score a goal on an empty field (with no opponents). We have provided starter code in the project directory. To find this code, look under:

`<assignment_root>/controllers/robot_one_controller/`

You should see the following files:

1. `robot_one_controller.py` is a Webots robot controller that provides a wrapper around the robot's sensors and actuators. Its purpose is to provide (noisy) observations of different landmarks relative to the robot's position and to add some noise to the controls of the robot. Do not modify this file.
2. `starter_controller.py` is where you will place your code. The main function in this class is called `step` and it is called by the `run()` function in `turtle_controller.py`. You will implement this function, and any necessary helper functions, to provide the desired capability. Specifically, this function should:
   - Take as input a `dict` of sensor values. The sensor values correspond to noisy, relative polar coordinate observations of different landmarks and objects that are within the robot's 90 degree field of view.
   - Included landmarks are the goals, the field corners, the penalty crosses, and the field center.
   - Note that the data association problem is not solved for you. For example, a corner observation does not tell you which of the four corners has been seen.
   - The robot can also perceive the ball and an opponent robot, when applicable.
   - Implement a controller that returns desired controls for the robot's two wheels to move the robot to the ball and then push the ball to the goal.

Note: you should not modify any provided files other than `starter_controller.py`. Doing so will cause your submission to fail when tested by the instructor.

### 4 Extra Credit

You may optionally participate in a final class tournament for the opportunity to earn extra points. For this tournament, we will use the world file `soccer_dual.wbt` that has been included in the starter code. Your code will be placed in either the sub-directory `robot_one_controller/` or `robot_two_controller/`. Because observations are always relative to the robot, there is no need to know in advance which robot you are controlling.

To participate in the tournament, you must first demonstrate that you can successfully score on the correct goal without an opponent on the field. You must submit your tournament code by Tuesday, April 28 at midnight (Central time) to be eligible - note that this is before the final deadline. The exact tournament format will be at the discretion of the instructor and will depend on the number of entrants.

### 5 Rubric

This assignment is worth 20 points. The code will be tested with two different initializations of the ball position and points will be awarded based on your robot's ability to complete a series of objectives. You should expect the test cases to be chosen to be difficult for control, localization, or both.

1. Objective 1: Get to the ball (6 points). Your robot demonstrates that it can find the ball and move within a radius of 0.5 meters of the ball. Maximum allowed time: 6 minutes.
2. Objective 2: Score a goal (6 points). Your robot demonstrates that it can push the ball to one of the two goals and move the ball entirely over the white goal line. Maximum allowed time: 6 minutes.
3. Objective 3: Score on the correct goal (8 points). Same as objective 2, except the goal must be the goal that the robot was facing to begin with.

Partial credit will be awarded for progress toward these objectives. For each test case, you have a maximum of 6 minutes to complete all objectives. The maximum allowed time is based on the time it would take the robot to move down and back on the field 3 times at full speed. It is expected that the task can be completed in significantly less time.

Extra credit: the tournament winner will receive 5 bonus points, second place will receive 3 bonus points, and all other competitors will have the opportunity to receive 1 bonus point for beating a course staff agent.

### 6 Hints and Answers to FAQs

More hints and answers will be added as they arise.

1. Your robot will always start 1 meter from the center and facing its goal. However, the ball may not start in the very center during testing.
2. You may find it helpful to represent the robot's behavior as a finite state machine, where each state represents a sub-behavior. For example, you could have a `LookForBall` state, a `MoveToBall` state, a `LineUpToBall` state, and a `DribbleBall` state. Transitions between states would be driven by the current condition, e.g., the robot looks for the ball until it sees it and then moves toward it. If the ball is lost then the robot switches back to the `LookForBall` state.
3. You have roughly 4 weeks to complete the assignment, not counting spring break. Start early. A good checkpoint would be to make sure your robot can robustly stay localized as it moves around. If you can reach that point in approximately 2 weeks, then you should be well-positioned to develop a controller that solves all objectives by the end of the semester. If you want to compete in the tournament, you'll need to accelerate that timeline.
4. Some type of localization module is necessary to complete this assignment. If you rely on object observations alone, your robot will likely fail if the ball goes behind them such that the robot is looking at the wrong goal. You can expect your code to be challenged in this respect.

### 7 Final Submission

When you have completed your implementation, submit the file `starter_controller.py` on Gradescope.
