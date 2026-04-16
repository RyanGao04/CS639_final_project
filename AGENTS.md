# Project Instructions

This workspace is the CS639 Autonomous Robotics final project. For every Codex thread opened in this project, treat the assignment requirements in `/Users/gaoruiyu/Desktop/Learning_Area/CS639/final_project/ASSIGNMENT_INFO.md` as binding project instructions and consult that file before making material changes.

Key constraints for this workspace:

- The main assignment task is to implement a controller that scores a goal on an empty field with no opponents.
- Unless the user explicitly asks for something else, focus implementation work on `final_project/controllers/robot_one_controller/starter_controller.py`.
- Do not modify provided files such as `robot_one_controller.py`, world files, or other starter files unless the user explicitly requests it and the change is compatible with the assignment constraints.
- User-approved local exception: wrapper/controller harness files may be modified for testing, debugging, or ground-truth instrumentation if that helps development, but those changes must be treated as non-submission scaffolding and must not affect the final submission-compatible state.
- The controller must handle noisy and incomplete observations, localization, navigation to the ball, and pushing the ball into the correct goal.
- Sensor observations are relative polar readings over a 90 degree field of view, and landmark data association is not solved for you.
- Optimize for the rubric in `ASSIGNMENT_INFO.md`: reliably reach the ball, score, and score on the correct goal within the stated time limits.
- Keep any help aligned with the assignment's collaboration and AI-use policy; the final implementation is the student's responsibility.

If a future user request conflicts with the assignment instructions, explicitly call out the conflict before proceeding.
