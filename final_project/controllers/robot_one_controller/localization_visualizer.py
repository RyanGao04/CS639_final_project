import json
import math
import queue
import signal
import sys
import threading

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


FIELD_X_HALF = 4.5
FIELD_Y_HALF = 3.0
CENTER_CIRCLE_RADIUS = 0.75
PENALTY_X = 3.25
GOAL_DEPTH = 0.6
GOAL_HALF_WIDTH = 0.8

LANDMARK_COLORS = {
    "goal": "#f28e2b",
    "center_circle": "#17becf",
    "penalty_cross": "#4e79a7",
    "corners": "#b07aa1",
}


class LocalizationVisualizer:
    def __init__(self):
        self.snapshot_queue = queue.Queue()
        self.latest_snapshot = None
        self.dynamic_artists = []
        self.stdin_closed = False
        self.shutdown_requested = False
        self.shutting_down = False

        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.fig.canvas.manager.set_window_title("Webots Localization Visualizer")
        self._draw_field()
        self._draw_legend()
        self._install_signal_handlers()

        reader = threading.Thread(target=self._stdin_reader, daemon=True)
        reader.start()

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._tick)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _install_signal_handlers(self):
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                continue

    def _handle_signal(self, _signum, _frame):
        self.shutdown_requested = True

    def _draw_field(self):
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-5.3, 5.3)
        self.ax.set_ylim(-3.7, 3.7)
        self.ax.set_facecolor("#7fbf7f")

        pitch = Rectangle(
            (-FIELD_X_HALF, -FIELD_Y_HALF),
            2 * FIELD_X_HALF,
            2 * FIELD_Y_HALF,
            fill=False,
            edgecolor="white",
            linewidth=2.0,
        )
        self.ax.add_patch(pitch)
        self.ax.plot([0.0, 0.0], [-FIELD_Y_HALF, FIELD_Y_HALF], color="white", linewidth=1.5)
        self.ax.add_patch(Circle((0.0, 0.0), CENTER_CIRCLE_RADIUS, fill=False, color="white", linewidth=1.5))

        self.ax.scatter(
            [PENALTY_X, -PENALTY_X],
            [0.0, 0.0],
            s=40,
            c=LANDMARK_COLORS["penalty_cross"],
            marker="P",
            zorder=3,
        )
        self.ax.scatter(
            [-FIELD_X_HALF, -FIELD_X_HALF, FIELD_X_HALF, FIELD_X_HALF],
            [FIELD_Y_HALF, -FIELD_Y_HALF, FIELD_Y_HALF, -FIELD_Y_HALF],
            s=35,
            c=LANDMARK_COLORS["corners"],
            marker="s",
            alpha=0.55,
            zorder=3,
        )
        self.ax.scatter(
            [-FIELD_X_HALF, FIELD_X_HALF],
            [0.0, 0.0],
            s=45,
            c=LANDMARK_COLORS["goal"],
            marker="X",
            alpha=0.75,
            zorder=3,
        )

        self.ax.add_patch(
            Rectangle(
                (-FIELD_X_HALF - GOAL_DEPTH, -GOAL_HALF_WIDTH),
                GOAL_DEPTH,
                2 * GOAL_HALF_WIDTH,
                fill=False,
                edgecolor="#dddddd",
                linewidth=1.5,
            )
        )
        self.ax.add_patch(
            Rectangle(
                (FIELD_X_HALF, -GOAL_HALF_WIDTH),
                GOAL_DEPTH,
                2 * GOAL_HALF_WIDTH,
                fill=False,
                edgecolor="#dddddd",
                linewidth=1.5,
            )
        )
        self.ax.grid(color="white", alpha=0.12, linewidth=0.7)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_title("Waiting for Webots controller data...")

    def _draw_legend(self):
        legend_items = [
            Line2D([0], [0], marker="o", color="w", label="Actual robot", markerfacecolor="#2ca02c", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Fused robot", markerfacecolor="#d62728", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Raw localizer", markerfacecolor="#1f77b4", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Actual ball", markerfacecolor="#ffdd57", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Estimated ball", markerfacecolor="#ff7f0e", markersize=9),
            Line2D([0], [0], marker=".", color="#666666", label="Pose hypotheses", markersize=8, linestyle="None"),
            Line2D([0], [0], marker="o", color="#ffffff", label="All estimated landmarks", markerfacecolor="none", markersize=7, linestyle="None"),
            Line2D([0], [0], marker="o", color="#ffffff", label="Estimated in FOV", markerfacecolor="#ffffff", markersize=7, linestyle="None"),
            Line2D([0], [0], marker="o", color="w", label="Projected landmark obs", markerfacecolor="#ffffff", markersize=7),
            Line2D([0], [0], marker="x", color="#ffffff", label="Matched field landmark", markersize=8, linestyle="None"),
            Line2D([0], [0], color="#00e5ff", label="Planner robot path", linewidth=2.2),
            Line2D([0], [0], color="#f2f2f2", label="Planned push lane", linewidth=1.8, linestyle="--"),
            Line2D([0], [0], color="#ff4fd8", label="Inflated ball obstacle", linewidth=1.5, linestyle=":"),
            Line2D([0], [0], marker="D", color="w", label="Planner target", markerfacecolor="#00e5ff", markersize=7),
            Line2D([0], [0], marker="s", color="w", label="Ball staging point", markerfacecolor="#ff4fd8", markersize=7),
        ]
        self.ax.legend(handles=legend_items, loc="upper center", ncol=4, framealpha=0.9)

    def _stdin_reader(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if payload.get("type") == "shutdown":
                self.shutdown_requested = True
                break

            while True:
                try:
                    self.snapshot_queue.get_nowait()
                except queue.Empty:
                    break
            self.snapshot_queue.put(payload)
        self.stdin_closed = True
        self.shutdown_requested = True

    def _tick(self):
        updated = False
        while True:
            try:
                self.latest_snapshot = self.snapshot_queue.get_nowait()
                updated = True
            except queue.Empty:
                break
        if updated and self.latest_snapshot is not None:
            self._render_snapshot(self.latest_snapshot)
        if self.shutdown_requested and self.snapshot_queue.empty():
            self._shutdown()
            return False
        return True

    def _clear_dynamic_artists(self):
        for artist in self.dynamic_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.dynamic_artists = []

    def _plot_points(self, points, color, marker, size, label_prefix=None, alpha=1.0, zorder=5):
        if not points:
            return
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        artist = self.ax.scatter(xs, ys, c=color, marker=marker, s=size, alpha=alpha, zorder=zorder)
        self.dynamic_artists.append(artist)
        if label_prefix is not None:
            for idx, point in enumerate(points):
                label = self.ax.text(
                    point[0] + 0.05,
                    point[1] + 0.05,
                    f"{label_prefix}{idx + 1}",
                    fontsize=7,
                    color=color,
                    zorder=zorder + 1,
                )
                self.dynamic_artists.append(label)

    def _draw_pose(self, pose, color, label, alpha=1.0, zorder=6):
        if pose is None:
            return
        x, y, theta = pose
        body = self.ax.scatter([x], [y], c=color, s=80, marker="o", edgecolors="black", linewidths=0.6, zorder=zorder, alpha=alpha)
        self.dynamic_artists.append(body)
        dx = 0.35 * math.cos(theta)
        dy = 0.35 * math.sin(theta)
        arrow = self.ax.arrow(
            x,
            y,
            dx,
            dy,
            width=0.03,
            head_width=0.16,
            head_length=0.14,
            length_includes_head=True,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
        self.dynamic_artists.append(arrow)
        text = self.ax.text(x + 0.1, y - 0.18, label, fontsize=8, color=color, weight="bold", zorder=zorder + 1)
        self.dynamic_artists.append(text)

    def _draw_ball(self, point, color, label, zorder=6):
        if point is None:
            return
        artist = self.ax.scatter([point[0]], [point[1]], c=color, s=95, marker="o", edgecolors="black", linewidths=0.6, zorder=zorder)
        self.dynamic_artists.append(artist)
        text = self.ax.text(point[0] + 0.08, point[1] + 0.08, label, fontsize=8, color=color, zorder=zorder + 1)
        self.dynamic_artists.append(text)

    def _is_point(self, point):
        return (
            isinstance(point, (list, tuple))
            and len(point) >= 2
            and point[0] is not None
            and point[1] is not None
        )

    def _draw_plan_line(self, points, color, linewidth=2.0, linestyle="-", alpha=1.0, zorder=6):
        valid = [point for point in points if self._is_point(point)]
        if len(valid) < 2:
            return
        artist = self.ax.plot(
            [point[0] for point in valid],
            [point[1] for point in valid],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )[0]
        self.dynamic_artists.append(artist)

    def _draw_plan_marker(self, point, color, marker, label, size=85, zorder=9):
        if not self._is_point(point):
            return
        artist = self.ax.scatter(
            [point[0]],
            [point[1]],
            c=color,
            s=size,
            marker=marker,
            edgecolors="black",
            linewidths=0.7,
            zorder=zorder,
        )
        self.dynamic_artists.append(artist)
        text = self.ax.text(
            point[0] + 0.07,
            point[1] - 0.16,
            label,
            fontsize=8,
            color=color,
            weight="bold",
            zorder=zorder + 1,
        )
        self.dynamic_artists.append(text)

    def _draw_motion_plan(self, snapshot):
        plan = snapshot.get("motion_plan") or {}
        if not plan:
            return

        robot_pose = snapshot.get("estimated_pose")
        robot_xy = robot_pose[:2] if self._is_point(robot_pose) else None
        ball = snapshot.get("estimated_ball")
        target = plan.get("target")
        staging = plan.get("staging")
        goal_target = plan.get("goal_target")
        path = plan.get("path") or []
        obstacle_radius = plan.get("obstacle_radius")

        robot_path = [robot_xy]
        if path:
            robot_path.extend(point for point in path if self._is_point(point))
        else:
            if self._is_point(target):
                robot_path.append(target)
            if self._is_point(staging) and target != staging:
                robot_path.append(staging)
        self._draw_plan_line(robot_path, color="#00e5ff", linewidth=2.2, zorder=6)

        if self._is_point(ball) and obstacle_radius is not None:
            obstacle = Circle(
                (ball[0], ball[1]),
                float(obstacle_radius),
                fill=False,
                edgecolor="#ff4fd8",
                linewidth=1.5,
                linestyle=":",
                alpha=0.95,
                zorder=6,
            )
            self.ax.add_patch(obstacle)
            self.dynamic_artists.append(obstacle)

        if self._is_point(staging) and self._is_point(ball):
            self._draw_plan_line([staging, ball], color="#ff4fd8", linewidth=1.7, linestyle="--", alpha=0.85, zorder=6)
        if self._is_point(ball) and self._is_point(goal_target):
            self._draw_plan_line([ball, goal_target], color="#f2f2f2", linewidth=1.8, linestyle="--", alpha=0.95, zorder=6)
        elif plan.get("mode") in ("push", "finish") and self._is_point(ball) and self._is_point(target):
            self._draw_plan_line([ball, target], color="#f2f2f2", linewidth=1.8, linestyle="--", alpha=0.95, zorder=6)

        self._draw_plan_marker(target, color="#00e5ff", marker="D", label="plan target", size=78, zorder=10)
        self._draw_plan_marker(staging, color="#ff4fd8", marker="s", label="staging", size=78, zorder=9)
        if self._is_point(goal_target):
            self._draw_plan_marker(goal_target, color="#f2f2f2", marker="*", label="goal target", size=115, zorder=9)

    def _render_snapshot(self, snapshot):
        self._clear_dynamic_artists()

        hypotheses = snapshot.get("hypotheses") or []
        if hypotheses:
            hypothesis_points = [[pose[0], pose[1]] for pose in hypotheses]
            self._plot_points(hypothesis_points, color="#666666", marker=".", size=18, alpha=0.45, zorder=4)

        landmark_points = snapshot.get("landmark_points", {})
        all_landmarks = landmark_points.get("all", {})
        visible = landmark_points.get("visible", {})
        projected = landmark_points.get("projected", {})
        matched = landmark_points.get("matched", {})

        for name, color in LANDMARK_COLORS.items():
            self._plot_points(all_landmarks.get(name, []), color=color, marker="o", size=88, alpha=0.18, zorder=4)
            self._plot_points(all_landmarks.get(name, []), color=color, marker="o", size=28, alpha=0.45, zorder=4)
            self._plot_points(visible.get(name, []), color=color, marker="o", size=120, alpha=0.18, zorder=5)
            self._plot_points(visible.get(name, []), color=color, marker="o", size=52, alpha=1.0, zorder=6)
            self._plot_points(projected.get(name, []), color=color, marker="o", size=55, alpha=0.9, zorder=6)
            self._plot_points(matched.get(name, []), color=color, marker="x", size=90, alpha=1.0, zorder=7)

        self._draw_ball(snapshot.get("actual_ball"), color="#ffdd57", label="ball true", zorder=7)
        self._draw_ball(snapshot.get("estimated_ball"), color="#ff7f0e", label="ball est", zorder=8)
        self._draw_motion_plan(snapshot)
        self._draw_pose(snapshot.get("actual_pose"), color="#2ca02c", label="robot true", alpha=0.9, zorder=8)
        self._draw_pose(snapshot.get("localizer_pose"), color="#1f77b4", label="localizer", alpha=0.65, zorder=8)
        self._draw_pose(snapshot.get("estimated_pose"), color="#d62728", label="fused", alpha=0.9, zorder=9)

        state = snapshot.get("state", "UNKNOWN")
        confidence = snapshot.get("confidence", 0.0)
        localization = snapshot.get("localization") or {}
        fused = localization.get("fused") or {}
        localizer = localization.get("localizer") or {}
        landmark = localization.get("landmark_bin", "unknown")
        plan = snapshot.get("motion_plan") or {}
        plan_mode = plan.get("mode", "none")
        trust = localization.get("pose_correction_trust", 0.0)
        ess = localization.get("effective_sample_size_norm", 0.0)
        fused_error = fused.get("pos_error")
        raw_error = localizer.get("pos_error")
        fused_text = "?" if fused_error is None else f"{fused_error:.2f}m"
        raw_text = "?" if raw_error is None else f"{raw_error:.2f}m"
        self.ax.set_title(
            f"State: {state} | plan={plan_mode} | bin={landmark} | conf={confidence:.2f} "
            f"trust={trust:.2f} ESS={ess:.2f} | fused err={fused_text} raw err={raw_text}"
        )
        self.fig.canvas.draw_idle()

    def _shutdown(self):
        if self.shutting_down:
            return
        self.shutting_down = True
        try:
            self.timer.stop()
        except Exception:
            pass
        plt.close(self.fig)

    def _on_close(self, _event):
        self.shutting_down = True
        try:
            self.timer.stop()
        except Exception:
            pass

    def run(self):
        try:
            plt.show()
        finally:
            self._shutdown()


def main():
    app = LocalizationVisualizer()
    app.run()


if __name__ == "__main__":
    main()
