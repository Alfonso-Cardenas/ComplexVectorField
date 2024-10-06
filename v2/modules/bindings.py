from itertools import cycle
import numpy as np

from modules import constants, helper_functions, parallel_program
from vispy import visuals


class Binding():
    def __init__(
        self,
        objects,
        key: str,
        function,
    ):
        self.objects = objects
        self.function = function
        self.key = key

    def act(self):
        self.function(self.objects)


class Toggle(Binding):
    def __init__(self, objects: list[visuals.Visual], key: str):
        super().__init__(objects, key, self.toggle)

    def toggle(self, objects: list[visuals.Visual]) -> None:
        for o in objects:
            o.visible = not o.visible


class ChangeProjection(Binding):
    def __init__(self, lines=[], line_coords=[], points=[], point_coords=[], point_colors=[], key: str='p'):
        self.lines = lines
        self.points = points
        self.line_coords = line_coords
        self.point_coords = point_coords
        self.point_colors = point_colors
        self.key = key
        self.projection_cycle = cycle(constants.PROJECTIONS)

    def act(self):
        projection = next(self.projection_cycle)

        for line, line_coords in zip(self.lines, self.line_coords):
            line_projection = helper_functions.sterographic_projection(
                line_coords, projection
            )
            line.set_data(pos=line_projection)

        for points, point_coords, colors in zip(self.points, self.point_coords, self.point_colors):
            size = np.array([d[3] for d in points._data])
            point_projections = helper_functions.sterographic_projection(
                point_coords, projection
            )
            points.set_data(pos=point_projections, size=size, face_color=colors, edge_color=colors)


class ChangeOpacity(Binding):
    def __init__(self, objects=[], key='l'):
        self.objects = objects
        self.preset_cycle = cycle(constants.OPACITY_PRESETS)
        self.key = key

    def act(self):
        preset = next(self.preset_cycle)
        for o in self.objects:
            o.set_gl_state(**preset)
        self.objects[0].parent.update()


class ChangeOrigin(Binding):
    def __init__(self, points, head_indices, key='o'):
        self.points = points
        self.head_indices = head_indices
        self.index = cycle(list(range(-2, max(head_indices) + 1)))
        next(self.index)
        self.pos = np.array([d[0] for d in self.points._data])
        self.colors = np.array([d[1] for d in self.points._data])
        self.key = key

    def act(self):
        index = next(self.index)
        if index == -2:
            size = 2
        else:
            size = np.zeros_like(self.head_indices)
            size[np.where(self.head_indices == index)] = 2
        self.points.set_data(pos=self.pos, size=size, face_color=self.colors, edge_color=self.colors)


class RecalcTrajectories():
    def __init__(
        self,
        initial_conditions,
        a,
        b,
        N,
        head_start,
        frame_amount,
        epsilon,
        intermediate_steps,
        fixed_points,
        lines,
        coefficient_exploration_speed,
        key='s'
    ):
        self.key = key
        self.initial_conditions = initial_conditions
        self.a = a
        self.b = b
        self.N = N
        self.head_start = head_start
        self.frame_amount = frame_amount
        self.epsilon = epsilon
        self.intermediate_steps = intermediate_steps
        self.lines = lines
        self.fixed_points = fixed_points
        self.fixed_points_colors = np.array([d[1] for d in fixed_points._data])
        self.coefficient_exploration_speed = coefficient_exploration_speed

    def act(self):
        self.a += (
            np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
            + 1.j * np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
        )
        self.b += (
            np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
            + 1.j * np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
        )

        helper_functions.print_coefficients(self.N, self.a, self.b)

        trajectories = parallel_program.point_trajectories_pos_and_neg(
            self.N,
            self.a,
            self.b,
            self.head_start,
            self.frame_amount,
            self.initial_conditions,
            self.epsilon,
            self.intermediate_steps
        )[0]
        trajectories_projection = helper_functions.sterographic_projection(trajectories)

        diff = trajectories[-1] - trajectories[-2]
        dist = np.linalg.norm(diff, axis=1)
        fixed_indices = np.where(dist < 0.0001)
        fixed_points_size = np.zeros(trajectories.shape[1])
        fixed_points_size[fixed_indices] = 2
        self.fixed_points.set_data(
            pos=trajectories_projection[-1],
            size=fixed_points_size,
            face_color=self.fixed_points_colors,
            edge_color=self.fixed_points_colors
        )
        self.lines.set_data(pos=trajectories_projection)


class RecalcTrajectoriesCuadratic():
    def __init__(
        self,
        initial_conditions,
        a,
        b,
        N,
        head_start,
        frame_amount,
        epsilon,
        intermediate_steps,
        fixed_points,
        lines,
        coefficient_exploration_speed,
        key='s'
    ):
        self.key = key
        self.initial_conditions = initial_conditions
        self.a = a
        self.b = b
        self.N = N
        self.head_start = head_start
        self.frame_amount = frame_amount
        self.epsilon = epsilon
        self.intermediate_steps = intermediate_steps
        self.lines = lines
        self.fixed_points = fixed_points
        self.fixed_points_colors = np.array([d[1] for d in fixed_points._data])
        self.coefficient_exploration_speed = coefficient_exploration_speed

    def act(self):
        self.a[[3, 5]] += (
            np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
            + 1.j * np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
        )
        self.b[[3, 5]] += (
            np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
            + 1.j * np.random.uniform(-self.coefficient_exploration_speed, self.coefficient_exploration_speed, 2)
        )

        helper_functions.print_coefficients(self.N, self.a, self.b)

        trajectories = parallel_program.point_trajectories_pos_and_neg(
            self.N,
            self.a,
            self.b,
            self.head_start,
            self.frame_amount,
            self.initial_conditions,
            self.epsilon,
            self.intermediate_steps
        )[0]
        trajectories_projection = helper_functions.sterographic_projection(trajectories)

        diff = trajectories[-1] - trajectories[-2]
        dist = np.linalg.norm(diff, axis=1)
        fixed_indices = np.where(dist < 0.0001)
        fixed_points_size = np.zeros(trajectories.shape[1])
        fixed_points_size[fixed_indices] = 2
        self.fixed_points.set_data(
            pos=trajectories_projection[-1],
            size=fixed_points_size,
            face_color=self.fixed_points_colors,
            edge_color=self.fixed_points_colors
        )
        self.lines.set_data(pos=trajectories_projection)