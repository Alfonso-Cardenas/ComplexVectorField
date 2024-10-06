import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory

frame_amount = 4000
initial_conditions_size = 50
epsilon = 0.001
head_start = 1000000 #100000 empieza lo bueno
Lambda = 0.24#0.202288
fixed_point_tol = 0.0001
orbit_tol = 0.01
intermidiate_steps = 20
coefficient_exploration_speed = 0.05
N = 3

np.random.seed(2901)

a = np.random.uniform(-5, 5, (N, N)) + 1.j * np.random.uniform(-5, 5, (N, N))
a[0, 0] = 0
a[1, 0] = 0
a[1, 1] = 0
a[2, 0] = -0.05 + 0.3j
a[2, 1] = 0
a[2, 2] = 0.54 - 0.46j
a = a[np.tril_indices(N)].astype(np.csingle)

b =  np.random.uniform(-5, 5, (N, N)) + 1.j * np.random.uniform(-5, 5, (N, N))
b[0, 0] = 0
b[1, 0] = 0
b[1, 1] = 0
b[2, 0] = 0.3 + 0.27j
b[2, 1] = 0
b[2, 2] = 0.1 - 0.72j
b = b[np.tril_indices(N)].astype(np.csingle)

# a = np.random.uniform(-5, 5, (N, N)) + 1.j * np.random.uniform(-5, 5, (N, N))
# a[0, 0] = 0
# a[1, 0] = 1 + .3j
# a[1, 1] = 0
# a[2, 0] = 2.92 + 1.71j
# a[2, 1] = 2.46 + 1.85j
# a[2, 2] = 1.16 + 2.84j
# a[2, :] *= Lambda
# a = a[np.tril_indices(3)].astype(np.csingle)

# b =  np.random.uniform(-5, 5, (N, N)) + 1.j * np.random.uniform(-5, 5, (N, N))
# b[0, 0] = 0
# b[1, 0] = 0
# b[1, 1] = 1.j
# b[2, 0] = 1.57 + 2.92j
# b[2, 1] = 1.20 + 1.31j
# b[2, 2] = 1.15 + 1.34j
# b[2, :] *= Lambda
# b = b[np.tril_indices(3)].astype(np.csingle)

helper_functions.print_coefficients(N, a, b)


def calculate_experiment(initial_conditions = None):
    if initial_conditions is None:
        initial_conditions = point_factory.randomly_sample_S3(initial_conditions_size)

    trajectories, color_matrix, line_colors, line_connections = (
        parallel_program.point_trajectories_pos_and_neg(
            N,
            a,
            b,
            head_start,
            frame_amount,
            initial_conditions,
            epsilon,
            intermidiate_steps,
        )
    )

    return initial_conditions, trajectories, color_matrix, line_colors, line_connections


def render_experiment(initial_conditions, trajectories, color_matrix, line_colors, line_connections):
    global a, b
    object_list = []
    binding_list = []
    color_matrix[:, -1] = 1

    trajectories_projection = helper_functions.sterographic_projection(trajectories)
    integral_curves = plot.Line(pos = trajectories_projection, color = line_colors, width = 3, connect = line_connections)
    integral_curves.set_gl_state(preset = "translucent", depth_test=False)
    object_list.append(integral_curves)

    diff = trajectories[-1] - trajectories[-2]
    dist = np.linalg.norm(diff, axis=1)
    fixed_indices = np.where(dist < fixed_point_tol)
    fixed_points_size = np.zeros(trajectories.shape[1])
    fixed_points_size[fixed_indices] = 2
    fixed_points = plot.Scatter3D(
        pos = trajectories_projection[-1],
        face_color=color_matrix,
        symbol="o",
        size=fixed_points_size,
        edge_color=color_matrix,
    )
    fixed_points.set_gl_state(preset = "translucent", depth_test=False)
    object_list.append(fixed_points)

    binding_list.append(bindings.ChangeOpacity(object_list))

    binding_list.append(bindings.RecalcTrajectoriesCuadratic(
        initial_conditions,
        a,
        b,
        N,
        head_start,
        frame_amount,
        epsilon,
        intermidiate_steps,
        fixed_points,
        integral_curves,
        coefficient_exploration_speed,
    ))

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list)
    experiment_plot.show()


if __name__ == '__main__':
    render_experiment(*calculate_experiment())