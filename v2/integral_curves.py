import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory

frame_amount = 400
initial_conditions_size = 20
epsilon = 0.001
head_start = 0 #100000 empieza lo bueno
Lambda = 0.24#0.202288
fixed_point_tol = 0.0001
orbit_tol = 0.01
intermidiate_steps = 10
N = 3

np.random.seed(2901)

a = np.random.uniform(-5, 5, (N, N)) + 1.j * np.random.uniform(-5, 5, (N, N))
a[0, 0] = 0
a[1, 0] = 1 + .3j
# a[1, 0] = 0
a[1, 1] = 0
a[2, 0] = 2.92 + 1.71j
a[2, 1] = 2.46 + 1.85j
# a[2, 1] = 0
a[2, 2] = 1.16 + 2.84j
a[2, :] *= Lambda
a = a[np.tril_indices(N)].astype(np.csingle)

b =  np.random.uniform(-5, 5, (N, N)) + 1.j * np.random.uniform(-5, 5, (N, N))
b[0, 0] = 0
b[1, 0] = 0
b[1, 1] = 1.j
# b[1, 1] = 0
b[2, 0] = 1.57 + 2.92j
b[2, 1] = 1.20 + 1.31j
# b[2, 1] = 0
b[2, 2] = 1.15 + 1.34j
b[2, :] *= Lambda
b = b[np.tril_indices(N)].astype(np.csingle)

helper_functions.print_coefficients(N, a, b)


def calculate_experiment():
    # initial_conditions = point_factory.randomly_sample_S3(initial_conditions_size)
    border_points = np.load('border_point_coords.npy')
    # initial_conditions = np.vstack((initial_conditions, black_points))
    initial_conditions = border_points
    color_matrix = np.zeros((len(initial_conditions), 4), dtype=np.float32)
    color_matrix[:] = np.array([0, 0, 0, 0.1])

    return parallel_program.point_trajectories(
            N,
            a,
            b,
            head_start,
            frame_amount,
            initial_conditions,
            epsilon,
            intermidiate_steps,
            color_matrix,
        )


def render_experiment(trajectories, color_matrix, line_colors, line_connections):
    object_list = []
    binding_list = []
    color_matrix[:, -1] = 1

    trajectories_projection = helper_functions.sterographic_projection(trajectories)
    integral_curves = plot.Line(pos = trajectories_projection, color = line_colors, width = 3, connect = line_connections)
    integral_curves.set_gl_state(preset = "translucent", depth_test=False)
    object_list.append(integral_curves)

    # diff = trajectories[-1] - trajectories[-2]
    # dist = np.linalg.norm(diff, axis=1)
    # fixed_indices = np.where(dist < fixed_point_tol)
    # fixed_point_coords = trajectories[-1][fixed_indices]
    # fixed_point_list = []
    # if len(fixed_point_coords) > 0:
    #     fixed_points = plot.Scatter3D(
    #         pos = trajectories_projection[-1][fixed_indices],
    #         face_color=color_matrix[fixed_indices],
    #         symbol="o",
    #         size=2,
    #         edge_color=color_matrix[fixed_indices]
    #     )
    #     fixed_points.set_gl_state(preset = "translucent", depth_test=False)
    #     object_list.append(fixed_points)
    #     fixed_point_list.append(fixed_points)

    border_points = plot.Scatter3D(
        pos = trajectories_projection[0],
        face_color=[1,1,1],
        symbol="o",
        size=4,
        edge_color=color_matrix
    )
    object_list.append(border_points)

    binding_list.append(
        bindings.ChangeProjection(
            lines=[integral_curves],
            line_coords=[trajectories],
            # points=fixed_point_list,
            # point_coords=[fixed_point_coords],
            # point_colors=[color_matrix[fixed_indices]],
        )
    )
    binding_list.append(bindings.ChangeOpacity(object_list))

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list)
    experiment_plot.show()
    # experiment_plot.film_rotating_animation(900, 'Caso A/trajectories')


if __name__ == '__main__':
    render_experiment(*calculate_experiment())