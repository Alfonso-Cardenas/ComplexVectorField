import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory

frame_amount = 2000
initial_conditions_size = 100
epsilon = 0.001
head_start = 2000000 #100000 empieza lo bueno
fixed_point_tol = 0.0001
gamma = np.random.uniform(-5, 5) + np.random.uniform(-5, 5) * 1.j
beta = np.random.uniform(-5, 5) + np.random.uniform(-5, 5) * 1.j
orbit_tol = 0.01
intermidiate_steps = 20
N = 3

# np.random.seed(2901)

a = np.zeros((N, N), dtype=np.csingle)
a[2,1] = -beta
a = a[np.tril_indices(N)].astype(np.csingle)

b = np.zeros((N, N), dtype=np.csingle)
b[1,1] = -gamma
b[2,1] = beta
b = b[np.tril_indices(N)].astype(np.csingle)

helper_functions.print_coefficients(N, a, b)


def calculate_experiment():
    initial_conditions = point_factory.randomly_sample_S3(initial_conditions_size)

    return parallel_program.point_trajectories_pos_and_neg_not_tangent(
            N,
            a,
            b,
            head_start,
            frame_amount,
            initial_conditions,
            epsilon,
            intermidiate_steps,
        )


def render_experiment(trajectories, color_matrix, line_colors, line_connections):
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
    fixed_point_coords = trajectories[-1][fixed_indices]
    fixed_point_list = []
    if len(fixed_point_coords) > 0:
        fixed_points = plot.Scatter3D(
            pos = trajectories_projection[-1][fixed_indices],
            face_color=color_matrix[fixed_indices],
            symbol="o",
            size=2,
            edge_color=color_matrix[fixed_indices]
        )
        fixed_points.set_gl_state(preset = "translucent", depth_test=False)
        object_list.append(fixed_points)
        fixed_point_list.append(fixed_points)

    binding_list.append(
        bindings.ChangeProjection(
            lines=[integral_curves],
            line_coords=[trajectories],
            points=fixed_point_list,
            point_coords=[fixed_point_coords],
            point_colors=[color_matrix[fixed_indices]],
        )
    )
    binding_list.append(bindings.ChangeOpacity(object_list))

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list)
    experiment_plot.show()


if __name__ == '__main__':
    render_experiment(*calculate_experiment())