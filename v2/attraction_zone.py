import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory

frame_amount = 2000
orbit_initial_conditions_size = 2000
S3_sample_initial_condition_size = 5000
epsilon = 0.001
head_start = 1000000 #100000 empieza lo bueno
Lambda = 0.24#0.202288
fixed_point_tol = 0.0001
orbit_tol = 0.008
intermidiate_steps = 20
max_iter = 200000
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
    random_conditions = point_factory.randomly_sample_S3(orbit_initial_conditions_size)
    black_points = np.load('black_points.npy')
    random_conditions = np.vstack((random_conditions, black_points))

    trajectories, _, _, line_connections = (
        parallel_program.point_trajectories(
            N,
            a,
            b,
            head_start,
            frame_amount,
            random_conditions,
            epsilon,
            intermidiate_steps,
        )
    )
    orbit_heads, line_colors = helper_functions.find_orbit_heads(trajectories, orbit_tol)

    S3_sample = point_factory.randomly_sample_S3(S3_sample_initial_condition_size)

    head_indices, S3_sample_colors = parallel_program.find_corresponding_head(
        N,
        a,
        b,
        head_start,
        S3_sample,
        orbit_heads,
        orbit_tol * orbit_tol,
        max_iter,
        epsilon,
    )

    # black_points = S3_sample[np.where(head_indices == -1)]

    # np.save('black_points.npy', black_points)

    # print(black_points)

    return (
        trajectories,
        line_colors[: orbit_initial_conditions_size],
        line_colors,
        line_connections,
        S3_sample,
        S3_sample_colors,
        head_indices,
    )

def render_experiment(trajectories, color_matrix, line_colors, line_connections, S3_sample, S3_sample_colors, head_indices):
    object_list = []
    binding_list = []
    color_matrix[:, -1] = 1

    trajectories_projection = helper_functions.sterographic_projection(trajectories)
    integral_curves = plot.Line(pos = trajectories_projection, color = line_colors, width = 3, connect = line_connections)
    integral_curves.set_gl_state(preset = 'opaque')
    object_list.append(integral_curves)

    S3_sample_projection = helper_functions.sterographic_projection(S3_sample)
    S3_sample_points = plot.Scatter3D(
        pos = S3_sample_projection,
        face_color=S3_sample_colors,
        symbol="o",
        size=2,
        edge_color=S3_sample_colors,
    )
    S3_sample_points.set_gl_state(preset = 'opaque')
    object_list.append(S3_sample_points)

    binding_list.append(
        bindings.ChangeProjection(
            lines=[integral_curves],
            line_coords=[trajectories],
            points=[S3_sample_points],
            point_coords=[S3_sample],
            point_colors=[S3_sample_colors],
        )
    )
    binding_list.append(bindings.Toggle([integral_curves], 'l'))
    binding_list.append(bindings.Toggle([S3_sample_points], 'space'))
    binding_list.append(bindings.ChangeOrigin(S3_sample_points, head_indices, 'o'))

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list)
    
    experiment_plot.show()


if __name__ == '__main__':
    render_experiment(*calculate_experiment())