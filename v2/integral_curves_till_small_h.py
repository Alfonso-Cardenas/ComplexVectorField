import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory

frame_amount = 2000
initial_conditions_size = 20
epsilon = 0.001
head_start = 0#100000 empieza lo bueno
Lambda = 0.24#0.202288
fixed_point_tol = 0.0001
orbit_tol = 0.01
intermidiate_steps = 10
dead_point_sample=500000
green_point_amount = 2
samples_per_green_point = 50
green_point_index = 2
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


def get_black_points(name, h_norm_threshold):
    vertices = np.load(f'saved_arrays/{name}_torus_points.npy')
    h_norms = np.load(f'saved_arrays/{name}_torus_h_norms.npy')
    return vertices[np.where(h_norms < h_norm_threshold)]


def calculate_experiment():
    # initial_conditions = point_factory.randomly_sample_S3(initial_conditions_size)[0]
    # green_points = np.load("saved_arrays/green_points.npy")[1:green_point_amount]
    # initial_conditions = (
    #     np.tile(green_points, (samples_per_green_point, 1))
    #     + np.random.uniform(-0.1, 0.1, (samples_per_green_point * len(green_points), 2))
    #     + 1.j * np.random.uniform(-0.1, 0.1, (samples_per_green_point * len(green_points), 2))
    # )
    # initial_conditions = (initial_conditions / np.linalg.norm(initial_conditions, axis=1)[:, None]).astype(np.csingle)
    # black_points = np.load('saved_arrays/black_points.npy')
    # initial_conditions = np.vstack((initial_conditions, black_points))
    # initial_conditions = border_points
    # print(len(border_points))
    black_points = []
    epsilons = []
    for name, h_norm_threshold, eps in zip(
        [
            # 'small_attractor',
            # 'big_attractor',
            'repulsor',
        ],
        [
            # 0.05,
            # 0.05,
            0.05,
        ],
        [
            # -epsilon,
            # -epsilon,
            epsilon,
        ]
    ):
        black_points.append(get_black_points(name, h_norm_threshold))
        epsilons += [eps] * len(black_points[-1])

    epsilons = np.array(epsilons, dtype=np.float32)
    black_points = np.vstack(black_points)
    initial_conditions = helper_functions.inverse_stereographic_projection(black_points)

    color_matrix = np.zeros((len(initial_conditions), 4), dtype=np.float32)
    color_matrix[:, 3] = 0.1

    return parallel_program.point_trajectories_till_small_h(
        N,
        a,
        b,
        head_start,
        frame_amount,
        initial_conditions,
        epsilons,
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

    # initial_conditions = point_factory.randomly_sample_S3(initial_conditions_size)

    # trajectories2, color_matrix2, line_colors2, line_connections2 = parallel_program.point_trajectories_pos_and_neg(
    #     N,
    #     a,
    #     b,
    #     100000,
    #     800,
    #     initial_conditions,
    #     epsilon,
    #     intermidiate_steps,
    # )
    # trajectories_projection2 = helper_functions.sterographic_projection(trajectories2)
    # integral_curves2 = plot.Line(pos = trajectories_projection2, color = line_colors2, width = 3, connect = line_connections2)
    # integral_curves2.set_gl_state(preset = "opaque", depth_test=True)
    # object_list.append(integral_curves2)

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
        edge_color=color_matrix,
    )
    green_points = np.load("saved_arrays/green_points.npy")#[1:green_point_amount]
    border_endpoints = plot.Scatter3D(
        pos = helper_functions.sterographic_projection(green_points),
        face_color=[0,1,0],
        symbol="o",
        size=4,
        edge_color=[0,1,0]
    )
    # np.save("green_points.npy", trajectories_projection[-1])
    object_list.append(border_points)
    border_endpoints.set_gl_state(preset = "translucent")
    object_list.append(border_endpoints)

    # np.save("initial_conditions.npy", trajectories[len(trajectories)//2])

    # initial_conditions = plot.Scatter3D(
    #     pos = np.load("torus_black_points.npy")[:2000],
    #     face_color=[1,1,1],
    #     symbol="o",
    #     size=4,
    #     edge_color=[0,0,0],
    # )
    # object_list.append(initial_conditions)

    # green_points = helper_functions.calc_dead_points(point_factory.randomly_sample_S3(dead_point_sample), N, a, b, 0.01)
    # border_endpoints = plot.Scatter3D(
    #     pos = helper_functions.sterographic_projection(green_points),
    #     face_color=[0,1,0],
    #     symbol="o",
    #     size=4,
    #     edge_color=[0,1,0],
    # )
    # object_list.append(border_endpoints)

    # binding_list.append(
    #     bindings.ChangeProjection(
    #         lines=[integral_curves],
    #         line_coords=[trajectories],
    #         # points=fixed_point_list,
    #         # point_coords=[fixed_point_coords],
    #         # point_colors=[color_matrix[fixed_indices]],
    #     )
    # )
    binding_list.append(bindings.ChangeOpacity(object_list))
    binding_list.append(bindings.Toggle([integral_curves], 'space'))
    binding_list.append(bindings.Toggle([border_endpoints], 'p'))

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list)
    experiment_plot.show()
    # experiment_plot.film_rotating_animation(900, 'Caso A/trajectories')


if __name__ == '__main__':
    render_experiment(*calculate_experiment())