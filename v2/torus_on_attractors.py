import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory, constants
from vispy.scene.visuals import Mesh

frame_amount = 2773
orbit_frame_amount = 2000
orbit_initial_conditions_size = 20
epsilon = 0.001
head_start = 1000000 #100000 empieza lo bueno
Lambda = 0.24#0.202288
intermidiate_steps = 5
max_iter = 200000
N = 3
torus_points_per_step = 80
orbit_tol = 0.01
torus_radius = 0.01

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

def include_integral_curves(object_list, binding_list, names, binding_keys, colors):
    for name, binding_key, color in zip(names, binding_keys, colors):
        trajectory = np.load(f'saved_arrays/{name}_trajectory.npy')
        line_connections = helper_functions.calc_line_connections(trajectory.shape)

        trajectory_projection = helper_functions.sterographic_projection(trajectory)
        integral_curves = plot.Line(pos = trajectory_projection, color = color, width = 3, connect = line_connections)
        integral_curves.set_gl_state(preset = 'opaque')
        object_list.append(integral_curves)

        binding_list.append(bindings.Toggle([integral_curves], binding_key))

def include_toruses(object_list, binding_list, names, binding_keys, colors, h_norm_thresholds, epsilons):
    for name, binding_key, color, h_norm_threshold, eps in zip(names, binding_keys, colors, h_norm_thresholds, epsilons):
        vertices = np.load(f'saved_arrays/{name}_torus_points.npy')
        faces = np.load(f'saved_arrays/{name}_torus_faces.npy')
        try:
            h_norms = np.load(f'saved_arrays/{name}_torus_h_norms.npy')
        except Exception:
            h_norms = parallel_program.find_min_h_norm(N, a, b, max_iter, helper_functions.inverse_stereographic_projection(vertices), eps)
            np.save(f'saved_arrays/{name}_torus_h_norms.npy', h_norms)

        h_norms[np.where(h_norms < h_norm_threshold)] = 0 
        h_norms[np.where(h_norms >= h_norm_threshold)] = 1 

        torus = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=color * h_norms[:, None],
        )
        object_list.append(torus)
        torus.set_gl_state(preset = 'opaque')

        binding_list.append(bindings.Toggle([torus], binding_key))

def create_toruses(radius):
    for name in ['small_attractor', 'big_attractor', 'repulsor']:

        trajectory = np.load(f'saved_arrays/{name}_trajectory.npy').reshape(-1, 2)
        trajectory_projection = helper_functions.sterographic_projection(trajectory)

        torus_points = helper_functions.calc_torus_points(trajectory_projection, torus_points_per_step, radius)
        np.save(f'saved_arrays/{name}_torus_points', torus_points)

        faces = helper_functions.calc_torus_faces(len(trajectory_projection) - 1, torus_points_per_step)
        np.save(f'saved_arrays/{name}_torus_faces', faces)

def calculate_experiment():
    pass

    # trajectory, color_matrix, line_colors, line_connections = (
    #     parallel_program.point_trajectories(
    #         N,
    #         a,
    #         b,
    #         head_start,
    #         1309,
    #         point_factory.randomly_sample_S3(20)[1:2],
    #         -epsilon,
    #         intermidiate_steps,
    #     )
    # )

    # np.save('saved_arrays/repulsor_trajectory', trajectory)
    # exit()

def render_experiment(
):
    object_list = []
    binding_list = []

    include_integral_curves(
        object_list,
        binding_list,
        ['small_attractor', 'big_attractor', 'repulsor'],
        ['s', 'b', 'r'],
        [
            constants.COLORS[1] * 0.8,
            constants.COLORS[0] * 0.8,
            [0, 0, 0.8],
        ],
    )

    for o in object_list:
        o.visible = False

    include_toruses(
        object_list,
        binding_list,
        ['small_attractor', 'big_attractor', 'repulsor'],
        ['1', '2', '3'],
        [
            np.tile([0, 0, 1], [(len(object_list[0].pos) - 1) * torus_points_per_step, 1]),
            np.tile([0, 0, 1], [(len(object_list[1].pos) - 1) * torus_points_per_step, 1]),
            np.load('saved_arrays/repulsor_torus_colors.npy')
        ],
        [0.18, 0.15, 0.08],
        [-epsilon, -epsilon, epsilon],
    )

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list)
    
    experiment_plot.show()


if __name__ == '__main__':
    # create_toruses(torus_radius)
    render_experiment()