import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory
from vispy.scene.visuals import Mesh
import imageio
from tqdm import tqdm

frame_amount = 2000
orbit_frame_amount = 2000
orbit_initial_conditions_size = 20
epsilon = 0.001
head_start = 0 #100000 empieza lo bueno
Lambda = 0.24#0.202288
intermidiate_steps = 20
max_iter = 200000
N = 3
torus_points_per_step = 40
orbit_tol = 0.01
torus_radius = 0.2

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
    torus_points = np.load("torus_points.npy")
    color_matrix = np.zeros((len(torus_points), 4), dtype=np.float32)
    color_matrix[:, 3] = 1

    trajectory, color_matrix, line_colors, line_connections = (
        parallel_program.point_trajectories(
            N,
            a,
            b,
            head_start,
            frame_amount,
            helper_functions.inverse_stereographic_projection(torus_points),
            -epsilon,
            intermidiate_steps,
            color_matrix,
        )
    )

    return (
        trajectory,
        color_matrix,
        line_colors,
        line_connections,
        # torus_points,
        # torus_colors,
    )

def render_experiment(
    trajectory,
    color_matrix,
    line_colors,
    line_connections,
    # torus_points,
    # torus_colors,
):
    object_list = []
    binding_list = []

    trajectory_projection = helper_functions.sterographic_projection(trajectory)
    line_connections = line_connections.reshape(1999, -1, 2)
    integral_curves = plot.Line(pos = trajectory_projection, color = line_colors, width = 3, connect = line_connections[:, :40].reshape(-1, 2))
    integral_curves.set_gl_state(preset = 'opaque')
    object_list.append(integral_curves)

    # vertices = np.load('torus_points.npy')
    # colors = np.load('torus_colors.npy')
    # faces = np.load('torus_faces.npy')

    # torus = Mesh(
    #     vertices=vertices,
    #     faces=faces,
    #     vertex_colors=colors,
    # )
    # object_list.append(torus)


    binding_list.append(bindings.Toggle([integral_curves], 'l'))
    # binding_list.append(bindings.Toggle([torus], 't'))

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list, decorate=False, size=(1920, 1088), show=False, autoswap=False, vsync=False)
    
    # experiment_plot.show()
    writer = imageio.get_writer('animations/torus_transversality_check.mp4', fps = 30)
    for starting_index in tqdm(range(0, trajectory_projection.shape[1], 40)):
        integral_curves.set_data(connect=line_connections[:, starting_index:starting_index + 40].reshape(-1, 2))
        experiment_plot.film_partial_stationary_animation(writer)


if __name__ == '__main__':
    render_experiment(*calculate_experiment())