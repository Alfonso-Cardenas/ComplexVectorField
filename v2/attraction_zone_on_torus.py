import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory
from vispy.scene.visuals import Mesh

frame_amount = 1320
orbit_frame_amount = 2000
orbit_initial_conditions_size = 20
epsilon = 0.001
head_start = 1000000 #100000 empieza lo bueno
Lambda = 0.24#0.202288
intermidiate_steps = 5
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
    color_matrix = np.zeros((1, 4), dtype=np.float32)
    color_matrix[:, 2] = 1

    trajectory, color_matrix, line_colors, line_connections = (
        parallel_program.point_trajectories(
            N,
            a,
            b,
            head_start,
            frame_amount,
            point_factory.randomly_sample_S3(1),
            -epsilon,
            intermidiate_steps,
            color_matrix,
        )
    )

    random_conditions = point_factory.randomly_sample_S3(orbit_initial_conditions_size)
    attractor_trajectories, _, _, attractor_connections = (
        parallel_program.point_trajectories(
            N,
            a,
            b,
            head_start,
            orbit_frame_amount,
            random_conditions,
            epsilon,
            20,
        )
    )

    orbit_heads, attractor_colors = helper_functions.find_orbit_heads(attractor_trajectories, orbit_tol)

    # torus_points = helper_functions.calc_torus_points(
    #     helper_functions.sterographic_projection(trajectory).reshape(-1, 3),
    #     torus_points_per_step,
    #     torus_radius,
    # )

    torus_points = np.load('saved_arrays/repulsor_torus_points.npy')

    _, torus_colors, h_norms = parallel_program.find_corresponding_head(
        N,
        a,
        b,
        head_start,
        helper_functions.inverse_stereographic_projection(torus_points),
        orbit_heads,
        orbit_tol * orbit_tol,
        max_iter,
        epsilon,
    )

    # np.save("torus_points.npy", torus_points)
    np.save("saved_arrays/repulsor_torus_colors.npy", torus_colors)
    exit()
    # np.save("h_norms.npy", h_norms)
    # exit()

    return (
        trajectory,
        color_matrix,
        line_colors,
        line_connections,
        attractor_trajectories,
        attractor_colors,
        attractor_connections,
        # torus_points,
        # torus_colors,
    )

def render_experiment(
    trajectory,
    color_matrix,
    line_colors,
    line_connections,
    attractor_trajectories,
    attractor_colors,
    attractor_connections,
    # torus_points,
    # torus_colors,
):
    object_list = []
    binding_list = []

    trajectory_projection = helper_functions.sterographic_projection(trajectory)
    integral_curves = plot.Line(pos = trajectory_projection, color = line_colors, width = 3, connect = line_connections)
    integral_curves.set_gl_state(preset = 'opaque')
    object_list.append(integral_curves)

    attractor_trajectories_projection = helper_functions.sterographic_projection(attractor_trajectories)
    attractor_colors -= 0.2
    attractor_colors[np.where(attractor_colors < 0)] = 0
    attractor_curves = plot.Line(pos = attractor_trajectories_projection, color = attractor_colors, width = 3, connect = attractor_connections)
    attractor_curves.set_gl_state(preset = 'opaque')
    object_list.append(attractor_curves)

    vertices = np.load('torus_points.npy')
    colors = np.load('torus_colors.npy')
    faces = np.load('torus_faces.npy')
    h_norms = np.load('h_norms.npy')
    h_norms[np.where(h_norms > 0.08)] = 1
    h_norms[np.where(h_norms <= 0.08)] = 0

    np.save("torus_black_points.npy", vertices[np.where(h_norms <= 0.08)])

    torus = Mesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors[:, :3] * h_norms[:, None],
    )
    object_list.append(torus)

    # torus_points_plot = plot.Scatter3D(
    #     pos = torus_points,
    #     face_color=torus_colors,
    #     symbol="o",
    #     size=2,
    #     edge_color=torus_colors,
    # )
    # torus_points_plot.set_gl_state(preset = 'opaque')
    # object_list.append(torus_points_plot)


    binding_list.append(bindings.Toggle([integral_curves], 'l'))
    binding_list.append(bindings.Toggle([attractor_curves], 'space'))
    binding_list.append(bindings.Toggle([torus], 't'))

    experiment_plot = plot.Plot(object_list=object_list, binding_list=binding_list)
    
    experiment_plot.show()


if __name__ == '__main__':
    render_experiment(*calculate_experiment())