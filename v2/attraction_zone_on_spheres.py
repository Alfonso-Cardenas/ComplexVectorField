import numpy as np
from modules import bindings, helper_functions, parallel_program, plot, point_factory
from vispy.geometry import create_sphere
from vispy.scene.visuals import Mesh
import imageio

frame_amount = 2000
orbit_initial_conditions_size = 1000
S3_sample_initial_condition_size = 1000
epsilon = 0.001
head_start = 1000000 #100000 empieza lo bueno
Lambda = 0.24#0.202288
fixed_point_tol = 0.0001
orbit_tol = 0.008
intermidiate_steps = 20
max_iter = 50000
N = 3
cols = 20
rows = 20
radiuses = np.linspace(0.1, 5, 20)
radius = 0.8

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
    # random_conditions = np.vstack((random_conditions, np.load('black_points.npy')))

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

    neg_trajectories, _, _, neg_line_connectopns = (
        parallel_program.point_trajectories(
            N,
            a,
            b,
            head_start,
            frame_amount,
            random_conditions,
            -epsilon,
            intermidiate_steps,
        )
    )

    orbit_heads, line_colors = helper_functions.find_orbit_heads(trajectories, orbit_tol)

    return (
        trajectories,
        line_colors[: orbit_initial_conditions_size],
        line_colors,
        line_connections,
        orbit_heads,
        neg_trajectories,
        neg_line_connectopns,
    )

def calc_sphere(radius, orbit_heads):
    sphere_mesh_data = create_sphere(cols=cols, rows=rows, radius=radius)
    S3_sample = helper_functions.inverse_stereographic_projection(sphere_mesh_data.get_vertices())

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

    sphere_mesh_data.set_vertex_colors(S3_sample_colors)

    return S3_sample, head_indices, sphere_mesh_data

def find_border_points(head_indices, triangle_indices, S3_sample):
    triangle_head_indices = head_indices[triangle_indices]

    border_indices = triangle_indices[
        np.where(
            np.any(triangle_head_indices != triangle_head_indices[:, 0, None], axis=1)
        )
    ]

    border_triangle_coords = S3_sample[border_indices]

    return np.mean(border_triangle_coords, axis=1)

def render_experiment(trajectories, color_matrix, line_colors, line_connections, orbit_heads, neg_trajectories, neg_line_connections):
    object_list = []
    binding_list = []
    color_matrix[:, -1] = 1

    trajectories_projection = helper_functions.sterographic_projection(trajectories)
    integral_curves = plot.Line(pos = trajectories_projection, color = line_colors, width = 3, connect = line_connections)
    integral_curves.set_gl_state(preset='opaque', depth_test=True)
    object_list.append(integral_curves)

    neg_trajectories_projection = helper_functions.sterographic_projection(neg_trajectories)
    neg_integral_curves = plot.Line(pos = neg_trajectories_projection, color = "blue", width = 3, connect = neg_line_connections)
    neg_integral_curves.set_gl_state(preset='opaque', depth_test=True)
    object_list.append(neg_integral_curves)

    sphere = Mesh()
    sphere.set_gl_state(preset='opaque', depth_test=True)
    object_list.append(sphere)

    binding_list.append(bindings.Toggle([integral_curves], 'l'))
    binding_list.append(bindings.Toggle([sphere], 'space'))

    S3_sample, head_indices, meshdata = calc_sphere(radius, orbit_heads)

    border_point_coords = find_border_points(head_indices, meshdata.get_faces(), S3_sample)
    # np.save("border_point_coords.npy", border_point_coords)

    border_points = plot.Scatter3D(
        pos = helper_functions.sterographic_projection(border_point_coords),
        face_color=[0,0,0],
        symbol="o",
        size=5,
        edge_color=[0,0,0],
    )
    object_list.append(border_points)

    sphere.set_data(meshdata=meshdata)

    green_points = np.load('green_points.npy')
    border_endpoints = plot.Scatter3D(
        pos = helper_functions.sterographic_projection(green_points),
        face_color=[0,1,0],
        symbol="o",
        size=4,
        edge_color=[0,1,0],
    )
    object_list.append(border_endpoints)

    experiment_plot = plot.Plot(vsync=False, object_list=object_list, binding_list=binding_list, create_axis=False)

    experiment_plot.show()
    # writer = imageio.get_writer('animations/Caso A/Atracción en esfera variando radio.mp4', fps = 30)
    # for radius in radiuses:
    #     sphere.set_data(meshdata=calc_sphere(radius, orbit_heads))
    #     experiment_plot.film_partial_rotating_animation(300, writer)
    # writer.close()


if __name__ == '__main__':
    render_experiment(*calculate_experiment())