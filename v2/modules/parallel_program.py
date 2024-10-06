import os

import numpy as np
import numpy.typing as npt
import pyopencl as cl
from tqdm import tqdm
from typing import Optional

from modules.helper_functions import timed_function
from modules import constants

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type = cl.device_type.GPU)
ctx = cl.Context(devices = my_gpu_devices )
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, open("modules/parallel_program.c").read()).build()


@timed_function
def point_trajectories(
    N: int,
    a: npt.NDArray[np.csingle],
    b: npt.NDArray[np.csingle],
    head_start: int,
    frame_amount: int,
    initial_conditions: npt.NDArray[np.csingle],
    epsilon: float,
    intermediate_steps: int,
    color_matrix: Optional[npt.NDArray[np.float32]] = None,
) -> tuple:

    initial_conditions_size = np.int32(len(initial_conditions))
    trajectories = np.zeros((initial_conditions_size, frame_amount, 2), dtype=np.csingle)

    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)\

    kernel = prg.point_trajectories
    kernel_args = [
        np.int32((N * (N + 1)) / 2),
        a_buf,
        b_buf,
        np.int32(head_start),
        np.int32(frame_amount),
        initial_conditions_size,
        np.float32(epsilon),
        np.int32(intermediate_steps),
    ]
    run_kernel(kernel, initial_conditions_size, [initial_conditions], [trajectories], *kernel_args)

    trajectories = np.swapaxes(trajectories, 0, 1)

    if color_matrix is None:
        color_matrix = np.array([
                np.tile([1, 0, 0, 0.1], (len(initial_conditions), 1)),
            ],
            dtype=np.float32,
        ).reshape(-1, 4)

    line_colors = np.tile(color_matrix, (frame_amount, 1))

    line_connections = np.zeros((len(color_matrix) * (frame_amount - 1), 2), dtype=np.int32)
    line_connections[:, 0] = np.arange(0, len(color_matrix) * (frame_amount - 1))
    line_connections[:, 1] = np.arange(len(color_matrix), len(color_matrix) * frame_amount)

    return trajectories, color_matrix, line_colors, line_connections


@timed_function
def point_trajectories_pos_and_neg(
    N: int,
    a: npt.NDArray[np.csingle],
    b: npt.NDArray[np.csingle],
    head_start: int,
    frame_amount: int,
    initial_conditions: npt.NDArray[np.csingle],
    epsilon: float,
    intermediate_steps: int,
    color_matrix: Optional[npt.NDArray[np.float32]] = None,
) -> tuple:

    initial_conditions_size = np.int32(len(initial_conditions))
    trajectories = np.zeros((initial_conditions_size * 2, frame_amount, 2), dtype=np.csingle)
    tmp_trajectories = np.zeros((initial_conditions_size, frame_amount, 2), dtype=np.csingle)

    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

    kernel = prg.point_trajectories
    kernel_args = [
        np.int32((N * (N + 1)) / 2),
        a_buf,
        b_buf,
        np.int32(head_start),
        np.int32(frame_amount),
        initial_conditions_size,
        np.float32(epsilon),
        np.int32(intermediate_steps),
    ]
    run_kernel(kernel, initial_conditions_size, [initial_conditions], [tmp_trajectories], *kernel_args)
    trajectories[:initial_conditions_size] = tmp_trajectories

    kernel_args[-2] = -np.float32(epsilon)
    run_kernel(kernel, initial_conditions_size, [initial_conditions], [tmp_trajectories], *kernel_args)
    trajectories[initial_conditions_size:] = tmp_trajectories

    trajectories = np.swapaxes(trajectories, 0, 1)

    if color_matrix is None:
        color_matrix = np.array([
                np.tile([1, 0, 0, 0.1], (len(initial_conditions), 1)),
                np.tile([0, 0, 1, 0.1], (len(initial_conditions), 1)),
            ],
            dtype=np.float32,
        ).reshape(-1, 4)

    line_colors = np.tile(color_matrix, (frame_amount, 1))

    line_connections = np.zeros((len(color_matrix) * (frame_amount - 1), 2), dtype=np.int32)
    line_connections[:, 0] = np.arange(0, len(color_matrix) * (frame_amount - 1))
    line_connections[:, 1] = np.arange(len(color_matrix), len(color_matrix) * frame_amount)

    return trajectories, color_matrix, line_colors, line_connections


@timed_function
def find_corresponding_head(
    N: int,
    a: npt.NDArray[np.csingle],
    b: npt.NDArray[np.csingle],
    head_start: int,
    initial_conditions: npt.NDArray[np.csingle],
    orbit_heads: npt.NDArray[np.csingle],
    orbit_tol: float,
    max_iter: int,
    epsilon: float,
):

    initial_conditions_size = np.int32(len(initial_conditions))
    orbit_head_amount = np.int32(len(orbit_heads))
    head_indices = np.zeros(initial_conditions_size, dtype=np.int32)

    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    orbit_heads_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=orbit_heads)

    kernel = prg.find_corresponding_head
    kernel_args = [
        np.int32((N * (N + 1)) / 2),
        a_buf,
        b_buf,
        np.int32(head_start),
        initial_conditions_size,
        orbit_heads_buf,
        orbit_head_amount,
        np.float32(orbit_tol),
        np.int32(max_iter),
        np.float32(epsilon),
    ]
    run_kernel(kernel, initial_conditions_size, [initial_conditions], [head_indices], *kernel_args)

    return head_indices, constants.COLORS[head_indices]


@timed_function
def point_trajectories_not_tangent(
    N: int,
    a: npt.NDArray[np.csingle],
    b: npt.NDArray[np.csingle],
    head_start: int,
    frame_amount: int,
    initial_conditions: npt.NDArray[np.csingle],
    epsilon: float,
    intermediate_steps: int,
) -> tuple:

    initial_conditions_size = np.int32(len(initial_conditions))
    trajectories = np.zeros((frame_amount, initial_conditions_size, 2), dtype=np.csingle)

    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    initial_conditions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=initial_conditions)
    trajectories_buf = cl.Buffer(ctx, mf.WRITE_ONLY, trajectories.nbytes)

    prg.point_trajectories_not_tangent(
        queue,
        [initial_conditions_size],
        None,
        np.int32((N * (N + 1)) / 2),
        a_buf,
        b_buf,
        np.int32(head_start),
        np.int32(frame_amount),
        initial_conditions_size,
        initial_conditions_buf,
        trajectories_buf,
        np.float32(epsilon),
        np.int32(intermediate_steps),
    )

    cl.enqueue_copy(queue, trajectories, trajectories_buf)

    color_matrix = np.array([
            np.tile([1, 0, 0, 0.1], (len(initial_conditions), 1)),
        ],
        dtype=np.float32,
    ).reshape(-1, 4)

    line_colors = np.tile(color_matrix, (frame_amount, 1))

    line_connections = np.zeros((len(color_matrix) * (frame_amount - 1), 2), dtype=np.int32)
    line_connections[:, 0] = np.arange(0, len(color_matrix) * (frame_amount - 1))
    line_connections[:, 1] = np.arange(len(color_matrix), len(color_matrix) * frame_amount)

    return trajectories, color_matrix, line_colors, line_connections


@timed_function
def point_trajectories_pos_and_neg_not_tangent(
    N: int,
    a: npt.NDArray[np.csingle],
    b: npt.NDArray[np.csingle],
    head_start: int,
    frame_amount: int,
    initial_conditions: npt.NDArray[np.csingle],
    epsilon: float,
    intermediate_steps: int,
) -> tuple:

    initial_conditions_size = np.int32(len(initial_conditions))
    trajectories = np.zeros((frame_amount, initial_conditions_size * 2, 2), dtype=np.csingle)
    tmp_trajectories = np.zeros((frame_amount, initial_conditions_size, 2), dtype=np.csingle)

    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    initial_conditions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=initial_conditions)
    trajectories_buf = cl.Buffer(ctx, mf.WRITE_ONLY, trajectories.nbytes)

    prg.point_trajectories_not_tangent(
        queue,
        [initial_conditions_size],
        None,
        np.int32((N * (N + 1)) / 2),
        a_buf,
        b_buf,
        np.int32(head_start),
        np.int32(frame_amount),
        initial_conditions_size,
        initial_conditions_buf,
        trajectories_buf,
        np.float32(epsilon),
        np.int32(intermediate_steps),
    )

    cl.enqueue_copy(queue, tmp_trajectories, trajectories_buf)
    trajectories[:, :initial_conditions_size] = tmp_trajectories

    prg.point_trajectories_not_tangent(
        queue,
        [initial_conditions_size],
        None,
        np.int32((N * (N + 1)) / 2),
        a_buf,
        b_buf,
        np.int32(head_start),
        np.int32(frame_amount),
        initial_conditions_size,
        initial_conditions_buf,
        trajectories_buf,
        -np.float32(epsilon),
        np.int32(intermediate_steps),
    )

    cl.enqueue_copy(queue, tmp_trajectories, trajectories_buf)
    trajectories[:, initial_conditions_size:] = tmp_trajectories

    color_matrix = np.array([
            np.tile([1, 0, 0, 0.1], (len(initial_conditions), 1)),
            np.tile([0, 0, 1, 0.1], (len(initial_conditions), 1)),
        ],
        dtype=np.float32,
    ).reshape(-1, 4)

    line_colors = np.tile(color_matrix, (frame_amount, 1))

    line_connections = np.zeros((len(color_matrix) * (frame_amount - 1), 2), dtype=np.int32)
    line_connections[:, 0] = np.arange(0, len(color_matrix) * (frame_amount - 1))
    line_connections[:, 1] = np.arange(len(color_matrix), len(color_matrix) * frame_amount)

    return trajectories, color_matrix, line_colors, line_connections


@timed_function
def find_corresponding_head_not_tangent(
    N: int,
    a: npt.NDArray[np.csingle],
    b: npt.NDArray[np.csingle],
    head_start: int,
    initial_conditions: npt.NDArray[np.csingle],
    orbit_heads: npt.NDArray[np.csingle],
    orbit_tol: float,
    max_iter: int,
    epsilon: float,
):

    initial_conditions_size = np.int32(len(initial_conditions))
    orbit_head_amount = np.int32(len(orbit_heads))
    head_indices = np.zeros(initial_conditions_size, dtype=np.int32)

    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    initial_conditions_buf = cl.Buffer(ctx, mf.READ_ONLY, initial_conditions[:constants.MAX_BUFFER_SIZE].nbytes)
    orbit_heads_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=orbit_heads)
    head_indices_buf = cl.Buffer(ctx, mf.WRITE_ONLY, head_indices[:constants.MAX_BUFFER_SIZE].nbytes)

    start_index = 0
    end_index = constants.MAX_BUFFER_SIZE
    for _ in tqdm(range(initial_conditions_size // constants.MAX_BUFFER_SIZE)):
        cl.enqueue_copy(queue, initial_conditions_buf, initial_conditions[start_index: end_index])

        prg.find_corresponding_head(
            queue,
            [constants.MAX_BUFFER_SIZE],
            None,
            np.int32((N * (N + 1)) / 2),
            a_buf,
            b_buf,
            np.int32(head_start),
            initial_conditions_buf,
            orbit_heads_buf,
            head_indices_buf,
            orbit_head_amount,
            np.float32(orbit_tol),
            np.int32(max_iter),
            np.float32(epsilon),
        )

        cl.enqueue_copy(queue, head_indices[start_index: end_index], head_indices_buf)

        start_index += constants.MAX_BUFFER_SIZE
        end_index += constants.MAX_BUFFER_SIZE

    remainder = np.int32(initial_conditions_size % constants.MAX_BUFFER_SIZE)
    end_index = start_index + remainder
    if remainder:
        cl.enqueue_copy(queue, initial_conditions_buf, initial_conditions[start_index: end_index])

        prg.find_corresponding_head(
            queue,
            [remainder],
            None,
            np.int32((N * (N + 1)) / 2),
            a_buf,
            b_buf,
            np.int32(head_start),
            initial_conditions_buf,
            orbit_heads_buf,
            head_indices_buf,
            orbit_head_amount,
            np.float32(orbit_tol),
            np.int32(max_iter),
            np.float32(epsilon),
        )

        cl.enqueue_copy(queue, head_indices[start_index: end_index], head_indices_buf)

    return head_indices, constants.COLORS[head_indices]



def run_kernel(kernel, buffer_size, reading_objs, writing_objs, *kernel_args):
    reading_bufs = [
        cl.Buffer(ctx, mf.READ_ONLY, obj[:constants.MAX_BUFFER_SIZE].nbytes)
        for obj in reading_objs
    ]
    writing_bufs = [
        cl.Buffer(ctx, mf.WRITE_ONLY, obj[:constants.MAX_BUFFER_SIZE].nbytes)
        for obj in writing_objs
    ]

    start_index = 0
    end_index = min(constants.MAX_BUFFER_SIZE, buffer_size)
    for _ in tqdm(range(int(np.ceil(buffer_size / constants.MAX_BUFFER_SIZE)))):
        for buf, obj in zip(reading_bufs, reading_objs):
            cl.enqueue_copy(queue, buf, obj[start_index: end_index])

        kernel(queue, [end_index - start_index], None, *kernel_args, *reading_bufs, *writing_bufs)

        for buf, obj in zip(writing_bufs, writing_objs):
            cl.enqueue_copy(queue, obj[start_index: end_index], buf)

        start_index += constants.MAX_BUFFER_SIZE
        end_index = min(end_index + constants.MAX_BUFFER_SIZE, buffer_size)