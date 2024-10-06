import numpy as np
from modules import bindings
from vispy import app, scene, visuals
import imageio
from tqdm import tqdm

Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)  # type: ignore
Line = scene.visuals.create_visual_node(visuals.LineVisual)  # type: ignore


class Plot():
    def __init__(
        self,
        decorate: bool = True,
        size: tuple[float, float] = (800, 600),
        show: bool = True,
        bgcolor: str = 'white',
        autoswap: bool = False,
        vsync: bool = True,
        distance: float = 10,
        camera: str = 'turntable',
        object_list = [],
        binding_list: list[bindings.Binding] = [],
        create_axis = True,
    ):
        if create_axis:
            self.axis = Line(
                pos=[
                    [-1, 0, 0], [1, 0, 0],
                    [0, -1, 0], [0, 1, 0],
                    [0, 0, -1], [0, 0, 1],
                ],
                connect=np.array([
                    [0, 1],
                    [2, 3],
                    [4, 5]
                ]),
                color=(
                    'cyan',
                    'cyan',
                    'green',
                    'green',
                    'black',
                    'black',
                ),
                width=3,
            )
            self.axis.set_gl_state(depth_test=True)
            binding_list.append(bindings.Toggle([self.axis], 'a'))
            object_list.append(self.axis)

        camera_keys = {
            "escape" : self.exit,
            "c" : self.reset_camera_center,
            "r" : self.reset_camera,
            "f1" : self.full_screen,
        }
        object_keys = {b.key: b.act for b in binding_list}

        self.canvas = scene.SceneCanvas(
            decorate=decorate,
            size=size,
            show=show,
            bgcolor=bgcolor,
            autoswap=autoswap,
            keys=camera_keys | object_keys,
            vsync=vsync,
        )

        self.view = self.canvas.central_widget.add_view(camera=camera)
        self.view.camera.distance = distance

        for o in object_list:
            o.parent = self.view.scene

    def reset_camera(self):
        self.view.camera.reset()

    def reset_camera_center(self):
        self.view.camera.center = [0,0,0]

    def exit(self):
        self.canvas.close()

    def full_screen(self):
        self.canvas.fullscreen = not self.canvas.fullscreen

    def show(self):
        app.run()

    def film_rotating_animation(self, frame_amount, video_name):
        writer = imageio.get_writer(f'animations/{video_name}.mp4', fps = 30)
        angle_step = 360/frame_amount
        for _ in tqdm(range(frame_amount)):
            im = self.canvas.render(alpha=True)
            writer.append_data(im)
            self.view.camera.orbit(angle_step, 0)
        writer.close()

    def film_partial_rotating_animation(self, frame_amount, writer):
        angle_step = 360/frame_amount
        for _ in tqdm(range(frame_amount)):
            im = self.canvas.render(alpha=True)
            writer.append_data(im)
            self.view.camera.orbit(angle_step, 0)