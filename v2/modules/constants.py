import numpy as np

PI_TIMES_2 = np.pi * 2
PI_OVER_2 = np.pi / 2

COLORS = np.array([
    (1.0, 0.0, 1.0, 1.0),   # Magenta
    (1.0, 0.5, 0.0, 1.0),   # Orange
    (0.5, 0.0, 1.0, 1.0),   # Purple
    (0.0, 1.0, 1.0, 1.0),   # Cyan
    (1.0, 1.0, 0.0, 1.0),   # Yellow
    (0.0, 1.0, 0.5, 1.0),   # Lime
    (0.5, 0.5, 0.5, 1.0),   # Gray
    (1.0, 0.5, 0.5, 1.0),   # Pink
    (0.5, 1.0, 0.5, 1.0),   # Light Green
    (0.5, 0.5, 1.0, 1.0),   # Light Blue
    (1.0, 1.0, 0.5, 1.0),   # Light Yellow
    (1.0, 0.5, 1.0, 1.0),   # Light Magenta
    (0.5, 1.0, 1.0, 1.0),   # Light Cyan
    (0.3, 0.2, 0.5, 1.0),   # Indigo
    (0.0, 0.5, 0.5, 1.0),   # Teal
    (0.5, 0.0, 0.5, 1.0),   # Maroon
    (0.5, 0.5, 0.0, 1.0),   # Olive
    (1.0, 0.0, 0.0, 1.0),   # Red
    (0.0, 1.0, 0.0, 1.0),   # Green
    (0.0, 0.0, 1.0, 1.0),   # Blue
    (0.0, 0.0, 0.0, 1.0),   # Black
])

PROJECTIONS = ['south', 'north']

OPACITY_PRESETS = [
    dict(preset='opaque', depth_test=True),
    dict(preset='translucent', depth_test=False),
]

MAX_BUFFER_SIZE = np.int32(1000)