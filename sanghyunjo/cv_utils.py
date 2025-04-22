# Copyright (C) 2025 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import os
import cv2
import copy
import cmapy
import requests
import warnings
import functools
import numpy as np

from io import BytesIO
from dataclasses import dataclass
from PIL import ImageFont, ImageDraw, Image

Image.MAX_IMAGE_PIXELS = None # to read unlimited pixels like a large tiff format

def deprecated(alternative_name):
    """ 
    A decorator to mark functions as deprecated and suggest an alternative function.
    
    Args:
        alternative_name (str): The recommended alternative function name.
    
    Returns:
        Wrapper function that issues a warning.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{func.__name__}' is deprecated and will be removed in the future. "
                f"Use '{alternative_name}' instead.",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@dataclass(frozen=True)
class KeyCode:
    ESC: int = 27
    SPACE: int = 32
    PLUS: int = ord('+')
    MINUS: int = ord('-')

@dataclass(frozen=True)
class MouseEvent:
    NONE: int = 0
    MOVE: int = 1
    LEFT_DOWN: int = 2
    LEFT_UP: int = 3
    RIGHT_DOWN: int = 4
    RIGHT_UP: int = 5
    WHEEL_UP: int = 6
    WHEEL_DOWN: int = 7
    LEFT_MOVE: int = 8
    RIGHT_MOVE: int = 9

@dataclass
class MouseState:
    x: int = 0
    y: int = 0
    event: int = MouseEvent.NONE
    
    @property
    def point(self):
        return self.x, self.y

class MouseEventHandler:
    def __init__(self, winname):
        self.state = MouseState()
        self.button = None  # 'left' or 'right'

        cv2.namedWindow(winname)
        cv2.setMouseCallback(winname, self)

    def get(self):
        current_state = copy.deepcopy(self.state)
        if self.state.event in [MouseEvent.WHEEL_DOWN, MouseEvent.WHEEL_UP]:
            self.state.event = MouseEvent.NONE
        return current_state

    def move(self, x, y):
        self.state.x, self.state.y = x, y
        if self.button == 'left':
            self.state.event = MouseEvent.LEFT_MOVE
        elif self.button == 'right':
            self.state.event = MouseEvent.RIGHT_MOVE
        else:
            self.state.event = MouseEvent.MOVE

    def leftdown(self, x, y):
        self.state = MouseState(x, y, MouseEvent.LEFT_DOWN)
        self.button = 'left'

    def leftup(self, x, y):
        self.state = MouseState(x, y, MouseEvent.LEFT_UP)
        self.button = None

    def rightdown(self, x, y):
        self.state = MouseState(x, y, MouseEvent.RIGHT_DOWN)
        self.button = 'right'

    def rightup(self, x, y):
        self.state = MouseState(x, y, MouseEvent.RIGHT_UP)
        self.button = None

    def wheelup(self):
        self.state.event = MouseEvent.WHEEL_UP

    def wheeldown(self):
        self.state.event = MouseEvent.WHEEL_DOWN

    def __call__(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.leftdown(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.leftup(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.rightdown(x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.rightup(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.move(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.wheelup() if flags > 0 else self.wheeldown()

def imread(path, backend='opencv', color=None):
    """
    Loads an image using OpenCV or Pillow.

    Args:
        path (str): Path to the image file.
        backend (str): Backend to use for loading ('opencv', 'pillow', or 'mask').
        color (str | None): Color mode ('gray', 'rgb', or None for unchanged).
    
    Returns:
        np.ndarray or PIL.Image.Image or None: Loaded image, or None if file not found.
    """
    color_modes = {
        'opencv': {
            'gray': cv2.IMREAD_GRAYSCALE,
            'rgb': cv2.IMREAD_COLOR,
            None: cv2.IMREAD_UNCHANGED,
        },
        'pillow': {
            'gray': 'L',
            'rgb': 'RGB',
            None: None,  # Leave as-is
        },
        'mask': {
            None: None,  # Treated as palette image via PIL
        }
    }

    try:
        if backend == 'opencv':
            # Load image using OpenCV with appropriate color mode
            flag = color_modes['opencv'].get(color, cv2.IMREAD_UNCHANGED)
            return cv2.imdecode(np.fromfile(path, np.uint8), flag)

        elif backend == 'mask':
            # Load image for mask purposes (e.g., with color palette)
            return np.asarray(Image.open(path)).copy()

        else:  # 'pillow' backend
            image = Image.open(path)
            mode = color_modes['pillow'].get(color)
            return image if mode is None else image.convert(mode)

    except FileNotFoundError:
        return None

def imwrite(path, image, palette=None):
    """
    Saves an image using OpenCV or Pillow. Supports saving with a custom color palette 
    for indexed color images (e.g., masks).

    Args:
        path (str): File path to save the image.
        image (np.ndarray or PIL.Image.Image): The image to be saved.
        palette (Union[np.ndarray, str, None]): Optional color palette for paletted images.
            - If None: save as a standard image using OpenCV.
            - If 'voc': use predefined VOC color palette (requires get_colors()).
            - If 'gray' or 'g': use 2-class grayscale palette.
            - If np.ndarray: must be shape (768,) or (256, 3), dtype=uint8.

    Returns:
        bool: True if the image is successfully saved, False otherwise.

    Examples:
        >>> imwrite("output.png", img_array)
        >>> imwrite("mask.png", mask_array, palette='voc')
    """
    try:
        if palette is None:
            return cv2.imwrite(path, image)

        # Resolve predefined palette string
        if isinstance(palette, str):
            if palette.lower() == 'voc':
                palette = get_colors()  # Must return shape (256, 3), dtype=uint8
            elif palette.lower() in ['gray', 'g']:
                palette = np.array([[0]*3, [255]*3], dtype=np.uint8)
            else:
                raise ValueError(f"Unsupported palette keyword: {palette}")

        # Validate palette as ndarray
        if not isinstance(palette, np.ndarray):
            raise TypeError("Palette must be a NumPy array, string keyword, or None.")
        
        if palette.dtype != np.uint8:
            raise TypeError("Palette must have dtype np.uint8.")

        # Convert and save as paletted PNG
        img = Image.fromarray(image.astype(np.uint8)).convert('P')
        img.putpalette(palette)
        img.save(path)
        return True

    except Exception as e:
        print(f"[imwrite error] {e}")  # Optional: log or silence
        return False

def imshow(winname, image, wait=-1, title=''):
    """
    Displays an image in an OpenCV window.

    Args:
        winname (str): Name of the OpenCV window.
        image (np.array): Image to display.
        wait (int): Time in milliseconds to wait for a key press (-1 for infinite).
        title (str): Optional title for the window.

    Returns:
        int or None: Key press value if `wait` >= 0, otherwise None.
    """
    cv2.imshow(winname, image)

    if title:
        cv2.setWindowTitle(winname, title)

    return cv2.waitKey(wait) if wait >= 0 else None
    
""" Deprecated aliases with warning """
@deprecated("imread")
def read_image(path, color=None, backend='opencv'):
    return imread(path, color, backend)

@deprecated("imwrite")
def write_image(path, image, palette=None):
    return imwrite(path, image, palette)

@deprecated("imshow")
def show_image(winname, image, wait=-1, title=''):
    return imshow(winname, image, wait, title)

class VideoReader:
    """
    Simplified video reader for easier frame extraction.

    Example 1: Read frames in a loop
        video = VideoReader("video.mp4")
        while True:
            frame = video()
            if frame is None:
                break
            cv2.imshow("Video", frame)
            cv2.waitKey(1)

    Example 2: Access frames by index
        video = VideoReader("video.mp4")
        for i in range(0, len(video), video.fps):
            frame = video[i]
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
    """
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

    def __len__(self, cast_fn=int):
        return cast_fn(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __getitem__(self, index=None):
        if index is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.video.read()
        return frame if ret else None
    
    def __call__(self):
        return self.__getitem__()
    
    def release(self):
        """ Releases the video capture object. """
        self.video.release()
    
class VideoWriter:
    """
    Simplified video writer for easy frame saving.

    Example:
        writer = VideoWriter("output.mp4", w, h, fps)
        for frame in frames:
            writer(frame)
        writer.release()
    """
    def __init__(self, path, width, height, fps):
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height)
        )

    def __call__(self, frame):
        """ Writes a frame to the video file. """
        self.writer.write(frame)

    def release(self):
        """ Releases the video writer object. """
        self.writer.release()

def viread(path):
    """ Creates a video reader instance (alias for VideoReader). """
    return VideoReader(path)

def viwrite(path, frames, fps):
    """ Writes a list of frames to a video file. """
    h, w = frames[0].shape[:2]
    writer = VideoWriter(path, w, h, fps)
    for frame in frames:
        writer(frame)
    writer.release()

""" Deprecated aliases with warning """
@deprecated("viread")
def read_video(path):
    return viread(path)

@deprecated("viwrite")
def write_video(path, frames, fps):
    return viwrite(path, frames, fps)

def blend_images(image1, image2_or_color, alpha, mask=None):
    """
    Blends an image with another image or a single color.

    Args:
        image1 (np.array): The base image to blend into.
        image2_or_color (np.array or tuple): Another image (same shape) or a color (e.g., (255, 0, 0)).
        alpha (float): Blending factor between image1 and image2_or_color.
        mask (np.array | None): Optional boolean mask for blending area.

    Returns:
        np.array: Blended image.

    Examples:
        # Case 1: image + color
        blended = blend_images(image, (0, 255, 0), 0.5)

        # Case 2: image + image
        blended = blend_images(image1, image2, 0.3)

        # Case 3: image + color + mask
        blended = blend_images(image, (255, 0, 0), 0.4, mask=box.mask)
    """
    if isinstance(image2_or_color, np.ndarray) and len(image2_or_color.shape) == 1:
        image2_or_color = tuple(image2_or_color)

    if isinstance(image2_or_color, tuple) or isinstance(image2_or_color, list):
        # Assume it's a color, create a solid image with same shape as image1
        overlay = np.full_like(image1, image2_or_color)
    else:
        # Assume it's an image (must be same shape)
        overlay = image2_or_color

    if mask is not None:
        if mask.dtype != np.bool_:
            mask = mask > 0

        image1[mask] = cv2.addWeighted(image1[mask], 1 - alpha, overlay[mask], alpha, 0)
        return image1
    else:
        return cv2.addWeighted(image1, 1 - alpha, overlay, alpha, 0)

""" Deprecated aliases with warning """
@deprecated("blend_images")
def overlay(image1, image2, alpha):
    return blend_images(image1, image2, alpha)

def draw_rect(image, xyxy, color=(79, 244, 255), thickness=1, dashed=False, step=10, circle=False, radius=5, filled=True):
    """
    Draws a rectangle on the given image, with an option for a dashed border and corner circles.

    Args:
        image (np.array): The image on which the rectangle is drawn.
        xyxy (tuple): The coordinates of the rectangle (xmin, ymin, xmax, ymax).
        color (tuple): The color of the rectangle in BGR format (default: light blue).
        thickness (int): The thickness of the rectangle's border (default: 1).
        dashed (bool): If True, draws a dashed rectangle instead of a solid one (default: False).
        step (int): The gap size for the dashed effect (default: 10 pixels).
        circle (bool): If True, draws circles at the corners of the rectangle (default: False).
        radius (int): The radius of the corner circles (default: 5).
        filled (bool): If True, fills the corner circles (default: True).

    Example:
        # Draw a solid rectangle with corner circles
        draw_rect(img, (50, 50, 200, 200), color=(0, 255, 0), thickness=2, circle=True)

        # Draw a dashed rectangle with corner circles
        draw_rect(img, (250, 50, 400, 200), color=(0, 255, 0), thickness=2, dashed=True, circle=True)
    """
    xmin, ymin, xmax, ymax = xyxy

    if not dashed:
        # Draw a solid rectangle using OpenCV's built-in function
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    else:
        # Draw a dashed rectangle by creating short line segments
        # Horizontal lines (top and bottom)
        for x in range(xmin, xmax, step * 2):
            cv2.line(image, (x, ymin), (min(x + step, xmax), ymin), color, thickness)
            cv2.line(image, (x, ymax), (min(x + step, xmax), ymax), color, thickness)

        # Vertical lines (left and right)
        for y in range(ymin, ymax, step * 2):
            cv2.line(image, (xmin, y), (xmin, min(y + step, ymax)), color, thickness)
            cv2.line(image, (xmax, y), (xmax, min(y + step, ymax)), color, thickness)

    if circle:
        # Draw circles at the corners of the rectangle
        circle_thickness = -1 if filled else thickness
        cv2.circle(image, (xmin, ymin), radius, color, circle_thickness)
        cv2.circle(image, (xmax, ymax), radius, color, circle_thickness)

def draw_point(image, point, size, color, edge_color=(0, 0, 0), shape='circle'):
    """
    Draws a shape (circle, star, or V) at a given point on the image.

    Args:
        image (np.array): The image on which the shape is drawn.
        point (tuple): (x, y) coordinates of the shape's center.
        size (int): Size of the shape.
        color (tuple): Fill color (B, G, R).
        edge_color (tuple): Edge color (B, G, R), default black.
        shape (str): Shape type ('circle', 'star', 'v').

    Example:
        draw_shape(img, (100, 100), 20, (255, 0, 0), shape='star')  # Draw a red star
        draw_shape(img, (200, 200), 15, (0, 255, 0), shape='v')  # Draw a green V shape
    """
    x, y = point
    edge_size = max(size // 5, 1)

    # Convert OpenCV image to PIL format
    pillow_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pillow_image)

    if shape == 'circle':
        # Draw a circle (same as original function)
        draw.ellipse(
            [(x - size, y - size), (x + size, y + size)],
            fill=color, outline=edge_color, width=edge_size
        )

    elif shape == 'star':
        # Define star points
        star_points = [
            (x, y - size),  # Top
            (x + size * 0.3, y - size * 0.3),
            (x + size, y - size * 0.2),
            (x + size * 0.5, y + size * 0.2),
            (x + size * 0.7, y + size),
            (x, y + size * 0.5),
            (x - size * 0.7, y + size),
            (x - size * 0.5, y + size * 0.2),
            (x - size, y - size * 0.2),
            (x - size * 0.3, y - size * 0.3),
        ]
        draw.polygon(star_points, fill=color, outline=edge_color)

    elif shape == 'v':
        # Define V shape points (left shorter, right longer)
        v_points = [
            (x - size * 0.5, y - size * 0.2),  # Left top
            (x, y + size),  # Bottom point
            (x + size * 0.7, y - size * 0.5),  # Right top
            (x + size * 0.5, y - size * 0.6),  # Right shorter inside
            (x, y + size * 0.5),  # Inner bottom point
            (x - size * 0.3, y - size * 0.1),  # Left shorter inside
        ]
        draw.polygon(v_points, fill=color, outline=edge_color)

    # Convert back to OpenCV format
    image[:, :, :] = np.asarray(pillow_image)

def denorm(image: np.ndarray) -> np.ndarray:
    """Convert a normalized image (0 to 1 range) to an 8-bit image (0 to 255 range).

    Args:
        image (np.ndarray): Input image with values in the range [0, 1].

    Returns:
        np.ndarray: Image with values scaled to the range [0, 255] as uint8.
    """
    return (image * 255).astype(np.uint8)


def resize(image, size=None, scale=None, mode='bicubic'):
    """Resize an image using a specified interpolation mode.
    
    Args:
        image (np.ndarray): Input image.
        size (tuple, optional): Target size (width, height). Default is None.
        scale (float, optional): Scale factor for resizing. Default is None.
        mode (str, optional): Interpolation mode ('bicubic' or 'nearest'). Default is 'bicubic'.

    Returns:
        np.ndarray: Resized image.
    """
    if size is None and scale is None:
        raise ValueError("Either 'size' or 'scale' must be provided.")

    if size is not None:
        size = get_size(size) if not isinstance(size, tuple) else size
    elif scale is not None:
        h, w = image.shape[:2]
        size = (int(w * scale), int(h * scale))

    interpolation_modes = {
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
    }
    if mode not in interpolation_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {list(interpolation_modes.keys())}.")

    return cv2.resize(image, size, interpolation=interpolation_modes[mode])

def resize_mask(image, size=None, scale=None, mode='nearest'):
    """Resize a mask image, ensuring it is in the correct format.

    Args:
        image (np.ndarray): Input mask image.
        size (tuple, optional): Target size (width, height). Default is None.
        scale (float, optional): Scale factor for resizing. Default is None.

    Returns:
        np.ndarray: Resized mask image.
    """
    if image.dtype in [np.float32, np.float64]:
        image = denorm(image)
        
    return resize(image, size, scale, mode)

def colorize(cam: np.ndarray, option: str = 'SEISMIC') -> np.ndarray:
    """Apply a colormap to a given grayscale or single-channel image.

    Args:
        cam (np.ndarray): Input image (grayscale or single-channel).
        option (str, optional): Colormap option. Defaults to 'SEISMIC'.
            Available options: 'JET', 'HOT', 'SUMMER', 'WINTER',
            'INFERNO', 'GRAY', 'SEISMIC', 'VIRIDIS'.

    Returns:
        np.ndarray: Colorized image.
    """
    color_dict = {
        'JET': cv2.COLORMAP_JET,
        'HOT': cv2.COLORMAP_HOT,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'WINTER': cv2.COLORMAP_WINTER,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'GRAY': cmapy.cmap('gray'),
        'SEISMIC': cmapy.cmap('seismic'),
        'VIRIDIS': cmapy.cmap('viridis'),
    }

    if option not in color_dict:
        raise ValueError(f"Invalid colormap option '{option}'. Choose from {list(color_dict.keys())}.")

    # Normalize float images to 0-255 range
    if cam.dtype in [np.float32, np.float64]:
        cam = denorm(cam)

    # If input has multiple channels, take the maximum projection
    if cam.ndim == 3:
        cam = np.max(cam, axis=0)

    # Apply colormap
    colors = color_dict[option]
    cam = cv2.applyColorMap(cam, colors)

    return cam

def clamp_coords(coords, width: int, height: int):
    """
    Clamps a bounding box or a point to stay within image boundaries.

    Args:
        coords (Tuple[float]): A tuple representing either a box (xmin, ymin, xmax, ymax)
                               or a point (cx, cy).
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        Tuple[int]: Clamped and rounded coordinates.
                    - (xmin, ymin, xmax, ymax) if input was a box
                    - (cx, cy) if input was a point

    Examples:
        >>> clamp_coords((10.6, -5, 1284, 970), width=1280, height=960)
        (11, 0, 1279, 959)

        >>> clamp_coords((640.2, -10), width=1280, height=720)
        (640, 0)
    """
    coords = tuple(int(round(x)) for x in coords)

    if len(coords) == 4:
        xmin, ymin, xmax, ymax = coords
        xmin = min(max(xmin, 0), width - 1)
        ymin = min(max(ymin, 0), height - 1)
        xmax = min(max(xmax, 0), width - 1)
        ymax = min(max(ymax, 0), height - 1)
        return xmin, ymin, xmax, ymax

    elif len(coords) == 2:
        cx, cy = coords
        cx = min(max(cx, 0), width - 1)
        cy = min(max(cy, 0), height - 1)
        return cx, cy

    else:
        raise ValueError("Input must be either a 2D point or a 4D bounding box.")

# TODO: optimize/add existing/new functions below
def draw_text(
        image: np.ndarray, text: str, coordinate: tuple, color: tuple=(0, 0, 0), 
        font_path: str=None, font_size: int=20, 
        background: tuple=(79, 244, 255), centering: bool=False, padding: int=5
    ):
    if font_path is None:
        font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'Times New Roman MT Std.otf')
    
    text = ' ' + text
    font = ImageFont.truetype(font_path, font_size)
    
    # tw, th = font.getsize(text) # for pillow==9.5.0
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top

    if centering:
        coordinate = list(coordinate)
        coordinate[0] = max(coordinate[0] - (tw // 2 + padding // 2), 0)
        coordinate[1] = max(coordinate[1] - (th // 2 + padding // 2), 0)
        coordinate = tuple(coordinate)
    
    background_box = None
    if background is not None:
        cv2.rectangle(image, coordinate, (coordinate[0] + tw + padding, coordinate[1] + th + padding), background, cv2.FILLED)
        
        xmin, ymin = coordinate
        xmax, ymax = (coordinate[0] + tw + padding, coordinate[1] + th + padding)
        
        background_box = [xmin, ymin, xmax, ymax]
    
    pillow_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pillow_image)
    draw.text(coordinate, text, font=font, fill=(color[0], color[1], color[2], 0))

    image[:, :, :] = np.asarray(pillow_image)
    return background_box

def interpolate_colors(c1, c2, n=256):
    c1 = np.asarray(c1, dtype=np.float32) / 255.
    c2 = np.asarray(c2, dtype=np.float32) / 255.

    colors = []

    for i in range(n):
        mix = i/(n-1)
        colors.append((1-mix)*c1 + mix*c2)
    
    colors = np.asarray(colors)
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    return colors[:, None, :]

def get_colors(num_classes=20, ignore_index=255, color_format='rgb', data=None):
    colors = []
    bitget = lambda v, i: (v & (1 << i)) != 0
    
    for i in range(num_classes):
        r = g = b = 0
        c = i

        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        
        colors.append([r, g, b])
    
    while len(colors) < ignore_index: colors.append([0, 0, 0])
    colors.append([224, 224, 192])

    colors = np.asarray(colors, dtype=np.uint8)

    if data is not None:
        if data == 'ADE2016':
            updated_colors = [
                # [0, 0, 0],
                [120, 120, 120],
                [180, 120, 120],
                [6, 230, 230],
                [80, 50, 50],
                [4, 200, 3],
                [120, 120, 80],
                [140, 140, 140],
                [204, 5, 255],
                [230, 230, 230],
                [4, 250, 7],
                [224, 5, 255],
                [235, 255, 7],
                [150, 5, 61],
                [120, 120, 70],
                [8, 255, 51],
                [255, 6, 82],
                [143, 255, 140],
                [204, 255, 4],
                [255, 51, 7],
                [204, 70, 3],
                [0, 102, 200],
                [61, 230, 250],
                [255, 6, 51],
                [11, 102, 255],
                [255, 7, 71],
                [255, 9, 224],
                [9, 7, 230],
                [220, 220, 220],
                [255, 9, 92],
                [112, 9, 255],
                [8, 255, 214],
                [7, 255, 224],
                [255, 184, 6],
                [10, 255, 71],
                [255, 41, 10],
                [7, 255, 255],
                [224, 255, 8],
                [102, 8, 255],
                [255, 61, 6],
                [255, 194, 7],
                [255, 122, 8],
                [0, 255, 20],
                [255, 8, 41],
                [255, 5, 153],
                [6, 51, 255],
                [235, 12, 255],
                [160, 150, 20],
                [0, 163, 255],
                [140, 140, 140],
                [250, 10, 15],
                [20, 255, 0],
                [31, 255, 0],
                [255, 31, 0],
                [255, 224, 0],
                [153, 255, 0],
                [0, 0, 255],
                [255, 71, 0],
                [0, 235, 255],
                [0, 173, 255],
                [31, 0, 255],
                [11, 200, 200],
                [255, 82, 0],
                [0, 255, 245],
                [0, 61, 255],
                [0, 255, 112],
                [0, 255, 133],
                [255, 0, 0],
                [255, 163, 0],
                [255, 102, 0],
                [194, 255, 0],
                [0, 143, 255],
                [51, 255, 0],
                [0, 82, 255],
                [0, 255, 41],
                [0, 255, 173],
                [10, 0, 255],
                [173, 255, 0],
                [0, 255, 153],
                [255, 92, 0],
                [255, 0, 255],
                [255, 0, 245],
                [255, 0, 102],
                [255, 173, 0],
                [255, 0, 20],
                [255, 184, 184],
                [0, 31, 255],
                [0, 255, 61],
                [0, 71, 255],
                [255, 0, 204],
                [0, 255, 194],
                [0, 255, 82],
                [0, 10, 255],
                [0, 112, 255],
                [51, 0, 255],
                [0, 194, 255],
                [0, 122, 255],
                [0, 255, 163],
                [255, 153, 0],
                [0, 255, 10],
                [255, 112, 0],
                [143, 255, 0],
                [82, 0, 255],
                [163, 255, 0],
                [255, 235, 0],
                [8, 184, 170],
                [133, 0, 255],
                [0, 255, 92],
                [184, 0, 255],
                [255, 0, 31],
                [0, 184, 255],
                [0, 214, 255],
                [255, 0, 112],
                [92, 255, 0],
                [0, 224, 255],
                [112, 224, 255],
                [70, 184, 160],
                [163, 0, 255],
                [153, 0, 255],
                [71, 255, 0],
                [255, 0, 163],
                [255, 204, 0],
                [255, 0, 143],
                [0, 255, 235],
                [133, 255, 0],
                [255, 0, 235],
                [245, 0, 255],
                [255, 0, 122],
                [255, 245, 0],
                [10, 190, 212],
                [214, 255, 0],
                [0, 204, 255],
                [20, 0, 255],
                [255, 255, 0],
                [0, 153, 255],
                [0, 41, 255],
                [0, 255, 204],
                [41, 0, 255],
                [41, 255, 0],
                [173, 0, 255],
                [0, 245, 255],
                [71, 0, 255],
                [122, 0, 255],
                [0, 255, 184],
                [0, 92, 255],
                [184, 255, 0],
                [0, 133, 255],
                [255, 214, 0],
                [25, 194, 194],
                [102, 255, 0],
                [92, 0, 255],
            ]
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'Cityscapes':
            updated_colors = [
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
            ]
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'COCO-Stuff':
            updated_colors = colors.copy()
            for i in range(num_classes-1):
                updated_colors[i] = updated_colors[i+1]
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'Mapillary-Vistas':
            updated_colors = np.asarray([
                [165, 42, 42],
                [0, 192, 0],
                [196, 196, 196],
                [190, 153, 153],
                [180, 165, 180],
                [102, 102, 156],
                [102, 102, 156],
                [128, 64, 255],
                [140, 140, 200],
                [170, 170, 170],
                [250, 170, 160],
                [96, 96, 96],
                [230, 150, 140],
                [128, 64, 128],
                [110, 110, 110],
                [244, 35, 232],
                [150, 100, 100],
                [70, 70, 70],
                [150, 120, 90],
                [220, 20, 60],
                [255, 0, 0],
                [255, 0, 0],
                [255, 0, 0],
                [200, 128, 128],
                [255, 255, 255],
                [64, 170, 64],
                [128, 64, 64],
                [70, 130, 180],
                [255, 255, 255],
                [152, 251, 152],
                [107, 142, 35],
                [0, 170, 30],
                [255, 255, 128],
                [250, 0, 30],
                [0, 0, 0],
                [220, 220, 220],
                [170, 170, 170],
                [222, 40, 40],
                [100, 170, 30],
                [40, 40, 40],
                [33, 33, 33],
                [170, 170, 170],
                [0, 0, 142],
                [170, 170, 170],
                [210, 170, 100],
                [153, 153, 153],
                [128, 128, 128],
                [0, 0, 142],
                [250, 170, 30],
                [192, 192, 192],
                [220, 220, 0],
                [180, 165, 180],
                [119, 11, 32],
                [0, 0, 142],
                [0, 60, 100],
                [0, 0, 142],
                [0, 0, 90],
                [0, 0, 230],
                [0, 80, 100],
                [128, 64, 64],
                [0, 0, 110],
                [0, 0, 70],
                [0, 0, 192],
                [32, 32, 32],
                [0, 0, 0],
                [0, 0, 0],
                ])
            for i, color in enumerate(updated_colors): colors[i] = color
        elif data == 'PascalContext':
            updated_colors = get_colors(num_classes+1)
            updated_colors[:num_classes] = updated_colors[1:num_classes+1]
            for i, color in enumerate(updated_colors): colors[i] = color
        else:
            updated_colors = None
            # raise ValueError(f'Please check {data}')

    if color_format.lower() == 'bgr': colors = colors[:, ::-1] # RGB to BGR

    return colors

def visualize_heatmaps(heatmaps, tags=None, image=None, option='SEISMIC', norm=False):
    vis_heatmaps = []

    if image is not None:
        draw_text(image, 'Input', (0, 0))
        vis_heatmaps.append(image)

    if tags is None:
        tags = [None for _ in heatmaps]

    for tag, heatmap in zip(tags, heatmaps):
        if norm: 
            min_v = heatmap.min()
            max_v = heatmap.max()
            heatmap = (heatmap - min_v) / (max_v - min_v + 1e-5)
        
        heatmap = colorize(heatmap, option)
        if tag is not None: draw_text(heatmap, tag, (0, 0), font_size=40)
        vis_heatmaps.append(heatmap)

    return np.concatenate(vis_heatmaps, axis=1)

def vstack(*images):
    return np.concatenate([image if len(image.shape) == 3 else convert(image) for image in images], axis=0)

def hstack(*images):
    return np.concatenate([image if len(image.shape) == 3 else convert(image) for image in images], axis=1)

def convert(image, code='gray2bgr'):
    if code == 'gray2bgr':
        code = cv2.COLOR_GRAY2BGR
    elif code == 'bgr2gray':
        code = cv2.COLOR_BGR2GRAY
    elif code == 'bgr2rgb':
        code = cv2.COLOR_BGR2RGB
    elif code == 'rgb2bgr':
        code = cv2.COLOR_RGB2BGR
    return cv2.cvtColor(image, code)

def cv2pil(image: np.ndarray) -> Image:
    return Image.fromarray(convert(image, 'bgr2rgb'))

def pil2cv(image: Image) -> np.ndarray:
    return convert(np.asarray(image), 'rgb2bgr')

def get_size(image) -> tuple:
    if isinstance(image, np.ndarray): # cv
        size = tuple(image.shape[:2][::-1])
    else: # pillow
        size = image.size
    return size

def write_gif(path, images, duration=1000): # 1s per image
    gif_images = []
    for image in images:
        gif_images.append(Image.fromarray(convert(image, 'bgr2rgb')))
    
    gif_images[0].save(
        path, append_images=gif_images[1:],
        save_all=True, loop=0xff, duration=duration
    )

def download_image_from_url(image_url: str) -> Image.Image:
    try:
        result = requests.get(image_url)
        result.raise_for_status()
        if 'image' not in result.headers['content-type']: 
            raise ValueError("ContentIsNotImage")
        image = Image.open(BytesIO(result.content))
    except requests.ConnectionError as identifier:
        raise requests.ConnectionError("ConnectionError")
    except requests.HTTPError as identifier:
        raise requests.HTTPError("HTTPError")
    except requests.ConnectTimeout as identifier:
        raise requests.ConnectTimeout("ConnectTimeout")
    except requests.TooManyRedirects as identifier:
        raise requests.TooManyRedirects("TooManyRedirects")
    except ValueError as identifier:
        raise ValueError("ContentIsNotImage")
    return image