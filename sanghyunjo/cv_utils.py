# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import os
import cv2
import cmapy
import numpy as np

from PIL import ImageFont, ImageDraw, Image

Image.MAX_IMAGE_PIXELS = None # to read unlimited pixels like a large tiff format

ESC = 27
SPACE = 32
PLUS = ord('+')
MINUS = ord('-')

class MouseEventHandler:
    def __init__(self): self.clear()

    def get(self): 
        event = self.event; self.event = None
        return self.x, self.y, event
    
    def clear(self): self.x, self.y, self.event, self.down = 0, 0, None, None
    def move(self, x, y): self.x, self.y, self.event = x, y, ('' if self.down is None else self.down)+'move'
    def leftdown(self, x, y): self.x, self.y, self.event, self.down = x, y, 'leftdown', 'left'
    def leftup(self, x, y): self.x, self.y, self.event, self.down = x, y, 'leftup', None
    def rightdown(self, x, y): self.x, self.y, self.event, self.down = x, y, 'rightdown', 'right'
    def rightup(self, x, y): self.x, self.y, self.event, self.down = x, y, 'rightup', None
    def wheelup(self): self.event = 'wheelup'
    def wheeldown(self): self.event = 'wheeldown'
    
    def __call__(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN: self.leftdown(x, y)
        elif event == cv2.EVENT_LBUTTONUP: self.leftup(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN: self.rightdown(x, y)
        elif event == cv2.EVENT_RBUTTONUP: self.rightup(x, y)
        elif event == cv2.EVENT_MOUSEMOVE: self.move(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL: 
            if flags > 0: self.wheelup()
            else: self.wheeldown()

def read_image(path, mode='opencv', unicode=False, mask=False):
    if mode == 'opencv': 
        if unicode: 
            return cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), 
                cv2.IMREAD_UNCHANGED
            )
        else: 
            return cv2.imread(path)
    else: 
        try: 
            image = Image.open(path)
            return image if mask else image.convert('RGB')
        except FileNotFoundError: 
            return None

def write_image(path, image, palette=None):
    if palette is None: cv2.imwrite(path, image)
    else:
        image = Image.fromarray(image.astype(np.uint8)).convert('P')
        image.putpalette(palette)
        image.save(path)

def read_video(path):
    return VideoReader(path)

def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    
    writer = VideoWriter(path, w, h, fps)
    for frame in frames:
        writer(frame)
    writer.close()

def set_mouse(winname, func):
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, func)

def get_colors(num_classes=20, ignore_index=255, color_format='RGB'):
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
    if color_format == 'BGR': colors = colors[:, ::-1] # RGB to BGR

    return colors

def colorize(cam, option='SEISMIC'):
    color_dict = {
        'JET': cv2.COLORMAP_JET,
        'HOT': cv2.COLORMAP_HOT,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'WINTER': cv2.COLORMAP_WINTER,
        'GRAY': cmapy.cmap('gray'),
        'SEISMIC': cmapy.cmap('seismic'),
    }
    
    if cam.dtype in [np.float32, np.float64]:
        cam = (cam * 255).astype(np.uint8)
    
    if len(cam.shape) == 3:
        cam = np.max(cam, axis=0)
    
    if option in color_dict:
        cam = cv2.applyColorMap(cam, color_dict[option])
    
    return cam

def get_default_font_path():
    # Get the absolute path to the font file included in the package
    return os.path.join(os.path.dirname(__file__), 'fonts', 'Times New Roman MT Std.otf')

def draw_text(
        image: np.ndarray, text: str, coordinate: tuple, color: tuple=(0, 0, 0), 
        font_path: str=None, font_size: int=20, 
        background: tuple=(79, 244, 255), centering: bool=True, padding: int=5
    ):
    if font_path is None:
        font_path = get_default_font_path()
    
    text = ' ' + text
    font = ImageFont.truetype(font_path, font_size)
    
    tw, th = font.getsize(text)
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

def draw_point(image, point, size, color, edge_color=(0, 0, 0)):
    x, y = point
    edge_size = max(size // 5, 1)

    b, g, r = color
    eb, eg, er = edge_color

    pillow_image = Image.fromarray(image)
    drw = ImageDraw.Draw(pillow_image)
    drw.ellipse(
        [(x-size, y-size), (x+size, y+size)], 
        (b, g, r, 0), (eb, eg, er, 0), edge_size
    )
    image[:, :, :] = np.asarray(pillow_image)

def draw_rect(image, xyxy, color=(79, 244, 255), thickness=1):
    cv2.rectangle(image, tuple(xyxy[:2]), tuple(xyxy[2:]), color, thickness)

def show_image(winname, image, wait=-1, title=''):
    cv2.imshow(winname, image)

    if len(title) > 0:
        cv2.setWindowTitle(winname, title)

    key = None
    if wait >= 0:
        key = cv2.waitKey(wait)

    return key

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

def resize(image, size=None, scale=None, mode='bicubic'):
    inp_dict = {
        'bicubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
    }
    if scale is not None:
        h, w = image.shape[:2]
        size = (int(w * scale), int(h * scale))
    return cv2.resize(image, size, interpolation=inp_dict[mode])

class VideoReader:
    """
    [Example 1]
    while True:
        frame = video()
        if frame is None:
            break

        cv2.imshow('Image', frame)
        cv2.waitKey(1)
    
    [Example 2]
    for i in range(0, len(video), video.fps):
        frame = video[i]
        
        cv2.imshow('Image', frame)
        cv2.waitKey(1)
    """
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)

        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __getitem__(self, index=None):
        if index is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, index)

        ret, frame = self.video.read()
        if not ret: frame = None

        return frame
    
    def __call__(self):
        return self.__getitem__()
    
class VideoWriter:
    def __init__(self, path, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps

        self.path = path
        self.open()

    def open(self):
        self.writer = cv2.VideoWriter(
            self.path, 
            cv2.VideoWriter_fourcc(*'MP4V'), 
            self.fps, (self.width, self.height)
        )

    def __call__(self, frame):
        self.writer.write(frame)
        
    def close(self):
        self.writer.release()
        self.writer = None

def vstack(*images):
    return np.concatenate(images, axis=0)

def hstack(*images):
    return np.concatenate(images, axis=1)

def gray2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def overlay(image1, image2, alpha):
    return cv2.addWeighted(image1, alpha, image2, 1. - alpha, 0.0)