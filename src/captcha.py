from PIL import Image, ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
import PIL
import random
import numpy as np


class SimpleCaptcha(object):
    def __init__(self, fonts):
        self._width = 136
        self._height = 50
        self._fonts = fonts
        self.font_sizes = 42
        self.truefonts = tuple([
            truetype(n, self.font_sizes)
            for n in self._fonts
        ])

    def write(self, chars, output, blur: bool, **kwargs):
        im = self.create_captcha_image(chars)
        if blur:
            im = im.filter(ImageFilter.SMOOTH)
        return im.save(output, format='gif')

    @staticmethod
    def background(w, h):
        arr = np.ones((h, w, 3), dtype=np.uint8) * 255
        im = Image.fromarray(arr, mode='RGB')
        return im

    @staticmethod
    def random_line_color():
        r = random.randint(180, 220)
        g = random.randint(180, 220)
        b = random.randint(180, 220)
        return (r, g, b)

    @staticmethod
    def random_dot_color():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r, g, b)

    def _generate_char(self, c, color, draw, font):
        w, h = draw.textsize(c, font=font)

        dx = random.randint(2, 4)
        dy = 0
        im = Image.new('RGBA', (w + dx, h + dy))
        Draw(im).text((dx, dy), c, font=font, fill=color)

        return im

    def add_line_noise(self, image, draw):
        color = self.random_line_color()
        x1 = random.randint(0, self._width - 1)
        y1 = random.randint(0, self._height - 1)
        x2 = random.randint(x1 + 1, self._width)
        y2 = random.randint(y1 + 1, self._height)
        draw.line(
            [
                x1, y1, x2, y2,
            ],
            width=1, fill=color
        )

    def add_dot_noise(self, image, draw):
        color = self.random_dot_color()
        x1 = random.randint(0, self._width)
        y1 = random.randint(0, self._height)
        x2 = x1 + 1
        y2 = y1 + 1
        draw.ellipse(
            [
                x1, y1, x2, y2,
            ],
            width=1, fill=color
        )

    def create_captcha_image(self, chars):
        image = self.background(self._width, self._height)

        draw = Draw(image)
        for i in range(75):
            self.add_line_noise(image, draw)

        r_arr = [20, 50, 80, 100, 110]
        g_arr = [50, 50, 20, 20, 20]
        b_arr = [196, 150, 120, 70, 50]

        y = random.randint(2, 4)
        base_x = random.randint(2, 4)
        font = random.choice(self.truefonts)
        for i, c in enumerate(chars):
            r = int(min(max(0, r_arr[i] + random.normalvariate(20, 5)), 255))
            g = int(min(max(0, g_arr[i] + random.normalvariate(20, 5)), 255))
            b = int(min(max(0, b_arr[i] + random.normalvariate(20, 5)), 255))
            img = self._generate_char(c, color=(r, g, b), draw=draw, font=font)

            w, h = img.size
            x = base_x + i * 23
            image.paste(img, (x, y), mask=img)  # alpha = 0

        for i in range(200):
            self.add_dot_noise(image, draw)

        image = image.resize((68, 25), PIL.Image.BICUBIC)
        return image
