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
        self._fonts = [font for font in fonts if 'PublicSans-SemiBoldItalic' in font]
        self.font_sizes = 42
        self.truefonts = [
            truetype(n, self.font_sizes)
            for n in self._fonts
        ]

    def write(self, chars, output, blur: bool, verbose: bool, **kwargs):
        im = self.create_captcha_image(chars)
        if blur:
            im = im.filter(ImageFilter.SMOOTH)
        if verbose:
            return im
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
        x1 = random.randint(-50, self._width - 1)
        y1 = random.randint(-50, self._height - 1)
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
        x1 = random.randint(-40, self._width)
        y1 = random.randint(-40, self._height)
        '''
        x2 = x1 + 1
        y2 = y1 + 1
        draw.ellipse(
            [
                x1, y1, x2, y2,
            ],
            width=1, fill=color
        )
        '''
        font = truetype(self._fonts[0], random.randint(6, 32))
        draw.text((x1, y1), '.', font=font, fill=color)

    def create_captcha_image(self, chars):
        image = self.background(self._width, self._height)

        draw = Draw(image)
        for i in range(45):
            self.add_line_noise(image, draw)

        r_arr = [20, 50, 80, 100, 110]
        g_arr = [50, 50, 20, 20, 20]
        b_arr = [216, 180, 120, 70, 50]

        for i in range(100):
            self.add_dot_noise(image, draw)
        # image = image.filter(ImageFilter.SMOOTH)
        y = random.randint(-2, 4)
        base_x = random.randint(2, 4)
        font = random.choice(self.truefonts)
        for i, c in enumerate(chars):
            r = int(min(max(0, r_arr[i] + random.normalvariate(20, 10)), 255))
            g = int(min(max(0, g_arr[i] + random.normalvariate(20, 10)), 255))
            b = int(min(max(0, b_arr[i] + random.normalvariate(20, 10)), 255))
            img = self._generate_char(c, color=(r, g, b), draw=draw, font=font)

            w, h = img.size
            x = base_x + i * 23 + random.randint(-4, 4)
            image.paste(img, (x, y), mask=img)  # alpha = 0

        for i in range(200):
            self.add_dot_noise(image, draw)
        for i in range(15):
            self.add_line_noise(image, draw)

        image = image.resize((68 * 10, 25 * 10), PIL.Image.BICUBIC)
        image = image.filter(ImageFilter.GaussianBlur(1))
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SHARPEN)
        # image = image.resize((68, 25), PIL.Image.BICUBIC)
        image = image.resize((610, 224), PIL.Image.BICUBIC)
        return image


class HardCaptcha(SimpleCaptcha):

    def __init__(self, fonts):
        super().__init__(fonts)
        self.chars = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.table = []
        for i in range(256):
            self.table.append(i * 1.97)

    @staticmethod
    def random_gray():
        r = random.randint(115, 130)
        g = r + random.randint(-5, 5)
        b = r + random.randint(-5, 5)
        return (r, g, b)

    @staticmethod
    def random_dark_color():
        r = random.randint(70, 120)
        g = int(random.gauss(60, 15))
        b = random.randint(30, 80)
        return (r, g, b)

    @staticmethod
    def noised_background(w, h):
        m = random.randint(115, 185)
        arr = np.random.normal(m, 40, size=(h, w, 3)).astype(np.uint8)
        im = Image.fromarray(arr, mode='RGB')
        return im

    @staticmethod
    def random_line_color():
        r = random.randint(35, 70)
        g = random.randint(35, 70)
        b = random.randint(35, 70)
        return (31, 33, 32)
        return (r, g, b)

    @staticmethod
    def random_curve_color():
        r = random.randint(35, 70)
        g = random.randint(35, 70)
        b = random.randint(35, 70)
        return (r, g, b)

    def _generate_char(self, c, factor, color, draw, font):
        w, h = draw.textsize(c, font=font)

        dx = random.randint(0, 2)
        dy = random.randint(0, 2)
        im = Image.new('RGBA', (w + dx, h + dy))
        Draw(im).text((dx, dy), c, font=font, fill=color)

        # warp
        dx = w * random.uniform(0.5, 0.9)
        dy = h * random.uniform(0.6, 0.9)
        x1 = int(random.uniform(-dx, dx))
        y1 = int(random.uniform(-dy, dy))
        x2 = int(random.uniform(-dx, dx))
        y2 = int(random.uniform(-dy, dy))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        im = im.resize((w2, h2))
        # im = im.resize((random.randint(250, 350), random.randint(150, 250)))
        im = im.resize((int(w2*factor), int(h2*factor)), PIL.Image.BILINEAR)

        return im

    def add_char_noise(self, image, c, factor, pos, draw):
        font = random.choice(self.truefonts)
        img = self._generate_char(c, factor, self.random_gray(), draw, font)
        w, h = img.size
        mask = img.convert('L').point(self.table)
        image.paste(img, pos, mask)

    def add_line_noise(self, image, draw):
        color = self.random_line_color()
        draw.line(
            [
                0,
                random.randint(0, self._height),
                self._width,
                random.randint(-self._height * 2, 2 * self._height),
            ],
            width=7, fill=color
        )

    def _generate_curve(self, draw, freq, height):
        if freq < 1:
            width = 10
        elif 3 < freq < 10:
            width = 6
        else:
            width = 3

        im = Image.new('RGBA', (5000, 200))
        color = self.random_curve_color()
        r = 70
        w = width
#         offset = w//2+3
        offset = 2
        for i in range(0, 50):

            Draw(im).arc([i*2*r, 0, (i*2)*r + r, r], 270, 0, width=w, fill=color)
            Draw(im).arc([(i*2)*r + r - offset, 0, (i+1)*2*r, r], 90, 180, width=w, fill=color)
            Draw(im).arc([(i*2)*r + r - offset, 0, (i+1)*2*r, r], 0, 90, width=w, fill=color)
            Draw(im).arc([(i+1)*2*r - offset, 0, (i+1)*2*r+r + offset, r], 180, 270, width=w, fill=color)
        im = im.crop(im.getbbox())
        im = im.resize((int(1624*(1+freq)), height), PIL.Image.BICUBIC)
        return im

    def add_curve_noise(self, image, draw):
        mode = random.choices(['thin', 'normal', 'heavy', 'flat'], [0.25, 0.25, 0.25, 0.25], k=1)[0]
        if mode == 'thin':
            img = self._generate_curve(draw, random.uniform(0.3, 1), random.randint(100, 900))
            x = 0
            y = random.randint(-self._height//2, self._height//2)
        elif mode == 'normal':
            img = self._generate_curve(draw, random.uniform(1, 10), random.randint(400, 900))
            x = random.randint(-50, 0)
            y = random.randint(-500, 500)
        elif mode == 'heavy':
            img = self._generate_curve(draw, random.uniform(7, 8), random.randint(300, 500))
            x = random.randint(-50, 0)
            y = random.randint(-50, 150)
        elif mode == 'flat':
            img = self._generate_curve(draw, random.uniform(50, 90), random.randint(100, 400))
            x = int(-2500 * random.random())
            y = random.randint(-self._height//2, self._height//2)

        w, h = img.size
        image.paste(img, (x, y), mask=img)

    def add_dot_noise(self, image, draw):
        color = self.random_dot_color()
        x1 = random.randint(0, self._width)
        y1 = random.randint(0, self._height)
        x2 = x1 + random.randint(1, 2)
        y2 = y1 + random.randint(1, 2)
        draw.ellipse(
            [
                x1, y1, x2, y2,
            ],
            width=1, fill=color
        )

    def create_captcha_image(self, chars):
        if random.random() < 0.5:
            image = self.noised_background(self._width, self._height)
        else:
            image = super().background(self._width, self._height)

        draw = Draw(image)
        #         gray char noise
        for i in range(50):
            self.add_char_noise(
                image,
                c=random.choice(self.chars),
                factor=random.uniform(0.1, 0.7),
                pos=(random.randint(0, self._width), random.randint(0, self._height)),
                draw=draw,
            )
        self.add_line_noise(image, draw)
        self.add_curve_noise(image, draw)

        for i in range(100):
            super().add_line_noise(image, draw)

        y = random.randint(-2, 4)
        base_x = random.randint(2, 4)
        font = random.choice(self.truefonts)
        r_arr = [20, 50, 80, 100, 110]
        g_arr = [50, 50, 20, 20, 20]
        b_arr = [196, 150, 120, 70, 50]
        for i, c in enumerate(chars):
            r = int(min(max(0, r_arr[i] + random.normalvariate(10, 15)), 255))
            g = int(min(max(0, g_arr[i] + random.normalvariate(10, 15)), 255))
            b = int(min(max(0, b_arr[i] + random.normalvariate(10, 15)), 255))
            img = super()._generate_char(c, color=(r, g, b), draw=draw, font=font)

            w, h = img.size
            x = base_x + i * 23 + random.randint(-2, 2)
            image.paste(img, (x, y), mask=img)  # alpha = 0

        for i in range(200):
            self.add_dot_noise(image, draw)

        for i in range(20):
            super().add_line_noise(image, draw)
        # image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.5)))
        # image = image.resize((203, 66), PIL.Image.BICUBIC)
        image = image.filter(ImageFilter.SMOOTH_MORE)
        # image = image.resize((self._width, self._height), PIL.Image.BICUBIC)
        if random.random() < 0.5:
            image = image.filter(ImageFilter.SMOOTH)
        else:
            image = image.filter(ImageFilter.SMOOTH_MORE)
        image = image.resize((self._width * 4, self._height * 4), PIL.Image.BICUBIC)
        if random.random() < 0.5:
            image = image.filter(ImageFilter.SMOOTH)
        return image
