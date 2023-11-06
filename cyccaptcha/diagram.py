from PIL import Image, ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
import PIL
import random
import numpy as np


class DiagramCaptcha(object):
    def __init__(self, fonts):
        self._width = 128
        self._height = 128
        self._fonts = fonts
        self.font_sizes = 18
        self.truefonts = [
            truetype(n, self.font_sizes)
            for n in self._fonts
            # if 'bold' not in n.lower() and 'black' not in n.lower()
        ]
        self.text_list = [
            '戶籍暨通訊資料變更申請書',
            '列印畫面',
            '對保檢核要向驗印',
            '交叉覆核',
            '簽名',
            '依申請書或經客服確認變更',
            '客戶是否辦理信託',
            '信用卡',
            '保管箱地址由分行保管',
            '文件歸檔',
            '分行作業流程',

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
        # arr = np.ones((h, w, 3), dtype=np.uint8) * 255
        arr = (np.random.rand(h, w, 3) * 5 + 250).astype(np.uint8)
        im = Image.fromarray(arr, mode='RGB')
        return im

    @staticmethod
    def random_line_color():
        r = random.randint(180, 220)
        g = random.randint(180, 220)
        b = random.randint(180, 220)
        return (r, g, b)

    @staticmethod
    def random_light_color():
        r = random.randint(200, 250)
        g = random.randint(200, 250)
        b = random.randint(200, 250)
        return (r, g, b)

    @staticmethod
    def random_dark_color():
        r = random.randint(0, 50)
        g = random.randint(0, 50)
        b = random.randint(0, 50)
        return (r, g, b)

    def _generate_char(self, c, color, draw, font, base_x, base_y):
        w, h = draw.textsize(c, font=font)

        dx = random.randint(2, 4)
        dy = 0
        im = Image.new('RGBA', (w + dx, h + dy))
        Draw(im).text((dx, dy), c, font=font, fill=color)

        return im

    def _generate_rectangle_char(self, c, color, draw, font, base_x, base_y):
        w, h = draw.textsize(c, font=font)

        dx = random.randint(2, 15)
        dy = random.randint(2, 15)

        im = Image.new('RGBA', (w + 2 * dx, h + 2 * dy))

        fill = None if random.random() < 0.8 else DiagramCaptcha.random_light_color()
        sub_draw = Draw(im)

        sub_draw.rounded_rectangle(
            [(0, 0), (w + 2 * dx, h + 2 * dy)],
            fill=fill,
            radius=random.randint(0, 10),
            outline=DiagramCaptcha.random_dark_color(),
            width=random.randint(1, 3),
        )
        sub_draw.text((dx, dy), c, font=font, fill=color)

        return im

    def _generate_inverted_rectangle_char(self, c, color, draw, font, base_x, base_y):
        w, h = draw.textsize(c, font=font)

        dx = random.randint(2, 15)
        dy = random.randint(2, 15)

        im = Image.new('RGBA', (w + 2 * dx, h + 2 * dy))

        sub_draw = Draw(im)
        fill = None if random.random() < 0.8 else DiagramCaptcha.random_light_color()

        r = w // 2
        offset = 40
        draw.polygon(
            [(0 + base_x, offset + base_y), (r + base_x, offset - r + base_y), (2 * r + base_x, offset + base_y), (r + base_x, offset + r + base_y)],
            fill=fill,
            outline=DiagramCaptcha.random_dark_color(),
            width=random.randint(1, 5),
            # width=5,
        )

        sub_draw.text((dx, dy), c, font=font, fill=color)
        return im

    def split_text(self, text):
        split_point = random.randint(5, 10)
        if len(text) > split_point:
            return text[:split_point] + '\n' + text[split_point:]
        else:
            return text

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

    def create_captcha_image(self, chars):
        image = self.background(self._width, self._height)

        draw = Draw(image)

        r = random.randint(0, 30)
        g = random.randint(0, 30)
        b = random.randint(0, 30)

        y = random.randint(-2, 2)

        fn = random.choice([self._generate_char, self._generate_rectangle_char, self._generate_inverted_rectangle_char])

        for i in range(1):
            font = random.choice(self.truefonts)
            x = random.randint(0, self._width - 100)
            y = random.randint(0, self._height - 70)
            img = fn(chars, color=(r, g, b), draw=draw, font=font, base_x=x, base_y=y)
            image.paste(img, (x, y), mask=img)  # alpha = 0

        image = image.resize((self._width * 10, self._height * 10), PIL.Image.BICUBIC)
        # image = image.filter(ImageFilter.GaussianBlur(1))
        image = image.filter(ImageFilter.SHARPEN)
        # image = image.filter(ImageFilter.SHARPEN)
        # image = image.filter(ImageFilter.SHARPEN)
        image = image.resize((self._width, self._height), PIL.Image.BICUBIC)
        draw = Draw(image)

        return image
