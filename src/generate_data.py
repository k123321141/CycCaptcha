# -*- coding: utf-8 -*-
from captcha import SimpleCaptcha
from tqdm import tqdm
import argparse
import os
from os.path import join
import random

VALID_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def output_random_image(n: int, fonts: list, length: int, output_dir: str, invalid_set: set):
    global VALID_CHARS
    image = SimpleCaptcha(fonts=fonts)
    for i in tqdm(range(n)):
        label = ''.join(random.choices(VALID_CHARS, k=length))
        while label in invalid_set:
            label = ''.join(random.choices(VALID_CHARS, k=length))
        invalid_set.add(label)
        image.write(
            label,
            os.path.join(output_dir, f'{label}.gif'),
            blur=False,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-l', '--length', type=int, default=5)
    parser.add_argument('-f', '--font_dir', type=str, default='../fonts/')
    parser.add_argument(
        '-o', '--output',
        default='./outputs',
        type=str
    )
    args = parser.parse_args()
    print(args)
    font_dir = args.font_dir
    fonts = [join(font_dir, f) for f in os.listdir(font_dir) if os.path.isfile(join(font_dir, f))]
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    output_random_image(args.n, fonts, args.length, args.output, set())