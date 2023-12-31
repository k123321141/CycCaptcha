# -*- coding: utf-8 -*-
import argparse
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from . import captcha


VALID_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def output_random_image(N: int, fonts: list, length: int, output_dir: str, invalid_set: set, ratio: float, verbose: bool):
    global VALID_CHARS
    assert 0 <= ratio <= 1
    generator_dict = {
        captcha.HardCaptcha(fonts=fonts): ratio,
        captcha.SimpleCaptcha(fonts=fonts): (1 - ratio),
    }

    for generator, ratio in generator_dict.items():
        n = int(N * ratio)
        print(f'Generating {n} images from {generator.__class__.__name__}')

        for i in tqdm(range(n)):
            label = ''.join(random.choices(VALID_CHARS, k=length))
            if verbose:
                label = '93518'
            while label in invalid_set:
                label = ''.join(random.choices(VALID_CHARS, k=length))
            invalid_set.add(label)
            im = generator.write(
                label,
                os.path.join(output_dir, f'{label}.gif'),
                blur=False,
                verbose=verbose,
            )
            if verbose:
                plt.imshow(im)
                plt.show()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-l', '--length', type=int, default=5)
    parser.add_argument('-f', '--font_dir', type=str, default='../fonts/')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.1)
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
    output_random_image(args.n, fonts, args.length, args.output, set(), args.ratio, args.verbose)
