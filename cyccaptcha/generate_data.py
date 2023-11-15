# -*- coding: utf-8 -*-
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from diagram import DiagramCaptcha


def output_random_image(fonts: list, output_dir: str, verbose: bool, i: int, j: int):
    generator = DiagramCaptcha(fonts=fonts)

    print(f'Generating images from {generator.__class__.__name__}')
    with open('/volume/payo-ml/tmp/sent.txt', 'r') as fin:
        for idx, label in enumerate(tqdm(fin, total=13464)):
            label = label.strip()
            if idx < i:
                continue
            if idx >= j:
                break
            im = generator.write(
                label,
                os.path.join(output_dir, f'{label}.gif'),
                blur=False,
                verbose=verbose,
            )
            if verbose:
                thresh = 127

                def fn(x):
                    return 255 if x > thresh else 0
                im = im.convert('L').point(fn, mode='1')
                plt.imshow(im)
                plt.show()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--font_dir', type=str, default='../zh_fonts/')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-j', type=int, default=100)
    parser.add_argument(
        '-o', '--output',
        default='~/Desktop/',
        type=str
    )
    args = parser.parse_args()
    print(args)
    font_dir = args.font_dir
    fonts = [join(font_dir, f) for f in os.listdir(font_dir) if os.path.isfile(join(font_dir, f))]
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    output_random_image(fonts, args.output, args.verbose, args.i, args.j)
