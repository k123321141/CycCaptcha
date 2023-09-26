from setuptools import setup, find_packages


install_requires = [
    'matplotlib>=3.7.1',
    'Pillow>=9.4.0',
    'tensorboard>=2.12.2',
    'tensorboardX>=2.6',
    'torch>=2.0.0',
    'torchvision>=0.15.1',
    'tqdm>=4.65.0',
    'transformers>=4.29.2',
    'protobuf==3.20.0',
    'numpy>=3.19.0',
]


setup(
    name='cyccaptcha',
    version='0.1.0',
    author='payo',
    author_email='kk123321141@gmail.com',
    description='It is a module to solve cyc captcha.',
    install_requires=install_requires,
    python_requires='>=3.8',
    packages=find_packages(),
)
