import os
import setuptools

# 如果readme文件中有中文，那么这里要指定encoding='utf-8'，否则会出现编码错误
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as readme:
    README = readme.read()

# 允许setup.py在任何路径下执行
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setuptools.setup(
    name="torch-frame",  # 库名, 需要在pypi中唯一
    version="1.7.5",  # 版本号
    author="Darkn Lxs",  # 作者
    author_email="1187220556@qq.com",  # 作看都将（方便使用索类现问图后成我我们）
    description="用于深度学习快速实现代码的框架",  # 简介
    long_description="见readme",  # 详细描述（一般会写在README.md中）
    long_description_content_type="text/markdown",  # README.md中描述的语法（一般为markdown)
    url="https://github.com/darknli/Pytorch-Frame/tree/main/torch_frame",  # 库/项目主页，放该项目的远程库地址即可
    packages=setuptools.find_packages(),  # 默认值即可，这个是方便以后我们给库拓展新功能的
    classifiers=[  # 指定该库依赖的Python版本、license、操作系统之类的
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[  # 该库需要的依前库
        "termcolor",
        "numpy>=1.17",
        "opencv-python",
        "tabulate",
        "torch",
        "transformers>=4.25.1",
        "accelerate>=0.16.0",
        "diffusers",
        "pyyaml",
        "tqdm"
    ],
    python_requires='>=3.6',
)

# python setup.py sdist bdist_wheel