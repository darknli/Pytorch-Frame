import logging
import os
import sys
from typing import Optional
from .dist_utils import get_rank
from termcolor import colored

logger_initialized = {}


class _ColorfulFormatter(logging.Formatter):
    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "magenta")
        elif record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(
    name: Optional[str] = None,
    output: Optional[str] = None,
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.INFO,
    color: bool = False,
) -> logging.Logger:
    """
    初始化 logger

    如果logger没被初始化, 这个函数会使用1~2个handlers来初始化logger。否则这个已经初始化过的logger会直接被返回。
    在初始化时只有主程序的logger会添加handlers，以:class:`StreamHandler`的形式添加。
    如果output被赋值, :class:`FileHandler` 也会被添加

    这里是常用的用法. 我们假设文件结构如下::

        project
        ├── module1
        └── module2

    - Only setup the parent logger (``project``), then all children loggers
      (``project.module1`` and ``project.module2``) will use the handlers of the parent logger.

    Example::

        >>> setup_logger(name="project")
        >>> logging.getLogger("project.module1")
        >>> logging.getLogger("project.module2")

    - Only setup the root logger, then all loggers will use the handlers of the root logger.

    Example::

        >>> setup_logger()
        >>> logging.getLogger(name="project")
        >>> logging.getLogger(name="project.module1")
        >>> logging.getLogger(name="project.module2")

    - Setup all loggers, each logger uses independent handlers.

    Example::

        >>> setup_logger(name="project")
        >>> setup_logger(name="project.module1")
        >>> setup_logger(name="project.module2")

    Parameters
    ----------
    name : str, default None
        Logger 名字。
    output : str, default None
        一个保存log的文件名或目录名
        * None: 不会保存log到文件
        * 后缀带.txt或.log : 将其设置成文件名
        * other : 文件名变为 output/log.txt
    console_log_level : int, default logging.INFO
        logger输出到控制台/终端的等级
    file_log_level : int, default logging.INFO
        logger输出到文件的等级
    color : bool, default False
        如果是True，logger将会有颜色

    Returns
    -------
        logging.Logger: 一个已初始化的logger
    """
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(console_log_level)
    # the messages of this logger are not propagated to its parent
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s %(name)s %(levelname)s]: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout and file logging: master only
    if get_rank() == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(console_log_level)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if output is not None:
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, "log.txt")
            # If a single file name is passed as argument, os.path.dirname() will return an empty
            # string. For example, os.path.dirname("log.txt") == "". This will cause an error
            # in os.makedirs(). So we need to wrap `filename` with os.path.abspath().
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            fh = logging.FileHandler(filename)
            fh.setLevel(file_log_level)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

    logger_initialized[name] = logger
    return logger
