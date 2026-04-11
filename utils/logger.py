import logging
import sys


def get_logger(name='DualSG', level=logging.INFO):
    """
    获取日志记录器

    参数:
        name: 日志记录器名称
        level: 日志级别

    返回:
        logging.Logger: 日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(console_handler)

    return logger


# 全局日志记录器
logger = get_logger()


def log(message):
    """
    记录信息日志

    参数:
        message: 日志消息
    """
    logger.info(message)


def log_debug(message):
    """
    记录调试日志

    参数:
        message: 日志消息
    """
    logger.debug(message)


def log_warning(message):
    """
    记录警告日志

    参数:
        message: 日志消息
    """
    logger.warning(message)


def log_error(message):
    """
    记录错误日志

    参数:
        message: 日志消息
    """
    logger.error(message)
