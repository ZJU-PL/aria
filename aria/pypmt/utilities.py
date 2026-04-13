import time

def log(message, level):
    from .config import global_config
    logger = global_config.get("logger")
    if level == 0:
        logger.critical(message)
    elif level == 1:
        logger.error(message)
    elif level == 2:
        logger.warning(message)
    elif level == 3:
        logger.info(message)
    elif level >= 4:
        logger.debug(message)

def timethis(level):
    """ Decorator to allow to cleanly time functions """
    def inner_decorator(f):
        def wrapped(*args, **kwargs):
            start_time = time.time()  # Record the start time
            res = f(*args, **kwargs)
            end_time = time.time()  # Record the end time
            time_elapsed = end_time - start_time  # Calculate time elapsed for this iteration
            log(f"{f.__module__}:{f.__name__}: {round(time_elapsed,2)}s", level)
            return res
        return wrapped
    return inner_decorator
