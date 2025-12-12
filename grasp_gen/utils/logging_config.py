import logging
import sys

class COLORS:
    # Reference:
    # https://talyian.github.io/ansicolors/
    # https://bixense.com/clicolors/
    def __init__(self) -> None:
        pass

    @property
    def GREEN(self):
        return "\x1b[38;5;119m"

    @property
    def BLUE(self):
        return "\x1b[38;5;159m"

    @property
    def YELLOW(self):
        return "\x1b[38;5;226m"

    @property
    def RED(self):
        return "\x1b[38;5;9m"

    @property
    def CORN(self):
        return "\x1b[38;5;11m"

    @property
    def GRAY(self):
        return "\x1b[38;5;247m"

    @property
    def MINT(self):
        return "\x1b[38;5;121m"


class FORMATS:
    def __init__(self) -> None:
        pass

    @property
    def BOLD(self):
        return "\x1b[1m"

    @property
    def ITALIC(self):
        return "\x1b[3m"

    @property
    def UNDERLINE(self):
        return "\x1b[4m"

    @property
    def RESET(self):
        return "\x1b[0m"

colors = COLORS()
formats = FORMATS()

# Global flag to track if logging is initialized
_logging_initialized = False


class GraspGenFormatter(logging.Formatter):
    def __init__(self, verbose_time=True):
        super().__init__()

        self.mapping = {
            logging.DEBUG: colors.GREEN,
            logging.INFO: colors.BLUE,
            logging.WARNING: colors.YELLOW,
            logging.ERROR: colors.RED,
            logging.CRITICAL: colors.RED,
        }

        if verbose_time:
            self.TIME = "%(asctime)s.%(msecs)03d"
            self.DATE_FORMAT = "%y-%m-%d %H:%M:%S"
            self.INFO_length = 41
        else:
            self.TIME = "%(asctime)s"
            self.DATE_FORMAT = "%H:%M:%S"
            self.INFO_length = 28

        self.LEVEL = "%(levelname)s"
        self.MESSAGE = "%(message)s"

        self.last_output = ""
        self.last_color = ""

    def colored_fmt(self, color):
        self.last_color = color
        return f"{color}[GraspGen] [{self.TIME}] [{self.LEVEL}] {self.MESSAGE}{formats.RESET}"

    def format(self, record):
        log_fmt = self.colored_fmt(self.mapping.get(record.levelno))
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        msg = formatter.format(record)
        self.last_output = msg
        return msg


def setup_logging():
    """
    Set up basic logging configuration for all scripts.
    Uses INFO level and outputs to both console and file.
    """
    global _logging_initialized

    # Only initialize once
    if _logging_initialized:
        return
    
    # Configure grasp_gen logger specifically
    logger = logging.getLogger("grasp_gen")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Create handler if not exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = GraspGenFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _logging_initialized = True


def get_logger(name):
    """
    Get a logger instance for a specific module.

    Args:
        name: The name of the module (usually __name__)

    Returns:
        A logger instance configured for the module
    """
    # Ensure logging is initialized before getting a logger
    setup_logging()
    return logging.getLogger(name)


# Initialize logging when this module is imported
setup_logging()
