import sys


def error_message_detail(error, error_detail: sys):
    """
    Builds a detailed error message including the file name, line number,
    and the actual error text. This is far more useful for debugging than
    Python's default traceback when you're scanning logs in production.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    A single custom exception class used across the entire project.
    Every component wraps its try/except blocks with this so that,
    no matter where something fails, the log/error message tells you
    exactly which file and line caused it.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
