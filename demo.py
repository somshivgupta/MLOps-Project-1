#from src.logger import logging

#logging.debug("This is a debug message.")
#logging.info("This is an info message.")
#logging.warning("This is an Warning Message")
#logging.error("This is an error message.")
#logging.critical("This is a critical message.")

#Below code is to check
from src.logger import logging
from src.exception import MyException
import sys

try:
    a = 1 + 'Z'
except Exception as e:
    logging.info(e)
    raise MyException(e, sys) from e