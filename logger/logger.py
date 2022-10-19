import logging
import sys

formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)

logger = logging.getLogger('global')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

