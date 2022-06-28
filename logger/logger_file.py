import os
import sys

import logging
from datetime import datetime

from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
# from logstash_async.formatter import LogstashFormatter
# from logstash_async.handler import AsynchronousLogstashHandler


# print(os.path)

log_folder_name = 'logs' # default is: logs ; secondary is logs_docker

def setup_file_logging():
    if not os.path.exists(f'./{log_folder_name}'):
        os.mkdir(f'./{log_folder_name}')

    filename = datetime.now().strftime(f"./{log_folder_name}/maBot_%d_%b_%Y_%H_%M.log") # "../logs/cryptoTrading_sys_%d_%b_%Y_%H_%M.%f.log"
    logging.StreamHandler(sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s - %(threadName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Define the RotatingFileHandler [rotate every midnight UTC]
    fh = TimedRotatingFileHandler(filename, when='midnight', interval=1, backupCount=365, utc=True)

    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Define the console handler (what we want to see being printed onto console)
    ch = logging.StreamHandler(stream=sys.stdout) # ch: consol handler
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# def setup_log_stash_logging():
#     host = get_config('LOGSTASH', 'host')
#     port = int(get_config('LOGSTASH', 'port'))
#     env = sys.argv[1:2][0].lower()
#     logger = logging.getLogger('python-logstash-logger')  # what does this do ?
#     logger.setLevel(logging.INFO)
#
#     logstash_formatter = LogstashFormatter(message_type='python-logstash',
#                                            extra_prefix='',
#                                            extra=dict(appName='sentiment-python',environment=env))
#
#     logstash_handler = AsynchronousLogstashHandler(host,port,database_path='')
#     logstash_handler.setFormatter(logstash_formatter)
#     logger.addHandler(logstash_handler)
#     return logger


# def setup_localhost_log_stash_logging():
#     if not os.path.exists('../logs'):
#         os.mkdir('../logs')
#
#     host = 'localhost'
#     port = 5959
#     # env = sys.argv[1:2][0].lower()
#     env = 'laptop'
#     logger = logging.getLogger('python-logstash-logger')  # what does this do ?
#     logger.setLevel(logging.INFO)
#
#     logstash_formatter = LogstashFormatter(message_type='python-logstash',
#                                            ensure_ascii=False,
#                                            extra_prefix='',
#                                            extra=dict(appName='sentiment-python', environment=env))
#
#
#     # If you don't want to write to a SQLite database, then you do
#     # not have to specify a database_path.
#     # NOTE: Without a database, messages are lost between process restarts.
#     # test_logger.addHandler(AsynchronousLogstashHandler(host, port))
#     logstash_handler = AsynchronousLogstashHandler(host, port, database_path="../logs/logstash.db")
#     logstash_handler.setFormatter(logstash_formatter)
#     logger.addHandler(logstash_handler)
#     return logger

print("logger is initialized...")
logger = setup_file_logging()  # log to local .log files

# logger = setup_localhost_log_stash_logging() # log to logstash.db

def run_logs():
    logger.info("print jere")
    logger.error("Warning or error")
    logger.warning("MAJOR EERROR")
    logger.info("ABCDEF")


if __name__=="__main__":
    run_logs()

##---------------------------------------------
# logger = None
# env = sys.argv[1:2][0].lower()
# if env == 'dev':
#     logger = setup_file_logging()  ## log to local files
# else:
#     logger = setup_log_stash_logging() ## log to online repo