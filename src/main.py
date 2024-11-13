from mdclogpy import Logger
import sys
import time
import signal

def main():

    global myLogger
    myLogger = Logger()

    global running
    running = True
    myLogger.mdclog_format_init(configmap_monitor=True)
    myLogger.error("This is an info log")
    myLogger.error("This is an error log")

    while(running):
        myLogger.info("HEALTHCHECK")
        time.sleep(10)

    return 0
