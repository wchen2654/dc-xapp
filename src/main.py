from mdclogpy import Logger
import sys
import time
import signal

def main():

    global myLogger
    myLogger = Logger()

    i = 0

    global running
    running = True
    myLogger.mdclog_format_init(configmap_monitor=True)
    myLogger.debug("This is an info log")
    myLogger.error("This is an error log")

    while(i < 5):
        myLogger.debug("HEALTHCHECK")
        myLogger.info("HEALTHCHECK")
        print("hi")
        time.sleep(5)

        i += 1

    return 0
