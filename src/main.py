from mdclogpy import Logger
import sys
import time
import signal

def add(num1, num2):

    myLogger.info("Result is: " + str(num1 + num2))

    myLogger.info("Python version:", sys.version)
    myLogger.info(sys.executable)

    return num1 + num2

def main():

    global myLogger
    myLogger = Logger()

    global running
    running = true
    myLogger.mdclog_format_init(configmap_monitor=False)
    myLogger.info("This is an info log")

    while(running):
        myLogger.info("HEALTHCHECK")
        time.sleep(10)

    return 0
