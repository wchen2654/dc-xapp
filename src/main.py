import mdclogpy
import sys
import time
import signal
import os

def sigUsr(sigNum, stack):
    print("SIGNAL HANDLER")

    global running
    running = False

def main():

    global myLogger
    myLogger = mdclogpy.Logger()

    i = 0

    global running
    running = True
    myLogger.mdclog_format_init(configmap_monitor=True)
    myLogger.error("This is an error log")

    signal.signal(signal.SIGUSR1, sigUsr)

    with open("process.txt", "w") as f:
        f.write(str(os.getpid()))

    while(running):
        time.sleep(1)

    return 0
