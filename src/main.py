from mdclogpy import Logger
import sys

global myLogger

def add(num1, num2):

    myLogger.info("Result is: " + str(num1 + num2))

    print("Python version", sys.version)
    print(sys.executable)

    return num1 + num2

def main(num1, num2):

    myLogger = Logger()
    myLogger.mdclog_format_init(configmap_monitor=True)
    myLogger.info("This is an info log")
    myLogger.error("This is an error log")

    add(num1, num2)
