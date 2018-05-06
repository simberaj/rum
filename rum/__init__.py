import argparse

from .core import Initializer, GridMaker, ExtentMaker
     
def defaultArgumentParser(description, schema=True):
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('-d', '--dbconf', metavar='conffile', help='database connection configuration file')
    # argparser.add_argument('-c', '--taskconf', metavar='file', help='task-specific configuration file')
    if schema:
        argparser.add_argument('schema', help='the RUM analysis schema')
    return argparser

