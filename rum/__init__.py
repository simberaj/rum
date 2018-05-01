import argparse

from .core import Initializer, GridMaker
     
def defaultArgumentParser(description, schema=True):
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('-c', '--conf', metavar='file', help='database connection configuration file')
    if schema:
        argparser.add_argument('schema', help='the RUM analysis schema')
    return argparser

