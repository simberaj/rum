import sys
import os
import logging
import contextlib
import json

import psycopg2
from psycopg2 import sql


ROOT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
LOG_PATH = os.path.join(ROOT_PATH, 'log')
DBINIT_PATH = os.path.join(ROOT_PATH, 'sql', 'dbinit.sql')

DEFAULT_DB_CONF_PATH = os.path.join(ROOT_PATH, 'dbconn.json')

EXTENT_TABLE = 'extent'
GRID_TABLE = 'grid'


class Task:
    def __init__(self, connConfig, schema='rum'):
        if 'schema' in connConfig:
            self.schema = connConfig['schema']
            del connConfig['schema']
        else:
            self.schema = schema
        self.connConfig = connConfig
        self._startLogging()
    
    def _startLogging(self):
        self.logname = 'rum.' + self.__class__.__name__.lower()
        if self.schema:
            self.logname += ('.' + self.schema)
        self.logger = logging.getLogger(self.logname)
        self.logger.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(os.path.join(LOG_PATH, self.logname + '.log'))
        fileHandler.setLevel(logging.DEBUG)
        stdoutHandler = logging.StreamHandler(sys.stdout)
        stdoutHandler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        for handler in (fileHandler, stdoutHandler):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.debug('logging started')

    @contextlib.contextmanager
    def _connect(self):
        self.logger.debug('connecting to database %s', self.connConfig.get('dbname'))
        conn = psycopg2.connect(**self.connConfig)
        self.logger.debug('entering database transaction')
        with conn:
            cur = conn.cursor()
            yield cur
        self.logger.debug('exitting database transaction')

    @classmethod
    def fromArgs(cls, args):
        return cls.fromConfig(
            args.conf,
            (args.schema if hasattr(args, 'schema') else None)
        )
        
    @classmethod
    def fromConfig(cls, confpath=None, schema=None):
        if confpath is None:
            confpath = DEFAULT_DB_CONF_PATH
        with open(confpath) as infile:
            conf = json.load(infile)
        return cls(conf, schema=schema)
        
    def run(self, *args, **kwargs):
        self.logger.debug('starting %s', self.logname)
        try:
            self.main(*args, **kwargs)
        except Exception as exc:
            self.logger.exception(exc)
            raise
        self.logger.debug('successfully finished %s', self.logname)
        
    def main(self, *args, **kwargs):
        raise NotImplementedError
        
        
class Initializer(Task):
    def main(self):
        with self._connect() as cur:
            with open(DBINIT_PATH) as dbInitFile:
                queries = [qry.strip() for qry in dbInitFile.read().split(';')]
            for query in queries:
                if query: # disregard empty queries
                    self.logger.debug('initialization query: %s', query)
                    cur.execute(query)

class GridMaker(Task):
    def main(self, gridSize=100):
        with self._connect() as cur:
            qry = sql.SQL('''
                DROP TABLE IF EXISTS {schema}.{targetTable};
                CREATE TABLE {schema}.{targetTable}
                    AS SELECT
                        makegrid(geometry,{gridSize}) AS geometry
                    FROM {schema}.{extentTable};
                SELECT Populate_Geometry_Columns('{schema}.{targetTable}'::regclass);
                '''
            ).format(
                schema=sql.Identifier(self.schema),
                extentTable=sql.Identifier(EXTENT_TABLE),
                targetTable=sql.Identifier(GRID_TABLE),
                gridSize=sql.Literal(gridSize),
            ).as_string(cur)
            self.logger.debug('grid create query: %s', qry)
            cur.execute(qry)

