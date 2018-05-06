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
CONFIG_PATH = os.path.join(ROOT_PATH, 'config')

def configPath(filename):
    return os.path.join(CONFIG_PATH, filename)

DEFAULT_DB_CONF_PATH = configPath('dbconn.json')

EXTENT_TABLE = 'extent'
GRID_TABLE = 'grid'
GEOMETRY_FIELD = 'geometry'

TYPES_TO_POSTGRE = {
    str : 'text',
    float : 'double precision',
    bool : 'boolean',
    int : 'bigint',
}


class Error(Exception):
    pass
    
class ConfigError(Error):
    pass


class Task:
    DEFAULT_CONF_PATH = None

    def __init__(self, schema=None):
        self.schema = schema if schema else 'rum'
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
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        for handler in (fileHandler, stdoutHandler):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.debug('logging started')

        
    @classmethod
    def fromConfig(cls, schema=None):
        return cls(schema=schema)
        
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
        
        
class DatabaseTask(Task):
    def __init__(self, connector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connector = connector
        self.connector.logTo(self.logger)
        
    @contextlib.contextmanager
    def _connect(self):
        with self.connector as cursor:
            yield cursor
            
    @classmethod
    def fromArgs(cls, args):
        return cls.fromConfig(
            args.dbconf,
            args.schema if hasattr(args, 'schema') else None
        )
        
    @classmethod
    def fromConfig(cls, connConfig, schema):
        return cls(Connector.fromConfig(connConfig), schema=schema)
   
   
class Initializer(DatabaseTask):
    def main(self):
        with self._connect() as cur:
            with open(DBINIT_PATH) as dbInitFile:
                queries = [qry.strip() for qry in dbInitFile.read().split(';')]
            for query in queries:
                if query: # disregard empty queries
                    self.logger.debug('initialization query: %s', query)
                    cur.execute(query)
            cur.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}')
                .format(sql.Identifier(self.schema)).as_string(cur)
            )

class ExtentMaker(DatabaseTask):
    def main(self, table, overwrite=False):
        with self._connect() as cur:
            if overwrite:
                self.logger.debug('dropping previous extent')
                delqry = sql.SQL('''DROP TABLE IF EXISTS {schema}.extent''').format(
                    schema=sql.Identifier(self.schema),
                ).as_string(cur)
                cur.execute(delqry)
            qry = sql.SQL(
                '''CREATE TABLE {schema}.extent AS SELECT
                ST_Union(geometry) as geometry
                FROM {schema}.{table}'''
            ).format(
                schema=sql.Identifier(self.schema),
                table=sql.Identifier(table)
            ).as_string(cur)
            self.logger.debug('creating extent: %s', qry)
            cur.execute(qry)
            
class GridMaker(DatabaseTask):
    def main(self, gridSize=100):
        with self._connect() as cur:
            qry = sql.SQL('''
                DROP TABLE IF EXISTS {schema}.{targetTable};
                CREATE TABLE {schema}.{targetTable}
                    AS SELECT
                        makegrid({geomField},{gridSize}) AS {geomField}
                    FROM {schema}.{extentTable};
                SELECT Populate_Geometry_Columns('{schema}.{targetTable}'::regclass);
                '''
            ).format(
                schema=sql.Identifier(self.schema),
                extentTable=sql.Identifier(EXTENT_TABLE),
                targetTable=sql.Identifier(GRID_TABLE),
                gridSize=sql.Literal(gridSize),
                geomField=sql.Identifier(GEOMETRY_FIELD),
            ).as_string(cur)
            self.logger.debug('grid create query: %s', qry)
            cur.execute(qry)


class Connector:
    def __init__(self, config):
        self.config = config
        self.logger = EmptyLogger()
    
    def logTo(self, logger):
        self.logger = logger
        
    @classmethod
    def fromConfig(cls, config):
        if config is None:
            config = DEFAULT_DB_CONF_PATH
        if isinstance(config, str):
            config = loadConfig(config)
        return cls(config)
    
    def __enter__(self):
        self.logger.debug('connecting to database %s', self.config.get('dbname'))
        self.connection = psycopg2.connect(**self.config)
        self.logger.debug('entering database transaction')
        return self.connection.cursor()
    
    def __exit__(self, exctype, *args):
        if not exctype:
            self.logger.debug('committing database transaction')
            self.connection.commit()

            
class EmptyLogger:
    def __bool__(self):
        return False

    def info(self, *args, **kwargs):
        pass
        
    def debug(self, *args, **kwargs):
        pass
        
    def warning(self, *args, **kwargs):
        pass
    
    
def loadConfig(path):
    with open(path, encoding='utf8') as infile:
        return json.load(infile)
