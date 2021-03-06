import sys
import os
import logging
import logging.handlers
import contextlib
import json
import datetime

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
ALL_FEATS_TABLE = 'all_feats'

FEATURE_PREFIX = 'feat_'
NEIGHBOUR_FEATURE_PREFIX = FEATURE_PREFIX + 'neigh_'

TYPES_TO_POSTGRE = {
    str : 'text',
    float : 'double precision',
    bool : 'boolean',
    int : 'integer',
    datetime.datetime : 'datetime',
    datetime.date : 'date',
    datetime.time : 'time',
}


class Error(Exception):
    pass

class ConfigError(Error):
    pass

class InvalidParameterError(Error):
    pass

class InvalidContentsError(Error):
    pass

class Task:
    DEFAULT_CONF_PATH = None
    activeLoggers = []

    def __init__(self, schema=None):
        self.schema = schema if schema else 'rum'
        self.schemaSQL = sql.Identifier(self.schema)
        self._startLogging()

    def _startLogging(self):
        self.logname = 'rum.' + self.__class__.__name__.lower()
        if self.schema:
            self.logname += ('.' + self.schema)
        self.logger = logging.getLogger(self.logname)
        if self.logger not in self.activeLoggers:
            self.logger.setLevel(logging.DEBUG)
            fileHandler = logging.handlers.RotatingFileHandler(
                os.path.join(LOG_PATH, self.logname + '.log'),
                maxBytes=10000000,
                backupCount=3
            )
            fileHandler.setLevel(logging.DEBUG)
            stdoutHandler = logging.StreamHandler(sys.stdout)
            stdoutHandler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
            for handler in (fileHandler, stdoutHandler):
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.activeLoggers.append(self.logger)
            self.logger.debug('logging started')


    @classmethod
    def fromConfig(cls, schema=None):
        return cls(schema=schema)

    def run(self, *args, **kwargs):
        self.logger.debug('starting %s', self.logname)
        try:
            result = self.main(*args, **kwargs)
        except Exception as exc:
            self.logger.exception(exc)
            raise
        self.logger.debug('successfully finished %s', self.logname)
        return result

    def main(self, *args, **kwargs):
        raise NotImplementedError


class DatabaseTask(Task):
    def __init__(self, connector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connector = connector
        self.connector.logTo(self.logger)

    @contextlib.contextmanager
    def _connect(self, autocommit=False):
        with self.connector.connect(autocommit=autocommit) as cursor:
            yield cursor

    @contextlib.contextmanager
    def _connect_raw(self, autocommit=False):
        with self.connector.connect_raw(autocommit=autocommit) as conn:
            yield conn

    @classmethod
    def fromArgs(cls, args):
        return cls.fromConfig(
            args.dbconf,
            args.schema if hasattr(args, 'schema') else None
        )

    @classmethod
    def fromConfig(cls, connConfig, schema):
        return cls(Connector.fromConfig(connConfig), schema=schema)

    def getTableNames(self, cur, where=None):
        qry = sql.SQL('''
            SELECT table_name FROM information_schema.tables
            WHERE table_schema={schema} AND ({where});
        ''').format(
            schema=sql.Literal(self.schema),
            where=(where if where else sql.SQL('TRUE')),
        ).as_string(cur)
        self.logger.debug('selecting grid columns: %s', qry)
        cur.execute(qry)
        return [row[0] for row in cur.fetchall()]

    def getColumnNames(self, cur, where=None, schema=None):
        qry = sql.SQL('''
            SELECT table_name, column_name FROM information_schema.columns
            WHERE table_schema={schema} AND {where}
            ORDER BY table_name, ordinal_position;
        ''').format(
            schema=sql.Literal(self.schema if schema is None else schema),
            where=where
        ).as_string(cur)
        self.logger.debug('selecting columns: %s', qry)
        cur.execute(qry)
        colnames = {}
        for table, column in cur.fetchall():
            colnames.setdefault(table, []).append(column)
        return colnames

    def getColumnNamesForTable(self, cur, table, where=None, schema=None):
        return self.getColumnNames(cur,
            where=sql.SQL('table_name={table} AND {where}').format(
                table=sql.Literal(table),
                where=(where if where else sql.SQL('TRUE'))
            ),
            schema=schema,
        ).get(table, [])

    def getFeatureNames(self, cur):
        return self.getColumnNames(cur,
            where=sql.SQL("table_name LIKE 'feat_%' AND column_name<>'geohash'")
        )

    def getSRID(self, cur, table, geomField=GEOMETRY_FIELD):
        qry = sql.SQL(
            "SELECT Find_SRID({schema}, {table}, {geomField});"
        ).format(
            schema=sql.Literal(self.schema),
            table=sql.Literal(table),
            geomField=sql.Literal(geomField),
        ).as_string(cur)
        self.logger.debug('determining SRID of %s: %s', table, qry)
        cur.execute(qry)
        # there is an extent defined, make the imported data conform to its CRS
        result = cur.fetchone()
        if result is None:
            raise ValueError('table {} does not exist'.format(table))
        else:
            return result[0]

    def getConsolidatedFeatureNames(self, cur, condition=False):
        return [col
            for col in self.getColumnNamesForTable(cur, 'all_feats')
            if col != 'geohash' and (condition or col != 'condition')
        ]

    def createTable(self, cur, table, coldef, overwrite=False):
        # TODO accept field sizes in coldef
        if overwrite:
            self.clearTable(cur, table, overwrite=overwrite)
        qry = sql.SQL('''CREATE TABLE {schema}.{table} ({fieldDefs})''').format(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
            fieldDefs=sql.SQL(', ').join(
                sql.SQL(' ').join([
                    sql.Identifier(defitem[0]),
                    sql.SQL(TYPES_TO_POSTGRE[defitem[1]])
                ])
                for defitem in coldef.items()
            ),
        ).as_string(cur)
        self.logger.debug('creating table: %s', qry)
        cur.execute(qry)

    def clearTable(self, cur, table, overwrite=False):
        if overwrite:
            qry = sql.SQL('DROP TABLE IF EXISTS {schema}.{table};').format(
                schema=self.schemaSQL,
                table=sql.Identifier(table)
            )
            self.logger.debug('clearing table %s', table)
            cur.execute(qry)

    def createPrimaryKey(self, cur, table):
        qry = sql.SQL(
            '''ALTER TABLE {schema}.{table} ADD PRIMARY KEY (geohash);'''
        ).format(
            schema=self.schemaSQL,
            table=sql.Identifier(table)
        ).as_string(cur)
        self.logger.debug('creating primary key: %s', qry)
        cur.execute(qry)

    def createSchema(self, cur):
        cur.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}')
            .format(self.schemaSQL).as_string(cur)
        )



class Initializer(DatabaseTask):
    def main(self):
        with self._connect() as cur:
            with open(DBINIT_PATH) as dbInitFile:
                queries = [qry.strip() for qry in dbInitFile.read().split(';')]
            for query in queries:
                if query: # disregard empty queries
                    self.logger.debug('initialization query: %s', query)
                    cur.execute(query)
            self.createSchema(cur)


class ExtentMaker(DatabaseTask):
    def main(self, table, overwrite=False):
        with self._connect() as cur:
            if overwrite:
                self.logger.debug('dropping previous extent')
                delqry = sql.SQL('''DROP TABLE IF EXISTS {schema}.extent''').format(
                    schema=self.schemaSQL,
                ).as_string(cur)
                cur.execute(delqry)
            qry = sql.SQL(
                '''CREATE TABLE {schema}.extent AS SELECT
                ST_Union(geometry) as geometry
                FROM {schema}.{table}'''
            ).format(
                schema=self.schemaSQL,
                table=sql.Identifier(table)
            ).as_string(cur)
            self.logger.debug('creating extent: %s', qry)
            cur.execute(qry)
            popqry = sql.SQL('''SELECT Populate_Geometry_Columns(
                '{schema}.extent'::regclass
            );''').format(schema=self.schemaSQL).as_string(cur)
            self.logger.debug('registering extent geometry: %s', popqry)
            cur.execute(popqry)


class GridMaker(DatabaseTask):
    createPattern = sql.SQL('''
        CREATE TABLE {schema}.{grid}
            AS (WITH rawgrid AS
                (SELECT
                    makegrid(geometry,{gridSize},{xoffset},{yoffset}) AS geometry
                    FROM {schema}.extent
                )
                SELECT
                    g.geometry,
                    ST_Within(g.geometry,e.geometry) as inside,
                    reverse(ST_GeoHash(
                        ST_Transform(ST_Centroid(g.geometry),4326)
                    )) as geohash
                FROM {schema}.extent e, rawgrid g
            );
        SELECT Populate_Geometry_Columns(('{schema}.' || {gridLit})::regclass);
        CREATE INDEX {indexName} ON {schema}.{grid} USING GIST (geometry);
    ''')

    def main(self, gridName='grid', gridSize=100, xoffset=0, yoffset=0, overwrite=False):
        with self._connect() as cur:
            indexName = '{}_{}_gix'.format(self.schema, gridName)
            self.clearTable(cur, gridName, overwrite=overwrite)
            qry = self.createPattern.format(
                schema=self.schemaSQL,
                indexName=sql.Identifier(indexName),
                gridSize=sql.Literal(gridSize),
                xoffset=sql.Literal(xoffset),
                yoffset=sql.Literal(yoffset),
                grid=sql.Identifier(gridName),
                gridLit=sql.Literal(gridName),
            ).as_string(cur)
            self.logger.debug('grid create query: %s', qry)
            cur.execute(qry)
            self.createPrimaryKey(cur, gridName)


class Connector:
    def __init__(self, config):
        self.config = config
        self.logger = EmptyLogger()

    def copy(self):
        return self.__class__(self.config.copy())

    def logTo(self, logger):
        self.logger = logger

    @classmethod
    def fromConfig(cls, config):
        if config is None:
            config = DEFAULT_DB_CONF_PATH
        if isinstance(config, str):
            config = loadConfig(config)
        return cls(config)

    @contextlib.contextmanager
    def connect(self, autocommit=False):
        with self.connect_raw(autocommit=autocommit) as conn:
            yield conn.cursor()

    @contextlib.contextmanager
    def connect_raw(self, autocommit=False):
        self.logger.debug('connecting to database %s', self.config.get('dbname'))
        self.connection = psycopg2.connect(**self.config)
        if autocommit:
            self.connection.autocommit = True
        else:
            self.logger.debug('entering database transaction')
        try:
            yield self.connection
        except:
            raise
        else:
            if not autocommit:
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
