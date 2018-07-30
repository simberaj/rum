

import numpy
from psycopg2 import sql
import pandas as pd

from . import core

class Handler(core.DatabaseTask):
    targetType = 'double precision'

    def selectValues(self, cur, fieldnames, inside=False, grid='grid'):
        qry = sql.SQL('SELECT {fieldnames} FROM {schema}.{grid}{where}').format(
            schema=self.schemaSQL,
            fieldnames=sql.SQL(', ').join(sql.Identifier(fn) for fn in fieldnames),
            where=sql.SQL(' WHERE inside' if inside else ''),
            grid=sql.Identifier(grid),
        ).as_string(cur)
        self.logger.debug('executing select query: %s', qry)
        cur.execute(qry)
        self.logger.debug('retrieving data')
        df = pd.DataFrame.from_records(
            cur.fetchall(),
            columns=fieldnames
        )
        df['intercept'] = 1
        return df

    
    def createField(self, cur, name, overwrite=False):
        self.createFields(cur, [name], overwrite=overwrite)
        # qry = sql.SQL('''ALTER TABLE {schema}.grid
            # ADD COLUMN {ifnex}{name} {coltype}
        # ''').format(
            # schema=self.schemaSQL,
            # ifnex=sql.SQL('IF NOT EXISTS ' if overwrite else ''),
            # name=sql.Identifier(name),
            # coltype=sql.SQL(self.targetType),
        # ).as_string(cur)
        # self.logger.debug('creating field: %s', name)
        # cur.execute(qry)
        
    def createFields(self, cur, names, overwrite=False):
        fieldChunks = [
            sql.SQL('ADD {ifnex}{colname} {coltype}').format(
                ifnex=sql.SQL('IF NOT EXISTS ' if overwrite else ''),
                colname=sql.Identifier(field),
                coltype=sql.SQL(self.targetType),
            )
            for field in names
        ]
        creator = sql.SQL('ALTER TABLE {schema}.grid {actions}').format(
            schema=self.schemaSQL,
            actions=sql.SQL(', ').join(fieldChunks)
        ).as_string(cur)
        self.logger.debug('creating fields: %s', creator)
        cur.execute(creator)
    

class UniquesGetter:
    def __init__(self, cursor, schema, table):
        self.cursor = cursor
        self.schemaSQL = sql.Identifier(schema)
        self.tableSQL = sql.Identifier(table)
    
    def get(self, field=None):
        uniques = []
        if field:
            uniqueQry = sql.SQL(
                'SELECT DISTINCT {field} FROM {schema}.{table}'
            ).format(
                schema=self.schemaSQL,
                table=self.tableSQL,
                field=sql.Identifier(field)
            ).as_string(cur)
            # self.logger.debug('retrieving unique values: %s', uniqueQry)
            cur.execute(uniqueQry)
            uniques = list(sorted(
                row[0] for row in cur.fetchall()
                if row[0] is not None
            ))
        return uniques if uniques else [None]
    
