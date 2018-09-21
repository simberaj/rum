

import numpy
from psycopg2 import sql
import pandas as pd

from . import core

class Handler(core.DatabaseTask):
    targetType = 'double precision'

    def selectValues(self, cur, table, fieldnames, where=None, order=None):
        qry = sql.SQL('SELECT {fieldnames} FROM {schema}.{table}{where} {order}').format(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
            fieldnames=sql.SQL(', ').join(sql.Identifier(fn) for fn in fieldnames),
            where=(where if where else sql.SQL('')),
            order=(
                sql.SQL('ORDER BY {schema}.{table}.{order}').format(
                    schema=self.schemaSQL,
                    table=sql.Identifier(table),
                    order=sql.Identifier(order),
                ) if order else sql.SQL('')
            ),
        ).as_string(cur)
        self.logger.debug('executing select query: %s', qry)
        cur.execute(qry)
        self.logger.debug('retrieving data')
        df = pd.DataFrame.from_records(
            cur.fetchall(),
            columns=fieldnames
        )
        return df
    
    def insideCondition(self, table):
        return sql.SQL('JOIN {schema}.grid g ON {table}.geohash=g.geohash WHERE g.inside').format(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
        )
    
    def selectConsolidatedFeatures(self, cur, fieldnames=None, inside=False):
        df = self.selectValues(cur,
            'all_feats',
            fieldnames=fieldnames if fieldnames else self.getConsolidatedFeatureNames(cur),
            where=self.insideCondition('all_feats') if inside else None,
            order='geohash'
        )
        df['intercept'] = 1
        return df
            
    def selectTarget(self, cur, tablename, inside=False):
        return self.selectValues(cur,
            tablename,
            ['target'],
            where=self.insideCondition(tablename) if inside else None,
            order='geohash'
        )['target']
    
    
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
            ).as_string(self.cursor)
            # self.logger.debug('retrieving unique values: %s', uniqueQry)
            self.cursor.execute(uniqueQry)
            uniques = list(sorted(
                row[0] for row in self.cursor.fetchall()
                if row[0] is not None
            ))
        return uniques if uniques else [None]
    
