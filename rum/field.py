

import numpy
from psycopg2 import sql
import pandas as pd

from . import core

SQL_AND = sql.SQL(' AND ')

class Handler(core.DatabaseTask):
    targetType = 'double precision'

    def selectValues(self, cur, table, fieldnames, where=None, order=None):
        qry = sql.SQL('SELECT {fieldnames} FROM {schema}.{table} {where} {order}').format(
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
        return self.resultToDF(cur, fieldnames)
    
    @staticmethod
    def resultToDF(cur, fieldnames):
        return pd.DataFrame.from_records(
            cur.fetchall(),
            columns=fieldnames,
            coerce_float=True
        )
    
    def where(self, table, inside=False, condition=True):
        joins = []
        conditions = []
        if inside:
            joins.append('grid')
            conditions.append(sql.SQL('{schema}.grid.inside').format(
                schema=self.schemaSQL
            ))
        if condition:
            if table != 'all_feats':
                joins.append('all_feats')
            conditions.append(sql.SQL('{schema}.all_feats.condition').format(
                schema=self.schemaSQL
            ))
        return self.createJoins(table, joins) + self.createConditions(conditions)
    
    def createJoins(self, source, joins):
        return sql.SQL('\n').join([
            sql.SQL('JOIN {schema}.{join} ON {schema}.{source}.geohash={schema}.{join}.geohash').format(
                schema=self.schemaSQL,
                join=sql.Identifier(join),
                source=sql.Identifier(source),
            )
            for join in joins
        ])
    
    def createConditions(self, conditions):
        if conditions:
            return sql.SQL(' WHERE ') + SQL_AND.join(conditions)
        else:
            return sql.SQL('')
    
    def selectConsolidatedFeatures(self, cur, fieldnames=None, **kwargs):
        df = self.selectValues(cur,
            'all_feats',
            fieldnames=fieldnames if fieldnames else self.getConsolidatedFeatureNames(cur),
            where=self.where('all_feats', **kwargs),
            order='geohash'
        )
        df['intercept'] = 1
        return df
            
    def selectTarget(self, cur, tablename, **kwargs):
        return self.selectValues(cur,
            tablename,
            ['target'],
            where=self.where(tablename, **kwargs),
            order='geohash'
        )['target']
        
    def createField(self, cur, name, overwrite=False):
        self.createFields(cur, [name], overwrite=overwrite)
        
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
    
