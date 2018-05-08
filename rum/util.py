import json
from psycopg2 import sql

from . import core

class PassThrough:
    pass

class Recategorizer(core.DatabaseTask):
    def main(self, table, source, target, translation, overwrite=False):
        transDict, default, leading = self.loadTranslation(translation)
        targetType = core.TYPES_TO_POSTGRE[type(next(iter(transDict.values())))]
        with self._connect() as cur:
            self.createTargetField(cur, table, target, targetType, overwrite)
            qry = sql.SQL('UPDATE {schema}.{table} SET {target} = ').format(
                schema=self.schemaSQL,
                table=sql.Identifier(table),
                target=sql.Identifier(target),
            ) + self.caseQuery(cur, source, transDict, default, leading)
            self.logger.debug('recategorizing with %s', qry)
            cur.execute(qry)
        
    def loadTranslation(self, transPath):
        with open(transPath, encoding='utf8') as infile:
            transDict = json.load(infile)
        if 'translation' in transDict:
            return (
                transDict['translation'],
                transDict.get('default', PassThrough),
                bool(transDict.get('leading', False))
            )
        else:
            return transDict, PassThrough, False
        
    def createTargetField(self, cur, table, colname, coltype, overwrite):
        opts = dict(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
            colname=sql.Identifier(colname)
        )
        if overwrite:
            delqry = sql.SQL(
                'ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS {colname}'
            ).format(**opts)
            self.logger.debug('dropping target column: %s', delqry)
            cur.execute(delqry)
        qry = sql.SQL('''ALTER TABLE {schema}.{table}
            ADD COLUMN {colname} {coltype}''').format(
            coltype=sql.SQL(coltype), **opts
        ).as_string(cur)
        self.logger.debug('adding recategorized column: %s', qry)
        cur.execute(qry)
        
    def caseQuery(self, cur, column, translation, default=PassThrough, leading=False):
        colSQL = sql.Identifier(column)
        cases = [
            sql.SQL('WHEN {column} {matcher} {fromval} THEN {toval}').format(
                column=colSQL,
                matcher=sql.SQL('LIKE' if leading else '='),
                fromval=sql.Literal((fromval + '%') if leading else fromval),
                toval=sql.Literal(toval),
            )
            for fromval, toval in translation.items()
        ]
        defaultSQL = colSQL if default is PassThrough else sql.Literal(default)
        return (
            sql.SQL('CASE\n') + 
            sql.SQL('\n').join(cases) + 
            sql.SQL('\nELSE {} END').format(defaultSQL)
        )

class ShapeCalculator(core.DatabaseTask):
    fields = [
        ('area', 'ST_Area(geometry)'),
        ('perim_index', 'ST_Perimeter(geometry) / (3.5449077 * sqrt(ST_Area(geometry)))'),
        ('fullness_index', 'ST_Area(ST_Buffer(geometry,0.177245385 * sqrt(ST_Area(geometry)))) / ST_Area(geometry)'),
        ('depth_index', 'ST_Area(ST_Buffer(geometry,-0.177245385 * sqrt(ST_Area(geometry)))) / ST_Area(geometry)'),
        ('concavity_index', '1 - (ST_Area(geometry) / ST_Area(ST_ConvexHull(geometry)))'),
        ('detour_index', '3.5449077 * sqrt(ST_Area(geometry)) / ST_Perimeter(ST_ConvexHull(geometry))'),
    ]
    createSQL = 'ALTER TABLE {schema}.{table} ADD COLUMN {ifnex} {colname} double precision;'
    computeSQL = 'UPDATE {schema}.{table} SET {colname}={expression};'

    def main(self, table, overwrite=False):
        with self._connect() as cur:
            for metric, expression in self.fields:
                params = dict(
                    schema=self.schemaSQL,
                    table=sql.Identifier(table),
                    ifnex=sql.SQL('IF NOT EXISTS' if overwrite else ''),
                    colname=sql.Identifier(metric),
                    expression=sql.SQL(expression),
                )
                self.logger.info('computing %s', metric)
                for pattern in (self.createSQL, self.computeSQL):
                    qry = sql.SQL(pattern).format(**params).as_string(cur)
                    cur.execute(qry)
    