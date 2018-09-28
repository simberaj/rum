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
    createSQL = 'ALTER TABLE {schema}.{table} ADD COLUMN {ifnex}{colname} double precision;'
    computeSQL = 'UPDATE {schema}.{table} SET {colname}={expression};'

    def main(self, table, overwrite=False):
        with self._connect() as cur:
            for metric, expression in self.fields:
                params = dict(
                    schema=self.schemaSQL,
                    table=sql.Identifier(table),
                    ifnex=sql.SQL('IF NOT EXISTS ' if overwrite else ''),
                    colname=sql.Identifier(metric),
                    expression=sql.SQL(expression),
                )
                self.logger.info('computing %s', metric)
                for pattern in (self.createSQL, self.computeSQL):
                    qry = sql.SQL(pattern).format(**params).as_string(cur)
                    cur.execute(qry)

class FeatureLister(core.DatabaseTask):
    def main(self, consolidated=False):
        with self._connect() as cur:
            if consolidated:
                names = self.getConsolidatedFeatureNames(cur, condition=True)
                if 'condition' in names:
                    names.remove('condition')
                    hasCondition = True
                for name in names:
                    print(name)
                namelist = names
            else:
                names = self.getFeatureNames(cur)
                namelist = []
                hasCondition = True
                if names:
                    for table, columns in names.items():
                        if table == 'condition':
                            continue
                        print(table)
                        for column in columns:
                            print(' ', column)
                            namelist.append(column)
            print()
            if namelist:
                print('*** {} {}features total'.format(
                    len(namelist),
                    'consolidated ' if consolidated else ''
                ))
            else:
                print('*** No feature fields found (or schema missing)')
            if hasCondition:
                print('*** Modeling entry condition detected')


class FeatureClearer(core.DatabaseTask):
    def main(self):
        with self._connect() as cur:
            tablenames = list(self.getFeatureNames(cur).keys())
            self.logger.info('dropping %d tables', len(tablenames))
            for name in tablenames:
                cur.execute(
                    sql.SQL('DROP TABLE {schema}.{name};').format(
                        schema=self.schemaSQL,
                        name=sql.Identifier(name),
                    )
                )

class FeatureConsolidator(core.DatabaseTask):    
    def main(self, overwrite=False):
        with self._connect() as cur:
            self.clearTable(cur, 'all_feats', overwrite)
            featnames = self.getFeatureNames(cur)
            if self.hasConditionTable(cur):
                featnames['condition'] = ['condition']
            qry = self.consolidationQuery(featnames).as_string(cur)
            self.logger.debug('consolidating features: %s', qry)
            cur.execute(qry)

    def consolidationQuery(self, featnames):
        fieldsSQLs = []
        tablesSQLs = []
        for table, columns in featnames.items():
            tableIdent = sql.Identifier(table)
            tablePrefix = self.getTablePrefix(table)
            for column in columns:
                fieldsSQLs.append(
                    sql.SQL('{schema}.{table}.{column} AS {newname}').format(
                        schema=self.schemaSQL,
                        table=tableIdent,
                        column=sql.Identifier(column),
                        newname=sql.Identifier(tablePrefix + ('_' + column if column != tablePrefix else ''))
                    )
                )
            if tablesSQLs:
                tablesSQLs.append(
                    sql.SQL('{schema}.{table} ON {schema}.{table}.geohash={first}.geohash').format(
                        schema=self.schemaSQL,
                        table=tableIdent,
                        first=tablesSQLs[0]
                    )
                )
            else:
                tablesSQLs.append(
                    sql.SQL('{schema}.{table}').format(
                        schema=self.schemaSQL,
                        table=tableIdent,
                    )
                )
        return (
            sql.SQL('CREATE TABLE {schema}.all_feats AS SELECT {first}.geohash,\n').format(
                schema=self.schemaSQL,
                first=tablesSQLs[0]
            )
            + sql.SQL(',\n').join(fieldsSQLs)
            + sql.SQL(' FROM ')
            + sql.SQL('\nJOIN ').join(tablesSQLs)
        )
    
    def getTablePrefix(self, table):
        for prefix in (core.NEIGHBOUR_FEATURE_PREFIX, core.FEATURE_PREFIX):
            if table.startswith(prefix):
                return table[len(prefix):]
        return table

    def hasConditionTable(self, cur):
        qry = sql.SQL('''SELECT FROM information_schema.tables
            WHERE table_schema={schema} AND table_name='condition'
        ''').format(
            schema=sql.Literal(self.schema)
        ).as_string(cur)
        self.logger.debug('detecting condition table: %s', qry)
        cur.execute(qry)
        line = cur.fetchone()
        return line is not None



class Disaggregator(core.DatabaseTask):
    disagPattern = sql.SQL('''
    WITH fweights AS (SELECT
        g.geohash,
        d.geometry AS dgeometry,
        d.{disagField} AS dval,
        g.{weightField} * st_area(st_intersection(g.geometry, d.geometry)) AS fweight
    FROM {schema}.grid g
        JOIN {schema}.{disagTable} d ON st_intersects(g.geometry, d.geometry)
    ),
    transfers AS (SELECT
        dgeometry, sum(fweight) AS coef
    FROM fweights
    GROUP BY dgeometry
    ),
    vals AS (SELECT
        fw.geohash, sum(
            CASE WHEN t.coef = 0 THEN 0 ELSE fw.dval * fw.fweight / t.coef END
        ) as val
    FROM fweights fw
        JOIN transfers t ON fw.dgeometry=t.dgeometry
    GROUP BY fw.geohash
    )
    UPDATE {schema}.grid SET {targetField} = val
    FROM vals
    WHERE vals.geohash={schema}.grid.geohash
    ''')

    def createField(self, cur, name, overwrite=False):
        qry = sql.SQL('''ALTER TABLE {schema}.grid
            ADD COLUMN {ifnex}{name} double precision
        ''').format(
            schema=self.schemaSQL,
            ifnex=sql.SQL('IF NOT EXISTS ' if overwrite else ''),
            name=sql.Identifier(name)
        ).as_string(cur)
        self.logger.debug('creating field: %s', name)
        cur.execute(qry)

    def main(self, disagTable, disagField, weightField, targetField, relative=False, overwrite=False):
        if relative:
            raise NotImplementedError
        with self._connect() as cur:
            self.disaggregateAbsolute(cur, disagTable, disagField, weightField, targetField, overwrite=overwrite)

    def disaggregateAbsolute(self, cur, disagTable, disagField, weightField, targetField, overwrite=False):
        self.createField(cur, targetField, overwrite=overwrite)
        disagQry = self.disagPattern.format(
            schema=self.schemaSQL,
            disagTable=sql.Identifier(disagTable),
            disagField=sql.Identifier(disagField),
            weightField=sql.Identifier(weightField),
            targetField=sql.Identifier(targetField),
        ).as_string(cur)
        self.logger.debug('disaggregating: %s', disagQry)
        cur.execute(disagQry)


class BatchDisaggregator(Disaggregator):
    def main(self, disagTable, disagField, weightFieldBase, relative=False, overwrite=False):
        if relative:
            raise NotImplementedError
        with self._connect() as cur:
            for weightField in self.getGridNames(cur):
                if weightField.startswith(weightFieldBase):
                    self.disaggregateAbsolute(
                        cur,
                        disagTable, disagField,
                        weightField,
                        '{}_disag_{}'.format(disagField, weightField),
                        overwrite=overwrite
                    )
