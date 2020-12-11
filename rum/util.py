from psycopg2 import sql

from . import core, attribute

class Recategorizer(core.DatabaseTask):
    def main(self, table, source, target, translation, overwrite=False):
        translator = attribute.Translator.load(translation)
        outType = translator.outputPGType()
        with self._connect() as cur:
            self.createTargetField(cur, table, target, outType, overwrite)
            qry = sql.SQL('UPDATE {schema}.{table} SET {target} = ').format(
                schema=self.schemaSQL,
                table=sql.Identifier(table),
                target=sql.Identifier(target),
            ) + translator.caseQuery(cur, source)
            self.logger.debug('recategorizing with %s', qry)
            cur.execute(qry)

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


class PolygonGeometryFixer(core.DatabaseTask):
    computeSQL = '''UPDATE {schema}.{table} SET geometry=(
        case when substring(GeometryType(geometry) for 5) = 'MULTI'
            then st_multi(st_buffer(geometry, 0))
            else st_buffer(geometry, 0)
        end
    );'''
    cleanSQL = 'VACUUM ANALYZE {schema}.{table};'

    def main(self, table, overwrite=False):
        params = dict(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
        )
        for pattern in (self.computeSQL, ):#self.cleanSQL):
            # todo fix: vacuum cannot run inside transaction
            with self._connect() as cur:
                qry = sql.SQL(pattern).format(**params).as_string(cur)
                cur.execute(qry)



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


class Dissolver(core.DatabaseTask):
    QRY = sql.SQL('''CREATE TABLE {schema}.{tgt} AS SELECT
        st_multi(
            (st_dump(st_union(geometry))).geom
        )::geometry(multipolygon,{srid}) as geometry
    FROM {schema}.{src};
    ''')

    def main(self, source_table, target_table, overwrite=False):
        with self._connect() as cur:
            qry = self.QRY.format(
                schema=self.schemaSQL,
                src=sql.Identifier(source_table),
                tgt=sql.Identifier(target_table),
                srid=sql.Literal(self.getSRID(cur, source_table)),
            ).as_string(cur)
            self.clearTable(cur, target_table, overwrite=overwrite)
            self.logger.debug('dissolving %s to %s: %s', source_table, target_table, qry)
            cur.execute(qry)


class FeatureLister(core.DatabaseTask):
    def main(self, consolidated=False):
        hasCondition = False
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
                if names:
                    for table, columns in names.items():
                        if table == 'condition':
                            continue
                        print(table)
                        for column in columns:
                            print(' ', column)
                            namelist.append(column)
                hasCondition = 'condition' in self.getTableNames(cur)
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
            hasCondition = self.hasConditionTable(cur)
            if hasCondition:
                featnames['condition'] = ['condition']
            qry = self.consolidationQuery(featnames).as_string(cur)
            self.logger.debug('consolidating features: %s', qry)
            cur.execute(qry)
            if not hasCondition:
                self.createDefaultConditionField(cur)
            self.createPrimaryKey(cur, 'all_feats')

    def createDefaultConditionField(self, cur):
        qry = sql.SQL('''ALTER TABLE {schema}.all_feats
            ADD COLUMN condition boolean NOT NULL DEFAULT TRUE;'''
        ).format(schema=self.schemaSQL).as_string(cur)
        self.logger.debug('adding default condition field: %s', qry)
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


class RawDisaggregator(core.DatabaseTask):
    disagPattern = sql.SQL('''CREATE TABLE {schema}.{outputTable} AS (
        WITH parts AS (SELECT
            {partSource},
            {partWeight},
            CASE WHEN t.geometry IS NULL THEN 1 ELSE
                st_area(st_intersection(t.geometry, s.geometry))
                / (
                    sum(st_area(st_intersection(t.geometry, s.geometry)))
                    OVER (PARTITION BY t.geometry)
                )
            END AS part_area_fraction,
            CASE WHEN t.geometry IS NULL THEN s.geometry ELSE t.geometry END AS tgt_geometry,
            s.geometry AS src_geometry
        FROM {schema}.{disagTable} s
            {jointype} {schema}.{weightTable} t ON st_intersects(t.geometry, s.geometry)
        ),
        tcoefs AS (SELECT
                src_geometry,
                {tcoef}
            FROM parts
            GROUP BY src_geometry
        ),
        grouped AS (SELECT
                {result},
                p.tgt_geometry as geometry
            FROM parts p
                JOIN tcoefs t ON p.src_geometry=t.src_geometry
            GROUP BY p.tgt_geometry
        )
        SELECT
            g.*, w.{transferColumns}
        FROM grouped g
            LEFT JOIN {schema}.{weightTable} w ON g.geometry=w.geometry
    )''')

    namePatterns = {
        'source' : '{source}',
        'weight' : '{weight}',
        'partsource' : 'src_{source}',
        'partweight' : '{source}_wt_{weight}',
        'result' : '{source}_disag_{weight}',
    }
    snippetPatterns = {
        'partSource' : sql.SQL('''s.{source} AS {partsource}'''),
        'partWeight' : sql.SQL('''CASE
            WHEN t.geometry IS NULL THEN 1
            WHEN t.{weight} IS NULL THEN 0
            ELSE t.{weight}
        END AS {partweight}'''),
        'tcoef' :      sql.SQL('''max({partsource}) / CASE
            WHEN sum({partweight} * part_area_fraction) = 0 THEN count(*)
            ELSE sum({partweight} * part_area_fraction)
        END AS {result}'''),
        'result' :     sql.SQL('''sum(p.{partweight} * p.part_area_fraction * t.{result}) AS {result}'''),
    }

    def main(self, disagTable, disagFields, outputTable, weightTable, weightFields, keepUnweighted=False, relative=False, overwrite=False):
        if relative:
            raise NotImplementedError
        namesets = [
            tuple(sql.Identifier(name) for name in nameset)
            for nameset in sorted(set(
                tuple(
                    pattern.format(source=src, weight=wt)
                    for key, pattern in self.namePatterns.items()
                )
                for src in disagFields for wt in weightFields
            ))
        ]
        snippets = {
            key : sql.SQL(',\n').join([
                snippet.format(**dict(zip(self.namePatterns.keys(), nameset)))
                for nameset in namesets
            ])
            for key, snippet in self.snippetPatterns.items()
        }
        with self._connect() as cur:
            self.clearTable(cur, outputTable, overwrite=overwrite)
            disagQry = self.disagPattern.format(
                schema=self.schemaSQL,
                disagTable=sql.Identifier(disagTable),
                weightTable=sql.Identifier(weightTable),
                outputTable=sql.Identifier(outputTable),
                transferColumns=sql.SQL(', w.').join(
                    sql.Identifier(col)
                    for col in self.getColumnNamesForTable(cur, weightTable)
                    if col != 'geometry'
                ),
                jointype=(sql.SQL('LEFT JOIN') if keepUnweighted else sql.SQL('INNER JOIN')),
                **snippets
            ).as_string(cur)
            self.logger.debug('disaggregating: %s', disagQry)
            cur.execute(disagQry)


class Disaggregator(core.DatabaseTask):
    disagPattern = sql.SQL('''CREATE TABLE {schema}.{outputTable} AS (
        WITH fweights AS (SELECT
                g.geohash,
                d.geometry AS dgeometry,
                d.{disagField} AS dval,
                CASE WHEN w.{weightField} IS NULL THEN 0
                    ELSE w.{weightField} * st_area(st_intersection(g.geometry, d.geometry))
                END AS fweight
            FROM {schema}.grid g
                JOIN {schema}.{disagTable} d ON st_intersects(g.geometry, d.geometry)
                LEFT JOIN {schema}.{weightTable} w ON g.geohash=w.geohash
        ),
        transfers AS (SELECT
                dgeometry, sum(fweight) AS coef
            FROM fweights
            GROUP BY dgeometry
        )
        SELECT
            fw.geohash, sum(
                CASE WHEN t.coef = 0 THEN 0 ELSE fw.dval * fw.fweight / t.coef END
            ) as value
        FROM fweights fw
            JOIN transfers t ON fw.dgeometry=t.dgeometry
        GROUP BY fw.geohash
    )''')

    def main(self, disagTable, disagField, outputTable, weightTable, weightField='weight', relative=False, overwrite=False):
        if relative:
            raise NotImplementedError
        with self._connect() as cur:
            self.clearTable(cur, outputTable, overwrite=overwrite)
            disagQry = self.disagPattern.format(
                schema=self.schemaSQL,
                disagTable=sql.Identifier(disagTable),
                disagField=sql.Identifier(disagField),
                weightTable=sql.Identifier(weightTable),
                weightField=sql.Identifier(weightField),
                outputTable=sql.Identifier(outputTable),
            ).as_string(cur)
            self.logger.debug('disaggregating: %s', disagQry)
            cur.execute(disagQry)
            self.createPrimaryKey(cur, outputTable)


class BatchDisaggregator(Disaggregator):
    disagPattern = sql.SQL('''
    CREATE TABLE {schema}.{outputTable} AS (
        WITH parts AS (SELECT
                g.geohash,
                d.geometry AS src_geometry,
                st_area(st_intersection(g.geometry, d.geometry)) AS part_area,
                d.{disagField} AS src_value,
                {fweights}
            FROM {schema}.grid g
                JOIN {schema}.{disagTable} d ON st_intersects(g.geometry, d.geometry)
                LEFT JOIN {schema}.{weightTable} w ON g.geohash=w.geohash
            WHERE st_area(st_intersection(g.geometry, d.geometry)) > 0
        ),
        transfers AS (SELECT
                src_geometry,
                sum(part_area) AS total_area,
                max(src_value) as src_value,
                {transfers}
            FROM parts
            GROUP BY src_geometry
        )
        SELECT
            p.geohash,
            {finals}
        FROM parts p
            JOIN transfers t ON p.src_geometry=t.src_geometry
        GROUP BY p.geohash
    )''')

    patterns = [
        ('fweights', sql.SQL('''(CASE WHEN w.{0} IS NULL THEN 0 ELSE w.{0} END
            * st_area(st_intersection(g.geometry, d.geometry))
            / (sum(st_area(st_intersection(g.geometry, d.geometry))) OVER (PARTITION BY g.geohash))
            ) AS {0}''')),
        ('transfers', sql.SQL('CASE WHEN sum({0}) = 0 THEN -1 ELSE 1 / sum({0}) END AS {0}')),
        ('finals', sql.SQL('''sum(t.src_value * CASE
                WHEN t.{0} = -1 THEN p.part_area / t.total_area
                ELSE p.{0} * t.{0}
            END) AS {0}''')),
    ]

    def getWeightColumns(self, cur, table):
        return [col for col in self.getColumnNamesForTable(cur, table)
            if col != 'geohash'
        ]

    def main(self, disagTable, disagField, weightTable, outputTable, relative=False, overwrite=False):
        if relative:
            raise NotImplementedError
        with self._connect() as cur:
            weightCols = self.getWeightColumns(cur, weightTable)
            self.clearTable(cur, outputTable, overwrite=overwrite)
            self.logger.info('disaggregating by %d fields from %s', len(weightCols), weightTable)
            params = dict(
                schema=self.schemaSQL,
                disagTable=sql.Identifier(disagTable),
                disagField=sql.Identifier(disagField),
                weightTable=sql.Identifier(weightTable),
                outputTable=sql.Identifier(outputTable),
            )
            for key, pattern in self.patterns:
                params[key] = sql.SQL(',\n').join(
                    pattern.format(sql.Identifier(col)) for col in weightCols
                )
            qry = self.disagPattern.format(**params).as_string(cur)
            self.logger.debug('disaggregating batch: %s', qry)
            cur.execute(qry)
            self.createPrimaryKey(cur, outputTable)


class TransferWeightCalculator(core.DatabaseTask):
    pattern = sql.SQL('''CREATE TABLE {schema}.{outputTable} AS (
        WITH chunks AS (
        SELECT f.{fromIDField} AS from_id,
               t.{toIDField} AS to_id,
               st_area(st_intersection(
                   st_intersection(g.geometry, f.geometry),
                   t.geometry
               )) * {auxValueSource}.{auxDensField} AS weight
          FROM {schema}.{fromTable} f
               JOIN {schema}.{auxTable} g ON st_intersects(g.geometry, f.geometry)
               {auxValueClause}
               JOIN {schema}.{toTable} t ON st_intersects(g.geometry, t.geometry)
         WHERE {auxValueSource}.{auxDensField} > 0
        ),
        source_sums AS (
        SELECT a.from_id,
               sum(a.weight) AS site_weight_sum
          FROM chunks a
         GROUP BY a.from_id
        )
        SELECT a.from_id,
               a.to_id,
               sum(a.weight) / s.site_weight_sum as transfer_weight
          FROM chunks a
               join source_sums s ON a.from_id=s.from_id
         GROUP BY a.from_id, a.to_id, s.site_weight_sum
        HAVING sum(a.weight) > 0
    )''')
    auxValueClausePattern = sql.SQL('''
        JOIN {schema}.{auxValueTable} d ON g.{auxGeomIDField}=d.{auxValueIDField}
    ''')

    def main(self, fromTable, fromIDField, toTable, toIDField, auxTable, outputTable, auxDensField='weight', auxValueTable=None, auxGeomIDField='geohash', auxValueIDField='geohash', overwrite=False):
        with self._connect() as cur:
            self.clearTable(cur, outputTable, overwrite=overwrite)
            disagQry = self.pattern.format(
                schema=self.schemaSQL,
                fromTable=sql.Identifier(fromTable),
                toTable=sql.Identifier(toTable),
                fromIDField=sql.Identifier(fromIDField),
                toIDField=sql.Identifier(toIDField),
                auxDensField=sql.Identifier(auxDensField),
                auxTable=sql.Identifier(auxTable),
                outputTable=sql.Identifier(outputTable),
                auxValueClause=(
                    self.auxValueClausePattern.format(
                        schema=self.schemaSQL,
                        auxValueTable=sql.Identifier(auxValueTable),
                        auxGeomIDField=sql.Identifier(auxGeomIDField),
                        auxValueIDField=sql.Identifier(auxValueIDField),
                    ) if auxValueTable else sql.SQL('')
                ),
                auxValueSource=(sql.SQL('d' if auxValueTable else 'g')),
            ).as_string(cur)
            self.logger.debug('creating transfer table: %s', disagQry)
            cur.execute(disagQry)


class TransferWeightApplier(core.DatabaseTask):
    pattern = sql.SQL('''CREATE TABLE {schema}.{outputTable} AS (
        SELECT {targetFields},
               sum(f.{valueField} * w.transfer_weight) AS {valueField}
          FROM eesti.{fromTable} f
               join eesti.{transferTable} w ON f.{fromIDField}=w.from_id
               join eesti.{toTable} t ON t.{toIDField}=w.to_id
         GROUP BY {targetFields}
    )''')

    def main(self, fromTable, fromIDField, valueField, toTable, toIDField, transferTable, outputTable, overwrite=False):
        with self._connect() as cur:
            toFields = self.getColumnNamesForTable(cur, toTable, schema=self.schema)
            self.clearTable(cur, outputTable, overwrite=overwrite)
            qry = self.pattern.format(
                schema=self.schemaSQL,
                fromTable=sql.Identifier(fromTable),
                toTable=sql.Identifier(toTable),
                fromIDField=sql.Identifier(fromIDField),
                valueField=sql.Identifier(valueField),
                toIDField=sql.Identifier(toIDField),
                transferTable=sql.Identifier(transferTable),
                outputTable=sql.Identifier(outputTable),
                targetFields=sql.SQL(', ').join(
                    sql.SQL('t.{}').format(sql.Identifier(fld))
                    for fld in toFields
                ),
            ).as_string(cur)
            self.logger.debug('transferring: %s', qry)
            cur.execute(qry)


class TrainingSchemaMerger(core.DatabaseTask):
    TARGET_TABLE = 'target'

    def main(self, source_schemas, target_tables, overwrite=False):
        with self._connect() as cur:
            self.createSchema(cur)
            self.mergeAllFeaturesTables(cur, source_schemas, overwrite=overwrite)
            self.logger.info('merging target tables')
            self.mergeTables(
                cur,
                source_schemas,
                target_tables,
                self.TARGET_TABLE,
                fields=['geohash', 'target'],
                doPrimaryKey=True,
                overwrite=overwrite,
            )
            self.logger.info('merging grids')
            self.mergeTables(
                cur,
                source_schemas,
                'grid',
                'grid',
                fields=['geohash', 'inside'],
                doPrimaryKey=True,
                overwrite=overwrite,
            )

    def mergeAllFeaturesTables(self, cur, source_schemas, overwrite=False):
        self.clearTable(cur, core.ALL_FEATS_TABLE, overwrite=overwrite)
        self.logger.info('merging feature tables')
        cols = {
            schema : self.getColumnNamesForTable(cur, core.ALL_FEATS_TABLE, schema=schema)
            for schema in source_schemas
        }
        all_cols = list(sorted(frozenset(
            col for values in cols.values() for col in values
        )))
        all_ident = sql.Identifier(core.ALL_FEATS_TABLE)
        qry = sql.SQL('''CREATE TABLE {schema}.{all_ident} AS ((
            {contents}
        ))''').format(
            schema=self.schemaSQL,
            all_ident=all_ident,
            contents=sql.SQL('\n) UNION ALL (\n').join(
                sql.SQL('''SELECT
                    {colsegment}
                FROM {schema}.{all_ident}''').format(
                    schema=sql.Identifier(schema),
                    all_ident=all_ident,
                    colsegment=sql.SQL(',\n').join(
                        (
                            sql.SQL('NULL AS ') if col not in collist
                            else sql.SQL('')
                        ) + sql.Identifier(col)
                        for col in all_cols
                    ),
                )
                for schema, collist in cols.items()
            ),
        ).as_string(cur)
        self.logger.debug('merging feature table query: %s', qry)
        cur.execute(qry)
        self.createPrimaryKey(cur, core.ALL_FEATS_TABLE)

    def mergeTables(self, cur, source_schemas, source_tables, target_table, fields, doPrimaryKey=False, overwrite=False):
        if isinstance(source_tables, str):
            source_tables = [source_tables] * len(source_schemas)
        self.clearTable(cur, target_table, overwrite=overwrite)
        qry = sql.SQL('''CREATE TABLE {schema}.{tblname} AS ((
            {contents}
        ))''').format(
            schema=self.schemaSQL,
            tblname=sql.Identifier(target_table),
            contents=sql.SQL('\n) UNION ALL (\n').join(
                sql.SQL('SELECT {fields} FROM {schema}.{tgt}').format(
                    schema=sql.Identifier(schema),
                    tgt=sql.Identifier(tgt),
                    fields=sql.SQL(', ').join(
                        sql.Identifier(field) for field in fields
                    ),
                )
                for schema, tgt in zip(source_schemas, source_tables)
            )
        ).as_string(cur)
        self.logger.debug('table merge query: %s', qry)
        cur.execute(qry)
        if doPrimaryKey:
            self.createPrimaryKey(cur, target_table)

