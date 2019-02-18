import itertools

import numpy
import scipy.ndimage
import psycopg2.extras
from psycopg2 import sql

from . import core
from . import field

PARTIAL_EXPRESSIONS = {
    'aux_item_length' : 'ST_Length({schema}.{table}.geometry)',
    'aux_item_area' : 'ST_Area({schema}.{table}.geometry)',
    'aux_common_length' : 'ST_Length(ST_Intersection({schema}.grid.geometry, {schema}.{table}.geometry))',
    'aux_common_area' : 'ST_Area(ST_Intersection({schema}.grid.geometry, {schema}.{table}.geometry))',
    'aux_cell_area' : 'ST_Area({schema}.grid.geometry)'
}

class Calculator(field.Handler):
    BASE_LEAD = sql.SQL('''CREATE {temp}TABLE {schema}.{target} AS (
    WITH overlaid AS (
        SELECT
            {schema}.grid.geohash AS geohash,
            ''')
    BASE_MID = sql.SQL('''
        FROM {schema}.grid LEFT JOIN {schema}.{table}
        ON ST_Intersects(
            {schema}.grid.geometry,
            {schema}.{table}.geometry
        )
    )
    SELECT
        geohash,
        ''')
    BASE_TRAIL = sql.SQL('''
    FROM overlaid GROUP BY geohash
    )''')
    SEPARATOR = sql.SQL(',\n        ')
    ALIASER = sql.SQL(' AS ')

    def getQuery(self, cur, source, target, partials, finals, temporary=False):
        finalFieldSelect = self.fieldSelect(finals)
        return sql.SQL((
            self.BASE_LEAD
            + self.fieldSelect(itertools.chain(
                partials, self.partialsFromFinals(finalFieldSelect.as_string(cur))
            ))
            + self.BASE_MID
            + self.fieldSelect(finals)
            + self.BASE_TRAIL
        ).as_string(cur)).format(
            schema=self.schemaSQL,
            table=sql.Identifier(source),
            target=sql.Identifier(target),
            temp=sql.SQL('TEMPORARY ' if temporary else ''),
        )

    def partialsFromFinals(self, finexpr):
        partials = []
        for partname, partexpr in PARTIAL_EXPRESSIONS.items():
            if partname in finexpr:
                partials.append((partname, sql.SQL(partexpr)))
        return partials

    def fieldSelect(self, fieldTuples):
        # for ftup in fieldTuples:
            # for item in ftup:
                # print(item)
            # print()
        return self.SEPARATOR.join(
            self.ALIASER.join([expr, sql.Identifier(name)])
            for name, expr in fieldTuples
        )

    def calculate(self, cur, table, partials, finals, target=None, overwrite=False, bannedNames=[]):
        if target is None:
            target = '{type}_{table}'.format(
                type=self.type,
                table=table
            )
            if not overwrite:
                target = self.uniqueTableName(cur, target)
            elif target in bannedNames:
                target = self.uniqueTableName(cur, target, names=bannedNames)
        self.logger.info('computing table %s from table %s', target, table)
        targetSQL = sql.Identifier(target)
        qry = self.getQuery(cur, table, target, partials, finals).as_string(cur)
        self.clearTable(cur, target, overwrite)
        self.logger.debug('computing table: %s', qry)
        cur.execute(qry)
        self.createPrimaryKey(cur, target)
        return target
        
    def uniqueTableName(self, cur, target, names=[]):
        if names:
            currentNames = [name for name in names if target in name]
        else:
            currentNames = self.getTableNames(
                cur,
                where=sql.SQL("table_name LIKE {target}").format(
                    target=sql.Literal(target + '%')
                )
            )
        maxSuffix = -1
        if currentNames:
            maxSuffix = 0
        for name in currentNames:
            suffix = name.rsplit('_', 1)[-1]
            if suffix.isdigit():
                if int(suffix) > maxSuffix:
                    maxSuffix = int(suffix)
        if maxSuffix >= 0:
            return target + '_{:d}'.format(maxSuffix + 1)
        else:
            return target

    def vacuumGrid(self):
        with self._connect(autocommit=True) as cur:
            self.logger.debug('vacuuming grid')
            cur.execute(
                sql.SQL('VACUUM ANALYZE {schema}.grid').format(
                    schema=self.schemaSQL
                )
            )

            
class ConditionCalculator(Calculator):
    def main(self, table, expression, overwrite=False):
        partials = [('*',
            sql.SQL('{schema}.{table}.*').format(
                schema=self.schemaSQL,
                table=sql.Identifier(table)
            )
        )]
        finals = [('condition',
            sql.SQL('sum(CASE WHEN {expr} THEN 1 ELSE 0 END) > 0').format(
                expr=sql.SQL(expression) # TODO any way to sanitize expression?
            )
        )]
        with self._connect() as cur:
            self.calculate(cur, table,
                partials, finals,
                target='condition',
                overwrite=overwrite
            )


class FeatureCalculator(Calculator):
    type = 'feat'

    def getSpecifiers(self, cur, table, sourceField):
        return [None]

    def main(self, table, methods, sourceFields=None, caseField=None, overwrite=False, bannedNames=[]):
        if sourceFields is None:
            sourceFields = [None]
        definers = self.createDefiners(methods)
        with self._connect() as cur:
            uniquesGetter = field.UniquesGetter(cur, self.schema, table)
            finals = []
            for definer in definers:
                for sourceField in sourceFields:
                    finals.extend(definer.get(
                        uniquesGetter,
                        sourceField=sourceField,
                        caseField=caseField
                    ))
            partialSet = set(sourceFields + [caseField])
            partialSet.discard(None)
            partials = [(field, sql.Identifier(field)) for field in sorted(partialSet)]
            target = self.calculate(
                cur, table,
                partials, finals,
                overwrite=overwrite,
                bannedNames=bannedNames
            )
            return target
            # if None in partials:

    def createDefiners(self, methods):
        return [ExpressionDefiner.create(method) for method in methods]


class ExpressionDefiner:
    CASE_PATTERN = sql.SQL('CASE WHEN {caseField}={value} THEN {numerator} ELSE 0 END')
    MAIN_PATTERN = sql.SQL('(CASE WHEN {denominator} = 0 THEN 0 ELSE sum({numerator}) / {denominator} END)')

    @classmethod
    def create(cls, code):
        try:
            return cls.CODES[code.lower()]()
        except KeyError as err:
            raise core.InvalidParameterError('invalid method {}, allowed: {}'.format(
                code, ', '.join(cls.CODES.keys())
            ))

    def get(self, uniquesGetter, sourceField=None, caseField=None, targetName=None):
        if targetName is None:
            targetName = self.code
            if sourceField:
                targetName += ('_' + sourceField)
        numeratorSQL = sql.SQL(self.numerator)
        if sourceField is not None:
            numeratorSQL = numeratorSQL.format(field=sql.Identifier(sourceField))
        if caseField is not None:
            uniques = uniquesGetter.get(caseField)
            caseFieldSQL = sql.Identifier(caseField)
            numerators = [
                self.CASE_PATTERN.format(
                    caseField=caseFieldSQL,
                    value=sql.Literal(value),
                    numerator=numeratorSQL
                )
                for value in uniques
            ]
            names = [targetName + '_' + caseField + '_' + str(value) for value in uniques]
        else:
            numerators = [numeratorSQL]
            names = [targetName]
        return [
            (
                name,
                self.MAIN_PATTERN.format(
                    numerator=numerator,
                    denominator=sql.SQL(self.denominator),
                )
            )
            for numerator, name in zip(numerators, names)
        ]



class DensityDefiner(ExpressionDefiner):
    code = 'dens'
    numerator = '1'
    denominator = 'sum(aux_cell_area)'

class LengthDefiner(ExpressionDefiner):
    code = 'len'
    numerator = 'aux_common_length'
    denominator = 'sum(aux_cell_area)'

class CoverageDefiner(ExpressionDefiner):
    code = 'cov'
    numerator = 'aux_common_area'
    denominator = 'max(aux_cell_area)'

class SumDefiner(ExpressionDefiner):
    code = 'sum'
    numerator = '{field}'
    denominator = '1'

class AverageDefiner(ExpressionDefiner):
    code = 'avg'
    numerator = '{field}'
    denominator = 'count(1)'

class WeightedAverageDefiner(ExpressionDefiner):
    code = 'wavg'
    numerator = '{field} * aux_item_area'
    denominator = 'sum(aux_item_area)'

class DistributedSumDefiner(ExpressionDefiner):
    code = 'dsum'
    numerator = '{field} * aux_common_area / aux_item_area'
    denominator = '1'

class DistributedAverageDefiner(ExpressionDefiner):
    code = 'davg'
    numerator = '{field} * aux_common_area'
    denominator = 'sum(aux_common_area)'


ExpressionDefiner.CODES = {calc.code : calc for calc in [
    DensityDefiner,
    LengthDefiner,
    CoverageDefiner,
    SumDefiner,
    AverageDefiner,
    WeightedAverageDefiner,
    DistributedSumDefiner,
    DistributedAverageDefiner
]}


class TargetCalculator(Calculator):
    type = 'tgt'
    DEFINERS = {
        ('POINT', False) : SumDefiner,
        ('POINT', True) : AverageDefiner,
        ('POLYGON', False) : DistributedSumDefiner,
        ('POLYGON', True) : DistributedAverageDefiner,
    }

    def main(self, table, sourceField, target=None, relative=False, overwrite=False):
        with self._connect() as cur:
            definer = self.DEFINERS[self.getGeometryType(cur, table), relative]()
            finals = definer.get(None, sourceField, targetName='target')
            partials = [(sourceField, sql.Identifier(sourceField))]
            self.calculate(cur, table, partials, finals, target=target, overwrite=overwrite)

    def getGeometryType(self, cur, table):
        qry = sql.SQL('''SELECT type FROM geometry_columns
            WHERE f_table_schema={schema}
                AND f_table_name={table}
                AND f_geometry_column='geometry';
        ''').format(
            schema=sql.Literal(self.schema),
            table=sql.Literal(table)
        ).as_string(cur)
        cur.execute(qry)
        row = cur.fetchone()
        if not row:
            raise InvalidContentsError('source table does not contain geometry')
        geomtype = row[0].upper()
        if geomtype.startswith('MULTI'):
            geomtype = geomtype[5:]
        return geomtype


class NeighbourhoodCalculator(Calculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._connect() as cur:
            self.locator = CellLocator.fromGrid(cur, self.schemaSQL)

    def main(self, multipliers=[], overwrite=False):
        with self._connect() as cur:
            allFeats = self.getFeatureNames(cur)
            multipliersAndNames = self.neighColumnNames(multipliers)
            for inTable, featName, outTable in self.findTodos(allFeats, overwrite=overwrite):
                self.logger.info('calculating neighbourhood from %s.%s', inTable, featName)
                raster = self.load(cur, inTable, featName)
                outrasters = []
                for mult, outfeat in multipliersAndNames:
                    self.logger.info('blurring %s.%s with %f multiplier', inTable, featName, mult)
                    outrasters.append((outfeat, self.gaussify(raster, mult)))
                self.save(cur, outTable, outrasters)


    def findTodos(self, allFeats, overwrite=False):
        allTables = set(allFeats.keys())
        for featTable, featColumns in allFeats.items():
            if not featTable.startswith('feat_neigh_'):
                for col in featColumns:
                    neighTable = self.neighTableName(featTable, col)
                    if overwrite or neighTable not in allTables:
                        yield featTable, col, neighTable

    def neighTableName(self, tableName, columnName):
        return 'feat_neigh_{tbl}_{col}'.format(
            tbl=tableName[5:], # cut the feat_ prefix
            col=columnName,
        )

    def neighColumnNames(self, multipliers):
        return [
            (mult, 'neigh_' + self.multiplierSuffix(mult))
            for mult in multipliers
        ]

    def load(self, cur, table, feature):
        qry = sql.SQL('SELECT geohash, {feature} FROM {schema}.{table}').format(
            feature=sql.Identifier(feature),
            table=sql.Identifier(table),
            schema=self.schemaSQL,
        ).as_string(cur)
        self.logger.debug('loading feature for neighbourhood analysis: %s', qry)
        cur.execute(qry)
        return self.locator.raster(cur.fetchall())

    def save(self, cur, table, outrasters):
        fields = (
            [self.locator.geohashes()] +
            [self.locator.insertseq(raster) for name, raster in outrasters]
        )
        qry = sql.SQL('''CREATE TABLE {schema}.{table} AS SELECT
            unnest(%s) AS geohash,
            {varpart}
        ''').format(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
            varpart=sql.SQL(',\n').join(
                sql.SQL(sql.SQL('unnest(%s) AS {name}').format(
                    name=sql.Identifier(name)
                ).as_string(cur))
                for name, raster in outrasters
            )
        ).as_string(cur)
        self.logger.debug('creating neighbourhood feature table: %s', qry)
        cur.execute(qry, fields)
        self.createPrimaryKey(cur, table)

    def gaussify(self, raster, multiplier):
        mask = numpy.array(~numpy.isnan(raster), dtype=float)
        filtered = scipy.ndimage.gaussian_filter(
            numpy.nan_to_num(raster), sigma=multiplier, mode='nearest'
        )
        coefmask = scipy.ndimage.gaussian_filter(mask, sigma=multiplier, mode='nearest')
        oldset = numpy.seterr(invalid='ignore')
        result = filtered / coefmask * numpy.where(mask, 1, numpy.nan)
        numpy.seterr(**oldset)
        return result

    @staticmethod
    def multiplierSuffix(multiplier):
        multpart = str(multiplier).replace('.', '_')
        if multpart.endswith('_0'):
            multpart = multpart[:-2]
        return multpart


class CellLocator:
    def __init__(self, locations):
        self.locations = locations
        self.shape = (
            max(loc[0] for loc in self.locations.values())+1,
            max(loc[1] for loc in self.locations.values())+1,
        )

    def raster(self, records):
        values = numpy.full(self.shape, numpy.nan)
        for geohash, value in records:
            values[self.locations[geohash]] = value
        return values

    def geohashes(self):
        return list(self.locations.keys())

    def insertseq(self, raster):
        return [raster[loc] for loc in self.locations.values()]

    @classmethod
    def fromGrid(cls, cur, schemaIdent):
        qry = sql.SQL('''WITH dims AS (
                SELECT
                    min(ST_XMin(geometry)) as xmin,
                    min(ST_YMin(geometry)) as ymin,
                    avg(sqrt(ST_Area(geometry))) as cellsize
                FROM {schema}.grid
            ) SELECT
                geohash,
                cast(round((ST_XMin(geometry) - dims.xmin) / dims.cellsize) as integer) AS cell_x,
                cast(round((ST_YMin(geometry) - dims.ymin) / dims.cellsize) as integer) AS cell_y
            FROM {schema}.grid, dims''').format(schema=schemaIdent)
        cur.execute(qry)
        locations = {}
        for geohash, x, y in cur.fetchall():
            locations[geohash] = (x, y)
        return cls(locations)
