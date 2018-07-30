import itertools

import numpy
import scipy.ndimage
import psycopg2.extras
from psycopg2 import sql

from . import core
from . import field

PARTIAL_EXPRESSIONS = {
    'item_length' : 'ST_Length({schema}.{table}.geometry)',
    'item_area' : 'ST_Area({schema}.{table}.geometry)',
    'common_length' : 'ST_Length(ST_Intersection({schema}.grid.geometry, {schema}.{table}.geometry))',
    'common_area' : 'ST_Area(ST_Intersection({schema}.grid.geometry, {schema}.{table}.geometry))',
    'cell_area' : 'ST_Area({schema}.{table}.geometry)'
}

class Calculator(core.DatabaseTask):
    BASE_LEAD = '''CREATE {temp}TABLE {schema}.{target} AS (
    WITH overlaid AS (
        SELECT
            {schema}.grid.geohash AS geohash,
            '''
    BASE_MID = '''        FROM {schema}.grid LEFT JOIN {schema}.{table}
        ON ST_Intersects(
            {schema}.grid.geometry,
            {schema}.{table}.geometry
        )
    )
    SELECT
        '''
    BASE_TRAIL = '''    FROM overlaid
    GROUP BY geohash
    )'''
    SEPARATOR = sql.SQL(',\n        ')
    ALIASER = sql.SQL(' AS ')
    
    def getQuery(self, source, target, partials, finals, temporary=False):
        return (
            BASE_LEAD
            + SEPARATOR.join(ALIASER.join(item) for item in itertools.chain(
                partials, self.partialsFromFinals(finals)
            )
            + BASE_MID
            + SEPARATOR.join(ALIASER.join(item) for item in finals)
            + BASE_TRAIL
        ).format(
            schema=self.schemaSQL,
            table=sql.Identifier(source),
            target=sql.Identifier(target),
            temp=('TEMPORARY ' if temporary else ''),
        )
    
    def partialsFromFinals(self, finals):
        partials = []
        for partname, partexpr in PARTIAL_EXPRESSIONS.items():
            if any(partname in finexpr for finname, finexpr in finals):
                partials.append((partname, partexpr))
        return partials
    
    def calculate(self, cur, table, partials, finals, target=None, overwrite=False):
        if target is None:
            target = '{type}_{table}'.format(
                type=self.type,
                table=table
            )
            if not overwrite:
                target = self.uniqueTableName(cur, target)
        self.logger.info('computing table %s from table %s', target, table)
        targetSQL = sql.Identifier(target)
        qry = self.getQuery(table, target, partials, finals)
        print(qry)
        raise RuntimeError
        if overwrite:
            dropqry = sql.SQL('DROP TABLE IF EXISTS {schema}.{target}').format(
                schema=self.schemaSQL, target=targetSQL
            ).as_string(cur)
            self.logger.debug('overwriting: %s', dropqry)
            cur.execute(dropqry)
        self.logger.debug('computing table: %s', qry)
        cur.execute(qry)
        
    def uniqueTableName(self, cur, target):
        currentNames = self.getTableNames(
            cur,
            where=sql.SQL("table_name LIKE {target}").format(
                target=sql.Literal(target + '%')
            )
        )
        if currentNames:
            maxSuffix = max(int(name.rsplit('_', 1)) for name in currentNames)
            return target + '_{:d}'.format(maxSuffix+1)
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

        
        
class FeatureCalculator(Calculator):
    type = 'feat'

    def getSpecifiers(self, cur, table, sourceField):
        return [None]
                
    def main(self, table, methods, sourceFields=[None], caseField=None, overwrite=False):
        definers = self.createDefiners(methods)
        finals = []
        with self._connect() as cur:
            uniquesGetter = field.UniquesGetter(cur, self.schema, table)
            for definer in definers:
                for sourceField in sourceFields:
                    finals.extend(definer.get(
                        uniquesGetter,
                        sourceField=sourceField,
                        caseField=caseField
                    ))
            partialSet = set(sourceFields + [caseField])
            partialSet.discard(None)
            partials = [(field, field) for field in partials]
            self.calculate(cur, table, partials, finals, overwrite=overwrite)
            # if None in partials:

    def createDefiners(self, methods):
        return [ExpressionDefiner.create(method) for method in methods]
            
               
class ExpressionDefiner(FeatureCalculator):
    CASE_PATTERN = sql.SQL('CASE WHEN {caseField}={value} THEN {numerator} ELSE 0 END')
    MAIN_PATTERN = sql.SQL('sum({numerator}) / {denominator}')
    
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
        numeratorSQL = sql.SQL(self.numerator)
        if sourceField is not None:
            numeratorSQL = numeratorSQL.format(field=sql.Identifier(sourceField))
        if caseField is not None:
            uniques = uniquesGetter.get(caseField)
            caseFieldSQL = sql.Identifier(caseField)
            numerators = [
                self.CASE_PATTERN.format(
                    caseField=caseFieldSQL,
                    value=value,
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
                self.code + '_' + name,
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
    denominator = 'cell_area'
           
class LengthDefiner(ExpressionDefiner):
    code = 'len'
    numerator = 'common_length'
    denominator = 'cell_area'
           
class CoverageDefiner(ExpressionDefiner):
    code = 'cov'
    numerator = 'common_area'
    denominator = 'cell_area'
    
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
    numerator = '{field} * item_area'
    denominator = 'sum(item_area)'
    
class DistributedSumDefiner(ExpressionDefiner):
    code = 'dsum'
    numerator = '{field} * common_area / item_area'
    denominator = '1'
        
class DistributedAverageDefiner(ExpressionDefiner):
    code = 'davg'
    numerator = '{field} * common_area'
    denominator = 'sum(common_area)'
        
        
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
    
    def main(self, table, target, sourceField, relative=False, overwrite=False):
        with self._connect() as cur:
            definer = self.DEFINERS[self.getGeometryType(cur, table), relative]()
            finals = definer.get(None, sourceField, targetName='target')
            partials = [sourceField]
            self.calculate(cur, table, partials, finals, overwrite=overwrite)
                    
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
    def main(self, multipliers=[], overwrite=False):
        with self._connect() as cur:
            self.retrieveGridDimensions(cur)
            if not self.hasGridCoordinateFields(cur):
                self.addCoordinatesToGrid(cur)
            allNames = self.getGridNames(cur)
            mainfeats, all_todos = self.generateTodoFields(
                allNames, multipliers, overwrite=overwrite
            )
        # self.logger.info('creating neighbourhood feature fields')
        # with self._connect() as cur:
            # self.createFields(
                # cur, 
                # [featname for chunk in all_todos for mult, featname in chunk],
                # overwrite=overwrite
            # )
        # self.vacuumGrid()
            for feat, todos in zip(mainfeats, all_todos):
                if todos:
                    self.logger.info('calculating neighbourhood from %s', feat)
                    raster = self.load(cur, feat)
                    outrasters = []
                    for mult, outfeat in todos:
                        self.createField(cur, outfeat, overwrite=overwrite)
                        self.logger.info('blurring %s with %f multiplier', feat, mult)
                        outrasters.append((outfeat, self.gaussify(raster, mult)))
                    self.save(cur, outrasters)
                        
    def generateTodoFields(self, allNames, multipliers, overwrite=False):
        mainfeats = [feat for feat in allNames if feat.startswith('f_')]
        all_todos = []
        for feat in mainfeats:
            all_todos.append([])
            for mult in multipliers:
                neighname = self.neighFeatureName(feat, mult)
                if overwrite or neighname not in allNames:
                    all_todos[-1].append((mult, neighname))
        return mainfeats, all_todos
        
    def emptyArray(self):
        return numpy.full(self.shape, numpy.nan)
    
    def retrieveGridDimensions(self, cur):
        qry = sql.SQL('''SELECT
                min(ST_XMin(geometry)) as xmin,
                min(ST_YMin(geometry)) as ymin,
                max(ST_XMin(geometry)) as xmax,
                max(ST_YMin(geometry)) as ymax,
                avg(sqrt(ST_Area(geometry))) as cellsize
            FROM {schema}.grid''').format(schema=self.schemaSQL).as_string(cur)
        self.logger.info('retrieving grid dimensions')
        self.logger.debug('retrieving grid dimensions: %s', qry)
        cur.execute(qry)
        self.xmin, self.ymin, xmax, ymax, self.cellsize = tuple(cur.fetchone())
        self.shape = (
            int(round((xmax - self.xmin) / self.cellsize)) + 1,
            int(round((ymax - self.ymin) / self.cellsize)) + 1,
        )
        self.logger.debug('grid dimensions: %s, starting at (%f;%f), cellsize %f',
            self.shape, self.xmin, self.ymin, self.cellsize
        )
        
    def hasGridCoordinateFields(self, cur):
        gridfields = self.getGridNames(cur)
        return 'cell_x' in gridfields and 'cell_y' in gridfields
        
    def addCoordinatesToGrid(self, cur):
        params = dict(
        )
        createQry = sql.SQL('''ALTER TABLE {schema}.grid
            ADD COLUMN cell_x double precision,
            ADD COLUMN cell_y double precision
        ''').format(schema=self.schemaSQL).as_string(cur)
        self.logger.info('adding coordinate fields to grid')
        self.logger.debug('adding coordinate fields to grid: %s', createQry)
        cur.execute(createQry)
        computeQry = sql.SQL('''UPDATE {schema}.grid SET 
            cell_x = cast(round((ST_XMin(geometry) - {xmin}) / {cellsize}) as integer),
            cell_y = cast(round((ST_YMin(geometry) - {ymin}) / {cellsize}) as integer)
        ''').format(
            schema=self.schemaSQL,
            xmin=sql.Literal(self.xmin),
            ymin=sql.Literal(self.ymin),
            cellsize=sql.Literal(self.cellsize),
        ).as_string(cur)
        self.logger.debug('computing grid coordinate fields: %s', computeQry)
        cur.execute(computeQry)
        
    
    def load(self, cur, feature):
        qry = sql.SQL('SELECT cell_x, cell_y, {feature} FROM {schema}.grid').format(
            feature=sql.Identifier(feature),
            schema=self.schemaSQL,
        ).as_string(cur)
        cur.execute(qry)
        xs, ys, fs = [
            numpy.array(list(it), dtype=(float if i == 2 else int))
            for i, it in enumerate(zip(*cur))
        ]
        array = self.emptyArray()
        array[xs,ys] = fs
        return array
            
    def save(self, cur, outrasters):
        insertflds = [list(item) for item in zip(*(
            indexes + (value, )
            for indexes, value in numpy.ndenumerate(raster)
            if not numpy.isnan(value)
        ))]
        insertQry = sql.SQL('''UPDATE {schema}.grid SET {feature} = value
            FROM (SELECT 
                unnest(%s) as x,
                unnest(%s) as y,
                unnest(%s) as value
            ) newvals
            WHERE {schema}.grid.cell_x = newvals.x AND 
                {schema}.grid.cell_y = newvals.y
        ''').format(
            schema=self.schemaSQL,
            feature=sql.Identifier(feature),
        ).as_string(cur)
        self.logger.debug('inserting field values: %s', feature)
        cur.execute(insertQry, insertflds)
          
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
    def neighFeatureName(featName, multiplier):
        multpart = str(multiplier).replace('.', '_')
        if multpart.endswith('_0'):
            multpart = multpart[:-2]
        return 'fn_' + featName[2:] + '_' + multpart

        
# class CoverageCalculator(CategorizableCalculator):
    # code = 'cov'
    # expressionTemplate = '''sum(ST_Area(ST_Intersection(
            # {{{{schema}}}}.grid.geometry, {{{{schema}}}}.{{{{table}}}}.geometry
        # )){definer}) / ST_Area({{{{schema}}}}.grid.geometry)
    # '''
    
# class DensityCalculator(CategorizableCalculator):
    # code = 'dens'
    # expressionTemplate = 'sum(1){definer} / ST_Area({{{{schema}}}}.grid.geometry)'
           
# class LengthCalculator(CategorizableCalculator):
    # code = 'len'
    # expressionTemplate = '''sum(ST_Length(ST_Intersection(
        # {{{{schema}}}}.grid.geometry, {{{{schema}}}}.{{{{table}}}}.geometry
    # )){definer}) / ST_Area({{{{schema}}}}.grid.geometry)
    # '''
       
       
# class AverageCalculator(FeatureCalculator):
    # code = 'avg'
    # denullify = False
    # expressionTemplate = 'sum({{{{schema}}}}.{{{{table}}}}.{{sourceField}}){definer} / count(1)'
            
# class WeightedAverageCalculator(AverageCalculator):
    # code = 'wavg'
    # expressionTemplate = '''sum(
            # {{{{schema}}}}.{{{{table}}}}.{{sourceField}}
            # * ST_Area({{{{schema}}}}.{{{{table}}}}.geometry){definer}
        # ) / sum(ST_Area({{{{schema}}}}.{{{{table}}}}.geometry))
    # '''
    
