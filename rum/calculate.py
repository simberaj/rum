import contextlib

import numpy
import scipy.ndimage
import psycopg2.extras
from psycopg2 import sql

from . import core


class Calculator(core.DatabaseTask):
    targetType = 'double precision'
    denullify = None
    BASE_TEMPLATE = '''UPDATE {{schema}}.grid SET {{targetField}} = (
        SELECT {expression}
        FROM {{schema}}.{{table}}
        WHERE ST_Intersects(
            {{schema}}.grid.geometry,
            {{schema}}.{{table}}.geometry
        ){where}
    )'''

    def calculateField(self, cur, table, targetField, sourceField=None, overwrite=False, **kwargs):
        targetFieldSQL = sql.Identifier(targetField)
        self.logger.info('computing grid field %s', targetField)
        self.createField(cur, targetField, overwrite=overwrite)
        expression = self.expression(cur, table, sourceField, **kwargs)
        where = self.where(cur, table, sourceField, **kwargs)
        template = sql.SQL(
            sql.SQL(self.BASE_TEMPLATE).format(
                expression=expression,
                where=(sql.SQL(' AND ') + where if where else sql.SQL(''))
            ).as_string(cur)
        )
        qry = template.format(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
            sourceField=(sql.Identifier(sourceField) if sourceField else None),
            targetField=targetFieldSQL,
        ).as_string(cur)
        self.logger.debug('computing field: %s', qry)
        cur.execute(qry)
        if self.denullify:
            zeroqry = sql.SQL('''UPDATE {schema}.grid SET {targetField} = 0
                WHERE {targetField} IS NULL''').format(
                schema=self.schemaSQL,
                targetField=targetFieldSQL,
            ).as_string(cur)
            self.logger.debug('denulling field: %s', zeroqry)
            cur.execute(zeroqry)
                
    def createField(self, cur, targetField, overwrite=False):
        self.createFields(cur, [targetField], overwrite=overwrite)
        # creator = sql.SQL('ALTER TABLE {schema}.grid ADD {ifnex}{colname} {coltype}').format(
            # schema=self.schemaSQL,
            # ifnex=sql.SQL('IF NOT EXISTS ' if overwrite else ''),
            # colname=sql.Identifier(targetField),
            # coltype=sql.SQL(self.targetType),
        # ).as_string(cur)
        # self.logger.debug('creating field: %s', creator)
        # cur.execute(creator)
    
    def expression(self, *args, **kwargs):
        raise NotImplementedError

    def where(self, *args, **kwargs):
        return None
        
    def vacuumGrid(self):
        with self._connect(autocommit=True) as cur:
            self.logger.debug('vacuuming grid')
            cur.execute(
                sql.SQL('VACUUM ANALYZE {schema}.grid').format(
                    schema=self.schemaSQL
                )
            )

        
        
class FeatureCalculator(Calculator):
    def getSpecifiers(self, cur, table, sourceField):
        return [None]
                
    @classmethod
    def create(cls, code, args):
        try:
            return cls.CODES[code.lower()].fromArgs(args)
        except KeyError as err:
            raise core.InvalidParameterError('invalid method {}, allowed: {}'.format(
                code, ', '.join(cls.CODES.keys())
            ))
        
    def main(self, table, sourceField=None, overwrite=False):
        baseTargetField = 'f_{table}_{method}'.format(
            table=table, method=self.code
        )
        if sourceField:
            baseTargetField += ('_' + sourceField)
        with self._connect() as cur:
            specifiers = self.getSpecifiers(cur, table, sourceField)
            if len(specifiers) > 1:
                self.logger.info('found %d subfields', len(specifiers))
        for specifier in specifiers:
            with self._connect() as cur:
                targetField = baseTargetField
                if specifier:
                    targetField += '_' + str(specifier)
                self.calculateField(
                    cur, table, targetField,
                    sourceField=sourceField,
                    overwrite=overwrite,
                    specifier=specifier
                )
            self.vacuumGrid()
    
    def expression(self, *litter, **garbage):
        return sql.SQL(self.expressionTemplate)
    
               
class CategorizableCalculator(FeatureCalculator):
    denullify = True

    def getSpecifiers(self, cur, table, sourceField=None):
        if sourceField:
            uniqueQry = sql.SQL('SELECT DISTINCT {sourceField} FROM {schema}.{table}').format(
                schema=self.schemaSQL,
                table=sql.Identifier(table),
                sourceField=sql.Identifier(sourceField)
            ).as_string(cur)
            self.logger.debug('retrieving unique values: %s', uniqueQry)
            cur.execute(uniqueQry)
            return list(sorted(row[0] for row in cur.fetchall() if row[0] is not None))
        else:
            return [None]
        
    def where(self, cur, table, sourceField=None, specifier=None):
        if sourceField:
            return sql.SQL('{{schema}}.{{table}}.{{sourceField}}={specifier}').format(
                specifier=sql.Literal(specifier)
            )
        else:
            return None
        
        
class CoverageCalculator(CategorizableCalculator):
    code = 'cov'
    expressionTemplate = '''sum(ST_Area(ST_Intersection(
            {schema}.grid.geometry, {schema}.{table}.geometry
        ))) / ST_Area({schema}.grid.geometry)
    '''
    
class DensityCalculator(CategorizableCalculator):
    code = 'dens'
    expressionTemplate = 'sum(1) / ST_Area({schema}.grid.geometry)'
           
class LengthCalculator(CategorizableCalculator):
    code = 'len'
    expressionTemplate = '''sum(ST_Length(ST_Intersection(
        {schema}.grid.geometry, {schema}.{table}.geometry
    ))) / ST_Area({schema}.grid.geometry)
    '''
       
       
class AverageCalculator(FeatureCalculator):
    code = 'avg'
    denullify = False
    expressionTemplate = 'sum({schema}.{table}.{sourceField}) / count(1)'
            
class WeightedAverageCalculator(AverageCalculator):
    code = 'wavg'
    expressionTemplate = '''sum(
            {schema}.{table}.{sourceField} * ST_Area({schema}.{table}.geometry)
        ) / sum(ST_Area({schema}.{table}.geometry))
    '''
        
FeatureCalculator.CODES = {calc.code : calc for calc in [
    CoverageCalculator,
    DensityCalculator,
    LengthCalculator,
    AverageCalculator,
    WeightedAverageCalculator,
]}


class TargetCalculator(Calculator):
    TEMPLATES = {
        ('POINT', False) : 'sum({sourceField}) / ST_Area({schema}.grid.geometry)',
        ('POINT', True) : 'avg({sourceField})',
        ('POLYGON', False) : '''sum({sourceField}
            * ST_Area(ST_Intersection({schema}.grid.geometry, {schema}.{table}.geometry))
            / ST_Area({schema}.{table}.geometry)
        )''',
        ('POLYGON', True) : '''sum(
            {sourceField} * ST_Area(ST_Intersection(
                {schema}.grid.geometry, {schema}.{table}.geometry
            ))
        ) / sum(ST_Area(ST_Intersection(
            {schema}.grid.geometry, {schema}.{table}.geometry
            ))
        )''',
    }
    
    def main(self, table, sourceField, targetField, relative=False, overwrite=False):
        with self._connect() as cur:
            self.calculateField(
                cur, table, targetField,
                sourceField=sourceField,
                overwrite=overwrite,
                relative=relative
            )
            
    def expression(self, cur, table, sourceField, relative=False, **garbage):
        return sql.SQL(self.TEMPLATES[self.getCalculationMode(cur, table, relative)])

        
    def getCalculationMode(self, cur, table, relative=False):
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
        return (geomtype, relative)

        
class NeighbourhoodCalculator(Calculator):
    def main(self, multipliers=[], overwrite=False):
        with self._connect() as cur:
            self.retrieveGridDimensions(cur)
            if not self.hasGridCoordinateFields(cur):
                self.addCoordinatesToGrid(cur)
            allNames = self.getGridNames(cur)
        mainfeats, all_todos = self.generateTodoFields(allNames, multipliers, overwrite=overwrite)
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
                with self._connect() as cur:
                    self.logger.info('calculating neighbourhood from %s', feat)
                    raster = self.load(cur, feat)
                for mult, outfeat in todos:
                    with self._connect() as cur:
                        self.createField(cur, outfeat, overwrite=overwrite)
                        self.logger.info('blurring %s with %f multiplier', feat, mult)
                        outrast = self.gaussify(raster, mult)
                        self.save(cur, outfeat, outrast)
                    self.vacuumGrid()
                        
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
            
    def save(self, cur, feature, raster):
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

        
