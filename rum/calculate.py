
from psycopg2 import sql

from . import core


class Calculator(core.DatabaseTask):
    targetType = 'double precision'
    denullify = None

    @classmethod
    def create(cls, code, args):
        try:
            return cls.CODES[code].fromArgs(args)
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
                targetField = baseTargetField
                if specifier:
                    targetField += '_' + str(specifier)
                targetFieldSQL = sql.Identifier(targetField)
                self.logger.info('computing grid field %s', targetField)
                self.createField(cur, targetField, overwrite=overwrite)
                subqry = self.query(cur, table, sourceField, specifier)
                qry = sql.SQL('UPDATE {schema}.grid SET {targetField} = ({subqry})').format(
                    schema=self.schemaSQL,
                    targetField=targetFieldSQL,
                    subqry=subqry
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

    def getSpecifiers(self, cur, table, sourceField):
        return [None]
                
    def createField(self, cur, targetField, overwrite=False):
        creator = sql.SQL('ALTER TABLE {schema}.grid ADD {ifnex} {colname} {coltype}').format(
            schema=self.schemaSQL,
            ifnex=sql.SQL('IF NOT EXISTS' if overwrite else ''),
            colname=sql.Identifier(targetField),
            coltype=sql.SQL(self.targetType),
        ).as_string(cur)
        self.logger.debug('creating field: %s', creator)
        cur.execute(creator)
                
    def query(self, cur, table, sourceField):
        raise NotImplementedError
        
class CategorizableCalculator(Calculator):
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
        
    def query(self, cur, table, sourceField=None, specifier=None):
        qryTempl = self.template
        formatArgs = dict(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
            specifier=sql.Literal(specifier)
        )
        if sourceField:
            self.template += ' AND {schema}.{table}.{sourceField}={specifier}'
            formatArgs['sourceField'] = sql.Identifier(sourceField)
        return sql.SQL(qryTempl).format(**formatArgs)
        
        
class CoverageCalculator(CategorizableCalculator):
    code = 'cov'
    template = '''SELECT (
        sum(ST_Area(ST_Intersection(
                {schema}.grid.geometry, {schema}.{table}.geometry
            ))) / ST_Area({schema}.grid.geometry)
        ) FROM {schema}.{table}
        WHERE ST_Intersects(
            {schema}.grid.geometry,
            {schema}.{table}.geometry
        )'''
    
    
class DensityCalculator(CategorizableCalculator):
    code = 'dens'
    template = '''SELECT sum(1) / ST_Area({schema}.grid.geometry)
        FROM {schema}.{table}
        WHERE ST_Intersects(
            {schema}.grid.geometry,
            {schema}.{table}.geometry
        )'''
           
class AverageCalculator(Calculator):
    code = 'avg'
    denullify = False
    template = '''SELECT sum({schema}.{table}.{sourceField}) / count(1)
        FROM {schema}.{table}
        WHERE ST_Intersects(
            {schema}.grid.geometry,
            {schema}.{table}.geometry
        )'''
    
    def query(self, cur, table, sourceField, specifier=None):
        return sql.SQL(self.template).format(
            schema=self.schemaSQL,
            table=sql.Identifier(table),
            sourceField=sql.Identifier(sourceField)
        )
        
class WeightedAverageCalculator(AverageCalculator):
    code = 'wavg'
    template = '''SELECT sum(
            {schema}.{table}.{sourceField} * ST_Area({schema}.{table}.geometry)
        ) / sum(ST_Area({schema}.{table}.geometry))
        FROM {schema}.{table}
        WHERE ST_Intersects(
            {schema}.grid.geometry,
            {schema}.{table}.geometry
        )'''
    
        
Calculator.CODES = {calc.code : calc for calc in [
    CoverageCalculator,
    DensityCalculator,
    AverageCalculator,
    WeightedAverageCalculator,
]}