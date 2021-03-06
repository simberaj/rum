import os
import csv
import math

import numpy
from psycopg2 import sql
import matplotlib
import matplotlib.pyplot as plt

from . import field, html


class BaseValidator(field.Handler):
    @staticmethod
    def validate(data, trueField, modelField, reportPath):
        models = data[modelField].values
        trues = data[trueField].values
        validator = Validator(models, trues)
        validator.validate()
        for ind, val in validator.results.items():
            print(ind.ljust(7), val)
        if reportPath:
            validator.output(reportPath)
        return validator.results


class RawValidator(BaseValidator):
    def main(self, table, trueField='target', modelField='value', reportPath=None):
        with self._connect() as cur:
            data = self.selectValues(cur, table, [trueField, modelField]).fillna(0)
        return self.validate(data, trueField, modelField, reportPath)


class ModelValidator(BaseValidator):
    def main(self, trueTable, modelTable, trueField='target', modelField='value', reportPath=None):
        with self._connect() as cur:
            data = self.select(cur, trueTable, modelTable, trueField, [modelField])
        return self.validate(data, trueField, modelField, reportPath)

    def select(self, cur, trueTable, modelTable, trueField, modelFields):
        qry = sql.SQL('''SELECT
            t.{trueField}, m.{modelFieldSeq}
        FROM {schema}.grid g
            LEFT JOIN {schema}.{trueTable} t on g.geohash=t.geohash
            LEFT JOIN {schema}.{modelTable} m on g.geohash=m.geohash
        ''').format(
            schema=self.schemaSQL,
            trueTable=sql.Identifier(trueTable),
            modelTable=sql.Identifier(modelTable),
            trueField=sql.Identifier(trueField),
            modelFieldSeq=sql.SQL(', m.').join(
                sql.Identifier(fld) for fld in modelFields
            ),
        ).as_string(cur)
        self.logger.debug('selecting validation data: %s', qry)
        cur.execute(qry)
        return self.resultToDF(cur, [trueField] + modelFields).fillna(0)



class ModelArrayValidator(ModelValidator):
    def main(self, trueTable, modelTable, trueField='target', reportPath=None):
        with self._connect() as cur:
            modelFields = self.getColumnNamesForTable(cur, modelTable)
            if 'geohash' in modelFields:
                modelFields.remove('geohash')
            data = self.select(cur, trueTable, modelTable, trueField, modelFields)
        if reportPath and not os.path.isdir(reportPath):
            self.logger.debug('creating report directory: %s', reportPath)
            os.mkdir(reportPath)
        results = {}
        for modelField in modelFields:
            print('Validation of', modelField)
            results[modelField] = self.validate(data, trueField, modelField,
                os.path.join(reportPath, modelField + '.html') if reportPath else None
            )
            print()
        return results


class ModelMultiscaleValidator(BaseValidator):
    defaultMultiples = [2, 5, 10]
    SELECTION = sql.SQL('''WITH multgrid AS
        (SELECT
            makegrid(geometry,{gridSize},{xoffset},{yoffset}) AS geometry
            FROM {schema}.extent
        )
    SELECT
        {aggreg}(t.{trueField}) AS {trueField},
        {aggreg}(m.{modelField}) AS {modelField}
    FROM multgrid mg
        LEFT JOIN {schema}.grid g ON ST_Within(g.geometry, mg.geometry)
        LEFT JOIN {schema}.{trueTable} t ON g.geohash=t.geohash
        LEFT JOIN {schema}.{modelTable} m ON g.geohash=m.geohash
    GROUP BY mg.geometry
    ''')

    def main(self, trueTable, modelTable, trueField='target', modelField='value', multiples=None, reportPath=None, xoffset=None, yoffset=None):
        if reportPath and not os.path.exists(reportPath):
            self.logger.debug('creating directory %s', reportPath)
            os.mkdir(reportPath)
        if not multiples:
            multiples = self.defaultMultiples
        gridSize = self.getGridSize()
        if xoffset is None or yoffset is None:
            newXOffset, newYOffset = self.getGridOffset()
            if xoffset is None: xoffset = newXOffset
            if yoffset is None: yoffset = newYOffset
        # validator = ModelValidator(connector=self.connector.copy(), schema=self.schema)
        results = {}
        relative = False # TODO
        with self._connect() as cur:
            for multiple in multiples:
                if reportPath:
                    reportFile = os.path.join(reportPath, 'level_{}.html'.format(multiple))
                print('Multiscale level {}'.format(multiple))
                size = gridSize * multiple
                qry = self.SELECTION.format(
                    schema=self.schemaSQL,
                    trueTable=sql.Identifier(trueTable),
                    modelTable=sql.Identifier(modelTable),
                    trueField=sql.Identifier(trueField),
                    modelField=sql.Identifier(modelField),
                    gridSize=sql.Literal(size),
                    aggreg=sql.SQL('avg' if relative else 'sum'),
                    xoffset=sql.Literal(xoffset % size),
                    yoffset=sql.Literal(yoffset % size),
                ).as_string(cur)
                self.logger.debug(
                    'retrieving validation data for multiple %d: %s',
                    multiple, qry
                )
                cur.execute(qry)
                results[multiple] = self.validate(
                    self.resultToDF(cur, [trueField, modelField]).fillna(0),
                    trueField, modelField,
                    reportFile,
                )
                print()
        self.logger.debug('validation results: %s', results)
        with open(os.path.join(reportPath, 'results.csv'), 'w', newline='') as outfile:
            wr = csv.DictWriter(
                outfile,
                ['multiple'] + list(results[multiples[0]].keys()),
                delimiter=';'
            )
            wr.writeheader()
            for multiple in sorted(results):
                results[multiple]['multiple'] = multiple
                wr.writerow(results[multiple])

    def getGridSize(self):
        with self._connect() as cur:
            qry = sql.SQL('''SELECT
                    avg(sqrt(ST_Area(geometry))) as cellsize
                FROM {schema}.grid''').format(schema=self.schemaSQL).as_string(cur)
            self.logger.debug('retrieving grid cell size: %s', qry)
            cur.execute(qry)
            return cur.fetchone()[0]

    def getGridOffset(self, **kwargs):
        with self._connect() as cur:
            qry = sql.SQL('''SELECT
                    min(ST_XMin(geometry)) as xoffset,
                    min(ST_YMin(geometry)) as yoffset
                FROM {schema}.grid''').format(schema=self.schemaSQL).as_string(cur)
            self.logger.debug('retrieving grid cell offset: %s', qry)
            cur.execute(qry)
            return cur.fetchone()


class Validator:
    templateFilePath = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'share', 'report.html')
    )
    ERRAGG_NAMES = [
        ('mismatch', 'Absolute value sum difference'),
        ('rmse', 'Root mean square error (RMSE)'),
        ('tae', 'Total absolute error (TAE)'),
        ('rtae', 'Relative total absolute error (RTAE)'),
        ('r2', 'Coefficient of determination (R<sup>2</sup>)'),
    ]
    DESC_NAMES = [
        ('set', 'Dataset'),
        ('sum', 'Sum'),
        ('min', 'Minimum'),
        ('q2l', 'Q2,5'),
        ('q10', 'Q10 - 1st Decile'),
        ('q25', 'Q25 - 1st Quartile'),
        ('median', 'Q50 - Median'),
        ('q75', 'Q75 - 3rd Quartile'),
        ('q90', 'Q90 - 9th Decile'),
        ('q2h', 'Q97,5'),
        ('max', 'Maximum'),
        ('mean', 'Mean'),
    ]
    FORMATS = {
        'mismatch' : '{:g}',
        'tae' : '{:.0f}',
        'rtae' : '{:.2%}',
        'r2' : '{:.3%}',
        'rmse' : '{:.2f}',
    }
    FORMATS.update({item[0] : '{:g}' for item in DESC_NAMES})

    def __init__(self, models, reals):
        self.models = models
        self.reals = reals

    def validate(self):
        self.realSum = self.reals.sum()
        self.realMean = self.realSum / len(self.reals)
        self.modelSum = self.models.sum()
        self.mismatch = self.modelSum - self.realSum
        self.resid = self.models - self.reals
        self.absResid = abs(self.resid)
        self.sqResidSum = (self.absResid ** 2).sum()
        self.tae = self.absResid.sum()
        self.rtae = self.tae / self.realSum
        self.r2 = 1 - self.sqResidSum / ((self.reals - self.realMean) ** 2).sum()
        self.rmse = math.sqrt(self.sqResidSum / len(self.reals))

    @property
    def results(self):
        return {
            'DIFF' : self.mismatch,
            'TAE' : self.tae,
            'RTAE' : self.rtae,
            'R2' : self.r2,
            'RMSE' : self.rmse,
        }

    def describe(self, data):
        return {
            'min' : data.min(),
            'max' : data.max(),
            'sum' : data.sum(),
            'mean' : data.mean(),
            'median' : numpy.median(data),
            'q25' : numpy.percentile(data, 25),
            'q75' : numpy.percentile(data, 75),
            'q10' : numpy.percentile(data, 10),
            'q90' : numpy.percentile(data, 90),
            'q2l' : numpy.percentile(data, 2.5),
            'q2h' : numpy.percentile(data, 97.5)
        }

    def descriptions(self):
        descs = []
        for name, data in (
                ('Modeled', self.models),
                ('Real', self.reals),
                ('Residuals', self.resid),
                ('Abs(Residuals)', self.absResid)
            ):
            desc = self.format(self.describe(data))
            desc['set'] = name
            descs.append(desc)
        return descs

    def globals(self):
        return dict(
            mismatch=self.mismatch,
            tae=self.tae,
            rtae=self.rtae,
            r2=self.r2,
            rmse=self.rmse
        )

    def format(self, fdict):
        return {
            key : self.FORMATS[key].format(float(value)).replace('.', ',')
            for key, value in fdict.items()
        }

    def output(self, fileName):
        try:
            with open(self.templateFilePath) as templFile:
                template = templFile.read()
        except IOError as exc:
            raise IOError('report template file not found') from exc
        template = template.replace('{', '[').replace('}', ']').replace('[[', '{').replace(']]', '}')
        fileNameBase = os.path.splitext(fileName)[0]
        text = template.format(
            erragg=html.dictToTable(
                self.format(self.globals()),
                self.ERRAGG_NAMES
            ),
            setdescstat=html.dictToTable(
                self.descriptions(),
                self.DESC_NAMES,
                rowHead=True
            ),
            fname=os.path.relpath(fileNameBase, os.path.dirname(fileNameBase))
        )
        with open(fileName, 'w') as outfile:
            outfile.write(text.replace('[', '{').replace(']', '}'))
        self.outputImages(fileNameBase)


    def outputImages(self, fileNameBase):
        plt.figure()
        plt.loglog(self.reals, self.models, 'b.')
        ax = plt.gca()
        lims = [
            numpy.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            numpy.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r-', alpha=0.25, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel('Real values')
        plt.ylabel('Modeled values')
        plt.savefig(fileNameBase + '_correl.png', bbox_inches='tight')