import os
import pickle
import gzip
import operator

import numpy
import psycopg2.extras
from psycopg2 import sql
import pandas as pd
import sklearn.linear_model
import sklearn.svm
import sklearn.neighbors
import sklearn.ensemble
import sklearn.neural_network
import sklearn.preprocessing

from . import core, field

class Model:
    TYPES = {
        'ols' : sklearn.linear_model.LinearRegression,
        'ridge' : sklearn.linear_model.Ridge,
        'lasso' : sklearn.linear_model.Lasso,
        # 'ard' : sklearn.linear_model.ARDRegression,
        'sgd' : sklearn.linear_model.SGDRegressor,
        'ann' : sklearn.neural_network.MLPRegressor,
        'svr' : sklearn.svm.SVR,
        'knn' : sklearn.neighbors.KNeighborsRegressor,
        'rfor' : sklearn.ensemble.RandomForestRegressor,
        'extra' : sklearn.ensemble.ExtraTreesRegressor,
        'gboost' : sklearn.ensemble.GradientBoostingRegressor,
    }

    def __init__(self, typename, **kwargs):
        self.type = self.TYPES[typename]
        self.typename = typename
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.regressor = self.type(**kwargs)
        self.featureNames = None
        self.targetName = None

    def setFeatureNames(self, names):
        self.featureNames = names

    def setTargetName(self, name):
        self.targetName = name

    def getFeatureNames(self):
        return self.featureNames
    
    def getRegressor(self):
        return self.regressor

    def fit(self, features, target):
        normalized = self.scaler.fit_transform(features)
        self.regressor.fit(normalized, target)

    def predict(self, features):
        preds = self.regressor.predict(self.scaler.transform(features))
        preds[preds < 0] = 0
        return preds

    def save(self, outfile):
        with gzip.open(outfile, 'wb') as outfileobj:
            pickle.dump(self, outfileobj)
    
    @classmethod
    def load(cls, infile):
        with gzip.open(infile, 'rb') as infileobj:
            return pickle.load(infileobj)


    
class ModelTrainer(field.Handler):
    def main(self, modeltype, targetTable, outpath, overwrite=False, **kwargs):
        if os.path.isfile(outpath) and not overwrite:
            raise IOError('model file {} already exists'.format(outpath))
        model = Model(modeltype, **kwargs)
        model.setTargetName(targetTable[4:])
        with self._connect() as cur:
            featureNames = self.getConsolidatedFeatureNames(cur)
            model.setFeatureNames(featureNames)
            self.logger.info('selecting training values')
            features, target = self.selectFeaturesAndTarget(cur, featureNames, targetTable)
            self.logger.info('training model')
            print(features[:,:3])
            print(target)
            model.fit(features, target)
            self.logger.info('saving model to %s', outpath)
            with open(outpath, 'wb') as outfile:
                model.save(outfile)

    def selectFeaturesAndTarget(self, cur, featureNames, targetTable):
        return (
            self.selectConsolidatedFeatures(cur, featureNames, 
                # inside=True
            ).fillna(0).values,
            self.selectTarget(cur, targetTable,
                # inside=True
            ).fillna(0).values
        )


class ModelArrayTrainer(ModelTrainer):
    def main(self, targetTable, outpath, overwrite=False, **kwargs):
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        with self._connect() as cur:
            featureNames = self.getConsolidatedFeatureNames(cur)
            self.logger.info('selecting training values')
            features, target = self.selectFeaturesAndTarget(
                cur, featureNames, targetTable
            )
            self.logger.info('training models to %s', outpath)
            for modeltype in Model.TYPES.keys():
                model = Model(modeltype, **kwargs)
                self.logger.info('training %s', model.typename)
                outmodpath = os.path.join(
                    outpath,
                    'model_{}.rum'.format(model.typename)
                )
                if os.path.isfile(outmodpath) and not overwrite:
                    raise IOError('model file {} already exists'.format(outmodpath))
                model.setTargetName(targetTable)
                model.setFeatureNames(featureNames)
                model.fit(features, target)
                with open(outmodpath, 'wb') as outfile:
                    model.save(outfile)


class ModelApplier(field.Handler):
    def main(self, modelPath, weightTable, overwrite=False):
        self.logger.info('loading models from %s', modelPath)
        model = Model.load(modelPath)
        with self._connect() as cur:
            self.logger.info('selecting features')
            features, ids = self.selectFeaturesAndIds(cur, model.getFeatureNames())
            self.logger.info('predicting weights')
            weights = model.predict(features)
            self.logger.info('saving weights')
            self.saveWeights(cur, weightTable, ids, weights, overwrite=overwrite)

    def selectFeaturesAndIds(self, cur, featureNames):
        data = self.selectConsolidatedFeatures(cur, featureNames + ['geohash'])
        ids = data['geohash'].tolist()
        data.drop('geohash', axis=1, inplace=True)
        data.fillna(0, inplace=True)
        return data.values, ids

    def saveWeights(self, cur, weightTable, ids, weights, overwrite=False):
        params = {
            'schema' : self.schemaSQL,
            'name' : sql.Identifier(weightTable)
        }
        self.clearTable(cur, weightTable, overwrite)
        createQry = sql.SQL('''CREATE TABLE {schema}.{name} (
            geohash text, weight double precision
        )''').format(**params).as_string(cur)
        self.logger.debug('creating weight table: %s', createQry)
        cur.execute(createQry)
        insertQry = sql.SQL(
            'INSERT INTO {schema}.{name} VALUES (%s, %s)'
        ).format(**params).as_string(cur)
        self.logger.debug('inserting weights: %s', insertQry)
        psycopg2.extras.execute_batch(cur, insertQry, zip(ids, weights))
        self.createPrimaryKey(cur, weightTable)
        

class ModelArrayApplier(ModelApplier):
    def main(self, modelDirPath, weightTable, overwrite=False):
        with self._connect() as cur:
            modelIter = self.loadModels(modelDirPath)
            self.logger.info('selecting features')
            model = next(modelIter)
            features, ids = self.selectFeaturesAndIds(cur, model.getFeatureNames())
            weightFields = []
            weightValues = [numpy.array(ids)]
            while True:
                self.logger.info('running %s', model.typename)
                weightField = 'weight_{}'.format(model.typename)
                weights = model.predict(features)
                weightFields.append(weightField)
                weightValues.append(weights)
                try:
                    model = next(modelIter)
                except StopIteration:
                    break
            self.saveMultipleWeights(
                cur, weightTable, weightFields, weightValues, overwrite=overwrite
            )


    def saveMultipleWeights(self, cur, table, fields, values, overwrite=False):
        self.clearTable(cur, table, overwrite)
        params = {
            'schema' : self.schemaSQL,
            'table' : sql.Identifier(table)
        }
        createQry = (
            sql.SQL('CREATE TABLE {schema}.{table} (geohash text, ').format(**params)
            + sql.SQL(' double precision, ').join([sql.Identifier(fld) for fld in fields])
            + sql.SQL(' double precision)')
        ).as_string(cur)
        self.logger.info('creating weight table %s', table)
        self.logger.debug('creating weight table: %s', createQry)
        cur.execute(createQry)
        insertQry = (
            sql.SQL('INSERT INTO {schema}.{table} VALUES (').format(**params)
            + sql.SQL(', ').join([sql.SQL('%s')] * (len(fields) + 1))
            + sql.SQL(')')
        ).as_string(cur)
        self.logger.debug('inserting weights: %s', insertQry)
        psycopg2.extras.execute_batch(cur, insertQry, list(zip(*values)))

        
    # def updateMultipleWeights(self, cur, ids, weightFields, weightValues):
        # namepart = sql.SQL(', ').join([
            # sql.SQL('{feat} = {tmpfeat}').format(
                # feat=sql.Identifier(feat),
                # tmpfeat=sql.Identifier('tmp_' + feat)
            # )
            # for feat in weightFields
        # ])
        # valpart = sql.SQL(', ').join([
            # sql.SQL('unnest(%s) as {tmpfeat}').format(
                # tmpfeat=sql.Identifier('tmp_' + feat)
            # )
            # for feat in weightFields
        # ])
        # insertQry = sql.SQL('''UPDATE {schema}.grid SET {namepart}
            # FROM (SELECT unnest(%s) as geohash, {valpart}) newvals
            # WHERE {schema}.grid.geohash = newvals.geohash
        # ''').format(
            # schema=self.schemaSQL,
            # namepart=namepart,
            # valpart=valpart
        # ).as_string(cur)
        # self.logger.debug('inserting weights: %s', insertQry)
        # cur.execute(insertQry, [ids] + weightValues)

    def loadModels(self, path):
        self.logger.info('loading models from %s', path)
        for fileName in os.listdir(path):
            try:
                yield Model.load(modelPath)
            except pickle.UnpicklingError:
                pass


class ModelIntrospector(core.Task):
    ITEMS = [
        ('feature_importances_', 'feature importances', None),
        ('coef_', 'coefficients', abs),
    ]

    def main(self, modelfile):
        model = Model.load(modelfile)
        coreModel = model.getRegressor()
        featnames = model.getFeatureNames()
        reported = False
        for attr, name, sorter in self.ITEMS:
            if hasattr(coreModel, attr):
                self.report(featnames, getattr(coreModel, attr), name, sorter)
                reported = True
        if not reported:
            print('*** No introspectable attribute found')
                
    def report(self, names, values, label, sorter=None):
        leftcolwidth = min(max(len(name) for name in names) + 1, 60)
        print('Model', label, '({} features)'.format(len(names)))
        print()
        rowiter = sorted(zip(names, values),
            key=((lambda t: sorter(t[1])) if sorter else operator.itemgetter(1)),
            reverse=True
        )
        for name, value in rowiter:
            if value:
                print(name.ljust(leftcolwidth), value)
