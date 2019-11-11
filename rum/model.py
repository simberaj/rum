import os
import pickle
import gzip
import operator
import re

import numpy
import scipy.optimize
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


class LADRegression:
    def __init__(self, epsilon=1e-7, delta=1e-3, **kwargs):
        self.epsilon = epsilon
        self.delta = delta
        self.inner = sklearn.linear_model.LinearRegression(**kwargs)

    def fit(self, X, y, sample_weights=None):
        if sample_weights is None:
            sample_weights = numpy.ones(y.shape[0])
        prev_tae = numpy.inf
        tgtsum = y.sum()
        while True:
            self.inner.fit(X, y, sample_weights)
            fitted = self.inner.predict(X)
            resid = abs(fitted - y)
            tae = resid.sum() / tgtsum
            if abs(tae - prev_tae) < self.epsilon:
                break
            prev_tae = tae
            sample_weights = 1 / numpy.where(resid < self.delta, self.delta, resid)
        self.coef_ = self.inner.coef_
        self.intercept_ = self.inner.intercept_

    def predict(self, X):
        return self.inner.predict(X)


class SumMatchingMultiplication:
    def __init__(self, fit_intercept=False):
        if fit_intercept:
            raise NotImplementedError

    def fit(self, X, y):
        self.multiplier = y.sum() / X.sum()
        self.coef_ = [self.multiplier]
        self.intercept_ = 0

    def predict(self, X):
        return (X * self.multiplier).sum(axis=1)


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
    PARAMETERS = {
        'rfor' : {'n_estimators' : 100},
        'extra' : {'n_estimators' : 100},
        'gboost' : {'n_estimators' : 200, 'loss' : 'lad'},
        'sgd' : {'max_iter' : 500, 'tol' : 1e-5},
        'ann' : {'max_iter' : 500, 'activation' : 'tanh'},
        'knn' : {'n_neighbors' : 5},
        'svr' : {'gamma' : 'auto'},
    }

    def __init__(self, typename, **kwargs):
        self.type = self.TYPES[typename]
        self.typename = typename
        self.scaler = sklearn.preprocessing.StandardScaler()
        params = self.PARAMETERS.get(typename, {}).copy()
        params.update(kwargs)
        self.regressor = self.type(**params)
        self.featureNames = None
        self.targetName = None

    def setFeatureNames(self, names):
        self.featureNames = names

    def setTargetName(self, name):
        self.targetName = name

    def getFeatureNames(self, intercept=False):
        return self.featureNames + (['intercept'] if intercept else [])

    def getRegressor(self):
        return self.regressor

    def fit(self, features, target):
        normalized = self.scaler.fit_transform(self.addIntercept(features))
        self.regressor.fit(normalized, target)

    def predict(self, features):
        preds = self.regressor.predict(self.scaler.transform(self.addIntercept(features)))
        preds[preds < 0] = 0
        return preds

    def addIntercept(self, features):
        return numpy.hstack((features, numpy.zeros((len(features), 1))))

    def save(self, outfile):
        with gzip.open(outfile, 'wb') as outfileobj:
            pickle.dump(self, outfileobj)

    @classmethod
    def load(cls, infile):
        with gzip.open(infile, 'rb') as infileobj:
            return pickle.load(infileobj)



class ModelTrainer(field.Handler):
    def main(self, modeltype, targetTable, outpath, seed=None, fraction=1, feature_regex=None, invert_regex=False, overwrite=False, condition=True, **kwargs):
        if os.path.isfile(outpath) and not overwrite:
            raise IOError('model file {} already exists'.format(outpath))
        if seed is not None:
            numpy.random.seed(seed)
        model = Model(modeltype, **kwargs)
        model.setTargetName(targetTable[4:])
        with self._connect() as cur:
            featureNames = self.getTrainingFeatureNames(cur, feature_regex, invert_regex)
            if not featureNames:
                raise ValueError('no features found')
            model.setFeatureNames(featureNames)
            features, target = self.selectFeaturesAndTarget(
                cur, featureNames, targetTable, fraction, condition=condition
            )
            self.logger.info('training model on %d features', len(featureNames))
            model.fit(features, target)
            self.logger.info('saving model to %s', outpath)
            with open(outpath, 'wb') as outfile:
                model.save(outfile)

    def getTrainingFeatureNames(self, cur, regex, invert=False):
        all_feats = self.getConsolidatedFeatureNames(cur)
        if regex:
            pattern = re.compile(regex)
            if invert:
                checker = lambda fn: not pattern.fullmatch(fn)
            else:
                checker = lambda fn: pattern.fullmatch(fn)
            return [feat for feat in all_feats if checker(feat)]
        else:
            return all_feats

    def selectFeaturesAndTarget(self, cur, featureNames, targetTable, fraction=1, condition=True):
        self.logger.info('selecting training values')
        feats = self.selectConsolidatedFeatures(
            cur,
            featureNames,
            inside=True,
            condition=condition
        ).fillna(0).values
        targets = self.selectTarget(
            cur,
            targetTable,
            inside=True,
            condition=condition
        ).fillna(0).values
        return self.restrictToFraction(feats, targets, fraction=fraction)

    def restrictToFraction(self, features, target, fraction=1):
        if fraction < 1:
            n_all = len(features)
            n_sel = int(n_all * fraction)
            if n_sel == 0:
                raise ValueError('fraction too low: no samples selected')
            rands = numpy.random.choice(n_all, n_sel, replace=False)
            return features[rands], target[rands]
        else:
            return features, target


class ModelArrayTrainer(ModelTrainer):
    def main(self, targetTable, outpath, seed=None, fraction=1, feature_regex=None, overwrite=False, condition=True, **kwargs):
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        if seed is not None:
            numpy.random.seed(seed)
        with self._connect() as cur:
            featureNames = self.getTrainingFeatureNames(cur, feature_regex)
            features, target = self.selectFeaturesAndTarget(
                cur, featureNames, targetTable, fraction, condition=condition
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
    def main(self, modelPath, weightTable, overwrite=False, condition=True):
        self.logger.info('loading models from %s', modelPath)
        model = Model.load(modelPath)
        with self._connect() as cur:
            self.logger.info('selecting features')
            features, ids = self.selectFeaturesAndIds(cur, model.getFeatureNames(), condition=condition)
            self.logger.info('predicting weights')
            weights = model.predict(features)
            self.logger.info('saving weights')
            self.saveWeights(cur, weightTable, ids, weights, overwrite=overwrite)

    def selectFeaturesAndIds(self, cur, featureNames, condition=True):
        currentNames = self.getConsolidatedFeatureNames(cur)
        missings = []
        founds = []
        for name in featureNames:
            if name in currentNames:
                founds.append(name)
            else:
                missings.append(name)
                self.logger.warn('feature %s from model not found in target schema, imputing zeros', name)
        data = self.selectConsolidatedFeatures(cur, founds + ['geohash'], condition=condition)
        ids = data['geohash'].tolist()
        data.drop('geohash', axis=1, inplace=True)
        data.fillna(0, inplace=True)
        for name in missings:
            data[name] = 0
        print(data[featureNames].values.shape, len(featureNames), len(currentNames))
        return data[featureNames].values, ids

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
    def main(self, modelDirPath, weightTable, modelNames=None, featuresDiffer=False, overwrite=False, condition=True, useModelFileNames=False):
        with self._connect() as cur:
            models, modelNames = self.loadAndNameModels(
                modelDirPath,
                modelNames=modelNames,
                useFileNames=useModelFileNames
            )
            if not featuresDiffer:
                self.logger.info('selecting features')
                features, ids = self.selectFeaturesAndIds(cur, models[0].getFeatureNames(), condition=condition)
                weightValues = [numpy.array(ids)]
            else:
                weightValues = []
            weightFields = []
            for model, modelName in zip(models, modelNames):
                self.logger.info('running %s', modelName)
                if featuresDiffer:
                    features, ids = self.selectFeaturesAndIds(cur, model.getFeatureNames(), condition=condition)
                    if not weightValues:
                        weightValues.append(numpy.array(ids))
                weights = model.predict(features)
                weightFields.append(modelName)
                weightValues.append(weights)
            self.saveMultipleWeights(
                cur, weightTable, weightFields, weightValues, overwrite=overwrite
            )

    def loadAndNameModels(self, path, modelNames=None, useFileNames=False):
        models = []
        names = []
        for name, mod in self.loadModels(path):
            models.append(mod)
            if useFileNames:
                names.append(name)
            elif modelNames is None:
                names.append(mod.typename)
            else:
                names.append(modelNames[len(names)])
        if len(frozenset(names)) < len(models):
            raise ValueError('could not identify model names from types, need explicit names')
        return models, names

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
        self.createPrimaryKey(cur, table)

    def loadModels(self, path):
        self.logger.info('loading models from %s', path)
        for fileName in sorted(os.listdir(path)):
            try:
                yield (
                    os.path.splitext(fileName)[0],
                    Model.load(os.path.join(path, fileName))
                )
            except pickle.UnpicklingError:
                pass


class Calibrator(field.Handler):
    MODEL_TYPES = {
        'lad' : LADRegression,
        'ols' : sklearn.linear_model.LinearRegression,
        'sum' : SumMatchingMultiplication,
    }

    def main(self, table, idField, rawField, calibField, outputField, type='sum', overwrite=False, fit_intercept=True):
        with self._connect() as cur:
            self.createField(cur, table=table, name=outputField, overwrite=overwrite)
            data = self.selectValues(cur, table, [idField, rawField, calibField])
            self.logger.debug('fitting calibrator')
            calibrator = self.MODEL_TYPES[type](fit_intercept=fit_intercept)
            feats = data[rawField].values.reshape(-1, 1)
            # print(feats)
            tgts = data[calibField].values
            # print(tgts)
            calibrator.fit(feats, tgts)
            self.logger.info(
                'calibrator fitted: linear coefficient %g, intercept %g',
                calibrator.coef_[0], calibrator.intercept_
            )
            fitted = calibrator.predict(feats)
            fitted = numpy.where(fitted < 0, 0, fitted)
            qry = sql.SQL('''UPDATE {schema}.{table} SET {outputField} = fitted
                FROM (SELECT
                    unnest(%s) as id,
                    unnest(%s) as fitted
                ) newvals
                WHERE {schema}.{table}.{idField} = newvals.id
            ''').format(
                schema=self.schemaSQL,
                table=sql.Identifier(table),
                outputField=sql.Identifier(outputField),
                idField=sql.Identifier(idField),
            ).as_string(cur)
            self.logger.debug('inserting fitted values: %s', qry)
            cur.execute(qry, [data[idField].values.tolist(), fitted.tolist()])


class ModelIntrospector(core.Task):
    ITEMS = [
        ('feature_importances_', ('feature importances', None, '{:.3%}')),
        ('coef_', ('coefficients', abs, '{:.4g}')),
    ]

    GROUPERS = [
        ('source', lambda name: name.split('_')[0]),
        ('range', lambda name: (
            name[name.rfind('neigh')+6:] if 'neigh' in name else 'none'
        )),
        ('core feature', lambda name: (
            name[:name.rfind('neigh')-1] if 'neigh' in name else name
        )),
    ]

    def main(self, modelfile, add_grouped=False):
        model = Model.load(modelfile)
        coreModel = model.getRegressor()
        featnames = model.getFeatureNames(intercept=True)
        reported = False
        for attr, settings in self.ITEMS:
            if hasattr(coreModel, attr):
                self.report(
                    featnames,
                    getattr(coreModel, attr),
                    *settings,
                    add_grouped=add_grouped
                )
                reported = True
        if not reported:
            print('*** No introspectable attribute found')

    def report(self, names, values, label, transformer=None, formatter='{}', add_grouped=False):
        df = pd.DataFrame({'feature' : names, 'value' : values})
        df['transformed'] = df.value.map(transformer) if transformer else df.value
        df.sort_values('transformed', ascending=False, inplace=True)
        self.display(df, 'feature', ['value'],
                     'Model {} ({} features)'.format(label, len(df.index)),
                     formatter
        )
        if add_grouped:
            for grpname, grouper in self.GROUPERS:
                df[grpname] = df['feature'].map(grouper)
                grouped_df = df.groupby(grpname).agg({
                    'value' : sum,
                    'transformed' : sum
                }).reset_index().sort_values('transformed', ascending=False)
                self.display(grouped_df, grpname, ['transformed', 'value'],
                             'Grouped {} by {}'.format(label, grpname),
                             formatter
                )

    def display(self, df, keyname, valnames, title, formatter):
        print()
        print(title)
        print()
        leftcolwidth = min(df[keyname].map(len).max(), 60)
        for index, row in df.iterrows():
            if row[valnames[0]]:
                print(row[keyname].ljust(leftcolwidth), end=' ')
                for valname in valnames:
                    print(formatter.format(row[valname]), end=' ')
                print()
