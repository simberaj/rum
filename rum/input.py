import xml.sax
import functools
import collections
import sqlite3
import contextlib
import tempfile
import os
import json
import bz2

import fiona
import json

import fiona.crs
import psycopg2
from psycopg2 import sql
import shapely.geometry
import shapely.speedups
if shapely.speedups.available:
    shapely.speedups.enable()

from . import core, attribute

WGS84_SRID = 4326


class DataImportError(core.Error):
    pass

        
class Importer(core.DatabaseTask):
    def getExtentBBox(self, cur, srid):
        self.logger.debug('retrieving extent bounding box')
        qry = sql.SQL(
            '''WITH bbox AS (
                SELECT Box2D(ST_Extent(ST_Transform({geomField},{srid}))) AS b
                FROM {schema}.extent
            )
            SELECT ST_XMin(b), ST_YMin(b), ST_XMax(b), ST_YMax(b) FROM bbox;'''
        ).format(
            schema=self.schemaSQL,
            geomField=sql.Identifier(core.GEOMETRY_FIELD),
            srid=sql.Literal(srid)
        ).as_string(cur)
        try:
            cur.execute(qry)
        except psycopg2.ProgrammingError as err:
            raise DataImportError('could not select extent bounding box') from err
        box = cur.fetchone()
        if not box:
            raise DataImportError('could not select extent bounding box')
        return box
        
        
    
# single importer: imports from a geojson feature iterator
# layer importer: uses fiona as feature iterator
# extractor importer: uses extractor as feature iterator

# osm importer: handles parser, extractors and extractor importers
    
class OSMImporter(Importer):
    DEFAULT_CONF_PATH = core.configPath('osmextract.json')

    def __init__(self, *args, transforms={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractors = []
        for name, conf in transforms.get('layers', {}).items():
            e = OSMExtractor.fromConfig(name, conf)
            e.logTo(self.logger)
            self.extractors.append(e)
        if not self.extractors:
            self.logger.warning('no extractors found, no output data will be produced (dry run)')
        self.buildListeners()
        
    @classmethod
    def fromConfig(cls, connConfig, transformConfig, schema):
        return cls(
            core.Connector.fromConfig(connConfig),
            transforms=core.loadConfig(transformConfig if transformConfig else cls.DEFAULT_CONF_PATH),
            schema=schema
        )
    
    def buildListeners(self):
        self.listeners = collections.defaultdict(list)
        for e in self.extractors:
            for geomtype in e.listenGeomTypes:
                self.listeners[geomtype].append(e)
    
    def main(self, path, targetSRID=None, clipExtent=False, overwrite=False, isTmpDB=False, isBZ2=True):
        with self._connect() as cur:
            if clipExtent:
                acceptor = self.createExtentAcceptor(cur)
            else:
                acceptor = None
            self.logger.debug('opening database output channels')
            for e in self.extractors:
                e.buildWriter(self.schema, targetSRID=targetSRID, overwrite=overwrite)
            with contextlib.ExitStack() as extractContext:
                for e in self.extractors:
                    extractContext.enter_context(e.open(cur))
                self.logger.info('parsing %s', path)
                for feature in OSMParser().parse(path, isTmpDB=isTmpDB, isBZ2=isBZ2):
                    try:
                        geom = feature['geometry']
                        geomtype = geom['type']
                        geom = shapely.geometry.shape(geom)
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                        if not acceptor or acceptor(geom):
                            feature['geometry'] = geom
                            for e in self.listeners[geomtype]:
                                e.process(feature)
                    except:
                        self.logger.debug('failed at %s', str(feature))
                        raise
                            
    def createExtentAcceptor(self, cur):
        self.logger.info('clipping input by analysis extent')
        bbox = shapely.geometry.geo.box(*self.getExtentBBox(cur, WGS84_SRID))
        return lambda geom: bbox.intersects(geom.envelope)
        
        
            
            
class OSMExtractor:
    inserter = None

    def __init__(self, name, geomtype, listenGeomTypes, selector=None, attributes=[]):
        self.name = name
        self.geomtype = geomtype
        self.listenGeomTypes = listenGeomTypes
        self.selector = selector
        self.attributes = attributes
        if self.geomtype == 'Point' and set(self.listenGeomTypes) != {'Point'}:
            self.transformGeometry = self.geometryToPoint
        else:
            self.transformGeometry = None
        self.logger = core.EmptyLogger()
    
    def logTo(self, logger):
        self.logger = logger
    
    def buildWriter(self, schema, targetSRID, overwrite):
        self.writer = Writer(
            schema,
            self.name,
            fields=self.getFieldDefs(),
            geomtype=self.geomtype,
            sourceSRID=WGS84_SRID, 
            targetSRID=targetSRID,
            overwrite=overwrite
        )
        if self.logger:
            self.writer.logTo(self.logger)
    
    def getFieldDefs(self):
        fields = collections.OrderedDict()
        for attr in self.attributes:
            fields[attr.name] = (core.TYPES_TO_POSTGRE[attr.type], None)
        return fields

    @classmethod
    def fromConfig(cls, name, conf):
        return cls(
            name=name,
            geomtype=conf['geometry'],
            listenGeomTypes=conf['input-geometries'],
            selector=attribute.selector(conf.get('selector')),
            attributes=[
                attribute.attribute(attrconf)
                for attrconf in conf.get('attributes', [])
            ]
        )
    
    @contextlib.contextmanager
    def open(self, cursor):
        with self.writer.open(cursor):
            yield self

    def process(self, feature):
        properties = feature['properties']
        if not self.selector or self.selector(properties):
            properties = self.transformProperties(properties)
            if properties:
                geometry = feature['geometry']
                if self.transformGeometry:
                    geometry = self.transformGeometry(geometry)
                for props in properties:
                    self.writer.write({
                        'geometry' : geometry,
                        'properties' : props,
                    })
    
    def geometryToPoint(self, geom):
        return geom.representative_point() # guaranteed to be within
        
    
    def transformProperties(self, props):
        transformed = {}
        maxlen = None
        for attr in self.attributes:
            value = attr.get(props)
            if isinstance(value, list):
                maxlen = len(value)
            elif value is None and attr.restrict:
                return None
            transformed[attr.name] = value
        if maxlen is None:
            return [transformed]
        else:
            return [{
                    key : (value[i] if isinstance(value, list) else value)
                    for key, value in transformed.items()
                } for i in range(maxlen)
            ]
      
class TempFile:
    """Temporary file context manager, wraps tempfile.mkstemp."""

    def __init__(self, dir=None, suffix=''):
        fd, name = tempfile.mkstemp(dir=dir, suffix=suffix)
        os.close(fd)
        self.name = os.path.normpath(name)
  
    def path(self):
        return self.name
  
    def __enter__(self):
        return self.name
  
    def __exit__(self, *args):
        if not args[0] and os.path.exists(self.name):
            os.unlink(self.name)
    
class MalformedRelationError(Exception):
    pass
      
class OSMParser(xml.sax.handler.ContentHandler):
    '''A OSM to GeoJSON converter. Uses XML SAX to read the OSM XML.
    
    Creates an auxiliary SQLite database to resolve geometries as in-memory
    node/way storage would crash on memory lack on country-level data.'''
    
    POLYGONAL_RELATIONS = set(('boundary', 'multipolygon')) # omit other relations
    SYSTEM_PROPS = set(('source', 'fixme', 'created_by'))
    REMOVE_PROPS = ['type']
    # do not use None as value, will break the checks
    FORBIDDEN_TAGS = [('disused', 'yes')]
    WINDINGS = {'outer' : False, 'inner' : True}
    CREATE_SCHEMA = [
        ('nodegeoms', 'id integer primary key, lat real, lon real', ['id']),
        ('nodes', 'id integer primary key, tags text', ['id']),
        ('wayrefs', 'id integer, noderef integer, ord integer', ['id']),
        ('ways', 'id integer primary key, tags text', ['id']),
        ('relrefs', 'id integer, wayref integer, outer integer', ['id']),
        ('rels', 'id integer primary key, tags text', ['id']),
        ('used_tags', 'id integer, tag text', ['id'])
    ]
    TABLE_NAMES = {'node' : 'nodes', 'way' : 'ways', 'relation' : 'rels'}
    TYPE_TRANSLATOR = {
        'point' : ('Point', ),
        'line' : ('LineString', 'MultiLineString'),
        'polygon' : ('Polygon', 'MultiPolygon')
    }

    def startDocument(self):
        self.mode = None
        self.wayOrder = None
        self.currentTags = {}
        self.nodeCache = []
        self.wayrefCache = []
    
    def checkForbidden(self, properties):
        for banKey, banVal in self.FORBIDDEN_TAGS:
            if properties.get(banKey) == banVal:
                return False
        return True

    def startElement(self, name, attrs):
        try:
            if name == 'node':
                self.mode = name
                self.curID = int(attrs['id'])
                self.nodeCache.append((
                    self.curID,
                    float(attrs['lat']),
                    float(attrs['lon'])
                ))
                if len(self.nodeCache) >= 1000:
                    self.cursor.executemany(
                        'INSERT INTO nodegeoms VALUES (?,?,?)',
                        self.nodeCache
                    )
                    self.nodeCache = []
            elif name == 'nd':
                if self.wayOrder:
                    self.wayrefCache.append((
                        self.curID,
                        int(attrs['ref']),
                        self.wayOrder
                    ))
                    if len(self.wayrefCache) >= 1000:
                        self.cursor.executemany(
                            'INSERT INTO wayrefs VALUES (?,?,?)',
                            self.wayrefCache
                        )
                        self.wayrefCache = []
                    self.wayOrder += 1
            elif name == 'tag':
                key = attrs['k']
                val = attrs['v']
                if key not in self.SYSTEM_PROPS:
                    self.currentTags[key] = val
            elif name == 'member': # TODO
                if self.mode == 'relation' and attrs['type'] == 'way':
                    self.cursor.execute(
                        'INSERT INTO relrefs VALUES (?,?,?)',
                        (self.curID, int(attrs['ref']), attrs['role'] != 'inner')
                    )
            elif name in ('way', 'relation'):
                self.wayOrder = 1
                self.mode = name
                self.curID = int(attrs['id'])
        except self.duplicateError:
            # duplicate node/way/relation, pass it, no interest
            # (happens when assembling more files with one feature in more of them)
            pass 
  
    def endElement(self, name):
        if name in ('node', 'way', 'relation'):
            if self.currentTags and self.condition(name, self.currentTags):
                try:
                    self.cursor.execute(
                        'INSERT INTO {} VALUES (?,?)'.format(self.TABLE_NAMES[name]),
                        (self.curID, json.dumps(self.currentTags))
                    )
                except self.duplicateError:
                    pass
            self.currentTags.clear()
            self.mode = None
            self.wayOrder = None
  
    def condition(self, name, tags):
        return name != 'relation' or (
            'type' in tags and tags['type'] in self.POLYGONAL_RELATIONS
        )
    
    def endDocument(self):
        self.cursor.executemany(
            'INSERT INTO nodegeoms VALUES (?,?,?)',
            self.nodeCache
        )
        self.cursor.executemany(
            'INSERT INTO wayrefs VALUES (?,?,?)',
            self.wayrefCache
        )
        self.connection.commit()
        self.createIndices()
        self.connection.commit()

    def toGeoJSON(self, geometry, properties, id):
        if (geometry and properties and self.checkForbidden(properties)):
            if id is not None:
                properties['osm_id'] = id
            return {
                'geometry' : geometry,
                'properties' : properties,
                'id' : id
            }
        
    def outputNodes(self):
        cur = self.cursor
        cur.execute('''SELECT
            nodes.id AS id,
            nodes.tags AS tags,
            nodegeoms.lat AS lat,
            nodegeoms.lon AS lon
        FROM nodes JOIN nodegeoms ON nodes.id=nodegeoms.id''')
        node = cur.fetchone()
        while node:
            feature = self.toGeoJSON({
                    'type' : 'Point',
                    'coordinates' : (node['lon'], node['lat'])
                },
                json.loads(node['tags']),
                'node/' + str(node['id'])
            )
            if feature:
                yield feature
            node = cur.fetchone()
  
    def generate(self, name):
        cur = self.connection.cursor()
        cur.execute('SELECT * FROM {}'.format(self.TABLE_NAMES[name]))
        item = cur.fetchone()
        while item:
            yield item
            item = cur.fetchone()
  
    def outputWays(self):
        cur = self.cursor
        for way in self.generate('way'):
            wayID = way['id']
            tags = json.loads(way['tags'])
            if tags:
                for tag in self.getDeletedTags(wayID):
                    if tag in tags:
                        del tags[tag]
                if tags:
                    cur.execute('''SELECT
                        nodegeoms.lat AS lat, nodegeoms.lon AS lon
                        FROM nodegeoms JOIN wayrefs ON nodegeoms.id=wayrefs.noderef
                        WHERE wayrefs.id=? ORDER BY wayrefs.ord''',
                        (str(wayID), )
                    )
                    coors = [(pt['lon'], pt['lat']) for pt in cur.fetchall()]
                    feature = self.toGeoJSON(
                        self.wayGeometry(coors, tags),
                        tags,
                        'way/' + str(wayID)
                    )
                    if feature:
                        yield feature
    
    def getDeletedTags(self, wayID):
        self.cursor.execute('SELECT tag FROM used_tags WHERE id=?', (wayID, ))
        res = self.cursor.fetchall()
        return [rec['tag'] for rec in res]
    
    
    def outputRelations(self):
        cur = self.cursor
        nodes = {}
        delWayTags = collections.defaultdict(set)
        for rel in self.generate('relation'):
            tags = json.loads(rel['tags'])
            relID = rel['id']
            cur.execute('''SELECT
                    nodegeoms.id as nodeid,
                    nodegeoms.lat as lat, nodegeoms.lon as lon,
                    wayrefs.id as wayid,
                    ways.tags as tags,
                    relrefs.outer as outer
                FROM ((relrefs JOIN wayrefs ON wayrefs.id=relrefs.wayref)
                        JOIN nodegeoms ON nodegeoms.id=wayrefs.noderef)
                          LEFT JOIN ways ON ways.id=relrefs.wayref
                WHERE relrefs.id=?
                ORDER BY wayrefs.id, wayrefs.ord
                ''',
                (str(relID), )
            )
            nodes.clear()
            waytags = set()
            wayrefs = {
                True : collections.defaultdict(list),
                False : collections.defaultdict(list)
            }
            for pt in cur.fetchall():
                if pt['outer']:
                    waytags.add(pt['tags'])
                wayrefs[pt['outer']][pt['wayid']].append(pt['nodeid'])
                nodes[pt['nodeid']] = (pt['lon'], pt['lat'])
            try:
                relProps, fromWays = self.relationProperties(
                    tags, [json.loads(ts) for ts in waytags if ts]
                )
                if relProps:
                    relGeom = self.relationGeometry(
                        list(wayrefs[True].values()),
                        list(wayrefs[False].values()),
                        nodes
                    )
                    feature = self.toGeoJSON(relGeom, relProps, 'rel/' + str(relID))
                    if feature:
                        yield feature
                if fromWays:
                    for wayid in wayrefs[True].keys():
                        delWayTags[wayid].update(fromWays)
                    if len(delWayTags) > 1000:
                        self.serializeUsedWayTags(delWayTags)
                        delWayTags = collections.defaultdict(set)
            except MalformedRelationError:
                pass
            if delWayTags:
                self.serializeUsedWayTags(delWayTags)

    def serializeUsedWayTags(self, usedTags):
        c = self.connection.cursor()
        c.executemany('INSERT INTO used_tags VALUES (?,?)',
            ((id, tag) for id, tags in usedTags.items() for tag in tags)
        )
    
    def parse(self, locator, isTmpDB=False, isBZ2=True):
        dbContext = self.openTempDB if isTmpDB else self.tempDB
        with dbContext(locator):
            if not isTmpDB:
                if isinstance(locator, str):
                    if isBZ2:
                        locator = bz2.open(locator, 'rt', encoding='utf8')
                    else:
                        locator = open(locator, encoding='utf8')
                firstline = locator.readline() # skip xml declaration
                # accepts a filelike object or a local file path
                xml.sax.parse(locator, self)
            yield from self.output()
    
    def output(self):
        yield from self.outputNodes()
        yield from self.outputRelations()
        yield from self.outputWays()

    @contextlib.contextmanager
    def tempDB(self, locator):
         # autodeletes the database file on end of file processing
        with TempFile() as fname:
            self.connect(fname)
            self.createTables()
            yield
            self.connection.close()
      
    @contextlib.contextmanager
    def openTempDB(self, fname):
        self.connect(fname)
        yield
        self.connection.close()
    
    def connect(self, fname):
        self.connection = sqlite3.connect(fname)
        self.connection.row_factory = sqlite3.Row
        self.duplicateError = sqlite3.IntegrityError
        self.cursor = self.connection.cursor()
        
    def createTables(self):
        for name, cols, indexCols in self.CREATE_SCHEMA:
            self.cursor.execute('CREATE TABLE {}({})'.format(name, cols))
  
    def createIndices(self):
        for name, cols, indexCols in self.CREATE_SCHEMA:
            for col in indexCols:
                self.cursor.execute(
                    'CREATE INDEX {0}_{1} on {0}({1})'.format(name, col)
                )     

    @classmethod
    def wayGeometry(cls, coors, tags=None):
        if coors[0] == coors[-1] and not cls.hasPolylineTags(tags):
            if len(coors) > 3:
                cls.setWinding(coors, False)
                return {'type' : 'Polygon', 'coordinates' : [coors]}
        elif len(coors) > 1:
            return {'type' : 'LineString', 'coordinates' : coors}
    
    @staticmethod
    def hasPolylineTags(tags):
        if 'area' in tags:
            return tags['area'] == 'no' or (
                'highway' in tags and tags['highway'] == 'pedestrian'
            )
        else:
            return ('highway' in tags or 'barrier' in tags or 'junction' in tags)
            
    @classmethod
    def relationGeometry(cls, outer, inner, nodes):
        # role is True (outer rings) or False (inner rings)
        outerRings = cls.noderefsToRings(outer, nodes, False)
        if not outerRings: # degenerate outer rings only
            return None
        holeLists = cls.matchHoles(
            outerRings,
            cls.noderefsToRings(inner, nodes, True)
        )
        coors = [[outerRings[i]] + holeLists[i] for i in range(len(outerRings))]
        if len(coors) == 1:
            return {'type' : 'Polygon', 'coordinates' : coors[0]}
        else:
            return {'type' : 'MultiPolygon', 'coordinates' : coors}

    @classmethod
    def relationProperties(cls, props, wayprops):
        allProps = props.copy()
        propsFromWays = []
        if wayprops:
            # merge properties from ways if they are the same and they are not
            # yet present
            allWayProps = set(wayprops[0].items())
            for wayPropDict in wayprops[1:]:
                allWayProps.intersection_update(wayPropDict.items())
            for prop, value in allWayProps: # update the main properties
                if prop not in allProps:
                    allProps[prop] = value
                    propsFromWays.append(prop)
        for key in cls.REMOVE_PROPS:
            if key in allProps:
                del allProps[key]
        return allProps, propsFromWays
        
    @classmethod
    def noderefsToRings(cls, noderefs, nodes, winding=False):
        # winding: False is counterclockwise
        # noderefs: list of sequences of node IDs
        # nodes: node ID: (lon, lat)
        rings = [
            cls.noderefsToLine(ring, nodes)
            for ring in cls.dropDegens(cls.ringify(noderefs))
        ]
        while [] in rings:
            rings.remove([])
        for ring in rings:
            cls.setWinding(ring, winding)
        return rings
        
    @staticmethod
    def dropDegens(ringlist):
        return [ring for ring in ringlist if len(ring) > 3]
        
    @staticmethod
    def noderefsToLine(wayrefs, nodes):
        return [nodes[ref] for ref in wayrefs]
    
    @classmethod
    def matchHoles(cls, outer, inner): # match inner to outer rings
        if len(outer) == 1:
            return [inner]
        else:
            holes = [[] for i in range(len(outer))]
            for hole in inner:
                onept = hole[0]
                for i in range(len(outer)):
                    # we leave out holes whose first point is not inside
                    if cls.pointInPolygon(onept, outer[i]):
                        holes[i].append(hole)
                        break
            return holes
    
    @staticmethod
    def isClockwise(ring):
        """Returns True if the ring's vertices are in clockwise order.
      
        Expects a list of 2-tuples or 2-lists on input."""
        return sum(
            (ring[i+1][0] - ring[i][0]) * (ring[i+1][1] + ring[i][1])
            for i in range(len(ring)-1)
        ) > 0

    @classmethod
    def setWinding(cls, ring, clockwise=True):
        '''Sets the direction of the ring to the given direction (True - clockwise).'''
        if cls.isClockwise(ring) != clockwise:
            ring.reverse()
        
    @staticmethod
    def pointInPolygon(pt, poly):
        """Returns True if the given point lies within the line segment ring.
      
        Expects a 2-tuple/2-list and a list of 2-tuples/2-lists on input."""
        inside = False
        x, y = pt
        p1x, p1y = poly[0]
        for i in range(len(poly)):
            p2x, p2y = poly[i % (len(poly)-1)]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y
        return inside

    
    @staticmethod
    def ringify(ways, id=None): # merges ways to rings
        rings = []
        i = 0
        while i < len(ways):
            if not ways[i]:
                ways.pop(i)
            elif ways[i][0] == ways[i][-1]: # closed ways are parts on their own
                rings.append(ways.pop(i))
            else:
                j = i + 1
                while j < len(ways):
                    if ways[i][-1] == ways[j][0]: # succesor found
                        ways[i] = ways[i][:-1] + ways.pop(j)
                        break
                    elif ways[i][0] == ways[j][-1]: # predecessor
                        ways[i] = ways.pop(j) + ways[i][1:]
                        break
                    elif ways[i][0] == ways[j][0]: # reverse predecessor
                        ways[i] = list(reversed(ways.pop(j))) + ways[i][1:]
                        break
                    elif ways[i][-1] == ways[j][-1]: # reverse successor
                        ways[i] = ways[i][:-1] + list(reversed(ways.pop(j)))
                        break
                    else:
                        j += 1
                else:
                    raise MalformedRelationError(
                        'open multipolygon' + 
                        ('' if id is None else (' for relation ' + str(id)))
                    )
        return rings


  
class LayerImporter(Importer):
    SIZELESS_TYPES = {
        'int' : 'int',
        'float' : 'double precision'
    }

    def main(self, path, table, encoding=None, sourceSRID=None, targetSRID=None, forcedGeometryType=None, clipExtent=False, overwrite=False):
        with fiona.drivers():
            with fiona.open(path, encoding=encoding) as source:
                fields = self.getFieldDefs(source.meta['schema']['properties'])
                geomtype = source.meta['schema']['geometry']
                if forcedGeometryType and forcedGeometryType != geomtype:
                    self.logger.info('forcing geometry to %s instead of %s', forcedGeometryType, geomtype)
                    geomtype = forcedGeometryType
                with self._connect() as cursor:
                    if not sourceSRID:
                        sourceSRID = self.getSourceSRID(cursor, source.meta['crs'])
                    writer = Writer(
                        self.schema, table,
                        fields=fields,
                        geomtype=geomtype,
                        sourceSRID=sourceSRID, 
                        targetSRID=targetSRID,
                        overwrite=overwrite
                    )
                    writer.logTo(self.logger)
                    if clipExtent:
                        featureIterator = source.filter(
                            bbox=self.getExtentBBox(cursor, sourceSRID)
                        )
                    else:
                        featureIterator = source
                    with writer.open(cursor):
                        self.logger.debug('starting feature import')
                        for feature in featureIterator:
                            writer.write(feature)
                        
    def getFieldDefs(self, fieldDict):
        fdict = collections.OrderedDict([])
        for name, typedef in fieldDict.items():
            typename, typesize = typedef.split(':')
            defsize = None
            if typename in self.SIZELESS_TYPES:
                posttype = self.SIZELESS_TYPES[typename]
            elif typename == 'str':
                posttype = 'varchar'
                defsize = int(typesize)
            else:
                raise DataImportError('unknown field data type: ' + typename)
            fdict[name] = (posttype, defsize)
        return fdict
    
    def getSourceSRID(self, cur, proj4dict):
        if not proj4dict:
            raise DataImportError('source SRID not specified and not found in layer')
        proj4string = fiona.crs.to_string(proj4dict)
        self.logger.debug('estimating source SRID from %s', ' '.join(proj4string))
        qry = sql.SQL('SELECT srid FROM spatial_ref_sys WHERE ')
        qry += sql.SQL(' AND ').join([
            sql.SQL('proj4text LIKE ') + sql.Literal('%' + param + '%')
            for param in proj4string.split()
        ])
        qrystr = qry.as_string(cur)
        try:
            cur.execute(qrystr)
            sridrow = cur.fetchone()
        except psycopg2.ProgrammingError as err:
            raise DataImportError('input layer SRID not found in PostGIS database') from err
        if not sridrow:
            raise DataImportError('input layer SRID not found in PostGIS database')
        srid = sridrow[0]
        self.logger.info('SRID %d detected on input', srid)
        return srid
    

class Writer:
    ALLOWED_GEOMETRY_TYPES = [
        'point', 'linestring', 'multilinestring', 'polygon', 'multipolygon'
    ]
    
    def __init__(self, schema, table, fields, geomtype, sourceSRID=None, targetSRID=None, overwrite=False):
        self.schema = schema
        self.schemaSQL = sql.Identifier(self.schema)
        self.table = table
        self.tablepath = '{}.{}'.format(self.schema, self.table)
        self.fields = fields
        self.geomtype = geomtype
        if self.geomtype.lower() not in self.ALLOWED_GEOMETRY_TYPES:
            raise DataImportError('invalid geometry type: ' + str(self.geomtype))
        self.sourceSRID = sourceSRID
        self.userTargetSRID = targetSRID
        self.targetSRID = None
        self.overwrite = overwrite
        self.cursor = None
        self.logger = core.EmptyLogger()
    
    def logTo(self, logger):
        self.logger = logger
    
    @contextlib.contextmanager
    def open(self, cursor):
        self.cursor = cursor
        self.targetSRID = self.correctTargetSRID()
        self.createTable()
        self.insertQuery = self.createInsertQuery()
        self.count = 0
        yield self
        self.logger.info('%d features written to %s', self.count, self.tablepath)
        self.count = 0
        self.createSpatialIndex()
        self.cursor = None
        
    def write(self, feature):
        properties = feature['properties']
        geometry = feature['geometry']
        if hasattr(geometry, 'keys'):
            geometry = shapely.geometry.shape(geometry)
        if geometry.has_z:
            raise NotImplementedError('geometry contains z-coordinates')
        properties['geometry'] = geometry.wkb
        self.cursor.execute(self.insertQuery, properties)
        self.count += 1
    
    def createTable(self):
        tabledef = sql.SQL('{schema}.{table}').format(
            schema=self.schemaSQL, table=sql.Identifier(self.table)
        )
        if self.overwrite:
            delqry = sql.SQL('''DROP TABLE IF EXISTS {tabledef}''').format(tabledef=tabledef)
            self.cursor.execute(delqry)
        start = sql.SQL('''CREATE TABLE {tabledef} (''').format(tabledef=tabledef)
        fieldDefs = sql.SQL(', ').join(
            list(self.createFieldPart(self.fields)) +
            [sql.SQL('{geomField} geometry({geomtype},{srid})').format(
                geomField=sql.Identifier(core.GEOMETRY_FIELD),
                geomtype=sql.Literal(self.geomtype),
                srid=sql.Literal(self.targetSRID),
            )]
        )
        qry = (start + fieldDefs + sql.SQL(');')).as_string(self.cursor)
        self.logger.debug('creating table %s with %s', self.tablepath, qry)
        try:
            self.cursor.execute(qry)
        except psycopg2.Error as err:
            raise DataImportError('cannot create table: ' + err.pgerror) from err
        self.logger.info('table %s created with SRID %s', self.tablepath, self.targetSRID)

    @staticmethod
    def createFieldPart(fields):
        for name, fdef in fields.items():
            ftype, size = fdef
            fieldDef = sql.Identifier(name) + sql.SQL(' ') + sql.SQL(ftype)
            if size:
                fieldDef += sql.SQL('({})').format(sql.Literal(size))
            yield fieldDef
            
    def createSpatialIndex(self):
        self.logger.debug('creating spatial index for %s', self.tablepath)
        indexName = '{}_{}_gix'.format(self.schema, self.table)
        qry = sql.SQL('''
            CREATE INDEX {indexname}
            ON {schema}.{table} USING GIST ({geomcol});
        ''').format(
            indexname=sql.Identifier(indexName),
            schema=self.schemaSQL,
            table=sql.Identifier(self.table),
            geomcol=sql.Identifier(core.GEOMETRY_FIELD),
        ).as_string(self.cursor)
        self.cursor.execute(qry)

    def correctTargetSRID(self):
        targetSRID = self.userTargetSRID
        extentSRID = self.getExtentSRID()
        if extentSRID:
            if targetSRID and extentSRID != targetSRID:
                self.logger.warning('forced import SRID %d but extent already present with SRID %d', targetSRID, extentSRID)
            else:
                targetSRID = extentSRID
        if not targetSRID:
            if not self.sourceSRID:
                raise DataImportError('SRID of input and output undefined')
            else:
                self.logger.info('import taking over source SRID %d', self.sourceSRID)
                return self.sourceSRID
        else:
            return targetSRID
        
    def getExtentSRID(self):
        if self.extentExists():
            qry = sql.SQL(
                "SELECT Find_SRID({schema}, 'extent', {geomField});"
            ).format(
                schema=sql.Literal(self.schema),
                geomField=sql.Literal(core.GEOMETRY_FIELD),
            ).as_string(self.cursor)
            self.logger.debug('determining extent SRID by %s', qry)
            self.cursor.execute(qry)
            # there is an extent defined, make the imported data conform to its CRS
            result = self.cursor.fetchone()
            if result is not None:
                srid = result[0]
                self.logger.debug('extent SRID determined as %s', srid)
                return srid
        # extent does not exist, keep current CRS
        self.logger.debug('extent SRID not found')
        return None
    
    def extentExists(self):
        qry = sql.SQL('''SELECT EXISTS (
           SELECT 1 FROM information_schema.tables 
           WHERE table_schema = {schema} AND table_name = 'extent'
        );''').format(schema=sql.Literal(self.schema)).as_string(self.cursor)
        self.cursor.execute(qry)
        result = self.cursor.fetchone()
        return bool(result and result[0])
        
    def createInsertQuery(self):
        fieldnames = self.fields.keys()
        return sql.SQL('''
            INSERT INTO {schema}.{table}({fields})
            VALUES ({values})
        ''').format(
            schema=self.schemaSQL,
            table=sql.Identifier(self.table),
            fields=sql.SQL(', ').join(
                [sql.Identifier(fld) for fld in fieldnames] + 
                [sql.Identifier(core.GEOMETRY_FIELD)]
            ),
            values=sql.SQL(', ').join(
                [sql.Placeholder(fld) for fld in fieldnames] +
                [self.geometryPlaceholder()]
            ),
        ).as_string(self.cursor)
    
    def geometryPlaceholder(self):
        geometryPlaceholder = sql.SQL(
            'ST_SetSRID(ST_GeomFromWKB({geom}),{sourceSRID})'
        ).format(
            geom=sql.Placeholder('geometry'),
            sourceSRID=sql.Literal(self.sourceSRID),
        )
        if self.geomtype.lower().startswith('multi'):
            geometryPlaceholder = sql.SQL('ST_Multi({})').format(geometryPlaceholder)
        if self.sourceSRID != self.targetSRID:
            return sql.SQL(
                'ST_Transform({geom},{targetSRID})'
            ).format(
                geom=geometryPlaceholder,
                targetSRID=sql.Literal(self.targetSRID),
            )
        else:
            return geometryPlaceholder