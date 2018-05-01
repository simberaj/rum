import fiona
import psycopg2
from psycopg2 import sql

from .core import Task


class Importer(Task):
    pass
    
class LayerImporter(Task):
    def main(self, filepath, table, srid=None, clipExtent=False):
        with self._connect() as cur:
            extentSRID = self.getExtentSRID(cur)
            if extentSRID:
                if srid and extentSRID != srid:
                    self.logger.warning('forced import SRID %d but extent already present with SRID %d, inconsistency will arise', srid, extentSRID)
                else:
                    srid = extentSRID
            print(srid)
            raise NotImplementedError
            with fiona.drivers():
                with fiona.open(filepath) as source:
                    self.createTable(cur, source.meta, srid)
                    if clipExtent:
                        featureIterator = source.filter(bbox=self.getExtentBBox(cur, srid))
                    else:
                        featureIterator = source
                    for feature in featureIterator:
                        geometry = self.transform(feature['geometry'], srid)
                        properties = feature['properties']
                        self.insert(cur, geometry, properties)
    
    def getExtentSRID(self, cur):
        try:
            qry = sql.SQL(
                "SELECT Find_SRID({schema}, 'extent', 'geometry');"
            ).format(schema=sql.Literal(self.schema)).as_string(cur)
            self.logger.debug('determining extent SRID by %s', qry)
            cur.execute(qry)
            # there is an extent defined, make the imported data conform to its CRS
            result = cur.fetchone()
            srid = result[0]
            self.logger.debug('extent SRID determined as %s', srid)
            return srid
        except psycopg2.InternalError:
            # extent does not exist, keep current CRS
            self.logger.debug('extent SRID not found')
            return None        
                