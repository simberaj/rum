import rum
import rum.input

DESCRIPTION = '''Imports a CSV file into PostGIS/RUM.

Autodetects delimiters, field names and types (either from an associated CSVT
file or the first rows of the file).
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing table'
)
argparser.add_argument('-C', '--encoding', metavar='encoding', default='utf8',
    help='input layer attribute character encoding'
)
argparser.add_argument('table', help='target table name in the given schema')
argparser.add_argument('file', help='CSV file')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.input.TableImporter.fromArgs(args).run(
        table=args.table,
        file=args.file,
        encoding=args.encoding,
        overwrite=args.overwrite
    )