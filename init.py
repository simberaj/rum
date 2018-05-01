import rum

DESCRIPTION = '''Initializes the analysis database.

Creates the required functions for use by other tools of RUM.
Can be safely run multiple times on the same database without loosing or
altering existing data.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION, schema=False)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.Initializer.fromArgs(args).run()