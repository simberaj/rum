'''Initialize the analysis schema.

Creates the required functions for use by other tools of RUM and the
analysis schema.

Can be safely run multiple times on the same database without loosing or
altering existing data.
'''

import rum

argparser = rum.defaultArgumentParser(__doc__)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.Initializer.fromArgs(args).run()