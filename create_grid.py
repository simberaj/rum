'''Creates a modeling grid.

Creates a square grid of polygons with the defined cell size that cover the
analysis extent as defined in the ``extent`` table of the analysis schema.
'''

import rum

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('-s', '--cellsize', metavar='size',
    help='grid cell size in the extent CRS', type=int, default=100
)
argparser.add_argument('-x', '--xoffset', metavar='distance',
    help='grid x-coordinate offset in the extent CRS', type=float, default=0
)
argparser.add_argument('-y', '--yoffset', metavar='distance',
    help='grid y-coordinate offset in the extent CRS', type=float, default=0
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing grid'
)
argparser.add_argument('-g', '--grid-name', metavar='name', default='grid',
    help='name of the grid to create (default: grid)'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.GridMaker.fromArgs(args).run(
        args.grid_name,
        args.cellsize,
        args.xoffset, args.yoffset,
        args.overwrite
    )