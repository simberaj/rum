set PATH=%PATH%;c:\program files\postgresql\10\bin
set PGPASSWORD=bezhesla
shp2pgsql -s 3035 -c -g multipolygon -k -N skip %1 %2 | psql -U postgres -d rum
set PGPASSWORD=