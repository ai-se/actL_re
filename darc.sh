#!/bin/bash
rsync -av --ignore-existing jchen37@arc.csc.ncsu.edu:/home/jchen37/actL_re/records/raw/__* records
cd records/
a=`ls  __*.txt|head -1`
b=${a:2}
cat _*.txt >> $b
rm __*.txt
mv $b raw
cd ..
unset a
unset b