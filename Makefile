null:
autosave:
    cp records/raw/ ~/tmp/
	- git add --all
	- git commit -m $(msg)
	- git push origin master
dArc:
	rsync -av --ignore-existing jchen37@arc.csc.ncsu.edu:/home/jchen37/actL_re/records/raw records
