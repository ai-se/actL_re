null:
	- echo do_nothing
Autosave:
	- cp -r records/raw/ ~/tmp/
	- git add --all
	- git commit -m $(msg)
	- git push origin master
ForTest:
	- cp -r records/raw/ ~/tmp
	- git add --all
	- git commit -m "add test scripts"
	- git push origin master
Download_Arc_Res:
	- sh darc.sh
Deploy_to_Arc:
	- rsync -rav -e ssh --exclude='*.dimacs *.augment .*' ../actL_re/ jchen37@arc.csc.ncsu.edu:/home/jchen37/actL_re
Run_At_Arc:
	- sh main.sh