null:
autosave:
	- cp -r records/raw/ ~/tmp/
	- git add --all
	- git commit -m $(msg)
	- git push origin master
forTest:
	- cp -r records/raw/ ~/tmp
	- git add --all
	- git commit -m "add test scripts"
	- git push origin master
dArc:
	- sh darc.sh
