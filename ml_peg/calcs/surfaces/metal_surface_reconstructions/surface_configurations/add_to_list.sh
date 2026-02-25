> list
for i in *.xyz; do
	echo ${i::-4} >> list
done
