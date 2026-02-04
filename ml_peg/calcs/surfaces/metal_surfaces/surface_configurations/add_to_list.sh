> list
for i in *.xyz; do
	echo ${i::-4} >> list
done

for i in bulk/*.xyz; do
	echo ${i::-4} >> list
done

for i in gas_phase*.xyz; do
	echo ${i::-4} >> list
done