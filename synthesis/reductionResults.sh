for f in "burglary" "csi" "eyecolor" "healthiness" "hurricaneVariation"; do
	echo $f
	python synthesizePPLmodel.py ../benchmarkSuite/datasets/$f.csv 0 outputs $SUBSTRING-$i n
done
wait