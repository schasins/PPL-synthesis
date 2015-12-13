for f in "burglary" "csi" "eyecolor" "healthiness" "hurricaneVariation"; do
	python synthesizePPLmodel.py ../benchmarkSuite/datasets/$f.csv 0 outputs $SUBSTRING-$i n > $f-reductionResults.csv &
done
wait