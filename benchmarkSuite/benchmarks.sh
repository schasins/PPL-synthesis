for strategy in "n" "d" "c"; do
    for i in {1..5}; do
	for entry in `ls datasets/`; do
		echo $entry
		SUBSTRING=$(echo $entry| cut -d '.' -f 1)
		echo $SUBSTRING
       		python ../synthesis/synthesizePPLmodel.py datasets/$entry 250 outputs $SUBSTRING-$i $strategy annealing f f t t holdoutsets/"$SUBSTRING"_holdout.csv &
	done
	wait
    done
done
