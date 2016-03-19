for strategy in "n"; do
	for entry in `ls datasets/`; do
		echo $entry
		SUBSTRING=$(echo $entry| cut -d '.' -f 1)
		echo $SUBSTRING
	    for i in {1}; do
			python ../synthesis/synthesizePPLmodel.py datasets/$entry 100 outputs $SUBSTRING-$i $strategy annealing f f f
	    done
	    wait
	done
done
