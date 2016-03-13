for strategy in "n" "d" "c"; do
	for entry in `ls datasets/`; do
		echo $entry
		SUBSTRING=$(echo $entry| cut -d '.' -f 1)
		echo $SUBSTRING
	    for i in {1..5}; do
			python ../synthesis/synthesizePPLmodel.py datasets/$entry 300 outputs $SUBSTRING-$i $strategy annealing f t
	    done
	    wait
	done
done
