for strategy in "n" "d" "c"
	for entry in `ls datasets/`; do
		echo $entry
		SUBSTRING=$(echo $entry| cut -d '.' -f 1)
		echo $SUBSTRING
	    for i in {1..5}; do
			python ../synthesis/synthesizePPLmodel.py datasets/$entry 100 outputs $SUBSTRING-$i $strategy &
	    done
	    wait
	done
done
