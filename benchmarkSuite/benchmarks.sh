for entry in `ls datasets/`; do
    for i in {1..1}; do
	echo $entry
	SUBSTRING=$(echo $entry| cut -d '.' -f 1)
	echo $SUBSTRING
	python ../synthesis/synthesizePPLmodel.py datasets/$entry 20 .2 outputs $SUBSTRING-$i
    done
done
