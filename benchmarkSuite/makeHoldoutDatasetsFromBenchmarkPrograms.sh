#!/bin/bash
for entry in `ls groundTruthBLOGPrograms/`; do
	echo $entry
	SUBSTRING=$(echo $entry| cut -d '.' -f 1)
	echo $SUBSTRING
	blog groundTruthBLOGPrograms/$entry --generate > BLOGOutputs/"$SUBSTRING"_holdout.output
	python blogOutputToCSV.py BLOGOutputs/"$SUBSTRING"_holdout.output holdoutsets/"$SUBSTRING"_holdout.csv
done
