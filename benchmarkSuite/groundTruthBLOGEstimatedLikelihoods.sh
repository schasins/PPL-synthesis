for entry in `ls groundTruthBLOGProgramsWithQueries/`; do
	python ../synthesis/BLOGScore.py groundTruthBLOGProgramsWithQueries/$entry 10000
done
