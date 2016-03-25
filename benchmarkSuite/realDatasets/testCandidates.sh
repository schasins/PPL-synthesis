for entry in `ls synthesizedBLOGPrograms/`; do
    echo $entry
    python evaluateSynthesizedUsingTestSet.py synthesizedBLOGPrograms/$entry testingData2.csv &
    done
wait
