To make a dataset from a blog model, run the following command:

```
blog <modelname>.blog --generate > <modelname>.output
```

Then run:

```
python blogOutputToCSV.py <modelname>.output <modelname>.csv
```
