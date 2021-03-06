# Movies-ETL 

## Project Overview

In module 8, I learned how to use the Extract, Transform, Load (ETL) process to create a data pipeline. It is a combined three database function that pulls data from a source database and puts it into another database (destination). Also, it creates data pipelines that can transform data along the way. ETL is essential to data analysis.

## Resources

- Software: Python Data, Jupyter Notebbok, Visual Studio Code 1.45.1, PostgreSQL:pgAdmin4
- Data Source:   wikipedia.movies.json, movies_metadata.csv, ratings.csv 

## Challenge Overview

In this module, I had to extract data from Wikipedia and Kaggle files, transform the datasets by cleaning up the joining the files together, and load the dataset into a SQL database. Since this process will be run without supervision, I didn't need to perform exploratory data analysis steps. Since there will be new data coming in which may contain errors, it may stop the process or corrupt data. I had to add a try-except block that will help ETL process become more robust to errors. The assumption was that the updated data will stay in the same format. The task of this challenge was:

- Create an automated ETL pipeline.
- Extract data from multiple sources.
- Clean and transform the data automatically using Pandas and regular expressions.
- Load new data into PostgreSQL.

## Assumptions
1. Release dates are entered in the following formats:  YYYY, MM DD YYYY, YYYY MM DD, MM YYYY
2. Budget and revenue has two formats:  $123.4 million/billion &/or 123,456,789
3. remove null values from Wikipedia dataset
4. replace NaN with zero to utilize the dataset
5. Runtime doesn't include second.  All entries measurement of hours, hours + minutes, or minutes
6. used .CSV and .JSON files for the function
7. remove all exploratory data that can be helpful to understand the clean dataset

Review the following file: Challenge.py


