-- This is a helper file for creating and populating the sqlite database.
-- Future iterations of the Cullen Colony Classifier will automate this 
-- process, but for now I manually copy/paste/execute these commands.

-- Export a query result to a CSV file using sqlite3 shell:
-- .headers on
-- .mode csv
-- .output ccc_classifications.csv
-- select filename,imagekey,avg(ruffled_prob),avg(smooth_prob) from classifications group by filename,imagekey;
-- .output stdout