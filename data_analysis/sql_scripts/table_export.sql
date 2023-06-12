#SELECT @@GLOBAL.secure_file_priv;
#SHOW VARIABLES LIKE "secure_file_priv";
secure-file-priv = ""
TABLE graham1 
INTO OUTFILE '/Users/wayne/Documents/EGGS-Aquacrop/data/agricLand/soils/gmd4_soils/MergedSoil/graham1.csv'
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
ESCAPED BY ''
LINES TERMINATED BY 'n';

#Error Code: 1290. The MySQL server is running with the --secure-file-priv option so it cannot execute this statement

SHOW VARIABLES LIKE "secure_file_priv"; 

#secure-file-priv='/Users/wayne/Documents/EGGS-Aquacrop/data/agricLand/soils/gmd4_soils/MergedSoil/'