SET GLOBAL local_infile=1;

#CREATE TABLE thetas(unknown int, uid int, depth text ,theta_s text);

CREATE TABLE fields_co(unknown int, uid int, county text, geo text);

LOAD DATA LOCAL INFILE '/Users/wayne/Downloads/POLARISMergedSoils_GMD4/fields_county.csv' 
INTO TABLE fields_co
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

# re-import the large files
CREATE TABLE graham1(unknown int, uid
int,
county
text,
geo
text,
depth
text,
log_alpha
text,
log_hb
text,
log_om
text,
log_ksat
text,
la
text,
clay_prc
text);

LOAD DATA LOCAL INFILE '/Users/wayne/Downloads/POLARISMergedSoils_GMD4/graham1.csv' 
INTO TABLE fields_co
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

