###### mutliple joins
CREATE TABLE logan1
AS
SELECT 
a.uid,
county,
geo,
a.depth,
logAlpha,
logHb,
logOm,
logKsat,
lambda_val,
clay_prc
FROM log_alpha AS a
INNER JOIN clay AS c ON a.uid=c.uid
    AND a.depth=c.depth
INNER JOIN log_hb AS hb ON a.uid=hb.uid
    AND a.depth=hb.depth
INNER JOIN log_om AS om ON a.uid=om.uid
    AND a.depth=om.depth
INNER JOIN log_ksat AS ksat ON a.uid=ksat.uid
    AND a.depth=ksat.depth
INNER JOIN lambda AS l ON a.uid=l.uid
    AND a.depth=l.depth
LEFT JOIN fields_co AS f ON a.uid=f.uid
WHERE county = 'Logan';


CREATE TABLE logan2
AS
SELECT
n2.uid,
county,
geo,
n2.depth,
n_val,
sand_prc,
silt_prc,
#clay_prc,
thetaR,
thetaS
FROM n_df AS n2
INNER JOIN sand AS sa ON n2.uid=sa.uid
    AND n2.depth=sa.depth
INNER JOIN silt AS sl ON n2.uid=sl.uid
    AND n2.depth=sl.depth
INNER JOIN theta_r AS r ON n2.uid=r.uid
    AND n2.depth=r.depth
INNER JOIN theta_s AS s ON n2.uid=s.uid
    AND n2.depth=s.depth
LEFT JOIN fields_co AS f ON n2.uid=f.uid
WHERE county = 'Logan';


CREATE TABLE logan
AS
SELECT
w1.uid,
w1.county,
w1.geo,
w1.depth,
logAlpha,
logHb,
logOm,
logKsat,
lambda_val,
n_val,
sand_prc,
silt_prc,
clay_prc,
thetaR,
thetaS
FROM logan1 AS w1
JOIN logan2 AS w2 ON w1.uid=w2.uid
    AND w1.depth=w2.depth;
#



############ one join
CREATE TABLE wallace
AS
SELECT 
a.uid,
county,
a.depth,
log_alpha,
log_hb,
log_om,
log_ksat,
lambda,
n,
sand_prc,
silt_prc,
clay_prc,
theta_r,
theta_s
FROM alpha AS a
INNER JOIN clay AS c ON a.uid=c.uid
    AND a.depth=c.depth
INNER JOIN hb AS hb ON a.uid=hb.uid
    AND a.depth=hb.depth
INNER JOIN om AS om ON a.uid=om.uid
    AND a.depth=om.depth
INNER JOIN ksat AS ksat ON a.uid=ksat.uid
    AND a.depth=ksat.depth
INNER JOIN la AS l ON a.uid=l.uid
    AND a.depth=l.depth
INNER JOIN n_val ON a.uid=n_val.uid
    AND a.depth=n_val.depth
INNER JOIN sand AS sa ON a.uid=sa.uid
    AND a.depth=sa.depth
INNER JOIN silt AS sl ON a.uid=sl.uid
    AND a.depth=sl.depth
INNER JOIN thetar AS r ON a.uid=r.uid
    AND a.depth=r.depth
INNER JOIN thetas AS s ON a.uid=s.uid
    AND a.depth=s.depth
JOIN fields_co AS f ON a.uid=f.uid
WHERE county = 'Wallace';

#Error Code: 3. Error writing file '/var/folders/zz/zyxvpxvq6csfxvn_n000009800002_/T/MYGhIBtd' (OS errno 28 - No space left on device)


