CREATE TABLE tb_config(
    config_id SERIAL PRIMARY KEY,
    id_type VARCHAR(70) DEFAULT NULL,
    id_version INTEGER DEFAULT 0,
    params VARCHAR(5000) DEFAULT NULL,
    params_dist NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    params_ratio NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    image_breath NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    image_length NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    key_breath VARCHAR(1000) DEFAULT 'Breath',
    key_length VARCHAR(1000) DEFAULT 'length',
    status CHAR(1) DEFAULT 'A',
    created_by INTEGER DEFAULT 0,
    created_date TIMESTAMP DEFAULT NULL,
    updated_by INTEGER DEFAULT 0,
    updated_date TIMESTAMP DEFAULT NULL
);

--- No use this All-----

CREATE TABLE tb_config(
    seq_id SERIAL PRIMARY KEY,
    id_type VARCHAR(70) DEFAULT NULL,
    id_version INTEGER DEFAULT 1,
    config_id INTEGER DEFAULT NULL,
    params VARCHAR(1000) DEFAULT NULL,
    params_type VARCHAR(50) DEFAULT 'OCR',
    params_dist NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    params_ratio NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    image_breath NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    image_length NUMERIC(10,4) NOT NULL DEFAULT 0.00,
    image_key VARCHAR(1000) DEFAULT 'Length_Breath',
    status CHAR(1) DEFAULT 'A',
    created_by INTEGER DEFAULT 0,
    created_date TIMESTAMP DEFAULT NULL,
    updated_by INTEGER DEFAULT 0,
    updated_date TIMESTAMP DEFAULT NULL
);

--- PAN id-version :1
INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('PAN',1, 1,'DEPARTMENT',406.1773,4.8421,306,699);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('PAN',1, 1,'GOVT.',406.1773,3.0513,306,699);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('PAN',1, 1,'Father''s',86.885,4.96,306,699);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('PAN',1, 1,'Date',86.885,1.1132,306,699);

--- PAN id-version :2
INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,image_breath,image_length) 
VALUES('PAN',2, 2,'DEPARTMENT',306.1773,316,619);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,image_breath,image_length) 
VALUES('PAN',2, 2,'GOVT.',306.1773,316,619);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,image_breath,image_length) 
VALUES('PAN',2, 2,'Father''s',82.1773,316,619);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,image_breath,image_length) 
VALUES('PAN',2, 2,'Date',84.1773,316,619);

--- AADHAR id-version :1
INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('AADHAR',1, 1,'DO',110.0227,1.8148,86,-83);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('AADHAR',1, 1,'MALE',110.0227,3.2963,86,-83);

--- AADHAR id-version :2
INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('AADHAR',2, 2,'DO',108.0227,2.8148,82,-82);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('AADHAR',2, 2,'MALE',108.0227,4.2963,82,-82);


--- Driving id-version :1

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',1, 1,'issue',154.013,3.6667,95,489);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',1, 1,'Validity',154.013,4.2727,95,489);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',1, 1,'Nationality',74.3303,5.2143,95,489);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',1, 1,'Birth',74.3303,2.6667,95,489);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',1, 1,'Group',404.9691,2.8462,95,489);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',1, 1,'Father''s',404.9691,4.8333,95,489);


--- Driving id-version :2
INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',2, 2,'issue',534.0459,3.3846,231,1320);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',2, 2,'Validity',534.0459,3.4615,231,1320);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',2, 2,'Nationality',257.6354,4.7778,231,1320);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',2, 2,'Birth',257.6354,2.5455,231,1320);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',2, 2,'Group',1371.6209,2.7857,231,1320);

INSERT INTO tb_config (id_type,id_version,config_id,params,params_dist,params_ratio,image_breath,image_length) 
VALUES('DRIVING',2, 2,'Father''s',1371.6209,4.5909,231,1320);