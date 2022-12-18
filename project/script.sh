#!/bin/bash
sleep 10
if ! mysql -h mariadb -u root -e "USE baseball"; then
    echo "Baseball Dataset Does Not Exist"
    mysql -h mariadb -u root -e "CREATE DATABASE IF NOT EXISTS baseball;"
    mysql -h mariadb -u root baseball < /baseball.sql
else
    echo "Baseball Dataset Exists"
    mysql -h mariadb -u root -e "CREATE OR REPLACE DATABASE baseball;"
    mysql -h mariadb -u root baseball < /baseball.sql

fi

echo "Initializing Files for Project"
mysql -h mariadb -u root baseball < /project.sql

#mysql -h mariadb -u root baseball -e 'SELECT * FROM FEATURES;' > ./results/Features.csv
