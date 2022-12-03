#!/bin/bash
sleep 10
if ! mysql -h mariadb -u root -e "USE baseball"; then
    echo "Baseball Dataset Does Not Exist"
    mysql -h mariadb -u root -e "CREATE DATABASE IF NOT EXISTS baseball;"
    mysql -h mariadb -u root baseball < /scripts/baseball.sql
else
    echo "Baseball Dataset Exists"
    mysql -h mariadb -u root -e "CREATE OR REPLACE DATABASE baseball;"
    mysql -h mariadb -u root baseball < /scripts/baseball.sql

fi

echo "Initializing HW6"
mysql -h mariadb -u root baseball < /scripts/cmd.sql
mysql -h mariadb -u root baseball -e 'SELECT * FROM rolling_average where game_id=12560;' > /results/batting_avg.txt
