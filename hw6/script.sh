if ! mysql -h mariadb -u root -p believe -e "USE baseball"; then
  echo "Baseball Dataset Does Not Exist"
  mariadb -u root -p believe -e "CREATE DATABASE IF NOT EXISTS baseball;"
  mariadb -u root -p believe baseball < baseball.sql
else
  echo "Baseball Dataset Exists"
fi

echo "Intializing HW6"
mariadb -u root -p believe baseball < cmd.sql
mariadb -u root -p believe baseball -e 'SELECT * FROM rolling_average where game_id=12560;' > /results/batting_avg.txt