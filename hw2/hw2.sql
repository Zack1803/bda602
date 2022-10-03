use baseball;
# Question 1 -  Annual

#create a new table

create or replace table annual_average
select DISTINCT a.batter,extract (year from g.local_date) as year ,(sum(a.Hit)/NULLIF(sum(a.atBat),0))*100 as batting_average
from
batter_counts as A
JOIN game g on A.game_id = g.game_id
group by a.batter, year;



#Question 2 - Historical


#create a new table

create or replace table historical_average
select DISTINCT a.batter,(sum(a.Hit)/NULLIF(sum(a.atBat),0))*100 as batting_average
from
batter_counts as A
JOIN game g on A.game_id = g.game_id
group by a.batter;

#Question 3

#Q3 - Probably Correct

#Step 1 - Create a temp table with all required columns

#drop table before creation

create or replace table rolling_average

create temporary table rolling_average_intermediate
(SELECT a.batter, b.local_date, A.hit,A.atBat
FROM batter_counts as A
LEFT JOIN GAME as B
ON A.game_id = B.game_id);
CREATE UNIQUE INDEX batter ON rolling_average_intermediate(batter);
CREATE UNIQUE INDEX date ON rolling_average_intermediate(local_date);
CREATE UNIQUE INDEX batter_date ON rolling_average_intermediate(batter,local_date);



#Step 2 - Use the Temp Table to create self join and find rolling average
#Hint 2 - for one batter

select a.batter, a.local_date, (sum(b.Hit)/NULLIF(sum(b.atBat),0))*100 as rolling_avg
from rolling_average_intermediate as a
join rolling_average_intermediate as b
on a.batter = b.batter and a.local_date > b.local_date and b.local_date between  a.local_date - INTERVAL 100 DAY and a.local_date
#where a.batter = 110029
group by a.batter,a.local_date
order by a.local_date DESC;