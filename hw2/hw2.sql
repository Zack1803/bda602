use baseball;
# Question 1

#drop table before creating a new

drop table yearly_average;

#create a new table

create table yearly_average
select DISTINCT a.batter,extract (year from g.local_date) as year ,(sum(a.Hit)/NULLIF(sum(a.atBat),0))*100 as batting_average
from
batter_counts as A
JOIN game g on A.game_id = g.game_id
group by a.batter, year;




#Question 2

#Q2 - v1
WITH batters as
    (
select  g.local_date ,a.batter, a.Hit,a.atBat
from
batter_counts as A
JOIN game g on A.game_id = g.game_id
group by a.batter)

select date(local_date) as Date,
       batter,
       AVG((Hit/atBat)*100) OVER(ORDER BY local_date ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)
        AS moving_average_100_days
from batters;


#Q2 - v2

WITH batter as
(select  cast(g.local_date as date) as date ,a.batter, a.Hit,a.atBat
from
batter_counts as A
JOIN game g
on A.game_id = g.game_id
where g.local_date >= Date_SUB((select max(local_date) from game),interval 100 day )
group by a.batter)

select distinct batter, (Hit)/(atBat)
from batter
group by  batter;






#Q3

#Step 1 - Create a temp table with all required columns

#drop table before creation

create or replace temporary table rolling_average_intermediate as
SELECT a.batter, b.local_date, A.hit,A.atBat
FROM batter_counts as A
LEFT JOIN GAME as B
ON A.game_id = B.game_id ;


#Step 2 - Use the Temp Table to create self join and find rolling average
#Hint 2 - for one batter - commented for submission
create or replace table rolling_average
select a.batter, a.local_date, (sum(a.Hit)/NULLIF(sum(a.atBat),0))*100 as rolling_avg
from rolling_average_intermediate as a
join rolling_average_intermediate as b
on a.batter = b.batter and a.local_date > b.local_date and b.local_date between a.local_date - INTERVAL 100 DAY and a.local_date
#where a.batter = 451088
group by a.batter,a.local_date
order by a.local_date DESC;