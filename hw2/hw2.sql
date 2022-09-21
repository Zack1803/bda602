
# Question 1
create temporary table q1 as(
select DISTINCT a.batter,extract (year from g.local_date) as year ,(a.Hit/a.atBat)*100 as batting_average
from
batter_counts as A
JOIN game g on A.game_id = g.game_id
group by a.batter, year);

select * from q1;




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





SELECT a.batter, b.local_date, (A.hit/A.atBat)
FROM batter_counts as A
LEFT JOIN GAME as B
ON A.game_id = B.game_id
WHERE B.local_date >= DATE_SUB((select local_date from game), interval 100 day);


 

#Q3

#Step 1 - Create a temp table with all required columns

WITH temp1 as
(
SELECT a.batter, b.local_date, A.hit,A.atBat
FROM batter_counts as A
LEFT JOIN GAME as B
ON A.game_id = B.game_id)


#Step 2 - Use the Temp Table to create self join and find rolling average

select a.batter, a.local_date, (sum(a.hit)/sum(a.atBat)) as rolling_avg, count(*) as cnt
from temp1 as a
join temp1 as b
on a.batter = b.batter and a.local_date > b.local_date and b.local_date between a.local_date - INTERVAL 100 DAY and a.local_date
where a.batter = 451088
group by a.batter,a.local_date
order by a.local_date DESC;