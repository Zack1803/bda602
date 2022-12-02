USE baseball;

create or replace temporary table rolling_average_intermediate
(SELECT a.batter, b.local_date, A.hit,A.atBat
FROM batter_counts as A
LEFT JOIN GAME as B
ON A.game_id = B.game_id);
CREATE INDEX batter ON rolling_average_intermediate(batter);
CREATE  INDEX date ON rolling_average_intermediate(local_date);
CREATE  INDEX batter_date ON rolling_average_intermediate(batter,local_date);



create or replace table rolling_average
select a.batter, a.local_date, (sum(b.Hit)/NULLIF(sum(b.atBat),0)) as rolling_avg
from rolling_average_intermediate as a
join rolling_average_intermediate as b
on a.batter = b.batter and a.local_date > b.local_date and b.local_date between  a.local_date - INTERVAL 100 DAY and a.local_date
group by a.batter,a.local_date
order by a.local_date DESC;