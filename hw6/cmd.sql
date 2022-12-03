USE baseball;
create or replace temporary table rolling_average_intermediate
(SELECT BC.game_id,BC.batter, GAME.local_date, BC.hit,BC.atBat
FROM batter_counts as BC
LEFT JOIN GAME as GAME
ON BC.game_id = GAME.game_id);
CREATE INDEX batter ON rolling_average_intermediate(batter);
CREATE  INDEX date ON rolling_average_intermediate(local_date);
CREATE  INDEX batter_date ON rolling_average_intermediate(batter,local_date);
CREATE  INDEX game_id ON rolling_average_intermediate(game_id);


create or replace table rolling_average
select b.game_id,a.batter, a.local_date, (sum(b.Hit)/NULLIF(sum(b.atBat),0)) as rolling_avg
from rolling_average_intermediate as a
join rolling_average_intermediate as b
on a.batter = b.batter
and a.local_date > b.local_date
and b.local_date between  (a.local_date - INTERVAL 100 DAY) and a.local_date
group by b.game_id,a.batter,a.local_date
order by a.local_date DESC;