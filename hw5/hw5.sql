use baseball;

CREATE OR REPLACE TABLE Features as
WITH HOME_TEAM as

(
SELECT
    g.game_id,
    g.local_date,
    TBC.team_id,
    'Home Team'  as HOME_AWAY,

	SUM(TPC.Strikeout) AS Strikeouts,
	SUM(TPC.plateApperance) as PlateApperance,

	SUM(TPC.Single) as Single,
	SUM(TPC.`Double`) as Double_,
	SUM(TPC.Triple) as Triple,

	COALESCE(ROUND(SUM(TBC.toBase) /SUM(NULLIF(TBC.atBat,0)),2),0) as Slugging_percentage,
	COALESCE(ROUND(SUM(TBC.Hit)/sum(NULLIF(TBC.atBat,0)),2),0) as Batting_Average,
	COALESCE(ROUND(sum(TBC.Walk) / SUM(NULLIF(TBC.Strikeout,0)),2),0) as Walk_strikeout_ratio,
	COALESCE(ROUND(SUM(TBC.Ground_Out) /SUM(NULLIF(TBC.Fly_Out,0)),2),0) as Ground_fly_ball_ratio,
	SUM(TBC.Intent_Walk) as Intentional_Walk,
    COALESCE(ROUND(SUM(TBC.atBat)/SUM(NULLIF(TBC.Home_Run,0)),2),0) as At_bats_per_home_run,
    COALESCE(ROUND(SUM(TBC.Home_Run) /SUM(NULLIF(TBC.Hit,0)),2),0) as Home_runs_per_hit,
    CASE WHEN b.winner_home_or_away = "H" THEN 1 ELSE 0 END as HomeTeamWins



FROM game g
LEFT JOIN  team_batting_counts as TBC
on TBC.game_id = g.game_id

LEFT JOIN team_pitching_counts TPC
on g.game_id = TPC.game_id

LEFT JOIN boxscore b
on g.game_id = b.game_id

WHERE TPC.homeTeam =1 and TBC.homeTeam =1

GROUP BY g.game_id,
         g.local_date,
         TBC.team_id
),

AWAY_TEAM AS(

SELECT
    g.game_id,
    g.local_date,
    TBC.team_id,
    'Away Team'  as HOME_AWAY,

	SUM(TPC.Strikeout) AS Strikeouts,
	SUM(TPC.plateApperance) as PlateApperance,

	SUM(TPC.Single) as Single,
	SUM(TPC.`Double`) as Double_,
	SUM(TPC.Triple) as Triple,

	COALESCE(ROUND(SUM(TBC.toBase) /SUM(NULLIF(TBC.atBat,0)),2),0) as Slugging_percentage,
	COALESCE(ROUND(SUM(TBC.Hit)/sum(NULLIF(TBC.atBat,0)),2),0) as Batting_Average,
	COALESCE(ROUND(sum(TBC.Walk) / SUM(NULLIF(TBC.Strikeout,0)),2),0) as Walk_strikeout_ratio,
	COALESCE(ROUND(SUM(TBC.Ground_Out) /SUM(NULLIF(TBC.Fly_Out,0)),2),0) as Ground_fly_ball_ratio,
	SUM(TBC.Intent_Walk) as Intentional_Walk,
    COALESCE(ROUND(SUM(TBC.atBat)/SUM(NULLIF(TBC.Home_Run,0)),2),0) as At_bats_per_home_run,
    COALESCE(ROUND(SUM(TBC.Home_Run) /SUM(NULLIF(TBC.Hit,0)),2),0) as Home_runs_per_hit,
    CASE WHEN b.winner_home_or_away = "H" THEN 1 ELSE 0 END as HomeTeamWins



FROM game g
LEFT JOIN  team_batting_counts as TBC
on TBC.game_id = g.game_id

LEFT JOIN team_pitching_counts TPC
on g.game_id = TPC.game_id

LEFT JOIN boxscore b
on g.game_id = b.game_id

WHERE TPC.awayTeam =1 and TBC.awayTeam =1

GROUP BY g.game_id,
         g.local_date,
         TBC.team_id

)
SELECT HT.game_id,
       CAST(HT.local_date as DATE) as local_date,
       HT.team_id as Home_Team_ID,
       AT.team_id as Away_Team_ID,
       HT.Strikeouts as Home_Team_Strikouts,
       AT.Strikeouts as Away_Team_Strikouts,
       HT.PlateApperance as Home_Team_PlateApperance,
       AT.PlateApperance as Away_Team_PlateApperance,
       HT.Single as Home_Team_Single,
       AT.Single as Away_Team_Single,
       HT.Double_ as Home_Team_Double,
       AT.Double_ as Away_Team_Double,
       HT.Triple as Home_Team_Triple,
       AT.Triple as Away_Team_Triple,

       HT.Slugging_percentage as Home_Team_Slugging_Percentage,
       AT.Slugging_percentage as Away_Team_Slugging_Percentage,

       HT.Batting_Average as Home_Team_Batting_Average,
       AT.Batting_Average as Away_Team_Batting_Average,
       HT.Walk_strikeout_ratio as Home_Team_Walk_strikeout_ratio,
       AT.Walk_strikeout_ratio as Away_Team_Walk_strikeout_ratio,
       HT.Ground_fly_ball_ratio as Home_Team_Ground_fly_ball_ratio,
       AT.Ground_fly_ball_ratio as Away_Team_Ground_fly_ball_ratio,
       HT.Intentional_Walk as Home_Team_Intentional_Walk,
       AT.Intentional_Walk as Away_Team_Intentional_Walk,
       HT.At_bats_per_home_run as Home_Team_At_bats_per_home_run,
       AT.At_bats_per_home_run as Away_Team_At_bats_per_home_run,
       HT.Home_runs_per_hit as Home_Team_Home_runs_per_hit,
       AT.Home_runs_per_hit as Away_Team_Home_runs_per_hit,
       HT.HomeTeamWins

FROM HOME_TEAM HT
JOIN AWAY_TEAM AT
ON HT.game_id = AT.game_id and HT.local_date = AT.local_date
ORDER BY HT.local_date;

SELECT  * FROM Features;