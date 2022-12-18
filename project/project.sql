
#################################################################################################################
USE baseball;

#Get Team Batter Stats
CREATE or REPLACE TABLE TEMP as
SELECT
TBC.game_id as game_id,
TBC.team_id as team_id,
DATE(G.local_date) as local_date,
TBC.atBat as AtBat_batter,
TBC.Hit as Hit_batter,
TBC.finalScore as Runs,
TBC.Home_Run as Home_Run_batter,
TBC.Sac_Fly as Sac_Fly_batter,
TBC.Strikeout as StrikeOut_batter,
TBC.Walk as Walk_batter,
TBC.Hit_By_Pitch as Hit_by_Pitch_batter,
TBC.plateApperance as plateApperance_batter,
TPC.atBat as AtBat_pitcher,
TPC.Hit as Hit_pitcher,
TPC.finalScore as Runs_pitcher,
TPC.Home_Run as Home_Run_pitcher,
TPC.Sac_Fly as Sac_Fly_pitcher,

TPC.Strikeout as StrikeOut_pitcher,
TPC.Walk as Walk_pitcher,
TPC.plateApperance as plateApperance_pitcher,

TBC.Ground_Out as Ground_Out_batter,
TPC.Ground_Out as Ground_Out_pitcher,

TBC.Fly_Out as Fly_Out_batter,
TPC.Fly_Out as Fly_Out_picther,

TBC.homeTeam as TBC_Home_Away,
TPC.homeTeam as TPC_Home_Away

FROM game G
LEFT JOIN team_batting_counts TBC
on G.game_id = TBC.game_id

LEFT JOIN team_pitching_counts TPC
on G.game_id = TPC.game_id;
CREATE INDEX team_id ON TEMP(team_id);
CREATE INDEX date ON TEMP(local_date);
CREATE INDEX game_id ON TEMP(game_id);
CREATE INDEX teamid_date ON TEMP(team_id,local_date);

CREATE OR REPLACE TABLE STARTING_PICTHER as
SELECT
PC.game_id,
PC.team_id,
DATE(G.local_date) as local_date,
PC.atBat,
PC.Hit,
PC.Home_Run,
PC.Strikeout,
PC.Walk,
PC.Ground_Out,
PC.Fly_Out,
PC.endingInning -(PC.startingInning -1) as innings_pitched,
PC.homeTeam

FROM pitcher_counts PC
JOIN game G
ON G.game_id = PC.game_id
WHERE PC.startingPitcher = 1
GROUP BY
PC.game_id,
PC.team_id,
G.local_date;
CREATE INDEX game_id_sp ON STARTING_PICTHER(game_id);
CREATE INDEX team_id_sp ON STARTING_PICTHER(team_id);
CREATE INDEX date_sp ON STARTING_PICTHER(local_date);
CREATE INDEX teamid_date_sp ON STARTING_PICTHER(team_id,local_date);

CREATE OR REPLACE TABLE HOME_TEAM_SP AS
SELECT
sp1.game_id,
sp1.team_id,
sp1.local_date,
SUM(sp2.Walk*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS HT_BB9,
SUM(sp2.Hit*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS HT_H9,
SUM(sp2.Home_Run*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS HT_HR9,
SUM(sp2.Strikeout*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS HT_S9,
(SUM(sp2.Hit)+SUM(sp2.Walk))/SUM(NULLIF(sp2.innings_pitched,0)) AS HT_WHIP
FROM
STARTING_PICTHER sp1
JOIN STARTING_PICTHER sp2
ON sp1.team_id = sp2.team_id
and sp1.local_date > sp2.local_date
and sp2.local_date between  (sp1.local_date - INTERVAL 100 DAY) and sp1.local_date
WHERE sp1.homeTeam =1
GROUP BY
sp1.game_id,
sp1.team_id,
sp1.local_date;
CREATE INDEX game_id_h_sp ON STARTING_PICTHER(game_id);
CREATE INDEX team_id_h_sp ON HOME_TEAM_SP(team_id);
CREATE INDEX date_h_sp ON HOME_TEAM_SP(local_date);
CREATE INDEX teamid_date_h_sp ON HOME_TEAM_SP(team_id,local_date);


CREATE OR REPLACE TABLE AWAY_TEAM_SP AS
SELECT
sp1.game_id,
sp1.team_id,
sp1.local_date,
SUM(sp2.Walk*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS AT_BB9,
SUM(sp2.Hit*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS AT_H9,
SUM(sp2.Home_Run*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS AT_HR9,
SUM(sp2.Strikeout*9)/SUM(NULLIF(sp2.innings_pitched,0)) AS AT_S9,
(SUM(sp2.Hit)+SUM(sp2.Walk))/SUM(NULLIF(sp2.innings_pitched,0)) AS AT_WHIP
FROM
STARTING_PICTHER sp1
JOIN STARTING_PICTHER sp2
ON sp1.team_id = sp2.team_id
and sp1.local_date > sp2.local_date
and sp2.local_date between  (sp1.local_date - INTERVAL 100 DAY) and sp1.local_date
WHERE sp1.homeTeam =0
GROUP BY
sp1.game_id,
sp1.team_id,
sp1.local_date;
CREATE INDEX game_id_a_sp ON STARTING_PICTHER(game_id);
CREATE INDEX team_id_a_sp ON HOME_TEAM_SP(team_id);
CREATE INDEX date_a_sp ON HOME_TEAM_SP(local_date);
CREATE INDEX teamid_date_a_sp ON HOME_TEAM_SP(team_id,local_date);


CREATE OR REPLACE TABLE HOME_TEAM as
SELECT
t1.game_id,
t1.team_id,
t1.local_date,

COALESCE(ROUND(SUM(t2.Hit_batter)/sum(NULLIF(t2.AtBat_batter,0)),2),0) as HT_Batter_Average,

COALESCE(ROUND(SUM(t2.Home_Run_batter) /SUM(NULLIF(t2.Hit_batter,0)),2),0) as HT_Home_runs_per_hit,

COALESCE(ROUND(sum(t2.Walk_batter) / SUM(NULLIF(t2.StrikeOut_batter,0)),2),0) as HT_Batter_Walk_strikeout_ratio,


(SUM(t2.Hit_batter) - SUM(t2.Home_Run_batter))/NULLIF((sum(t2.AtBat_batter) - SUM(t2.StrikeOut_batter) - SUM(t2.Home_Run_batter) + SUM(t2.Sac_Fly_batter)),0) as HT_Batting_BABIP,


COALESCE(ROUND(sum(t2.StrikeOut_batter) / SUM(NULLIF(t2.Walk_batter,0)),2),0) as HT_Batter_strikeout_Walk_ratio,

COALESCE(ROUND(SUM(t2.Ground_Out_batter) /SUM(NULLIF(t2.Fly_Out_batter,0)),2),0) as HT_Batter_Ground_fly_ball_ratio,

SUM(t2.plateApperance_batter)/SUM(NULLIF(t2.StrikeOut_batter,0)) as HT_Batter_Plate_App_Strikeout


FROM TEMP as t1
JOIN TEMP as t2
ON t1.team_id = t2.team_id
and t1.local_date > t2.local_date
and t2.local_date between  (t1.local_date - INTERVAL 100 DAY) and t1.local_date
WHERE t1.TBC_Home_Away = 1 and t1.TPC_Home_Away = 1
group by
t1.team_id,
t1.local_date,
t1.game_id
order by t1.local_date DESC;
CREATE INDEX game_id_ht ON STARTING_PICTHER(game_id);
CREATE INDEX team_id_ht ON HOME_TEAM(team_id);
CREATE INDEX date_ht ON HOME_TEAM(local_date);
CREATE INDEX teamid_date_ht ON HOME_TEAM(team_id,local_date);


CREATE OR REPLACE TABLE AWAY_TEAM as
SELECT
t1.game_id,
t1.team_id,
t1.local_date,

COALESCE(ROUND(SUM(t2.Hit_batter)/sum(NULLIF(t2.AtBat_batter,0)),2),0) as AT_Batter_Average,
COALESCE(ROUND(SUM(t2.Home_Run_batter) /SUM(NULLIF(t2.Hit_batter,0)),2),0) as AT_Home_runs_per_hit,
COALESCE(ROUND(sum(t2.Walk_batter) / SUM(NULLIF(t2.StrikeOut_batter,0)),2),0) as AT_Batter_Walk_strikeout_ratio,
(SUM(t2.Hit_batter) - SUM(t2.Home_Run_batter))/NULLIF((sum(t2.AtBat_batter) - SUM(t2.StrikeOut_batter) - SUM(t2.Home_Run_batter) + SUM(t2.Sac_Fly_batter)),0) as AT_Batting_BABIP,
COALESCE(ROUND(sum(t2.StrikeOut_batter) / SUM(NULLIF(t2.Walk_batter,0)),2),0) as AT_Batter_strikeout_Walk_ratio,
COALESCE(ROUND(SUM(t2.Ground_Out_batter) /SUM(NULLIF(t2.Fly_Out_batter,0)),2),0) as AT_Batter_Ground_fly_ball_ratio,
SUM(t2.plateApperance_batter)/SUM(NULLIF(t2.StrikeOut_batter,0)) as AT_Batter_Plate_App_Strikeout

FROM TEMP as t1
JOIN TEMP as t2
ON t1.team_id = t2.team_id
and t1.local_date > t2.local_date
and t2.local_date between  (t1.local_date - INTERVAL 100 DAY) and t1.local_date
WHERE t1.TBC_Home_Away = 0 and t1.TPC_Home_Away = 0
group by
t1.game_id,
t1.team_id,
t1.local_date
order by t1.local_date DESC;
CREATE INDEX game_id_at ON HOME_TEAM(game_id);
CREATE INDEX team_id_at ON HOME_TEAM(team_id);
CREATE INDEX date_at ON HOME_TEAM(local_date);
CREATE INDEX teamid_date_at ON HOME_TEAM(team_id,local_date);

#Combine Home Team Stats

CREATE OR REPLACE TABLE HOME_TEAM_STATS
SELECT
HOME_TEAM.game_id,
HOME_TEAM.team_id,
HOME_TEAM.local_date,
HOME_TEAM.HT_Batter_Average,
HOME_TEAM.HT_Batter_Ground_fly_ball_ratio,
HOME_TEAM.HT_Batter_Plate_App_Strikeout,
HOME_TEAM.HT_Batter_strikeout_Walk_ratio,
HOME_TEAM.HT_Batter_Walk_strikeout_ratio,
HOME_TEAM.HT_Batting_BABIP,
HOME_TEAM.HT_Home_runs_per_hit,
HOME_TEAM_SP.HT_BB9,
HOME_TEAM_SP.HT_H9,
HOME_TEAM_SP.HT_HR9,
HOME_TEAM_SP.HT_S9,
HOME_TEAM_SP.HT_WHIP

FROM HOME_TEAM
JOIN HOME_TEAM_SP
on HOME_TEAM.team_id = HOME_TEAM_SP.team_id
AND HOME_TEAM.local_date = HOME_TEAM_SP.local_date;

CREATE INDEX game_id_hts ON HOME_TEAM(game_id);
CREATE INDEX team_id_hts ON HOME_TEAM(team_id);
CREATE INDEX date_hts ON HOME_TEAM(local_date);
CREATE INDEX teamid_date_hts ON HOME_TEAM(team_id,local_date);

CREATE OR REPLACE TABLE AWAY_TEAM_STATS
SELECT
AWAY_TEAM.game_id,
AWAY_TEAM.team_id,
AWAY_TEAM.local_date,
AWAY_TEAM.AT_Batter_Average,
AWAY_TEAM.AT_Batter_Ground_fly_ball_ratio,
AWAY_TEAM.AT_Batter_Plate_App_Strikeout,
AWAY_TEAM.AT_Batter_strikeout_Walk_ratio,
AWAY_TEAM.AT_Batter_Walk_strikeout_ratio,
AWAY_TEAM.AT_Batting_BABIP,
AWAY_TEAM.AT_Home_runs_per_hit,
AWAY_TEAM_SP.AT_BB9,
AWAY_TEAM_SP.AT_H9,
AWAY_TEAM_SP.AT_HR9,
AWAY_TEAM_SP.AT_S9,
AWAY_TEAM_SP.AT_WHIP

FROM AWAY_TEAM
JOIN AWAY_TEAM_SP
on AWAY_TEAM.team_id = AWAY_TEAM_SP.team_id
AND AWAY_TEAM.local_date = AWAY_TEAM_SP.local_date;
CREATE INDEX game_id_ats ON HOME_TEAM(game_id);
CREATE INDEX team_id_ats ON HOME_TEAM(team_id);
CREATE INDEX date_ats ON HOME_TEAM(local_date);
CREATE INDEX teamid_date_ats ON HOME_TEAM(team_id,local_date);


CREATE OR REPLACE TABLE FEATURES_Rolling as
SELECT HOME_TEAM_STATS.game_id,
HOME_TEAM_STATS.team_id as home_team_id,
AWAY_TEAM_STATS.team_id as away_team_id,
HOME_TEAM_STATS.local_date,

HOME_TEAM_STATS.HT_Batter_Average,
AWAY_TEAM_STATS.AT_Batter_Average,

HOME_TEAM_STATS.HT_Batter_Ground_fly_ball_ratio,
AWAY_TEAM_STATS.AT_Batter_Ground_fly_ball_ratio,

HOME_TEAM_STATS.HT_Batter_Plate_App_Strikeout,
AWAY_TEAM_STATS.AT_Batter_Plate_App_Strikeout,

HOME_TEAM_STATS.HT_Batter_strikeout_Walk_ratio,
AWAY_TEAM_STATS.AT_Batter_strikeout_Walk_ratio,

HOME_TEAM_STATS.HT_Batter_Walk_strikeout_ratio,
AWAY_TEAM_STATS.AT_Batter_Walk_strikeout_ratio,

HOME_TEAM_STATS.HT_Batting_BABIP,
AWAY_TEAM_STATS.AT_Batting_BABIP,

HOME_TEAM_STATS.HT_Home_runs_per_hit,
AWAY_TEAM_STATS.AT_Home_runs_per_hit,

HOME_TEAM_STATS.HT_BB9,
AWAY_TEAM_STATS.AT_BB9,

HOME_TEAM_STATS.HT_H9,
AWAY_TEAM_STATS.AT_H9,

HOME_TEAM_STATS.HT_HR9,
AWAY_TEAM_STATS.AT_HR9,

HOME_TEAM_STATS.HT_S9,
AWAY_TEAM_STATS.AT_S9,

HOME_TEAM_STATS.HT_WHIP,
AWAY_TEAM_STATS.AT_WHIP,

CASE WHEN b.winner_home_or_away = "H" THEN 1 ELSE 0 END as HomeTeamWins

FROM HOME_TEAM_STATS
JOIN AWAY_TEAM_STATS
ON HOME_TEAM_STATS.game_id = AWAY_TEAM_STATS.game_id
JOIN boxscore b
on HOME_TEAM_STATS.game_id = b.game_id;

CREATE INDEX game_id_FR ON FEATURES_Rolling(game_id);
CREATE INDEX date_FR ON FEATURES_Rolling(local_date);
CREATE INDEX gameid_date_FR ON FEATURES_Rolling(game_id,local_date);

CREATE OR REPLACE TABLE FEATURES_Normal as
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
    TBC.finalScore as Runs_Scored,
    TPC.finalScore as Runs_Allowed,
    POWER(TBC.finalScore,2) / NULLIF((SUM(POWER(TBC.finalScore,2)) + SUM(POWER(TPC.finalScore,2))),0) as Pythagorean_Win_Ratio,
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
    TBC.finalScore as Runs_Scored,
    TPC.finalScore as Runs_Allowed,
    POWER(TBC.finalScore,2) / NULLIF((SUM(POWER(TBC.finalScore,2)) + SUM(POWER(TPC.finalScore,2))),0) as Pythagorean_Win_Ratio,
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
       HT.Strikeouts as Home_Team_Strikouts_Normal,
       AT.Strikeouts as Away_Team_Strikouts_Normal,
       HT.PlateApperance as Home_Team_PlateApperance_Normal,
       AT.PlateApperance as Away_Team_PlateApperance_Normal,
       HT.Single as Home_Team_Single_Normal,
       AT.Single as Away_Team_Single_Normal,
       HT.Double_ as Home_Team_Double_Normal,
       AT.Double_ as Away_Team_Double_Normal,
       HT.Triple as Home_Team_Triple_Normal,
       AT.Triple as Away_Team_Triple_Normal,

       HT.Slugging_percentage as Home_Team_Slugging_Percentage_Normal,
       AT.Slugging_percentage as Away_Team_Slugging_Percentage_Normal,

       HT.Batting_Average as Home_Team_Batting_Average_Normal,
       AT.Batting_Average as Away_Team_Batting_Average_Normal,
       HT.Walk_strikeout_ratio as Home_Team_Walk_strikeout_ratio_Normal,
       AT.Walk_strikeout_ratio as Away_Team_Walk_strikeout_ratio_Normal,
       HT.Ground_fly_ball_ratio as Home_Team_Ground_fly_ball_ratio_Normal,
       AT.Ground_fly_ball_ratio as Away_Team_Ground_fly_ball_ratio_Normal,
       HT.Intentional_Walk as Home_Team_Intentional_Walk_Normal,
       AT.Intentional_Walk as Away_Team_Intentional_Walk_Normal,
       HT.At_bats_per_home_run as Home_Team_At_bats_per_home_run_Normal,
       AT.At_bats_per_home_run as Away_Team_At_bats_per_home_run_Normal,
       HT.Home_runs_per_hit as Home_Team_Home_runs_per_hit_Normal,
       AT.Home_runs_per_hit as Away_Team_Home_runs_per_hit_Normal,
       HT.Pythagorean_Win_Ratio as Home_Team_Pythagorean_Win_Ratio,
       AT.Pythagorean_Win_Ratio as Away_Team_Pythagorean_Win_Ratio,
       HT.HomeTeamWins

FROM HOME_TEAM HT
JOIN AWAY_TEAM AT
ON HT.game_id = AT.game_id and HT.local_date = AT.local_date
ORDER BY HT.local_date;

CREATE INDEX game_id_FN ON FEATURES_Normal(game_id);
CREATE INDEX date_FN ON FEATURES_Normal(local_date);
CREATE INDEX gameid_date_FN ON FEATURES_Normal(game_id,local_date);

CREATE OR REPLACE TABLE FEATURES as
SELECT
FEATURES_Rolling.game_id,
FEATURES_Rolling.home_team_id,
FEATURES_Rolling.away_team_id,
FEATURES_Rolling.local_date,

FEATURES_Rolling.HT_Batter_Average,
FEATURES_Rolling.AT_Batter_Average,

ABS(FEATURES_Rolling.HT_Batter_Average - FEATURES_Rolling.AT_Batter_Average) as Diff_Batter_Average,

FEATURES_Rolling.HT_Batter_Ground_fly_ball_ratio,
FEATURES_Rolling.AT_Batter_Ground_fly_ball_ratio,

ABS(FEATURES_Rolling.HT_Batter_Ground_fly_ball_ratio -FEATURES_Rolling.AT_Batter_Ground_fly_ball_ratio) as Diff_Batter_Ground_fly_ball_ratio ,

FEATURES_Rolling.HT_Batter_Plate_App_Strikeout,
FEATURES_Rolling.AT_Batter_Plate_App_Strikeout,

ABS(FEATURES_Rolling.HT_Batter_Plate_App_Strikeout - FEATURES_Rolling.AT_Batter_Plate_App_Strikeout) as Diff_Batter_Plate_App_Strikeout,

FEATURES_Rolling.HT_Batter_strikeout_Walk_ratio,
FEATURES_Rolling.AT_Batter_strikeout_Walk_ratio,

ABS(FEATURES_Rolling.HT_Batter_strikeout_Walk_ratio - FEATURES_Rolling.AT_Batter_strikeout_Walk_ratio) as Diff_Batter_strikeout_Walk_ratio,

FEATURES_Rolling.HT_Batter_Walk_strikeout_ratio,
FEATURES_Rolling.AT_Batter_Walk_strikeout_ratio,

ABS(FEATURES_Rolling.HT_Batter_Walk_strikeout_ratio - FEATURES_Rolling.AT_Batter_Walk_strikeout_ratio) as Diff_Batter_Walk_strikeout_ratio,


FEATURES_Rolling.HT_Batting_BABIP,
FEATURES_Rolling.AT_Batting_BABIP,

ABS(FEATURES_Rolling.HT_Batting_BABIP - FEATURES_Rolling.AT_Batting_BABIP) as Diff_Batting_BABIP,

FEATURES_Rolling.HT_Home_runs_per_hit,
FEATURES_Rolling.AT_Home_runs_per_hit,

ABS(FEATURES_Rolling.HT_Home_runs_per_hit -FEATURES_Rolling.AT_Home_runs_per_hit) as Diff_Home_runs_per_hit,

FEATURES_Rolling.HT_BB9,
FEATURES_Rolling.AT_BB9,

ABS(FEATURES_Rolling.HT_BB9 -FEATURES_Rolling.AT_BB9) as Diff_BB9,

FEATURES_Rolling.HT_H9,
FEATURES_Rolling.AT_H9,

ABS(FEATURES_Rolling.HT_H9 -FEATURES_Rolling.AT_H9) as Diff_H9,

FEATURES_Rolling.HT_HR9,
FEATURES_Rolling.AT_HR9,

ABS(FEATURES_Rolling.HT_HR9 - FEATURES_Rolling.AT_HR9) as DIff_HR9 ,

FEATURES_Rolling.HT_S9,
FEATURES_Rolling.AT_S9,

ABS(FEATURES_Rolling.HT_S9 -FEATURES_Rolling.AT_S9) as Diff_S9,

FEATURES_Rolling.HT_WHIP,
FEATURES_Rolling.AT_WHIP,

ABS(FEATURES_Rolling.HT_WHIP -FEATURES_Rolling.AT_WHIP) as Diff_WHIP,

FN.Home_Team_Strikouts_Normal,
FN.Away_Team_Strikouts_Normal,

ABS(FN.Home_Team_Strikouts_Normal - FN.Away_Team_Strikouts_Normal) as Diff_Strikouts_Normal,

FN.Home_Team_PlateApperance_Normal,
FN.Away_Team_PlateApperance_Normal,

ABS(FN.Home_Team_PlateApperance_Normal - FN.Away_Team_PlateApperance_Normal) as Diff_Team_PlateApperance_Normal,

FN.Home_Team_Single_Normal,
FN.Away_Team_Single_Normal,

ABS(FN.Home_Team_Single_Normal - FN.Away_Team_Single_Normal) as Diff_Single_Normal,

FN.Home_Team_Double_Normal,
FN.Away_Team_Double_Normal,

ABS(FN.Home_Team_Double_Normal - FN.Away_Team_Double_Normal) as Diff_Team_Double_Normal,

FN.Home_Team_Triple_Normal,
FN.Away_Team_Triple_Normal,

ABS(FN.Home_Team_Triple_Normal -FN.Away_Team_Triple_Normal) as Diff_Triple_Normal,

FN.Home_Team_Slugging_Percentage_Normal,
FN.Away_Team_Slugging_Percentage_Normal,

ABS(FN.Home_Team_Slugging_Percentage_Normal -FN.Away_Team_Slugging_Percentage_Normal) as Diff_Slugging_Percentage_Normal,

FN.Home_Team_Batting_Average_Normal,
FN.Away_Team_Batting_Average_Normal,

ABS(FN.Home_Team_Batting_Average_Normal - FN.Away_Team_Batting_Average_Normal) as Diff_Batting_Average_Normal,

FN.Home_Team_Walk_strikeout_ratio_Normal,
FN.Away_Team_Walk_strikeout_ratio_Normal,

ABS(FN.Home_Team_Walk_strikeout_ratio_Normal - FN.Away_Team_Walk_strikeout_ratio_Normal) as Diff_Walk_strikeout_ratio_Normal,

FN.Home_Team_Ground_fly_ball_ratio_Normal,
FN.Away_Team_Ground_fly_ball_ratio_Normal,

ABS(FN.Home_Team_Ground_fly_ball_ratio_Normal - FN.Away_Team_Ground_fly_ball_ratio_Normal) as Diff_Ground_fly_ball_ratio_Normal,

FN.Home_Team_Intentional_Walk_Normal,
FN.Away_Team_Intentional_Walk_Normal,

ABS(FN.Home_Team_Intentional_Walk_Normal - FN.Away_Team_Intentional_Walk_Normal) as Diff_Intentional_Walk_Normal,

FN.Home_Team_At_bats_per_home_run_Normal,
FN.Away_Team_At_bats_per_home_run_Normal,

ABS(FN.Home_Team_At_bats_per_home_run_Normal - FN.Away_Team_At_bats_per_home_run_Normal) as Diff_At_bats_per_home_run_Normal,

FN.Home_Team_Home_runs_per_hit_Normal,
FN.Away_Team_Home_runs_per_hit_Normal,

ABS(FN.Home_Team_Home_runs_per_hit_Normal - FN.Away_Team_Home_runs_per_hit_Normal) as Diff_Home_runs_per_hit_Normal,
#FN.Home_Team_Pythagorean_Win_Ratio,
#FN.Away_Team_Pythagorean_Win_Ratio,

FEATURES_Rolling.HomeTeamWins
FROM FEATURES_Rolling
JOIN FEATURES_Normal FN
on FEATURES_Rolling.game_id = FN.game_id;
