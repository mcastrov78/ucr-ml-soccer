-- GENERAL
select * from country -- EPL 1729

select * from league -- EPL 1729

select count(*) from match -- 25,979
select * from match

select count(*) from player -- 11,060
select * from player
select * from player where player_name like "%ronaldo%" -- id=1995 / player_api_id=30893

select count(*) from player_attributes -- 183,978
select * from player_attributes
select * from player_attributes where player_api_id = 30894 order by date

select count(*) from team -- 299
select * from team
select * from team where team_long_name like "%juventus%" -- id=20522 / team_api_id=9885 / team_fifa_api_id=45

select count(*) from team_attributes --1,458
select * from team_attributes
select * from team_attributes where team_api_id = 9885 order by date


-- EPL
select count(*) from match where country_id = 1729 -- 3040 games (8 * 380)
select * from match where country_id = 1729 order by season

select count(*) from match where country_id = 1729 and season = "2008/2009" -- 380
select count(*) from match where country_id = 1729 and season = "2009/2010" -- 380
select count(*) from match where country_id = 1729 and season = "2010/2011" -- 380
select count(*) from match where country_id = 1729 and season = "2011/2012" -- 380
select count(*) from match where country_id = 1729 and season = "2012/2013" -- 380
select count(*) from match where country_id = 1729 and season = "2013/2014" -- 380
select count(*) from match where country_id = 1729 and season = "2014/2015" -- 380
select count(*) from match where country_id = 1729 and season = "2015/2016" -- 380


-- POSSESSION
select count(*) from match where country_id = 1729 and (possession is not null and length(trim(possession)) > 0) -- 3040 games
select possession from match where country_id = 1729 -- 3040 games
select id, country_id, season, possession from match where country_id = 1729 limit 100

select count(*) from match where country_id = 1729 and possession like "%<elapsed>90</elapsed>%" --2244
select id, country_id, season, possession from match where country_id = 1729 and possession like "%<elapsed>90</elapsed>%"

select count(*) from match where country_id = 1729 and possession NOT like "%<elapsed>90</elapsed>%" --796
select id, country_id, season, possession from match where country_id = 1729 and possession NOT like "%<elapsed>90</elapsed>%" --796


-- GAME
select * from team where team_long_name like "%leicester%" -- team_api_id=8197
select * from team where team_long_name like "%swansea%" -- team_api_id=10003
select * from match where league_id = 1729 and home_team_api_id = 8197
select * from match where league_id = 1729 and away_team_api_id = 8197

-- LEI games in 2015/2016
select m.id, m.league_id, m.season, m.stage, m.date, m.match_api_id, m.home_team_api_id, m.away_team_api_id, t1.team_long_name as HOME, t2.team_long_name as AWAY, m.home_team_goal, m.away_team_goal
from match as m, team as t1, team as t2
where m.league_id = 1729 and m.season = "2015/2016"
and (m.home_team_api_id = 8197 or m.away_team_api_id = 8197)
and (m.home_team_api_id = t1.team_api_id)
and (m.away_team_api_id = t2.team_api_id)
order by date


-- MATCH SWA vs LEI
select * from match where match_api_id = 1989053
select id from match where league_id = 1729 and season = "2015/2016" and (home_team_api_id = 8197 or away_team_api_id = 8197)


-- PLAYERS SWA vs LEI
select p.player_api_id, p.player_name, m.match_api_id, m.home_team_api_id as team_api_id, t.team_long_name from player as p, match as m, team as t
where m.match_api_id = 1989053
and p.player_api_id in (m.home_player_1, m.home_player_2, m.home_player_3, m.home_player_4, m.home_player_5, m.home_player_6, m.home_player_7, m.home_player_8, m.home_player_9, m.home_player_10, m.home_player_11)
and m.home_team_api_id = t.team_api_id
UNION
select p.player_api_id, p.player_name, m.match_api_id, m.away_team_api_id as team_api_id, t.team_long_name from player as p, match as m, team as t
where m.match_api_id = 1989053
and p.player_api_id in (m.away_player_1, m.away_player_2, m.away_player_3, m.away_player_4, m.away_player_5, m.away_player_6, m.away_player_7, m.away_player_8, m.away_player_9, m.away_player_10, m.away_player_11)
and m.away_team_api_id = t.team_api_id
ORDER BY team_api_id, player_name


-- OTHER FEATURES
select * from match where country_id = 1729 -- 3040 games
select id, country_id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, goal, shoton, shotoff, possession 
from match where league_id = 1729
                            
select count(*) from match where country_id = 1729 and (possession is not null and length(trim(possession)) > 0) -- 3040 games
select count(*) from match where country_id = 1729 and (goal is not null and length(trim(goal)) > 0) -- 3040 games
select count(*) from match where country_id = 1729 and (shoton is not null and length(trim(shoton)) > 0) -- 3040 games
select count(*) from match where country_id = 1729 and (shotoff is not null and length(trim(shotoff)) > 0) -- 3040 games
select count(*) from match where country_id = 1729 and (corner is not null and length(trim(corner)) > 0) -- 3040 games

-- https://www.footballcritic.com/premier-league-leicester-city-fc-swansea-city-afc/match-stats/508454
select id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, goal from match where match_api_id = 1989053
select id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, shoton from match where match_api_id = 1989053
select id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, shotoff from match where match_api_id = 1989053
select id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, corner from match where match_api_id = 1989053

----------------------------------------
-- AVERAGES

select * from country -- 11
select * from league -- 11 - country_id
select * from team  -- 299 - NO league or country id

select * from match where league_id = 1 order by season
select * from match m where home_team_api_id = 8342 and m.season="2015/2016" -- Club Brugge KV - CLB 
select distinct(m.home_team_api_id), m.country_id, m.league_id, t.team_long_name, t.team_short_name from match m, team t where m.league_id = 1 and m.season="2015/2016" and m.home_team_api_id = t.team_api_id

-- NO ENOUGH INFO!!!
select * from match m where league_id = 1 order by season       -- Belgium -> NO INFO!!!
select * from match m where league_id = 4769 order by season    -- France -> NO INFO on 50% of matches!!!
select * from match m where league_id = 13274 order by season   -- Netherlands -> NO INFO on 75% of matches!!!
select * from match m where league_id = 15722 order by season   -- Poland -> NO INFO !!!
select * from match m where league_id = 17642 order by season   -- Portugal -> NO INFO !!!
select * from match m where league_id = 19694 order by season   -- Scotland -> NO INFO !!!
select * from match m where league_id = 24558 order by season   -- Switzerland -> NO INFO on 90% of matches!!!

-- ENOUGH INFO
select * from match m where league_id = 1729 order by season    -- England -> ALMOST ALL INFO
select * from match m where league_id = 7809 order by season    -- Germany -> ALMOST ALL INFO
select * from match m where league_id = 10257 order by season   -- Italy -> ALMOST ALL INFO
select * from match m where league_id = 21518 order by season   -- Spain -> ALMOST ALL INFO

-- 78 teams * 8 seasons = 624 averages
select distinct(m.home_team_api_id), m.country_id, m.league_id, t.team_long_name, t.team_short_name from match m, team t where m.league_id = 1729  and m.season="2015/2016" and m.home_team_api_id = t.team_api_id -- 20 teams
select distinct(m.home_team_api_id), m.country_id, m.league_id, t.team_long_name, t.team_short_name from match m, team t where m.league_id = 7809  and m.season="2015/2016" and m.home_team_api_id = t.team_api_id -- 18 teams
select distinct(m.home_team_api_id), m.country_id, m.league_id, t.team_long_name, t.team_short_name from match m, team t where m.league_id = 10257 and m.season="2015/2016" and m.home_team_api_id = t.team_api_id -- 20 teams
select distinct(m.home_team_api_id), m.country_id, m.league_id, t.team_long_name, t.team_short_name from match m, team t where m.league_id = 21518 and m.season="2015/2016" and m.home_team_api_id = t.team_api_id -- 20 teams

-- 129 DIFFERENT teams ALL seasons
select distinct(m.home_team_api_id), t.team_long_name, l.name from match m, team t, league l
where m.league_id in (1729, 7809, 10257, 21518) and m.home_team_api_id = t.team_api_id and m.league_id = l.id
order by m.league_id

-- Games per league per season
select count(*) from match where league_id in (1729) and season = "2008/2009" -- 20 * 38/2 = 380
select count(*) from match where league_id in (7809) and season = "2008/2009" -- 18 * 34/2 = 306
select count(*) from match where league_id in (10257) and season = "2008/2009" -- 20 * 38/2 = 380
select count(*) from match where league_id in (21518) and season = "2008/2009" -- 20 * 38/2 = 380
select count(*) from match where league_id in (1729, 7809, 10257, 21518) and season = "2008/2009" -- 380*3 + 306 = 1446

-- ALL games in ALL 4 seasons
select count(*) from match where league_id in (1729, 7809, 10257, 21518) -- 1446 * 8 = 11568 ~ 11545
select * from match where league_id in (1729, 7809, 10257, 21518) -- 1446 * 8 = 11568 ~ 11545
select home_team_possession, possession from match where league_id in (1729, 7809, 10257, 21518) -- 1446 * 8 = 11568 ~ 11545

select id, country_id, league_id, season, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, goal, shoton, shotoff, possession 
from match where league_id in (1729, 7809, 10257, 21518) order by league_id, season, home_team_api_id

select id, country_id, league_id, season, home_team_api_id, home_team_goal, away_team_goal, (home_team_goal - away_team_goal) as GD, possession
from match where league_id in (1729, 7809, 10257, 21518) order by league_id, season, home_team_api_id

-- 78 teams * 8 seasons = 624 averages
select m.league_id, l.name, m.season, m.home_team_api_id, t.team_long_name, sum(home_team_goal), avg(home_team_goal), avg(away_team_goal), avg(home_team_goal - away_team_goal) as AVG_GD
from match m, team t, league l
where league_id in (1729, 7809, 10257, 21518) and m.home_team_api_id = t.team_api_id and m.league_id = l.id
group by m.country_id, m.league_id, m.season, m.home_team_api_id

select id, country_id, league_id, season, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, goal, shoton, shotoff, possession 
from match where home_team_api_id = 8455 and season = "2008/2009"
order by league_id, season, home_team_api_id -- CHELSEA games in 2008/2009

ALTER TABLE match ADD home_team_possession INTEGER NULL
select * from match
ALTER TABLE match DROP COLUMN home_team_possession -- NOT allowed in SQLite

select count(*) from match where league_id not in (1729, 7809, 10257, 21518) -- 14434
select count(*) from match where league_id not in (1729, 7809, 10257, 21518) and home_team_possession is null -- 14434
select count(*) from match where league_id in (1729, 7809, 10257, 21518) -- 1446 * 8 = 11568 ~ 11545
select count(*) from match where league_id in (1729, 7809, 10257, 21518) and home_team_possession is not null -- 11545
select count(*) from match where league_id in (1729, 7809, 10257, 21518) and home_team_possession is null -- 0
select count(*) from match -- 14434 + 11545 = 25979

select * from match where league_id in (1729, 7809, 10257, 21518) 
select home_team_possession, possession from match where league_id in (1729, 7809, 10257, 21518)

select count(*) from match where league_id in (1729, 7809, 10257, 21518) and home_team_possession = 0  -- 4227
select * from match where league_id in (1729, 7809, 10257, 21518) and home_team_possession = 0
select * from match where league_id in (1729, 7809, 10257, 21518) and home_team_possession < 30

-- AVG(home_team_possession)
select m.league_id, l.name, m.season, m.home_team_api_id, t.team_long_name, 
        avg(home_team_goal) as home_team_goal_avg, avg(away_team_goal) as away_team_goal_avg, avg(home_team_goal - away_team_goal)  as goal_difference_avg, avg(home_team_possession) as home_team_possession_avg
from match m, team t, league l 
where league_id in (1729, 7809, 10257, 21518) and home_team_possession > 0 and m.home_team_api_id = t.team_api_id and m.league_id = l.id
group by m.country_id, m.league_id, m.season, m.home_team_api_id

-- THE AVG QUERY
select league_id, season, home_team_api_id, 
        avg(home_team_goal) as home_team_goal_avg, avg(away_team_goal) as away_team_goal_avg, avg(home_team_goal - away_team_goal)  as goal_difference_avg, avg(home_team_possession) as home_team_possession_avg
from match
where league_id in (1729, 7809, 10257, 21518) and home_team_possession > 0
group by country_id, league_id, season, home_team_api_id

select id, country_id, league_id, season, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_team_possession
from match where home_team_api_id = 8455 and season = "2015/2016"
order by league_id, season, home_team_api_id -- CHELSEA games in 2015/2016

select id, country_id, league_id, season, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_team_possession
from match where home_team_api_id = 8667 and season = "2008/2009"
order by league_id, season, home_team_api_id -- team with matches with ZERO possession

select league_id, season, home_team_api_id, 
        avg(home_team_goal) as home_team_goal_avg, avg(away_team_goal) as away_team_goal_avg, avg(home_team_goal - away_team_goal)  as goal_difference_avg, avg(home_team_possession) as home_team_possession_avg
from match
where league_id in (1729, 7809, 10257, 21518) and home_team_api_id = 8667 and home_team_possession > 0
group by country_id, league_id, season, home_team_api_id

