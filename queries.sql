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
