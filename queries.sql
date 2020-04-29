-- GENERAL
select * from country -- EPL 1729

select * from league -- EPL 1729

select count(*) from match
select * from match

select count(*) from player
select * from player
select * from player where player_name like "%ronaldo%" --30893

select count(*) from player_attributes
select * from player_attributes
select * from player_attributes where player_api_id = 30894

select count(*) from team
select * from team
select * from team where team_long_name like "%juventus%" --9885

select count(*) from team_attributes
select * from team_attributes
select * from team_attributes where team_api_id = 9885


-- EPL
select count(*) from match where country_id = 1729 -- 3040 games
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
select count(*) from match where country_id = 1729 and possession is not null and length(trim(possession)) > 0 -- 3040 games
select possession from match where country_id = 1729 -- 3040 games
select possession from match where country_id = 1729 -- 3040 games
select id, country_id, season, possession from match where country_id = 1729 limit 100

select count(*) from match where country_id = 1729 and possession like "%<elapsed>90</elapsed>%" --2244
select possession from match where country_id = 1729 and possession like "%<elapsed>90</elapsed>%"

select count(*) from match where country_id = 1729 and possession NOT like "%<elapsed>90</elapsed>%" --796
select possession from match where country_id = 1729 and possession NOT like "%<elapsed>90</elapsed>%" --796
