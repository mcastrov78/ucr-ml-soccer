import sqlite3
import pandas as pd
import xml.etree.ElementTree as ET


def get_dataframe(league):
    conn = sqlite3.connect("soccer.sqlite")
    soccer_df = pd.read_sql("select * from match where league_id = {}".format(league), conn)
    conn.close()
    return soccer_df


def save_data_to_csv(soccer_df):
    print("len(soccer_df): %s " % len(soccer_df))
    print("soccer_df: %s " % soccer_df)
    possession_elapsed_array = [0] * len(soccer_df)
    possession_home_array = [0] * len(soccer_df)
    possession_away_array = [0] * len(soccer_df)

    for row_number, row in enumerate(soccer_df["possession"]):
        possession_tree = ET.fromstring(row)
        number_of_values = len(list(possession_tree))

        for child_number, child in enumerate(possession_tree):
            if child_number == number_of_values - 1:
                if child.find("elapsed") is not None and child.find("homepos") is not None and child.find("awaypos") is not None:
                    print("row: %s - elapsed: %s - homepos: %s - awaypos: %s" %
                          (row_number, child.find("elapsed").text, child.find("homepos").text, child.find("awaypos").text))
                    possession_elapsed_array[row_number] = child.find("elapsed").text
                    possession_home_array[row_number] = child.find("homepos").text
                    possession_away_array[row_number] = child.find("awaypos").text

    soccer_df["possession_elapsed"] = possession_elapsed_array
    soccer_df["possession_home"] = possession_home_array
    soccer_df["possession_away"] = possession_away_array

    soccer_df.to_csv("soccer.csv", index=False,
                     columns=["id", "country_id", "league_id", "season", "stage", "date", "match_api_id",
                              "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal",
                              "possession_elapsed", "possession_home", "possession_away"])


def main():
    soccer_df = get_dataframe(1729)
    save_data_to_csv(soccer_df)


if __name__ == "__main__":
    main()
