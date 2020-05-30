import sqlite3
import pandas as pd
import xml.etree.ElementTree as ET
import torch

DB_FILENAME = "soccer.sqlite"
CSV_FILENAME = "soccer.csv"
LEAGUE_ID = 1729


def get_dataframe(db_filename, league_id):
    """
    Query the DB and return a Pandas dataframe with all matches data related to league_id.

    :param db_filename: DB filename to read from
    :param league_id: league ID for wich we need matches data
    :return: a Pandas dataframe with all matches data related to league_id
    """
    conn = sqlite3.connect(db_filename)
    soccer_df = pd.read_sql("select id, country_id, league_id, season, stage, date, match_api_id, home_team_api_id, "
                            "away_team_api_id, home_team_goal, away_team_goal, goal, shoton, shotoff, possession "
                            "from match where league_id = {}".format(league_id), conn)
    conn.close()
    return soccer_df


def save_data_to_csv(soccer_df, csv_filename):
    """
    Saves relevant data in the dataframe to a CSV file.

    :param soccer_df: dataframe
    :param csv_filename: name of the CSV file to write to
    :return: None
    """
    print("\nlen(soccer_df): %s " % len(soccer_df))
    print("\nsoccer_df: %s " % soccer_df)
    possession_elapsed_array = [0] * len(soccer_df)
    possession_home_array = [0] * len(soccer_df)
    possession_away_array = [0] * len(soccer_df)
    goal_diff_array = [0] * len(soccer_df)

    for row_number in range(len(soccer_df)):
        # process possession value
        possession_tree = ET.fromstring(soccer_df.loc[row_number, "possession"])
        number_of_values = len(list(possession_tree))

        for child_number, child in enumerate(possession_tree):
            if child_number == number_of_values - 1:
                if child.find("elapsed") is not None \
                        and child.find("homepos") is not None and child.find("awaypos") is not None:
                    #print("row: %s - elapsed: %s - homepos: %s - awaypos: %s" %
                    #      (row_number, child.find("elapsed").text, child.find("homepos").text,
                    #       child.find("awaypos").text))
                    possession_elapsed_array[row_number] = child.find("elapsed").text
                    possession_home_array[row_number] = child.find("homepos").text
                    possession_away_array[row_number] = child.find("awaypos").text

        # process score
        home_team_goal = soccer_df.loc[row_number, "home_team_goal"]
        away_team_goal = soccer_df.loc[row_number, "away_team_goal"]
        goal_diff_array[row_number] = home_team_goal - away_team_goal

    soccer_df["possession_elapsed"] = possession_elapsed_array
    soccer_df["possession_home"] = possession_home_array
    soccer_df["possession_away"] = possession_away_array
    soccer_df["goal_diff"] = goal_diff_array

    soccer_df.to_csv(csv_filename, index=False,
                     columns=["id", "country_id", "league_id", "season", "stage", "date", "match_api_id",
                              "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal",
                              "possession_elapsed", "possession_home", "possession_away", "goal_diff"])


def read_csv_enhanced(filename, columns):
    """
    Read CSV file.

    :param filename: name of the file to read
    :param columns: name of columns to read
    :return: two tensors, the second one contains the first column data, the first one all the other columns
    """
    print("\nREADING CSV %s ..." % filename)
    data = pd.read_csv(filename)
    y = torch.tensor(data[columns[0]])
    x = torch.empty(len(data[columns[0]]), len(columns) - 1)
    for n, column in enumerate(columns[1:]):
        print("column %s '%s'" % (n, column))
        x[:,n] = torch.tensor(data[column])
    return x, y


def create_train_test_sets(x_tensor, y_tensor):
    """
    Create randomized training set with 80% of records and test set with 20% of the records.

    :param x_tensor: complete X tensor
    :param y_tensor: complete Y tensor
    :return: training and test sets
    """
    print("\nCREATING TEST SETS ...")
    # create random permutation if integers from 0 to remaining_size and calculate index where training set ends
    random_perm = torch.randperm(x_tensor.shape[0])
    train_percent_index = (x_tensor.shape[0] * 80) // 100
    print("\nrandom_perm (%s): %s" % (random_perm.shape, random_perm))
    print("train_percent_index: %s" % train_percent_index)

    # create randomized training sets
    x_train_tensor = x_tensor[random_perm[:train_percent_index]]
    y_train_tensor = y_tensor[random_perm[:train_percent_index]]
    print("\nx_train_tensor (%s): %s" % (x_train_tensor.shape, x_train_tensor))
    print("y_train_tensor (%s): %s" % (y_train_tensor.shape, y_train_tensor))

    # create randomized testing sets
    x_test_tensor = x_tensor[random_perm[train_percent_index:]]
    y_test_tensor = y_tensor[random_perm[train_percent_index:]]
    print("\nx_test_tensor (%s): %s ..." % (x_test_tensor.shape, x_test_tensor[:10]))
    print("y_test_tensor (%s): %s ..." % (y_test_tensor.shape, y_test_tensor[:10]))

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor


def main():
    """
    Main method.

    :return: None
    """
    soccer_df = get_dataframe(DB_FILENAME, LEAGUE_ID)
    save_data_to_csv(soccer_df, CSV_FILENAME)

    x_tensor, y_tensor = read_csv_enhanced(CSV_FILENAME, ("goal_diff", "possession_home"))
    print("x_tensor(%s): %s" % (x_tensor.size(), x_tensor))
    print("y_tensor(%s): %s" % (y_tensor.size(), y_tensor))

    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = create_train_test_sets(x_tensor, y_tensor)

if __name__ == "__main__":
    main()
