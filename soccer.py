import sqlite3
import pandas as pd
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

DB_FILENAME = "database.sqlite"         # European Soccer Database filename (https://www.kaggle.com/hugomathien/soccer)
CSV_FILENAME = "soccer.csv"             # output file for Approach 1 (only ELP)
CSV_AVG_FILENAME = "soccer_avg.csv"     # output file for Approach 2 (ELP, )

LEAGUE_ID = 1729                            # league IDs for Approach 1 (only ELP)
LEAGUES_IDS = (1729, 7809, 10257, 21518)    # league IDs for Approach 2 (ELP, Bundesliga, Italy Serie A, La Liga BBVA)


# ----------------- APPROACH 1: ANALYZE ONLY ONE LEAGUE -----------------

def get_dataframe(db_filename, league_id):
    """
    Query the DB and return a Pandas dataframe with all matches data related to league_id.

    :param db_filename: DB filename to read from
    :param league_id: league ID for which we need matches data
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
    shots_on_diff_array = [0] * len(soccer_df)
    goal_diff_array = [0] * len(soccer_df)

    for row_number in range(len(soccer_df)):
        # *** POSSESSION processing ***
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

        # *** SHOTS ON processing ***
        shoton_tree = ET.fromstring(soccer_df.loc[row_number, "shoton"])

        home_team_api_id = str(soccer_df.loc[row_number, "home_team_api_id"])
        away_team_api_id = str(soccer_df.loc[row_number, "away_team_api_id"])
        shots_on = {home_team_api_id: 0, away_team_api_id: 0}
        #print("\n-- home_team_api_id: %s - away_team_api_id: %s" % (home_team_api_id, away_team_api_id))

        for child_number, child in enumerate(shoton_tree):
            if child.find("team") is not None and child.find("id") is not None:
                #print("team: %s - event: %s" % (child.find("team").text, child.find("id").text))
                shots_on[child.find("team").text] += 1

        shots_on_diff_array[row_number] = shots_on[home_team_api_id] - shots_on[away_team_api_id]
        #shots_on_diff_array[row_number] = shots_on[home_team_api_id]
        #print("shots_on: %s - DIFF: %s" % (shots_on, shots_on_diff_array[row_number]))

        # *** SCORE processing ***
        home_team_goal = soccer_df.loc[row_number, "home_team_goal"]
        away_team_goal = soccer_df.loc[row_number, "away_team_goal"]
        goal_diff_array[row_number] = home_team_goal - away_team_goal

    # add NEW columns to dataframe before creating th CSV file
    soccer_df["possession_elapsed"] = possession_elapsed_array
    soccer_df["possession_home"] = possession_home_array
    soccer_df["possession_away"] = possession_away_array
    soccer_df["shots_on_diff"] = shots_on_diff_array
    soccer_df["goal_diff"] = goal_diff_array

    soccer_df.to_csv(csv_filename, index=False,
                     columns=["id", "country_id", "league_id", "season", "stage", "date", "match_api_id",
                              "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal",
                              "possession_elapsed", "possession_home", "possession_away", "shots_on_diff", "goal_diff"])


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
    #print("\nrandom_perm (%s): %s" % (random_perm.shape, random_perm))
    #print("train_percent_index: %s" % train_percent_index)

    # create randomized training sets
    x_train_tensor = x_tensor[random_perm[:train_percent_index]]
    y_train_tensor = y_tensor[random_perm[:train_percent_index]]
    #print("\nx_train_tensor (%s): %s" % (x_train_tensor.shape, x_train_tensor))
    #print("y_train_tensor (%s): %s" % (y_train_tensor.shape, y_train_tensor))

    # create randomized testing sets
    x_test_tensor = x_tensor[random_perm[train_percent_index:]]
    y_test_tensor = y_tensor[random_perm[train_percent_index:]]
    #print("\nx_test_tensor (%s): %s ..." % (x_test_tensor.shape, x_test_tensor[:10]))
    #print("y_test_tensor (%s): %s ..." % (y_test_tensor.shape, y_test_tensor[:10]))

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor


class TwoLayerNN(nn.Module):
    """
    Two layer NN.
    """

    def __init__(self, input_n, hidden_n, output_n, activation_fn):
        """
        Initialize NN.
        :param input_n: number of inputs
        :param hidden_n: number of hidden neurons
        :param output_n: number of outputs
        :param activation_fn: activation function for the hidden layer
        """
        super(TwoLayerNN, self).__init__()
        print("\n*** TwoLayerNN i: %s - h: %s - o: %s ***" % (input_n, hidden_n, output_n))
        self.hidden_linear = nn.Linear(input_n, hidden_n)
        self.hidden_activation = activation_fn
        self.output_linear = nn.Linear(hidden_n, output_n)

    def forward(self, input):
        """
        Pass the input through the NN layers.
        :param input: input to the module
        :return: output from the module
        """
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t


def train_nn(iterations, nn_model, optimizer, nn_loss_fn, x_tensor, y_tensor, input_n, output_n):
    """
    Train Neural Network.
    :param iterations: epochs
    :param nn_model: NN model
    :param optimizer: optimizer
    :param nn_loss_fn: loss function
    :param x_tensor: X tensor
    :param y_tensor: Y tensor
    :param input_n: number of inputs
    :param output_n: number of outputs
    :return:
    """
    print("\n*** TRAINING NN ***")
    #print("\nx_tensor (%s): %s" % (x_tensor.shape, x_tensor))
    x_tensor_reshaped = x_tensor.view(-1, input_n)
    #print("\nx_tensor_reshaped (%s): %s" % (x_tensor_reshaped.shape, x_tensor_reshaped))

    for it in range(1, iterations + 1):
        y_tensor_pred = nn_model(x_tensor_reshaped)
        y_tensor_pred_reshaped = y_tensor_pred.view(-1, output_n)
        loss = nn_loss_fn(y_tensor.view(-1, output_n), y_tensor_pred_reshaped)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 100 == 0: print("N: %s\t | Loss: %f\t" % (it, loss))


def plot_data_set_and_function(x_tensor, y_tensor, fn_x_tensor=None, fn_y_tensor=None, x_label="X", y_label="Y", fig_name="Figure"):
    """
    Plot a given test set.

    :param x_tensor: tensor X
    :param y_tensor: tensor y
    :param fn_x_tensor: tensor x for function line plotting
    :param fn_y_tensor: tensor y for function line plotting
    :param y_label: X-axis label
    :param x_label: X-axis label
    :param fig_name: figure name
    """
    print("\n*** plot_data_set_and_model: %s ***" % fig_name)
    plt.figure(num=fig_name)
    plt.scatter(x_tensor, y_tensor)
    if fn_y_tensor is not None:
        x_tensor = x_tensor
        if fn_x_tensor is not None:
            x_tensor = fn_x_tensor
        plt.plot(x_tensor, fn_y_tensor, color="red")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def nn_plot_test_set(x_tensor, y_tensor, nn_model, set_name):
    """
    Scatterplot test set X and Y values and the function graph.

    :param x_tensor: X tensor
    :param y_tensor: Y tensor
    :param nn_model: NN model
    :param set_name: test set name for displaying purposes
    :return: nothing
    """
    fn_x_tensor = torch.tensor(np.linspace(x_tensor.min(), x_tensor.max(), 1000), dtype=torch.float)
    fn_y_tensor = nn_model(fn_x_tensor.view(-1, 1))
    plot_data_set_and_function(x_tensor, y_tensor, fn_x_tensor, fn_y_tensor.detach().numpy(), fig_name=set_name)


def print_statistics(x_tensor, y_tensor):
    """
    Prints some info and  statistics.

    :param x_tensor: X tensor
    :param y_tensor: Y tensor
    :return: None
    """
    print("\nMax Pos: %s - GD: %s" % (torch.max(x_tensor), torch.max(y_tensor.type(torch.FloatTensor))))
    print("Min Pos: %s - GD: %s" % (torch.min(x_tensor), torch.min(y_tensor.type(torch.FloatTensor))))
    print("Mean Pos: %s - GD: %s" % (torch.mean(x_tensor), torch.mean(y_tensor.type(torch.FloatTensor))))
    print("StdDev Pos: %s - GD: %s" % (torch.std(x_tensor), torch.std(y_tensor.type(torch.FloatTensor))))
    print("Corr Pos-GD: %s" % (np.corrcoef(x_tensor, y_tensor.type(torch.FloatTensor))))


def train_and_test_linear_possession(x_tensor, y_tensor):
    """
    Train and test LINEAR model.

    :param x_tensor: X tensor
    :param y_tensor: Y tensor
    :return: None
    """
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = create_train_test_sets(x_tensor, y_tensor)

    model = nn.Linear(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    nn_loss_fn = nn.MSELoss()
    print("\nmodel: %s" % model)
    train_nn(3000, model, optimizer, nn_loss_fn, x_train_tensor, y_train_tensor, 1, 1)

    # model each set and calculate loss
    y_predic = model(x_train_tensor.view(-1, 1))
    model_loss = nn_loss_fn(y_predic, y_train_tensor.view(-1, 1))
    print("\nLOSS for Train Set (train_and_test_nn_possession): %s " % model_loss)
    nn_plot_test_set(x_train_tensor, y_train_tensor, model, "LINEAR - TRAIN SET")

    y_predic = model(x_test_tensor.view(-1, 1))
    model_loss = nn_loss_fn(y_predic, y_test_tensor.view(-1, 1))
    print("\nLOSS for Test Set (train_and_test_nn_possession): %s " % model_loss)
    nn_plot_test_set(x_test_tensor, y_test_tensor, model, "LINEAR - TEST SET")


def train_and_test_nn_possession(x_tensor, y_tensor):
    """
    Train and test NEURAL NETWORK with linear model.

    :param x_tensor: X tensor
    :param y_tensor: Y tensor
    :return: None
    """
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = create_train_test_sets(x_tensor, y_tensor)

    model = TwoLayerNN(1, 20, 1, nn.ReLU())
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    print("\nmodel: %s" % model)
    train_nn(3000, model, optimizer, loss_fn, x_train_tensor, y_train_tensor, 1, 1)

    # model each set and calculate loss
    y_predic = model(x_train_tensor.view(-1, 1))
    model_loss = loss_fn(y_predic, y_train_tensor.view(-1, 1))
    print("\nLOSS for Train Set (train_and_test_nn_possesion): %s " % model_loss)
    nn_plot_test_set(x_train_tensor, y_train_tensor, model, "NN1 - TRAIN SET")

    y_predic = model(x_test_tensor.view(-1, 1))
    model_loss = loss_fn(y_predic, y_test_tensor.view(-1, 1))
    print("\nLOSS for Test Set (train_and_test_nn_possesion): %s " % model_loss)
    nn_plot_test_set(x_test_tensor, y_test_tensor, model, "NN1 - TEST SET")


def train_and_test_nn_multi():
    """
    Train and test neural network with linear model using multiple features.

    :return: None
    """

    # get dataframe from DB and save relevant data to a CSV file
    soccer_df = get_dataframe(DB_FILENAME, LEAGUE_ID)
    save_data_to_csv(soccer_df, CSV_FILENAME)

    # predict based on a single feature: possession
    x_tensor, y_tensor = read_csv_enhanced(CSV_FILENAME, ("goal_diff", "possession_home", "shots_on_diff"))
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = create_train_test_sets(x_tensor, y_tensor)

    model = TwoLayerNN(2, 20, 1, nn.ReLU())
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    print("\nmodel: %s" % model)
    train_nn(3000, model, optimizer, loss_fn, x_train_tensor, y_train_tensor, 2, 1)

    # model each set and calculate loss
    y_predic = model(x_train_tensor.view(-1, 2))
    model_loss = loss_fn(y_predic, y_train_tensor.view(-1, 1))
    print("\nLOSS for Train Set (train_and_test_nn_multi): %s " % model_loss)
    # nn_plot_test_test(x_train_tensor, y_train_tensor, model, "NN2 - TRAIN SET")

    y_predic = model(x_test_tensor.view(-1, 2))
    model_loss = loss_fn(y_predic, y_test_tensor.view(-1, 1))
    print("\nLOSS for Train Set (train_and_test_nn_multi): %s " % model_loss)
    # nn_plot_test_test(x_test_tensor, y_test_tensor, model, "NN2 - TEST SET")


def analyze_one_league():
    """
    First version of the project. Analyze results in only one league.

    :return: None.
    """
    # get dataframe from DB and save relevant data to a CSV file
    soccer_df = get_dataframe(DB_FILENAME, LEAGUE_ID)
    save_data_to_csv(soccer_df, CSV_FILENAME)

    # predict based on a single feature: possession
    x_tensor, y_tensor = read_csv_enhanced(CSV_FILENAME, ("goal_diff", "possession_home"))
    x_tensor = x_tensor.view(-1)
    print_statistics(x_tensor, y_tensor)

    # NOTE: we implemented several approaches, but for the final version we only care about the NN with one feature
    #train_and_test_linear_possession(x_tensor, y_tensor)
    train_and_test_nn_possession(x_tensor, y_tensor)
    #train_and_test_nn_multi()


# ----------------- APPROACH 2: ANALYZE AVERAGES FOR 4 LEAGUES -----------------

def add_columns_to_db(db_filename):
    """
    Add new columns to the database that are required for further analysis.

    :param db_filename: DB filename to read from
    :return: None.
    """
    conn = sqlite3.connect(db_filename)

    # alter table to add new home_team_possession column, if not already there
    try:
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE match ADD home_team_possession INTEGER NULL")
        conn.commit()
        print("home_team_possession was added successfully!!!")
    except sqlite3.OperationalError:
        print("home_team_possession was added previously ...")

    conn.close()


def populate_new_columns(db_filename):
    """
    Populate new columns that are required for further analysis.

    :param db_filename: DB filename to read from
    :return: None.
    """
    conn = sqlite3.connect(db_filename)

    # select all games in the chosen leagues
    select_statement = """select id, league_id, season, home_team_api_id, home_team_goal, away_team_goal, (home_team_goal - away_team_goal) as GD, possession
                from match where league_id in {} order by league_id, season, home_team_api_id"""
    results = conn.execute(select_statement.format(LEAGUES_IDS))

    # update selected rows to include the extracted home_team_possession value
    current_row = results.fetchone()
    row_count = 0
    while current_row is not None:
        print(current_row)
        # set home_team_possession to zero where possession is null, get real value where available
        home_team_possession_xml = current_row[7]
        home_team_possession = 0
        if home_team_possession_xml is not None:
            home_team_possession = get_match_home_possession(current_row[7])

        update_statement = "UPDATE match SET home_team_possession = {} WHERE id = {}"
        conn.execute(update_statement.format(home_team_possession, current_row[0]))
        conn.commit()

        current_row = results.fetchone()
        row_count += 1

    print("Number of rows: %s" % row_count)
    conn.close()


def get_match_home_possession(possession_xml):
    """
    Extract home possession value from original XML string provided.

    :param possession_xml:
    :return: home possession value as in integer
    """
    home_team_possession = 0
    possession_tree = ET.fromstring(possession_xml)
    number_of_values = len(list(possession_tree))

    for child_number, child in enumerate(possession_tree):
        if child_number == number_of_values - 1:
            if child.find("elapsed") is not None \
                    and child.find("homepos") is not None and child.find("awaypos") is not None:
                home_team_possession = child.find("homepos").text

    return home_team_possession


def get_averages_dataframe(db_filename, leagues_ids):
    """
    Query the DB and return a Pandas dataframe with all data related to leagues_ids.

    :param db_filename: DB filename to read from
    :param leagues_ids: leagues IDs for which we need matches data
    :return: a Pandas dataframe with all data related to leagues_ids
    """
    conn = sqlite3.connect(db_filename)
    soccer_df = pd.read_sql("select league_id, season, home_team_api_id, avg(home_team_goal) as home_team_goal_avg, "
                            "avg(away_team_goal) as away_team_goal_avg, avg(home_team_goal - away_team_goal)  as goal_difference_avg, "
                            "avg(home_team_possession) as home_team_possession_avg "
                            "from match where league_id in {} and home_team_possession > 0 "
                            "group by country_id, league_id, season, home_team_api_id".format(leagues_ids), conn)
    conn.close()
    return soccer_df


def analyze_leagues_averages():
    """
    Second version of the project. Analyze results in several leagues.

    :return: None.
    """
    #add_columns_to_db(DB_FILENAME)
    #populate_new_columns(DB_FILENAME)
    soccer_df = get_averages_dataframe(DB_FILENAME, LEAGUES_IDS)

    print("\nlen(soccer_df): %s " % len(soccer_df))
    print("\nsoccer_df: %s " % soccer_df)

    soccer_df.to_csv(CSV_AVG_FILENAME, index=False,
                     columns=["league_id", "season", "home_team_api_id", "home_team_goal_avg", "away_team_goal_avg",
                              "goal_difference_avg", "home_team_possession_avg"])

    # predict based on a single feature: possession
    x_tensor, y_tensor = read_csv_enhanced(CSV_AVG_FILENAME, ("goal_difference_avg", "home_team_possession_avg"))
    x_tensor = x_tensor.view(-1)
    print_statistics(x_tensor, y_tensor)

    train_and_test_nn_possession(x_tensor, y_tensor)


# ----------------- ANALYZE AVERAGES -----------------

def main():
    """
    Main method.

    :return: None
    """
    analyze_one_league()
    analyze_leagues_averages()


if __name__ == "__main__":
    main()
