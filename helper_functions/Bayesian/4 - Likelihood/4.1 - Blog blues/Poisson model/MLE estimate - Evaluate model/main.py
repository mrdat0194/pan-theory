import pandas as pd
from poisson_model import *
from helper_functions import plotter
from Bayesian import DATA_DIR
import os

def loadData(data_path):

    data_frame = pd.read_csv(data_path)

    data_values = reshape_data(data_frame)

    return data_values


def reshape_data(data_frame):

    data_values = data_frame.values.reshape(-1)

    return data_values


def main():
    """ Load and pre-process data. """
    data_path = os.path.join(DATA_DIR,"likelihood_blogVisits.csv")
    data_path = (data_path)
    time_between_visits = loadData(data_path)

    """ Solution to question 4.1.4 """
    MLE_first_time_visit_rate = compute_MLE_mean_time_between_visits(time_between_visits)

    print("The MLE of the first time visit rate for a blog, using 50 data points.")
    print("MLE estimate for lambda: ", MLE_first_time_visit_rate)


    """Generating data using the MLE parameter for the visit rate to evaluate our model"""
    generated_beer_visits = generate_data_samples(1/MLE_first_time_visit_rate, len(time_between_visits))

    # Plotting the generated data against the actual data
    plotter(generated_beer_visits, time_between_visits)



if __name__ == '__main__':
    main()
