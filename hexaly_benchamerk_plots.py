import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_data_profiles(file_path, success_criterion="gap",tau=0.01,upper=1.0):
    """
    The x-axis represents a dimensionless quality.
    The y-axis represents the fraction of problems successfully solved for a given quality.
    """
    plt.figure(figsize=(10, 6))  
    tau_range = np.arange(0,upper,tau)

    if success_criterion == "gap":
        column_name='gap quality'
        x_name='Optimality gap'
    elif success_criterion == "objective":
        column_name='Obj quality'
        x_name='Dimensionless distance between the objective and the best known objective'
    elif success_criterion == "bound":
        column_name='Best bound quality'
        x_name='Dimensionless distance between the lower bound and the best known lower bound'
    elif success_criterion == "time":
        column_name='time quality'
        x_name='Dimensionless time'
    else:
        raise ValueError("Invalid success_criterion! Choose from 'time','gap', 'objective', or 'bound'.") 
    
    # Dictionary to store performance data for each method
    performance_data = {}

    # Get all sheet names
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    model_names=['MIP','MInP(1)','MInLiP(1)','MInP(2)','MInLiP(2)']

    for k in range(len(model_names)):
        sheet=sheet_names[k]
        model=model_names[k]

        # Load the data
        df = pd.read_excel(xls, sheet_name=sheet)
        
        # Compute the fraction of successfully solved problems
        success_rates = []
        total_problems = len(df)
        
        for it in tau_range:
            success_count=((df[column_name]<=it)).sum()
            success_rate = success_count / total_problems
            success_rates.append(success_rate)
        
        # Store the results
        performance_data[model] = success_rates

        # Plot the performance profile for this method
        plt.plot(tau_range, success_rates, label=model)

    # Formatting the plot
    plt.xlabel("Desired DQ")
    plt.ylabel("Fraction of Problems successfully solved:\nDQ<=Desired DQ")
    plt.title("Dimensionless quality (DQ):\n"+x_name)
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    file_name="original.xlsx"
    file_path = Path("./hexaly_benchmarking_results/"+file_name)

    plot_data_profiles(file_path, success_criterion="time",tau=1e-3,upper=1)

    plot_data_profiles(file_path, success_criterion="gap",tau=1e-3,upper=1)

    plot_data_profiles(file_path, success_criterion="objective",tau=1e-3,upper=1)

    plot_data_profiles(file_path, success_criterion="bound",tau=1e-3,upper=1)

    file_name="known_n.xlsx"
    file_path = Path("./hexaly_benchmarking_results/"+file_name)


    plot_data_profiles(file_path, success_criterion="time",tau=1e-3,upper=1)

    plot_data_profiles(file_path, success_criterion="gap",tau=1e-3,upper=1)

    plot_data_profiles(file_path, success_criterion="objective",tau=1e-3,upper=1)

    plot_data_profiles(file_path, success_criterion="objective",tau=1e-8,upper=1e-5)

    plot_data_profiles(file_path, success_criterion="bound",tau=1e-3,upper=1)
