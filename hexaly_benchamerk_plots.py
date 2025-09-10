import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def is_real_numeric(value):
    return np.issubdtype(type(value), np.number) and (not np.iscomplexobj(value)) and (np.isfinite(value))

def performance_plots(# file info
                        file_paths: list[str],
                        sheet_names: list[str],
                       alg_names: list[str],
                       best_obj_header: str,
                       time_header: str,
                       bound_header: str,
                       gap_header: str,
                       # definition of success is assumed to be in the file
                       success_header: str,
                       success_input: list,    

                        # performance plot to generate
                       success_criterion: str="objective",
                       tau: float=0.01,
                       dimensionless_x_axis=False
                       ):


    # 1. Read files
    dfs=[]
    for indx,file in enumerate(file_paths):
        # Read file
        df=pd.read_excel(file,sheet_name=sheet_names[indx])
        # add success column based on input
        df['__success_header__'] = np.where(df[success_header].isin(success_input), 'ok', 'fail')

        # Add file to dictionary
        dfs.append(df)



    # 2. define target columns based on success_criterion
    if success_criterion == "time":
        column_name=time_header
        x_name='Time [s]'
        more_is_better=False
    elif success_criterion== "objective":
        column_name=best_obj_header
        x_name='Objective'
        more_is_better=False
    elif success_criterion== "bound":
        column_name=bound_header
        x_name='Obj. lower bound'
        more_is_better=True
    elif success_criterion== "gap":
        column_name=gap_header
        x_name='Obj. gap [%]'
        more_is_better=False
    elif success_criterion=="objective-distance":
        column_name=success_criterion
        x_name='Obj. - Best obj.'
        more_is_better=False
    elif success_criterion=="bound-distance":
        column_name=success_criterion
        x_name='Best bound - Bound'
        more_is_better=False
    else:
        raise ValueError("Invalid success_criterion") 

    # Adding new headers for bound distance and objective distance
    if success_criterion=="bound-distance" or success_criterion=="objective-distance":

        # Stack the column from each df into a 2D array
        if success_criterion=="objective-distance":
            #  use np.inf for failed rows
            column_matrix = np.array([
                np.where(
                    (df['__success_header__'] == 'fail') | (~df[best_obj_header].apply(is_real_numeric)),
                    np.inf,
                    df[best_obj_header].values
                )
                for df in dfs
            ])

            row_min = np.min(column_matrix, axis=0)


            for df in dfs:
                df[column_name] = df[best_obj_header]- row_min

        else:
            #  use np.inf for failed rows
            column_matrix = np.array([
                np.where(
                    (df['__success_header__'] == 'fail') | (~df[bound_header].apply(is_real_numeric)),
                    -np.inf,
                    df[bound_header].values
                )
                for df in dfs
            ])
            
            row_max = np.max(column_matrix, axis=0)
            # print(row_max)

            for df in dfs:
                df[column_name] = row_max - df[bound_header]


    #3. Find minimum and maximum

    # Filter for succesful runs
    filter_success_for_min_max = lambda df: df[(df['__success_header__'] == 'ok') & df[column_name].apply(is_real_numeric)][column_name]


    min_candidates = [filter_success_for_min_max(df).min() for df in dfs]
    max_candidates = [filter_success_for_min_max(df).max() for df in dfs]

    min_value = min(min_candidates, default=0)
    max_value = max(max_candidates, default=1)

    if not min_candidates:
        dimensionless_x_axis=True



    # 4. Define dimensionless success value
    for df in dfs:
        df['__dimensionless__']=np.where(df['__success_header__'] == 'ok',(df[column_name]-min_value)/(max_value-min_value),-np.inf if more_is_better else np.inf)

    # 5. Compute the fraction of successfully solved problems and plot

    plt.figure(figsize=(10, 6))  
    tau_range = np.arange(0, 1, tau)
    if 1 not in tau_range:
        tau_range = np.append(tau_range, 1)

    for index,name in enumerate(alg_names):
        df=dfs[index]
        success_rates = []
        x_axes_val=[]
        total_problems = len(df)
        
        for it in tau_range:
            if more_is_better:
                condition=df['__dimensionless__']>=it
            else:
                condition=df['__dimensionless__']<=it
            success_count=(condition).sum()
            success_rate = success_count / total_problems
            success_rates.append(success_rate)
            x_axes_val.append(min_value+it*(max_value-min_value))
        

        # Plot the performance profile for this method
        if dimensionless_x_axis:
            plt.plot(tau_range, success_rates, label=name)
            add='Dimensionless'
        
        else:
            plt.plot(x_axes_val, success_rates, label=name)
            add=''

    # Formatting the plot
    if more_is_better:
        sign='>='
    else:
        sign='<='

    if len(success_input)>1:
        success_str = ' or '.join(str(s) for s in success_input)
    else:
        success_str = str(success_input[0])


    plt.xlabel(f"x: {add} {x_name}")
    plt.ylabel(
        f"Fraction of problems with: \n"
        f"{success_header} = {success_str}\n"
        f"{add} {x_name} {sign} x"
    )
    plt.legend()
    # plt.xscale("log")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':


    # ---Fixed configuration---
    sheet_names=['Formulation_1','Formulation_2','Formulation_3','Formulation_4','Formulation_5']
    alg_names=['MIP','MInP(1)','MInLiP(1)','MInP(2)','MInLiP(2)']
    best_obj_headers='Objective'
    time_header='Time [s]'
    bound_header='Obj. bound'
    gap_header='Obj. gap'
    success_header='Status'
    success_input=['HxSolutionStatus.OPTIMAL','HxSolutionStatus.FEASIBLE']
    tau=0.001
    dimensionless_x_axis=False

    # ---Original formulation---

    file_name="original.xlsx"
    file_path = Path("./hexaly_benchmarking_results/"+file_name)
    file_paths=[str(file_path)]*5

    # obj distance
    success_criterion="objective-distance"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)

    # bound distance
    success_criterion="bound-distance"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)

    # time
    success_criterion="time"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)

    # gap
    success_criterion="gap"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)




    # ---Simplified formulation---

    file_name="known_n.xlsx"
    file_path = Path("./hexaly_benchmarking_results/"+file_name)
    file_paths=[str(file_path)]*5


    # obj distance
    success_criterion="objective-distance"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)

    # bound distance
    success_criterion="bound-distance"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)

    # time
    success_criterion="time"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)

    # gap
    success_criterion="gap"
    performance_plots(file_paths,sheet_names,alg_names,best_obj_headers,time_header,bound_header,gap_header,success_header,success_input,success_criterion,tau=tau,dimensionless_x_axis=dimensionless_x_axis)
