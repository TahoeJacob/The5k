import matplotlib.pyplot as plt
import pandas as pd

# Import formatted data in excel format. Use this to complete data analaysis


# Import excel file
df = pd.read_excel('/Users/jacobsaunders/Desktop/EngineDev/The5k/CEAParsed.xlsx')


def create_plot(x_axis,y_axis, x_label, y_label, title):
    plt.figure()  # Create a new figure
    plt.plot(x_axis, y_axis)
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    plt.title(f'{title}')
    plt.grid(True)
    
def calc_exit_mach_num(df):
    # Calculate the mach number at the exit of the nozzle
    
    # Use the following equation to calculate the exit mach number
    # M_e = sqrt(2/(gamma - 1) * ((P_c/P_e)^((gamma - 1)/gamma) - 1))

def engine_geometry(df):
    # Calculate the geometry of a rocket engine based off key parameters from CEA

    # First step is to calculate mach numbers in the exit


def main():
    # Main function

    # Calculate the engine geometry based off CEA data
    engine_geometry(df)

    # result = (df['CF'] * df['Cstar']) / 9.81
    # create_plot(df['O/F'], df['T_Chamber'], 'O/F', 'T Chamber', 'T Chamber vs O/F')
    # create_plot(df['O/F'], df['Gamma_Chamber'], 'O/F', 'Gamma Chamber', 'Gamma Chamber vs O/F')
    # create_plot(df['O/F'], df['Cstar'], 'O/F', 'CStar', 'CStar vs O/F')
    # create_plot(df['O/F'], df['Isp'], 'O/F', 'Isp', 'Isp vs O/F')
    # create_plot(df['O/F'], result, 'O/F', 'Isp [Sec]', 'Isp vs O/F')
    # plt.show()

main()