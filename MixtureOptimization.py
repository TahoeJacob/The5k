import matplotlib.pyplot as plt
from FlowModel import df


def plot_y_axis(x_axis,y_axis):
    plt.figure()  # Create a new figure
    plt.plot(x_axis, y_axis)
    plt.xlabel(f'{x_axis}')
    plt.ylabel(f'{y_axis}')
    plt.title(f'{y_axis} vs {x_axis}')
    plt.grid(True)
    

def main():
    result = (df['CF'] * df['Cstar']) / 9.81
    print(result)
    plot_y_axis(df['O/F'],df['Cstar'])
    plot_y_axis(df['O/F'],df['CF'])
    plot_y_axis(df['O/F'],result)
    plot_y_axis(df['O/F'],df['Isp'])
    plt.show()

main()