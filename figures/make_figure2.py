import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

tol_colors = [
    "#332288",  # azul escuro
    "#117733",  # verde escuro
    "#44AA99",  # verde água
    "#88CCEE",  # azul claro
    "#DDCC77",  # amarelo
    "#CC6677",  # rosa
    "#AA4499",  # roxo
]

# List of CSV files to read
path = "overfit/"
csv_files = [
    "v2-autoencoder-0.01k-32Lat-linear-64Batch_validation.csv",
    "v2-autoencoder-0.01k-32Lat-linear-64Batch_train.csv",
]

labels = ["Validation", "Train"]

figure_name = "overfit/v2-autoencoder-0.01k-32Lat-linear-64Batch_train.pdf"

# Initialize a dictionary to store data
data = {}

# Read data from each CSV file
for file in csv_files:
    file_data = pd.read_csv(path + file)
    
    # remove first column
    file_data = file_data.iloc[:, 1:]
    
    # Rename columns for clarity
    file_data.columns = ['step', 'value']
    
    # remove first row
    file_data = file_data.iloc[1:]

    # Store the DataFrame in the dictionary
    data[file] = file_data


plt.figure()


plt.figure()


pparam = {
    "xlabel": "Step", 
    "ylabel": "Loss Function",
}

with plt.style.context(["science", "ieee"]):
    fig, ax = plt.subplots()
    for file, df in data.items():
        print("Plotting data from:", file)
        l = labels.pop(0)
        ax.plot(df['step'], df['value'], label=f"{l}")
    ax.legend(title="Dataset")
    ax.autoscale(tight=True)
    ax.set(**pparam)
    fig.savefig(figure_name, dpi=300)
    plt.close()
