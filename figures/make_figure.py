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
path = "latent_dimension_analysis/validation/"
csv_files = [
    "autoencoder-1k-linear-32Batch-8LAT_validation.csv",
    "autoencoder-1k-linear-32Batch-16LAT_validation.csv",
    "autoencoder-1k-linear-32Batch-32LAT_validation.csv",
    "autoencoder-1k-linear-32Batch-64LAT_validation.csv",
]
labels = [8, 16, 32, 64]

figure_name = "latent_dimension_analysis/new-autoencoder-1k-linear-32Batch-16LAT_validation.pdf"

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
    ax.legend(title="Latent Dimension")
    ax.title.set_text("Validation Loss Function vs Step")
    ax.autoscale(tight=True)
    ax.set(**pparam)
    fig.savefig(figure_name, dpi=300)
    plt.close()
