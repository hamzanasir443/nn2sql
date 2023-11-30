import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


numpy_data = pd.read_csv('numpy_nn.csv', skipinitialspace=True)

# Print the first few rows to verify the data
print(numpy_data.head())

# Print the column names
print(numpy_data.columns)

numpy_pdf_path = 'numpy_error_rates.pdf'
with PdfPages(numpy_pdf_path) as numpy_pdf:
    plt.figure(figsize=(10, 6))
    # Box plot for each unique value in 'atts' column
    plt.boxplot([numpy_data[numpy_data['atts'] == atts]['error'] for atts in numpy_data['atts'].unique()], labels=numpy_data['atts'].unique())
    plt.xlabel('Number of Attributes (NumPy)')
    plt.ylabel('Error Rate')
    plt.title('Box Plot of Error Rates for Different Attribute Counts (NumPy Implementation)')
    plt.grid(True)
    numpy_pdf.savefig()
    plt.close()

print(f"Box plot saved as {numpy_pdf_path}")