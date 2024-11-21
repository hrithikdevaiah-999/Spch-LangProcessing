import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_into_matrix(csv_path):
    
    df = pd.read_csv(csv_path)
    
    Y_axis = df["recipe_train_dataset"]
    X_axis = df.columns[2:]
    
    df = df.drop(columns=["recipe_train_dataset","datetime"])
    
    print(f"Matrix for Y_axis = {Y_axis} and X_axis = {X_axis}")
    
    matrix = df.to_numpy()
    
    return matrix, Y_axis, X_axis
    
def plot_perplexity_matrix(matrix, Y_axis, X_axis):
    ax = plt.gca()
    
    # Plot the matrix
    ax.imshow(matrix, aspect='equal', cmap='viridis')
     
    # Set axis labels
    ax.set_xticks(np.arange(len(X_axis)))
    ax.set_xticklabels(X_axis, ha='center')
    ax.set_yticks(np.arange(len(Y_axis)))
    ax.set_yticklabels(Y_axis)
    
    # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='white' if matrix[i, j] < matrix.max() / 2 else 'black')
    

    plt.ylabel('Text Category for model training')
    plt.xlabel('Text Category for evaluation')
    plt.title('Perplexity scores by training and test text category')

    plt.tight_layout()
    plt.show()
    
    