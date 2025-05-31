import matplotlib.pyplot as plt
import seaborn as sns

def plot_residuals(y_true, y_pred, output_path="outputs/residuals_plot.png"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='purple', alpha=0.6)
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted")
    plt.savefig(output_path)
    plt.close()


def plot_prediction_distribution(y_true, y_pred, output_path="outputs/prediction_dist.png"):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_true, label="Actual", fill=True)
    sns.kdeplot(y_pred, label="Predicted", fill=True)
    plt.title("Actual vs. Predicted Distribution")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

