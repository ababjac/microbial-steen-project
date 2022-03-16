import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cf_matrix, title, filename, color):
    plt.gca().set_aspect('equal')
    #cf_matrix = metrics.confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    #print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap=color)

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_confusion_matrix([[9,1],[10,0]], 'Drought Tolerance', 'images/confusion-matrix/Rhizo/PCA/drought_tolerance_CM.png', 'Greens')
plot_confusion_matrix([[5,4],[5,5]], 'Drought Tolerance', 'images/confusion-matrix/Rhizo/Lasso/drought_tolerance_CM.png', 'Reds')
plot_confusion_matrix([[5,4],[3,7]], 'Drought Tolerance', 'images/confusion-matrix/Rhizo/AE/drought_tolerance_CM-AE100.png', 'Blues')

plot_confusion_matrix([[13000,0],[0,3000]], 'cultured', 'images/confusion-matrix/GEM/Lasso/cultured_CM.png', 'Reds')
plot_confusion_matrix([[13000,0],[3000,0]], 'cultured', 'images/confusion-matrix/GEM/Lasso/cultured_CM-nometa.png', 'Reds')
