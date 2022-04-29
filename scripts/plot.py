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

#plot_confusion_matrix([[9,1],[10,0]], 'Drought Tolerance', 'images/confusion-matrix/Rhizo/PCA/drought_tolerance_CM.png', 'Greens')
#plot_confusion_matrix([[5,4],[5,5]], 'Drought Tolerance', 'images/confusion-matrix/Rhizo/Lasso/drought_tolerance_CM.png', 'Reds')
#plot_confusion_matrix([[5,4],[3,7]], 'Drought Tolerance', 'images/confusion-matrix/Rhizo/AE/drought_tolerance_CM-AE100.png', 'Blues')

#plot_confusion_matrix([[13000,0],[0,3000]], 'cultured', 'images/confusion-matrix/GEM/Lasso/cultured_CM.png', 'Reds')
#plot_confusion_matrix([[13000,0],[3000,0]], 'cultured', 'images/confusion-matrix/GEM/Lasso/cultured_CM-nometa.png', 'Reds')
#plot_confusion_matrix([[31,1],[3,3]], 'NP', 'images/confusion-matrix/TARA/Lasso/NP_CM.png', 'Reds')

# plot_confusion_matrix([[4,5],[6,4]], 'Drought Tolerance', 'images/RF/confusion-matrix/Rhizo/tSNE/drought_tolerance_CM.png', 'Purples')
# plot_confusion_matrix([[2,7],[3,7]], 'Drought Tolerance', 'images/RF/confusion-matrix/Rhizo/tSNE/drought_tolerance_CM-nometa.png', 'Purples')
#
# plot_confusion_matrix([[14,18],[13,12]], 'AO', 'images/RF/confusion-matrix/TARA/tSNE/AO_CM.png', 'Purples')
# plot_confusion_matrix([[23,18],[11,17]], 'IO', 'images/RF/confusion-matrix/TARA/tSNE/IO_CM.png', 'Purples')
# plot_confusion_matrix([[28,8],[25,13]], 'MS', 'images/RF/confusion-matrix/TARA/tSNE/MS_CM.png', 'Purples')
# plot_confusion_matrix([[20,17],[16,11]], 'NAT', 'images/RF/confusion-matrix/TARA/tSNE/NAT_CM.png', 'Purples')
# plot_confusion_matrix([[27,9],[24,6]], 'NP', 'images/RF/confusion-matrix/TARA/tSNE/NP_CM.png', 'Purples')
# plot_confusion_matrix([[30,7],[31,5]], 'RS', 'images/RF/confusion-matrix/TARA/tSNE/RS_CM.png', 'Purples')
# plot_confusion_matrix([[28,8],[19,11]], 'SAT', 'images/RF/confusion-matrix/TARA/tSNE/SAT_CM.png', 'Purples')
# plot_confusion_matrix([[30,7],[29,7]], 'SO', 'images/RF/confusion-matrix/TARA/tSNE/SO_CM.png', 'Purples')
# plot_confusion_matrix([[16,5],[22,14]], 'SP', 'images/RF/confusion-matrix/TARA/tSNE/SP_CM.png', 'Purples')

# plot_confusion_matrix([[9, 0],[10,0]], 'drought_tolerance', 'images/SVM/confusion-matrix/Rhizo/tSNE/drought_tolerance_CM.png', 'Purples')
# plot_confusion_matrix([[1.2e+04, 7.1e+02],[2.7e+03,3.2e+02]], 'cultured', 'images/RF/confusion-matrix/GEM/tSNE/cultured_CM.png', 'Purples')
# plot_confusion_matrix([[1.2e+04, 8.7e+02],[2.6e+03,3.4e+02]], 'cultured', 'images/RF/confusion-matrix/GEM/tSNE/cultured_CM-nometa.png', 'Purples')

plot_confusion_matrix([[1.3e+04, 18],[1.6e+03,1.3e+03]], 'Cultured', 'images/RF/tmp/cultured_CM.png', 'Oranges')
plot_confusion_matrix([[1.3e+04, 56],[2e+03,1e+03]], 'Cultured (excluding metadata)', 'images/RF/tmp/cultured_CM-nometa.png', 'Oranges')
