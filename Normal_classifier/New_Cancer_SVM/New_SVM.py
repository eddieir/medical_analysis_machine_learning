import pandas as pd
import numpy as np
import time
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier



# ----------- Data Preparation Stage --------------

# Loading the data into a dataframe
df = pd.read_csv('data.csv')
print('Dimension of the dataset : ', df.shape)
print(df.head(n=50))

del df['Unnamed: 32']
df.info()

# Separating the feature variables and class variable(target variable)

X = df.iloc[:, 2:].values       # Feature variable
Y = df.iloc[:, 1].values        # Actual class label
print(type(Y))
print("\n Actual Class Labels : ", Y)

# Class Label encoding M & B to 1 & 0
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
print('After Encoding : ', Y)

plt.hist(Y, bins=3)  # arguments are passed to np.histogram
plt.title("Histogram with for Diagnosis Class Labels ")
plt.savefig("Histogram_with_for_Diagnosis_Class_Labels.png")

mean_columns = list(df.columns[2:12])
print(mean_columns)

plt.figure(figsize=(10, 10))
sns.heatmap(df[mean_columns].corr(), annot = True, square = True, cmap ='coolwarm' )
plt.savefig('correlation')
#plt.show()

# Splitting data into test and training sets and randomly selecting in order to bias
# (sometimes they are highly correlated data)

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3, random_state=0)
# Scaling our training data (feature scaling)
# Each feature in our dataset now will have a mean = 0 and standard deviation = 1

sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.fit_transform(X_test1)

# Model Building
# 1. Stochastic Gradient Descent (SGD)
# 2. Support Vector Machines (SVM- Linear Kernel)
# 3. Support Vector Machines (SVM- Gaussian Kernel)

def StochasticGradient(train, label, test, label_test):
    start = time.time()

    model = SGDClassifier()
    model.fit(train, label)

    pred = model.predict(test)

    end = time.time()

    confusion_mat = confusion_matrix(label_test, pred)

    def get_confusion_matrix_values2(label_test, pred):
        cm = confusion_matrix(label_test, pred)
        return (cm[0][0], cm[0][1], cm[1][0], cm[1][1])

    TN1, FP1, FN1, TP1 = get_confusion_matrix_values2(label_test, pred)

    denom1 = TP1 + FP1

    print('TP', TP1, 'FP', FP1, 'FN', FN1, 'TN', TN1)
    print(classification_report(label_test, pred))

    print('Accuracy of Stochastic Gradient classifier on test set: {:.2f}\n'.format(
        float(confusion_mat[0, 0] + confusion_mat[1, 1]) * 100 / confusion_mat.sum()))
    print(confusion_mat)

    print('Time took for training and predicting the results {0:.5} in seconds\n'.format(float(end - start)))

    recall = float(confusion_mat[1, 1] / (confusion_mat[1, 1] + confusion_mat[1, 0]))
    prec = float(confusion_mat[1, 1] / (confusion_mat[1, 1] + confusion_mat[0, 1]))

    f_score = (2 * recall * prec) / (recall + prec)

    # print('real prec', prec)

    print('Recall ', recall)

    print('F1- Score  ', f_score)

    roc_auc4 = roc_auc_score(label_test, pred)

    fpr4, tpr4, thresholds4 = roc_curve(label_test, pred)
    plt.figure()
    plt.plot(fpr4, tpr4, label='Stochastic Gradient Classifier (area = %0.2f)' % roc_auc4)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

    print('Area Under the Curve ', roc_auc4)
    sns.heatmap(confusion_mat, annot=True)
    plt.savefig('SGD.png')
    plt.show()


StochasticGradient(X_train1, Y_train1, X_test1, Y_test1)

# --------- Support vector machine

def SupportVectorMachines(train, label, test, label_test):
    start = time.time()

    model1 = svm.SVC(kernel='linear', C=0.3)
    model1.fit(train, label)

    pred1 = model1.predict(test)

    end = time.time()

    confusion_mat1 = confusion_matrix(label_test, pred1)

    def get_confusion_matrix_values2(label_test, pred1):
        cm1 = confusion_matrix(label_test, pred1)
        return (cm1[0][0], cm1[0][1], cm1[1][0], cm1[1][1])

    TN1, FP1, FN1, TP1 = get_confusion_matrix_values2(label_test, pred1)

    denom1 = TP1 + FP1

    print('TP', TP1, 'FP', FP1, 'FN', FN1, 'TN', TN1)
    print(classification_report(label_test, pred1))

    print('Accuracy of Support vector Machines classifier on test set: {:.2f}\n'.format(
        float(confusion_mat1[0, 0] + confusion_mat1[1, 1]) * 100 / confusion_mat1.sum()))
    print(confusion_mat1)

    print('Time took for training and predicting the results {0:.5} in seconds\n'.format(float(end - start)))

    recall1 = float(confusion_mat1[1, 1] / (confusion_mat1[1, 1] + confusion_mat1[1, 0]))
    prec1 = float(confusion_mat1[1, 1] / (confusion_mat1[1, 1] + confusion_mat1[0, 1]))

    f_score1 = (2 * recall1 * prec1) / (recall1 + prec1)

    # print('real prec', prec)

    print('Recall ', recall1)

    print('F1- Score  ', f_score1)

    roc_auc4 = roc_auc_score(label_test, pred1)

    fpr4, tpr4, thresholds4 = roc_curve(label_test, pred1)
    plt.figure()
    plt.plot(fpr4, tpr4, label='Support Vector Machines Classifier (area = %0.2f)' % roc_auc4)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for SVM with Linear Kernel')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC_SVM')
    plt.show()

    print('Area Under the Curve ', roc_auc4)
    sns.heatmap(confusion_mat1, annot=True)
    plt.savefig('SVM.png')
    plt.show()


SupportVectorMachines(X_train1, Y_train1, X_test1, Y_test1)


def SupportVectorMachines1(train, label, test, label_test):
    start = time.time()

    model1 = svm.SVC(kernel='rbf', C=0.3, gamma=0.02)
    model1.fit(train, label)

    pred1 = model1.predict(test)

    end = time.time()

    confusion_mat1 = confusion_matrix(label_test, pred1)

    def get_confusion_matrix_values2(label_test, pred1):
        cm1 = confusion_matrix(label_test, pred1)
        return (cm1[0][0], cm1[0][1], cm1[1][0], cm1[1][1])

    TN1, FP1, FN1, TP1 = get_confusion_matrix_values2(label_test, pred1)

    denom1 = TP1 + FP1

    print('TP', TP1, 'FP', FP1, 'FN', FN1, 'TN', TN1)
    print(classification_report(label_test, pred1))

    print('Accuracy of Support vector Machines classifier on test set: {:.2f}\n'.format(
        float(confusion_mat1[0, 0] + confusion_mat1[1, 1]) * 100 / confusion_mat1.sum()))
    print(confusion_mat1)

    print('Time took for training and predicting the results {0:.5} in seconds\n'.format(float(end - start)))

    recall1 = float(confusion_mat1[1, 1] / (confusion_mat1[1, 1] + confusion_mat1[1, 0]))
    prec1 = float(confusion_mat1[1, 1] / (confusion_mat1[1, 1] + confusion_mat1[0, 1]))

    f_score1 = (2 * recall1 * prec1) / (recall1 + prec1)

    # print('real prec', prec)

    print('Recall ', recall1)

    print('F1- Score  ', f_score1)

    roc_auc4 = roc_auc_score(label_test, pred1)

    fpr4, tpr4, thresholds4 = roc_curve(label_test, pred1)
    plt.figure()
    plt.plot(fpr4, tpr4, label='Support Vector Machines Classifier (area = %0.2f)' % roc_auc4)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for SVM with Gaussian Kernel')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC_SVM_gaussian')
    plt.show()

    print('Area Under the Curve for Gaussian Kernel ', roc_auc4)
    sns.heatmap(confusion_mat1, annot=True)
    plt.savefig('SVM_gaussian.png')
    plt.show()


SupportVectorMachines1(X_train1, Y_train1, X_test1, Y_test1)

