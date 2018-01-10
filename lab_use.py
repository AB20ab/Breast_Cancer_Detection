import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB as MNB
#from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf_RF = RandomForestClassifier()
clf_ET = ExtraTreesClassifier()
clf_svc= SVC()
clf_DT = DecisionTreeClassifier()
#clf_MNB= MNB()
eclf = EnsembleVoteClassifier(clfs=[clf_RF, clf_ET, clf_svc,clf_DT], weights=[1,1,1,1])

labels = ['Random Forest', 'Extra Trees', 'Support Vector','Decision Tree','Ensemble Vote']
for clf, label in zip([clf_RF, clf_ET, clf_svc,clf_DT, eclf], labels):

    scores = cross_val_score(clf, X, y,
                                              cv=5,
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

eclf.fit(X_train, y_train)
confidence = eclf.score(X_test, y_test)
print(confidence)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = eclf.predict(example_measures)
print(prediction)

col_dict=dict(list(enumerate(df.columns)))
col_dict

X = np.array(df.drop(['class'], 1),dtype=np.float64)
y = np.array(df['class'],dtype=np.int64)
plot_decision_regions(X=X,
                      y=y,
                      clf=clf,
                      filler_feature_values=col_dict,
                      filler_feature_ranges=col_dict,
                      )

#plt.xlabel(X.columns[0], size=14)
#plt.ylabel(X.columns[1], size=14)
#plt.title('SVM Decision Region Boundary', size=16)
