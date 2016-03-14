import pandas
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
import cPickle
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.decomposition import TruncatedSVD

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

def dround(x):
    if x > 0.0:
        return round(x)
    else:
        return 0

vround = np.vectorize(dround, otypes=[np.int64])

class RoundedLinearRegression(LinearRegression):
    def predict(self, X):
        return vround(super(LinearRegression, self).predict(X))


all_data = pandas.read_csv("data/500-a.csv", names=["deleted", "votes", "message"])
msg_train, msg_test, vote_train, vote_test = train_test_split(all_data['message'], all_data['votes'], test_size=0.1)
print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('regression', RoundedLinearRegression()),
])


# build the estimator from the training data
trained_estimator = pipeline.fit(msg_train, vote_train)

# predict the test set
test_predictions = trained_estimator.predict(msg_test)
trained_estimator.score(msg_test, vote_test)


print 'r2', r2_score(vote_test, test_predictions)
print 'explained variance', explained_variance_score(vote_test, test_predictions)
print 'MSE', mean_squared_error(vote_test, test_predictions)

scores = cross_val_score(pipeline, all_data['message'], all_data['votes'], scoring='r2')
print scores
