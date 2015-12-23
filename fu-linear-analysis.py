import pandas
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
import cPickle
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin

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

# any < 0 => 0
# else int(round(x))
class RoundedLinearRegression(LinearRegression):
    def predict(self, X):
        return vround(super(LinearRegression, self).predict(X))

class LengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, data, **transform_params):
        return [{'length': len(text)} for text in data]

    def fit(self, X, y=None, **fit_params):
        return self

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def transform(self, data, **transform_params):
        return data[self.col]

    def fit(self, X, y=None, **fit_params):
        return self

all_data = pandas.read_csv("500-a.csv", names=["deleted", "votes", "message"])

# test, limit the influence of 0s-- all_data[all_data.votes == 0].sample(frac=0.1)
zeroes = all_data[all_data.votes == 0].sample(frac=0.03)
with_votes = all_data[all_data.votes > 0]
all_data = pandas.concat([zeroes, with_votes])



#all_data.votes.plot(bins=1, kind='hist')
#plt.show()

train, test = train_test_split(all_data, test_size=0.2)

pipeline = Pipeline([
    ('feats', FeatureUnion(transformer_list=[
        ('lemmas', Pipeline([
            ('message', ColumnExtractor('message')),
            ('bow', CountVectorizer(analyzer=split_into_lemmas)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('message_length', Pipeline([
            ('message', ColumnExtractor('message')),
            ('length', LengthTransformer()),
            ('hashed', FeatureHasher()),
        ])),
    ])),
    ('classifier', RoundedLinearRegression()),
])


# build the estimator from the training data
trained_estimator = pipeline.fit(train, train['votes'])

# predict the test set
print "Predicting..."
test_predictions = trained_estimator.predict(test)
print test_predictions[:20]
print test.votes[:20]

print "Preliminary score: {}".format(trained_estimator.score(test, test['votes']))


print 'r2', r2_score(test['votes'], test_predictions)
print 'explained variance', explained_variance_score(test['votes'], test_predictions)
print 'MSE', mean_squared_error(test['votes'], test_predictions)

scores = cross_val_score(pipeline, all_data, all_data.votes, scoring='r2')
print scores
