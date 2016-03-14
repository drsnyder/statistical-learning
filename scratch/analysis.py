import pandas
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
import cPickle
import numpy as np
from sklearn.svm import SVC, LinearSVC

def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1,
            color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    plt.legend(loc="best")
    return plt

posts = pandas.read_csv("500-a.csv", names=["deleted", "votes", "message"])
posts["length"] = posts['message'].map(lambda message: len(message))
posts["bin_votes"] = posts['votes'].map(lambda vote: 1 if vote > 0 else 0)

posts.hist(column="length", by="bin_votes")
posts.length.plot(bins=20, kind='hist')

posts.message.head().apply(split_into_tokens)
posts.message.head().apply(split_into_lemmas)

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(posts['message'])
message4 = posts['message'][3]
bow4 = bow_transformer.transform([message4])
print bow4
print bow4.shape

print bow_transformer.get_feature_names()[3983]
print bow_transformer.get_feature_names()[1941]

messages_bow = bow_transformer.transform(posts['message'])
print 'sparse matrix shape:', messages_bow.shape
print 'number of non-zeros:', messages_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4

messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape

vote_detector = MultinomialNB().fit(messages_tfidf, posts['bin_votes'])
print 'predicted:', vote_detector.predict(tfidf4)[0]
print 'expected:', posts.votes[3]


bow29 = bow_transformer.transform([posts['message'][29]])
tfidf29 = tfidf_transformer.transform(bow29)
print 'predicted:', vote_detector.predict(tfidf29)[0]
print 'expected:', posts.votes[29]

all_predictions = vote_detector.predict(messages_tfidf)
print 'accuracy', accuracy_score(posts['bin_votes'], all_predictions)
print 'confusion matrix\n', confusion_matrix(posts['bin_votes'], all_predictions)
print '(row=expected, col=predicted)'

print classification_report(posts['bin_votes'], all_predictions)

# split out into training
msg_train, msg_test, label_train, label_test = train_test_split(posts['message'], posts['votes'], test_size=0.2)
print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    #('bow-simple', CountVectorizer(analyzer=split_into_lemmas, strip_accents='unicode', stop_words='english', dtype='double', ngram_range=(1,1))),
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores

plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

nb_detector = grid.fit(msg_train, label_train)
nb_detector.grid_scores_

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(msg_train, label_train)
print svm_detector.grid_scores_


###############
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.decomposition import TruncatedSVD


pca = TruncatedSVD(n_components=1000)
reduced = pca.fit_transform(messages_tfidf)

sample = pandas.read_csv("500.csv", names=["deleted", "votes", "message"])
sample_bow = bow_transformer.transform(sample['message'])
print 'sparse matrix shape:', sample_bow.shape
print 'number of non-zeros:', sample_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * sample_bow.nnz / (sample_bow.shape[0] * sample_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(sample_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4

sample_tfidf = tfidf_transformer.transform(sample_bow)

# set data to the tfidf data you want to model
data_df = sample
data = sample_tfidf

vote_estimator = LinearRegression().fit(data, data_df['votes'])
print 'predicted:', vote_estimator.predict(tfidf4)[0]
print 'expected:', posts.votes[3]


all_vote_predictons = vote_estimator.predict(data)
print 'r2', r2_score(data_df['votes'], all_vote_predictons)
print 'explained variance', explained_variance_score(data_df['votes'], all_vote_predictons)
print 'MSE', mean_squared_error(data_df['votes'], all_vote_predictons)
print 'Explained', mean_squared_error(data_df['votes'], all_vote_predictons)

vote_estimator.coef_


from sklearn import cross_validation
from sklearn.metrics import explained_variance_score
loo = cross_validation.LeaveOneOut(len(msg_train))
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('regression', LinearRegression()),
])
scores = cross_validation.cross_val_score(pipeline, msg_train, label_train, scoring='mean_squared_error', cv=loo,)


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('regression', LinearRegression()),
])

scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         'r2',
                         cv=10,
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores


##### what does the data look like
feature_names = bow_transformer.get_feature_names()
dense = messages_tfidf.todense()
posts.loc[posts['votes'] >= 4].head()
messages = dense[7790].tolist()[0] # have to find a message that has a high vote count
phrase_scores = [pair for pair in zip(range(0, len(messages)), messages)]
sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
       print('{0: <20} {1}'.format(phrase, score))
