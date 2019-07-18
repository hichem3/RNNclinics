

from whole_preprocessing import X_train, X_test, y_train, y_test, logger

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, recall_score


logger.info("Training model")

# hyperparameters found  by gridsearc --> text_reports.ipynb
tfidf = TfidfVectorizer(max_df=0.9, min_df=0.025,
                        ngram_range=(4, 4),
                        norm='l1', lowercase=False)
clf = RandomForestClassifier(max_features='sqrt',
                             min_samples_leaf=0.17,
                             n_estimators=82)

pipeline = Pipeline([('vectorizer', tfidf ),
                     ('classifier', clf)])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Recall score wiwth TFIDF and RF = {:.3f}".format(recall_score()))
print("Confusion matrix:", cm.tolist())

