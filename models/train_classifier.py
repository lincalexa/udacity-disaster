import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """Load data from the database ready for machine learning

    Args:
        database_filepath - path and filename for the database

    Returns:
        X - array of messages to use for ML test/train feature data
        y - array of target data
        target_names - names for the target classes

    """
    # load_data('DisasterMessageCategories.db', 'MessageCategories')
    # data/DisasterResponse.db
    # load data from database
    full_db_name = ('sqlite:///{}'.format(database_filepath))
    engine = create_engine(full_db_name)

    df = pd.read_sql_table('MessageCategories', con=engine)

    X = df['message'].values
    target_names = list(set(df.columns) - set({'id', 'message', 'original', 'genre'}))
    y = df.drop(['message', 'id','original','genre'], axis=1).values
    return X, y, target_names


def tokenize(text):
    """Given text, normalize and tokenize into words

    Args:
        text - a string of text to be tokenized

    Returns:
        array - an array of tokens

    """
    tokens = nltk.tokenize.word_tokenize(text)
    # initiate lemmatizer
    lemma = nltk.stem.WordNetLemmatizer()
    # iterate through tokens lemmatizing and cleaning
    clean_tokens = []
    for token in tokens:
        clean_token = lemma.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """Define the ML pipeline and tune for best parameters

    Args:
        None

    Returns:
        model - returns a tuned classifier model

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidt', TfidfTransformer(sublinear_tf=True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [100,200],
        'vect__max_features': [None,5000]
    }

    cv = GridSearchCV(pipeline, param_grid= parameters, cv=2, n_jobs=-1, verbose=1)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Evaluate a model for accuracy reporting for each target category

    Args:
        model - a classification model
        X_test - test data
        y_test - test target data
        category_names - names for the target categories

    Returns:
        y_pred - predicted target data

    """
    # predict on test data
    y_pred = model.predict(X_test)
    for i in range(y_test.shape[1]):
        print('Classification results for target category {}:'.format(category_names[i]))
        print(classification_report(y_test[:, i], y_pred[:, i]))
        print('------------------------------------------------------------------\n')
    return y_pred


def save_model(model, model_filepath):
    """Create a pickle file for a ML model

    Args:
        model - a ML model
        model_filepath - path and name for the pickle file

    Returns:
        None

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ML pipeline for disaster data modeling

    Args:
        database_filepath - path and filename for the database
        model_filepath - path and filename for the model pickle file

    Returns:
        None

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
