import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import sys


app = Flask(__name__)

def tokenize(text):
    """Given some text, normalize and tokenize it returning word tokens

    Args:
        text - a string of text to tokenize

    Returns:
        array: an array of words

    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Extract and format data for visualizations

    Args:
        none

    Returns:
        array: an array of plot data and layout specifications

    """

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # restructure data for visualizations
    df_genre = df.drop(['message', 'id','original'], axis=1)
    df_genre = pd.DataFrame(df_genre.groupby('genre').sum())
    df_genre.reset_index(inplace=True)
    df_genre = df_genre.T
    df_genre.columns =(['direct','news','social'])
    df_genre = df_genre.drop('genre')
    df_genre.index

    # get top 10 message categories from Social
    df_social = df_genre['social'].sort_values().tail(10)
    social_counts = df_social.values
    social_names = list(df_social.index)
    # get top 10 message categories from Direct
    df_direct = df_genre['direct'].sort_values().tail(10)
    direct_counts = df_direct.values
    direct_names = list(df_direct.index)
    # get top 10 message categories from News
    df_news = df_genre['news'].sort_values().tail(10)
    news_counts = df_news.values
    news_names = list(df_news.index)

    # natural disasters related disasters by genre
    df_news = df_genre['news']
    df_natural_news = df_news.loc[[ 'earthquake', 'fire', 'storm', 'floods', 'weather_related', 'other_weather']]
    nat_news_counts = df_natural_news.values
    nat_news_names = list(df_natural_news.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_names,
                    y=social_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories From Social',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_names,
                    y=direct_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories From Direct',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_names,
                    y=news_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories From News',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=nat_news_names,
                    y=nat_news_counts
                )
            ],

            'layout': {
                'title': 'News Genre Counts by Natural Disaster Message Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        }
    ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Given some text from the web app, call the predictive model for a classification

    Args:
        none directly - gets text from the web page

    Returns:
        array: an array of categories predicted based on the text provided

    """
    # save user input in query
    query = request.args.get('query', '')
    print('query being sent to predict model: {}'.format(query), file=sys.stdout)
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print('query resultsl: {}'.format(classification_results), file=sys.stdout)

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
