# udacity-disaster
Udacity Data Scientist - Disaster project


## Project Motivation
Take a data set from raw data to a web application with a predictive model and visualizations.  Demonstrate an understanding of the full end to end process of gathering raw data, performing any necessary data cleaning and feature engineering via an ETL pipeline, and building a predictive model via an ML pipeline.  Then take that model and use it in a web application to make predictions. Use the cleaned data for visualizations on the web site.

The data for this project is disaster data from Figure Eight.  The project uses 'message' and 'category' datasets.


## Installations
Jupyter notebook and helper files build using Python v3.8.3

Libraries Included:
* pandas
* json
* plotly - Bar
* nltk
* nltk.stem - WordNetLemmatizer
* nltk.tokenize - word_tokenize
* flask - Flask, render_template, request, ,jsonify
* sklearn.externals - jbolib
* sqlalchemy - create_engine
* sys
* sklearn.model_selection - train_test_split
* sklearn.ensemble - RandomForestClassifier
* sklearn.feature_extraction.text - CountVectorizer, TfidfTransformer
* sklearn.pipeline - pipeline
* sklearn.multioutput - MultiOutputClassifier
* sklearn.metrics - classification_report
* sklearn.model_selection - GridSearchCV
* pickle


## File Descriptions
* app/run.py Python file to run the web application
* app/templates/go.html - Flask html file to add Plotly visualizations to the apply
* app/templates/master.html - Flask html file to build the web application page
* data/process_data.py - Python file to gather, clean and engineer the raw data
* data/disaster_categories.csv - raw disaster categories data in CSV format
* data/disaster_messages.csv - raw disaster message data in CSV format
* data/DisasterResponse.db - Sqlite database containing the cleaned disaster data
* models/classifier.pkl - Pickle file containing the disaster classifier predictive model
* models/train_classifier.py - Python file to train and test the predictive model

## How to use
* Download the project files from Github
* Install libraries as documented above

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## Authors, Acknowledgements, Etc
* Author:  Lincoln Alexander
* Acknowledgements:  Udacity made me do it
