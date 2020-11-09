import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Given paths and filenames for the disaster CSV data, return a merged dataframe

    Args:
        messages_filepath - the path and file name of the messages data
        categories_filepath - the path and filename of the categories data

    Returns:
        dataframe: a dataframe containing merged message and category data

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df


def clean_data(df):
    """Given a dataframe with disaster data, clean and transform to get it ready for modeling

    Args:
        dataframe - a dataframe containing merged message and category data

    Returns:
        dataframe: a dataframe containing cleaned message and category data

    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='last')
    return df

def save_data(df, database_filename):
    """Save cleaned data to a sqlite database

    Args:
        df - dataframe with data to save
        text - the path and filename of the sqlite datatabase

    Returns:
        None

    """
    full_db_name = ('sqlite:///{}'.format(database_filename))
    engine = create_engine(full_db_name)
    df.to_sql('MessageCategories', engine, index=False)

def main():
    """ETL pipeline for disaster data preparation

    Args:
        messages_filepath - path and filename of the message data
        categories_filepath - path and filename of the category data
        database_filepath - path and filename for the database 

    Returns:
        None

    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
