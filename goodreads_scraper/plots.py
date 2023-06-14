import pandas as pd
import matplotlib.pyplot as plt

import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from flair.models import TextClassifier
from flair.data import Sentence

'''download when running for first time
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
'''

# reading in all my csv files
book_reviews_df = pd.read_csv('Top_book_reviews.csv')
book_data_df = pd.read_csv('Top_book_details.csv')
book_price_df = pd.read_csv('Top_book_prices.csv')

book_reviews_df = book_reviews_df.dropna().reset_index(drop=True)
book_price_df = book_price_df.dropna()
# worst_books list
control_reviews_df = pd.read_csv('Worst_book_reviews.csv')
control_data_df = pd.read_csv('Worst_book_details.csv')
control_price_df = pd.read_csv('Worst_book_prices.csv')

control_reviews_df = control_reviews_df.dropna()
control_price_df = control_price_df.dropna()
# worst_rated list
worst_rated_reviews_df = pd.read_csv('Worst_rated_book_reviews.csv')
worst_rated_data = pd.read_csv('Worst_rated_book_details.csv')
worst_rated_price = pd.read_csv('Worst_rated_book_prices.csv')

worst_rated_reviews_df = worst_rated_reviews_df.dropna()
worst_rated_price = worst_rated_price.dropna()


#
#
# rc= runtime configuration
# makes the whole figure fit in
plt.rcParams["figure.autolayout"] = True
def convert_dates(df):
    # convert date published here to make it faster than in program
    # y-m-d
    try:
        # book_data pd timestamp (print(pd.Timestamp.min) can only represent up to the year 1677, anything before
        # then will cause error, i use ‘coerce’, to set as NaT
        df['Date published'] = pd.to_datetime(df['Date published'], errors='coerce')
    except KeyError:
        # book_reviews
        df['Review Date'] = pd.to_datetime(df['Review Date'])


# how well is each genre represented - number of books in each genre
def plt_genres(df):
    genre_count = df.groupby('Top Genre')['Title'].apply(lambda title: list(title.unique())).to_dict()
    # print(genre_count)
    # this gives all the labels for the X-axis
    x_labels = [genre for genre in genre_count]
    # this gets the count of all books in the genre
    number = [len(genre_count[genre]) for genre in genre_count]

    plt.bar(x_labels, number)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Genre')
    plt.ylabel('Number of Books')
    plt.title('Amount of books in each genre')
    plt.show()


# PIECHART distribution of ratings
def rating_distribution(df):
    ratings = df["Rating"].value_counts().sort_index()
    numbers = ratings.index

    # plt.pie(quantity, labels=numbers) without value display
    # -to show actual values not percentages, autopct
    plt.pie(ratings, labels=numbers, autopct='%1.1f%%', startangle=90)
    plt.rc('font', size=12)
    plt.title('distribution of ratings')
    plt.show()


# avg rating for books published in same year - this works
def year_to_rating(df):
    # inplace will sort the original dataframe
    book_year = df.sort_values('Date published')
    yearly_grouping = book_year.groupby(book_year['Date published'].dt.year)['Rating'].apply(list)
    # print(yearly_grouping.head())
    '''
    index is the year, and the values are the lists of value column values for each year.
    '''
    # rating list = [[3.0, 5.0, 4.0, 5.0, 5.0], [values for year 2], ect]
    rating_list = [yearly_grouping[year] for year in yearly_grouping.index]
    # print(rating_list)

    # yearly = [1.2, 4.0, 4.5]
    yearly_avg = [sum(ratings) / len(ratings) for ratings in rating_list]
    # print(yearly_avg)

    plt.figure(figsize=(10, 6))
    plt.bar(yearly_grouping.index, yearly_avg)
    plt.xlabel('Year')
    plt.ylabel('Avg rating')
    plt.title('Avg rating of books published per year')
    plt.show()


def ratings_over_years(df):
    year_list = sorted(df['Review Date'].dt.year.unique().tolist())
    print(year_list)

    # this group each book together, puts the reviews in ascending order of review date
    sorted_df = df.sort_values('Review Date')
    # yearly_rating = sorted_df.groupby(['Top Genre', sorted_df['Review Date'].dt.year])['Rating'].apply(lambda x: sum(x) / len(x))  # list)
    yearly_rating = sorted_df.groupby(['Top Genre', sorted_df['Review Date'].dt.year])['Rating'].agg(['count', 'mean'])
    print(yearly_rating)

    yearly_rating = yearly_rating.reset_index()
    print(yearly_rating)

    plt.figure(figsize=(15, 6))
    plt.gca().set_xticks(year_list)
    plt.yticks([0, 1, 2, 3, 4, 5])

    for genre in yearly_rating['Top Genre'].unique():
        separate_genre = yearly_rating[yearly_rating['Top Genre'] == genre]
        plt.plot(separate_genre['Review Date'], separate_genre['mean'], label=genre, marker='o')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Rating')
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.title('Avg rating value received by genre over the years')
    plt.show()


def price_per_genre(price_reviews):
    genre_price = price_reviews.groupby('Top Genre')['Price'].mean().round(2).reset_index()
    # print(genre_price)

    plt.figure(figsize=(10, 6))
    plt.bar(genre_price['Top Genre'], genre_price['Price'])
    plt.xticks(rotation=45, ha='right')
    # adds actual values to each bar
    for i, v in enumerate(genre_price['Price']):
        # v+ (distance from bar)
        plt.text(i, v - 1, str(v), ha='center')

    plt.xlabel('Genre')
    plt.ylabel('Average Price')
    plt.title('Average Price by Genre')
    plt.show()

# total amount of rating/review for each genre
def no_reviews(df):
    total_reviews = df.groupby('Top Genre')['Total ratings'].count().reset_index()
    # print(total_reviews)
    plt.bar(total_reviews['Top Genre'], total_reviews['Total ratings'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Genre')
    plt.ylabel('Ratings')
    plt.title('Rating by Genre')
    plt.show()


# analyse whats poppin in 2023 - no books in 2023
'''
yearly_rating = book_data_df.groupby(book_data_df['Date published'].dt.year)
mask = yearly_rating.get_group(2023)
print(mask)
'''

# boxplot of number of reviews per every 20 books
def review_boxplot(df):
    # this splits the df into seperate elements including 20 books each
    lst = [df.iloc[i:i + 20] for i in range(0, len(df) - 20 + 1, 20)]
    # print(lst)

    titles = ['top 20', 'top 40', 'top 60', 'top 80', 'top 100']
    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    test = {}
    for i in range(len(lst)):
        test.update({titles[i]: lst[i]['Total ratings']})

    print(test)
    fig, ax = plt.subplots()
    ax.boxplot(test.values())
    ax.set_xticklabels(test.keys())

    plt.title('Boxplot of Value by Group')
    plt.xlabel('Group')
    plt.ylabel('Value')
    plt.show()


# create wordcloud of most frequent words

#
'''
new - i need to balance out the dataset - maybe remove common books and control amount of 
books per genre
'''

#show books in common
common = pd.merge(book_data_df, control_data_df, on=['Title'], how='inner')
#common.to_csv("common_books.csv")
#the reason theres 2 'to kill a mockingbird' - the descriptions are diff

common_titles = common['Title'].tolist()
#negation operator ~ 
new_best = book_data_df[~book_data_df['Title'].isin(common_titles)]
new_control = control_data_df[~control_data_df['Title'].isin(common_titles)]


#now i removed all the common books - allow only 10 books per genre
new_best = new_best.groupby('Top Genre')
new_control = new_control.groupby('Top Genre')

best_balanced = pd.DataFrame()
control_balanced = pd.DataFrame()

def append_10(grouped, emptydf):
    for group_name, group_df in grouped:
        #groupdf represents every genre group
        emptydf = emptydf.append(group_df.head(10))
    return emptydf

best_balanced = append_10(new_best, best_balanced)
control_balanced = append_10(new_control, control_balanced)
#   BOOK REVIEWS
#book reviews - need to be balanced, not book data lol
common_best = best_balanced['Title'].tolist()
common_worst = control_balanced['Title'].tolist()

balanced_best_reviews = book_reviews_df[book_reviews_df['Title'].isin(common_best)].reset_index(drop=True)
balanced_worst_reviews = control_reviews_df[control_reviews_df['Title'].isin(common_worst)].reset_index(drop=True)


# remove special char, lowercase, create tokens, stop words, lemm
# NLTK provides the sent_tokenize() function to split text into words
# ¡¡¡¡¡
stop_words = stopwords.words('english')

added_stopwords = ['book', 'review', 'read', 'author', 'character', 'story', 'one', 'novel', 'even', 'though', 'year old']
stop_words.extend(added_stopwords)
punctuation = string.punctuation


def custom_stopwords(df_data):
    titles = list(df_data['Title'].unique())
    authors = list(df_data['Author'].unique())
    book_list = authors + titles
    book_stopwords = []
    for word in book_list:
        book_stopwords += word.lower().split()
    stop_words.extend(book_stopwords)


def clean_text(review):
    # token, stopword, lemmenize

    # remove https
    review = re.sub(r"http\S+", "", review)
    # remove numbers
    review = re.sub(r'\d+', '', review)

    tokens = word_tokenize(review.lower())
    # print(tokens)
    # after cleaning initial punctuation, some stand-alone punctuation is left
    cleaning_tokens = [word for word in tokens if word not in stop_words and word.isalpha() and word not in punctuation]
    cleaning_tokens = [word.replace('\n', ' ') for word in cleaning_tokens]
    # print(cleaning_tokens)

    # better -> good
    lemm = WordNetLemmatizer()
    lemm_words = [lemm.lemmatize(word) for word in cleaning_tokens]

    # cleaned_review = ' '.join(cleaning_tokens)
    cleaned_review = ' '.join(lemm_words)
    return cleaned_review

def add_reviews(df):
    all_descriptions = [description for description in df['Description']]
    text = ' '.join(all_descriptions)
    # print(text)
    wordcloud = WordCloud(width=800, height=800, collocations=True, min_word_length=2, collocation_threshold=2,
                          min_font_size=15).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Common words in most read", fontsize=20)
    plt.show()

    text_dictionary = wordcloud.process_text(text)
    # sort the dictionary
    word_freq = {k: v for k, v in sorted(text_dictionary.items(), reverse=True, key=lambda item: item[1])}
    # print results
    print(list(word_freq.items())[:5])


balanced_best_reviews = balanced_best_reviews.drop(['Username', 'Review Date'], axis=1)
# book_reviews_df['Description'] rows are objects, but i need to convert them to string
custom_stopwords(best_balanced)
balanced_best_reviews['Description'] = balanced_best_reviews['Description'].apply(lambda x: clean_text(str(x)))
add_reviews(balanced_best_reviews)


balanced_worst_reviews = balanced_worst_reviews.drop(['Username', 'Review Date'], axis=1)
custom_stopwords(control_balanced)
balanced_worst_reviews['Description'] = balanced_worst_reviews['Description'].apply(lambda x: clean_text(str(x)))
add_reviews(balanced_worst_reviews)


# sentiment pre-trained set trained on IMDB reviews.
# i won't include sentiment as my laptop can't run it fast enough
classifier = TextClassifier.load('en-sentiment')


def flair_get_score_tweet(text):
    if not text:
        return 0
    # creates sentence obj from the entire review - flair tokenization
    sentence = Sentence(text)
    # stacked_embeddings.embed(text)
    classifier.predict(sentence)

    # sentence.labels = [pos (0.98)]
    # the sentiment value
    value = sentence.labels[0].to_dict()['value']
    # confidence score is combined with sentiment
    if value == 'POSITIVE':
        # 0.990 means its 0.990 confident that it is a positive sentiment
        result = float(sentence.to_dict()['all labels'][0]['confidence'])
    else:
        # -0.991
        result = float(-(sentence.to_dict()['all labels'][0]['confidence']))
    return round(result, 3)


# flair sentiment on the whole tweet

# Check the distribution of the score
# print(book_reviews_df['flair_confidence'].describe())
# Check the counts of labels
# print(book_reviews_df['flair_confidence'].value_counts())
def sentiment_pie(df):
    # Counts the sum of the positive/negative
    g = df.groupby(df['flair_confidence'].apply(lambda x: 'NEG' if x < -0.0 else 'POS')).size()

    sentiment = g.index
    values = [sentiment for sentiment in g]
    print("sentiment", values)
    plt.pie(values, labels=sentiment, autopct='%1.1f%%', startangle=90)
    plt.show()



#CODE TO RUN PLOTS FOR TOP RATED BOOKS
'''
convert_dates(book_data_df)
convert_dates(book_reviews_df)

plt_genres(book_data_df)
top_merged_df = pd.merge(book_data_df, book_reviews_df, on='Title')
rating_distribution(top_merged_df)
year_to_rating(top_merged_df)
ratings_over_years(top_merged_df)
top_merged_df = pd.merge(top_merged_df, book_price_df, on='Title')
price_per_genre(top_merged_df)
no_reviews(top_merged_df)
review_boxplot(book_data_df)

#NLTK
# axis=1 means drop columns not row
book_reviews_df = book_reviews_df.drop(['Username', 'Review Date'], axis=1)
# book_reviews_df['Description'] rows are objects, but i need to convert them to string
custom_stopwords(book_data_df)
book_reviews_df['Description'] = book_reviews_df['Description'].apply(lambda x: clean_text(str(x)))
# wordcloud
add_reviews(book_reviews_df)
# sentiment
book_reviews_df['flair_confidence'] = book_reviews_df['Description'].apply(flair_get_score_tweet)
print(book_reviews_df['flair_confidence'])
sentiment_pie(book_reviews_df)
'''


# CODE FOR WORST BOOKS
'''
print("worst books")
convert_dates(control_data_df)
convert_dates(control_reviews_df)

control_merged_df = pd.merge(control_data_df, control_reviews_df, on='Title')

plt_genres(control_data_df)
rating_distribution(control_reviews_df)
year_to_rating(control_merged_df)
ratings_over_years(control_merged_df)

# MERGE AGAIN WITH PRICE
control_merged_df = pd.merge(control_merged_df, control_price_df, on='Title')
price_per_genre(control_merged_df)
no_reviews(control_merged_df)

# axis=1 means drop columns not row
# Cleaning text
control_reviews_df = control_reviews_df.drop(['Username', 'Review Date'], axis=1)
custom_stopwords(control_data_df)
control_reviews_df['Description'] = control_reviews_df['Description'].apply(lambda x: clean_text(str(x)))

# worcloud
add_reviews(control_reviews_df)

'''

# worst rated book as - 2nd control list
'''
print("worst books")
convert_dates(worst_rated_data)
convert_dates(worst_rated_reviews_df)

worst_rated_merged_df = pd.merge(worst_rated_data, worst_rated_reviews_df, on='Title')

plt_genres(worst_rated_data)
rating_distribution(worst_rated_reviews_df)
year_to_rating(worst_rated_merged_df)
ratings_over_years(worst_rated_merged_df)

# MERGE AGAIN WITH PRICE
worst_rated_merged_df = pd.merge(worst_rated_merged_df, worst_rated_price, on='Title')
price_per_genre(worst_rated_merged_df)
no_reviews(worst_rated_merged_df)

# axis=1 means drop columns not row
# Cleaning text
worst_rated_reviews_df = worst_rated_reviews_df.drop(['Username', 'Review Date'], axis=1)
custom_stopwords(worst_rated_data)
worst_rated_reviews_df['Description'] = worst_rated_reviews_df['Description'].apply(lambda x: clean_text(str(x)))

# worcloud
add_reviews(worst_rated_reviews_df)
'''
