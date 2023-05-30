from plots import book_reviews_df, top_merged_df, control_reviews_df, control_merged_df
import matplotlib.pyplot as plt
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.metrics import accuracy_score
# download this as well
'''
import nltk
nltk.download('stopwords')
nltk.download('punkt')
'''

#
# scatterplot of want to read vs read

# Box plot of review ratings by product: You could create a box plot that shows the distribution of review ratings for each product. This would allow you to compare the review ratings for each product and see if there are any significant differences in the distribution of ratings between products.
# avg currently reading vs want to read

# ratins and reviews over time per genre per month from date published to see when theres spikes
# line plot of ratings for each book over time - per genre?

'''
# CODE FOR ONLY REVIEWS - CLEANING TEXT
'''
# axis=1 means drop column not row
book_reviews_df = book_reviews_df.drop(['Username', 'Review Date'], axis=1)


# create wordcloud of most frequent words

# remove special char, lowercase, create tokens, stop words, stemming, remove neutral
# NLTK provides the sent_tokenize() function to split text into sentences.
# ¡¡¡¡¡
stop_words = stopwords.words('english')
# i could add these custom words to the wordcloud specifically so for sentiment it includes it
added_stopwords = ["book", "review", "read", "author", "character", "story", "one", "novel", "even", "though"]

authors = list(top_merged_df['Author'].unique())
book_stopwords = []
for word in authors:
    book_stopwords += word.lower().split()
added_stopwords = added_stopwords + book_stopwords
'''
titles = list(top_merged_df['Title'].unique())

book_list = authors + titles

for word in book_list:
    book_stopwords += word.lower().split()
'''
stop_words.extend(added_stopwords)
punctuation = string.punctuation


#removes stopwords and punctuation
def clean_text(review):
    # token, stopword, lemmenize

    # remove https
    review = re.sub(r"http\S+", "", review)
    # remove numbers
    review = re.sub(r'\d+', '', review)
    # third argument is characters to remove during translation
    # table = str.maketrans('', '', string.punctuation)
    # review = review.translate(table)

    tokens = word_tokenize(review.lower())
    # print(tokens)
    # after cleaning initial punctuation, some stand alone punctuation is left
    cleaning_tokens = [word for word in tokens if word not in stop_words and word.isalpha() and word not in punctuation]
    cleaning_tokens = [word.replace('\n', ' ') for word in cleaning_tokens]
    # print(cleaning_tokens)

    # better -> good
    lemm = WordNetLemmatizer()
    lemm_words = [lemm.lemmatize(word) for word in cleaning_tokens]

    # cleaned_review = ' '.join(cleaning_tokens)
    cleaned_review = ' '.join(lemm_words)
    return cleaned_review


# book_reviews_df['Description'] rows are objects, but i need to convert them to string
book_reviews_df['Description'] = book_reviews_df['Description'].apply(lambda x: clean_text(str(x)))


def add_reviews(df):
    # or in book[desc]
    all_descriptions = [description for description in df['Description']]
    text = ' '.join(all_descriptions)
    #print(text)
    wordcloud = WordCloud(width=800, height=800, collocations=True, min_word_length=3, collocation_threshold=3,
                          min_font_size=15).generate(text)
    plt.imshow(wordcloud, interpolation='bilInear')
    plt.axis('off')
    plt.title("Common words in most read", fontsize=20)
    plt.show()

    text_dictionary = wordcloud.process_text(text)
    # sort the dictionary
    word_freq = {k: v for k, v in sorted(text_dictionary.items(), reverse=True, key=lambda item: item[1])}
    # use words_ to print relative word frequencies
    rel_freq = wordcloud.words_
    # print results
    print(list(word_freq.items())[:5])
    #print(list(rel_freq.items())[:5])
    return

# each_book = book_reviews_df.groupby('Title')
# all_book_reviews = each_book.apply(add_reviews)
# DO IT FOR ALL THE BOOKS, NOT EACH ONE
'''
from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))
'''
add_reviews(book_reviews_df)
print("end")
# now you have wordcloud, actually calculate most frequent words e.g with library collections

'''
#sentiment analysis with the cleaned text in VADER

sid = SentimentIntensityAnalyzer()


def get_sentiment(text):
    sentiment_score = sid.polarity_scores(text)
    return sentiment_score


book_reviews_df['sentiment_scores'] = book_reviews_df['Description'].apply(get_sentiment)
#df[scores_col] = df[target_col].apply(lambda tweet: sid.polarity_scores(tweet))
print(book_reviews_df['sentiment_scores'])
'''
# IMDB review trained set
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

print("sentiment")
# flair sentiment on the whole tweet
book_reviews_df['flair_confidence'] = book_reviews_df['Description'].apply(flair_get_score_tweet)
#book_reviews_df.to_csv('flair.csv', index=True)
print(book_reviews_df['flair_confidence'])

# Check the distribution of the score
# print(book_reviews_df['flair_confidence'].describe())
# Check the counts of labels
# print(book_reviews_df['flair_confidence'].value_counts())

# Counts the sum of the positive/negative
# g = book_reviews_df.groupby('flair_confidence').agg([('positive', lambda x:x[x > 0].value_counts()), ('negative', lambda x:x[x < 0].value_counts())])
g = book_reviews_df.groupby(book_reviews_df['flair_confidence'].apply(lambda x: 'NEG' if x < -0.0 else 'POS')).size()

sentiment = g.index
values = [sentiment for sentiment in g]
print("sentiment", values)
plt.pie(values, labels=sentiment, autopct='%1.1f%%', startangle=90)
plt.show()
