import time, random
from tweeterpy import TweeterPy
from tweeterpy import config
from tweeterpy.util import RateLimitError
from detoxify import Detoxify

# Check Configuration docs for the available settings.
# config.PROXY = {"http":"127.0.0.1","https":"127.0.0.1"}
# config.TIMEOUT = 10

twitter = TweeterPy()
# login if required

def gather_tweets(username):
    '''
    Gathers tweets of a username from Twitter.
    '''
    user_tweets = []
    has_more = True
    cursor = None
    while has_more:
        try:
            response = None
            response = twitter.get_user_tweets(username, end_cursor=cursor, pagination=False)
            user_tweets.extend(response['data'])
            has_more = response.get('has_next_page')
            api_rate_limits = response.get('api_rate_limit')
            limit_exhausted = api_rate_limits.get('rate_limit_exhausted')
            if has_more:
                cursor = response.get('cursor_endpoint')
            ## YOUR CUSTOM CODE HERE (DATA HANDLING, REQUEST DELAYS, SESSION SHUFFLING ETC.)
            ## time.sleep(random.uniform(7,10))
            if limit_exhausted:
                raise RateLimitError
        except Exception as error:
            print(error)
            break
    tweet_contents = []

    for tweet in user_tweets:
        # Extract the tweet content from the 'full_text' key
        content = tweet['content']['itemContent']['tweet_results']['result']['legacy']['full_text']
        # Append the tweet content to the list
        tweet_contents.append(content)

    # Now, tweet_contents contains the tweet contents of the desired username.
    return tweet_contents

if __name__ == '__main__':
    user = input("Enter username: ")
    tweets = gather_tweets(user)
    print(tweets)