'''
Collects Twitter posts using the API
Type the following as a command line argument to save the posts
in a text file:
python collect_tweets.py > name.txt
where name.txt is the name of the output file to be saved
'''

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

access_token = "1319538764-f8mQyoOfVdOzVvpFiEf5zYYIPxFYIElaMzsifzO"
access_token_secret = "bqxA9j6rApYyQ2BicQinv95ZgelJ5RiXbIJVpk1cAiTPl"
consumer_key = "AkbXmFGhEkVjjBtUmyKM8AsS8"
consumer_secret = "v5OR96FOKXhPioSsTqBuR5w9d8V94Nqe3c5YltpvxoC4dhJVq2"

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords
    stream.filter(track=['obama'])
