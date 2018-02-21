#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:33:20 2017

@author: rincygeorge
"""

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# user credentials to access Twitter API 
api_key = " "
api_secret = " "
access_token = " "
token_secret = " "

tweets=[]
class StdOutListener(StreamListener):
    
    def __init__(self, api=None):
        super(StdOutListener, self).__init__()
        self.num_tweets = 0
        
    def on_data(self, data):
        tweets.append(data)
        print('Tweet #', len(tweets))
        if len(tweets)>=20000:
           with open('tweet_data.json','w') as f:
               json.dump(tweets,f, indent=4)
               return False
        return True
  
    def on_error(self, status):
        print (status)
        
if __name__ == '__main__': # twitter authentification & connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, token_secret)
    stream = Stream(auth, l)

    #filter tweets to capture data by keywords
    stream.filter(track=['#TakeTheKnee', '#TakeAKnee'])


