import googleapiclient.discovery
import pandas as pd

class Youtube:
    def __init__(self, api_service, api_version, key):
        self.youtube = googleapiclient.discovery.build(api_service, api_version,developerKey=key)

    def get_channelByCountry(self, query):
        # request for channel list
        search = self.youtube.search().list(q = query, type='Channel', order='viewCount', part = "id, snippet", maxResults=100000).execute()
        # extracting the results from search response
        results = search['items']

        # empty dictionary to store channel title and Id
        channels = {}
        while 'nextPageToken' in  search.keys():
            # extracting required info from each result object
            for result in results:
                if result['id']['kind'] == "youtube#channel":
                    channels[result['snippet']['channelTitle']] = result['snippet']['channelId']
            search = self.youtube.search().list(q = query, type='Channel', order='viewCount', part = "id, snippet", pageToken=search['nextPageToken'], maxResults=10000).execute()
            results = search['items']
        return channels

    def get_channelDetails(self, channels):
        # empty dictionary to store channelDetails
        channelDetails = {}

        # loop through channels
        # request for channel details
        for title, id in channels.items():
            data = self.youtube.channels().list(part="snippet,contentDetails,statistics", id=id).execute()
            channelDetails[title] = data['items'][0]['statistics']
        return channelDetails

    def df_channelDetails(self, query):
        channel = self.get_channelByCountry(query)
        channelDetails = self.get_channelDetails(channel)
        # get df index
        index = [j for j in channelDetails.keys()]

        # empty dictionary
        data={}
        # get df columns
        data['viewCount'] = [int(j['viewCount']) for j in channelDetails.values()]
        data['subscriberCount'] = [int(j['subscriberCount']) if 'subscriberCount' in j.keys() else None for j in channelDetails.values()]
        data['videoCount'] = [int(j['videoCount']) for j in channelDetails.values()]
        
        df = pd.DataFrame(data, index=index)
        return df.sort_values(by='viewCount', ascending=False)