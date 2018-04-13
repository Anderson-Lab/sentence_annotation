#!/usr/bin/python

import sys
import urllib
import re
import json

from bs4 import BeautifulSoup

import socket
socket.setdefaulttimeout(10)

erase = '\x1b[1A\x1b[2K'

cache = {}

file = open(sys.argv[2], 'w')

start = int(sys.argv[3])

tweet_lines = open(sys.argv[1]).readlines()

end = len(tweet_lines)
if len(sys.argv) >= 5:
    end = int(sys.argv[4])

tot = end - start
div = tot // 30
num = 0

if tot > 0:
    for i in range(start, end):
        line = tweet_lines[i]
        fields = line.rstrip('\n').split('\t')
        sid = fields[0]
        uid = fields[1]

        #url = 'http://twitter.com/%s/status/%s' % (uid, sid)
        # print url

        tweet = None
        text = "Not Available"
        if sid in cache:
            text = cache[sid]
        else:
            try:
                f = urllib.request.urlopen(
                    "http://twitter.com/%s/status/%s" % (uid, sid))
                # Thanks to Arturo!
                html = f.read()
                soup = BeautifulSoup(html)

                jstt = soup.find_all("p", "js-tweet-text")
                tweets = list(set([x.get_text() for x in jstt]))
                # print(len(tweets))
                # print(tweets)
                if(len(tweets)) > 1:
                    continue

                text = tweets[0]
                cache[sid] = tweets[0]

                for j in soup.find_all("input", "json-data", id="init-data"):
                    js = json.loads(j['value'])
                    if ('embedData' in js):
                        tweet = js["embedData"]["status"]
                        text = js["embedData"]["status"]["text"]
                        cache[sid] = text
                        break
            except Exception:
                continue

        if (tweet != None and tweet["id_str"] != sid):
            text = "Not Available"
            cache[sid] = "Not Available"
        text = text.replace('\n', ' ',)
        text = re.sub(r'\s+', ' ', text)
        # print json.dumps(tweet, indent=2)
        # print("\t".join(fields + [text]).encode('utf-8'))
        file.write("\t".join(fields + [text, '\n']))
        num += 1
        stars = num // div
        underscores = tot // div - stars
        print(' %s%s Finished %s/%s%s' %
              (stars * '*', underscores * '_', num, tot, erase))
