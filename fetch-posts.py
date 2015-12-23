import argparse
import requests
import pprint
import unicodecsv as csv

pp = pprint.PrettyPrinter(indent=4)

TRENDING_URL = "https://services.wikia.com/discussion/{0}/forums/{1}?viewableOnly=false&sortKey=trending&limit=900"
THREAD_URL = "https://services.wikia.com/discussion/{0}/threads/{1}?responseGroup=full&sortDirection=descending&sortKey=creation_date&limit=10&viewableOnly=false"


def get_thread_id_from_trending(data):
    thread_ids = []
    for post in data['_embedded']['doc:threads']:
        thread_ids.append(post['_embedded']['firstPost'][0]['threadId'])

    return thread_ids

def cookie_header(access_token):
    return {"Cookie": "access_token={}".format(access_token)}

def get_trending_data(site_id, forum_id, access_token):
    print(TRENDING_URL.format(site_id, forum_id))
    return requests.get(TRENDING_URL.format(site_id, forum_id), headers=cookie_header(access_token)).json()

def get_thread(site_id, thread_id, access_token):
    return requests.get(THREAD_URL.format(site_id, thread_id), headers=cookie_header(access_token)).json()

def get_posts_from_thread(data):
    if 'doc:posts' in data['_embedded']:
        return data['_embedded']['doc:posts']
    else:
        return []





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', "--site_id", dest="site_id", help='site id')
    argparser.add_argument('-f', "--forum_id", dest="forum_id", help='forum id')
    argparser.add_argument('-t', "--thread_id", dest="thread_id", help='thread id')
    argparser.add_argument('-a', "--access_token", dest="access_token", help='users access token')
    argparser.add_argument('-o', "--csv", dest="csv", help='csv file to write to')

    options = argparser.parse_args()
    trending_threads = get_thread_id_from_trending(get_trending_data(options.site_id, options.forum_id, options.access_token))
    threads = [get_posts_from_thread(get_thread(options.site_id, thread_id, options.access_token)) for thread_id in trending_threads]
    non_empty_threads = [thread for thread in threads if len(thread) > 0]
    posts = [post for sublist in non_empty_threads for post in sublist]

    with open(options.csv, "wb") as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for post in posts:
            csvwriter.writerow([post['isDeleted'], post['upvoteCount'], post['rawContent']])




