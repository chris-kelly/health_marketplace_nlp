########################################
## IMPORT LIBRARIES
########################################

import requests, urllib, re, json as js, numpy as np, pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

########################################
## CREATE FUNCTIONS
########################################

def data_import(query = '', category = '', page_number = 1):
    """
    Imports the search results on a given page, passing a query/category
    :param query: Freetext query to filter results (string)
    :param category: Category to filter results to, should be exact (string)
    :param page_number: Number of page to scrape (integer)
    :return the_dict: Dictionary of the json response
    """
    if category != '':
        category = 'category:\"' + category + '\"'
    url = "http://techfinder.stanford.edu/api/searchResults"
    querystring = {"qs": query
                   ,"fs": category
                   ,"p": str(page_number)
                   ,"docType":"pub-opportunities"
                   ,"noSample":"false"
                   ,"precision":"100"}
    headers = {'accept': "application/json"} # "application/json, text/plain, */*"
    response = requests.request("GET", url, headers=headers, params=querystring)
    the_dict = js.loads(response.text)
    return(the_dict)

def data_parse(json_import):
    """
    Parses the json object returned from data_import()
    """
    df = pd.DataFrame()
    for i in range(len(json_import['results'])):
        the_dict = dict()
        the_dict['id'] = json_import['results'][i]['id']
        the_dict['fileslug'] = json_import['results'][i]['opportunities'][0]['fileNumberSlug']
        the_dict['title'] = json_import['results'][i]['opportunities'][0]['title']
        the_dict['sf_abstract'] = json_import['results'][i]['opportunities'][0]['abstract']
        tag_string = str('')
        for j in json_import['results'][i]['opportunities'][0]['tags']:
            tag_string = tag_string + j['id'] + '; '
        the_dict['tags'] = tag_string
        df = df.append(the_dict, ignore_index=True)
    return(df)

def import_all_data(query = ''):
    """
    Loops through data_import() and data_parse() to return data for all pages given a query/category
    (parallize it?)
    """
    json_import = data_import(query = query, page_number = 1)
    df = data_parse(json_import)
    total_items = np.ceil(json_import['totalItems']/json_import['itemsPerPage'])
    for i in tqdm(range(1, int(total_items))):
        json_import = data_import(query = query, page_number = i + 1)
        df = df.append(data_parse(json_import), ignore_index=True)
    return(df)

def paper_page_request(fileslug = 'S01-139A_t-cell-regulatory-genes-associated'):
    """
    Given a paper file slug, returns the paper's own page specific json object 
    """
    url = 'http://techfinder.stanford.edu/api/pub-opportunity/' + fileslug
    headers = {'accept': 'application/json'
               ,'referer': 'http://techfinder.stanford.edu/technologies/' + fileslug}
    response = requests.request("GET", url, headers=headers)
    return(js.loads(response.text))

def find_links_in_html_string(string = None):
    """
    Extract all html links in a string
    """    
    if string is not None:
        string = re.findall('href=[\"][^\"]*', string)
        string = [i[6:] for i in string]
        return(string)
    
def find_links_in_non_html_string(string = None):
    """
    Extract all non-html links in a string
    """    
    if string is not None:
        string = re.findall('https*:\/\/[^\>|^<|^\"]*', string)
        string = [i[:-2] for i in string]
        return(string)
    
def extract_publication_text(string):
    z = string
    first = z.find('<h2>Publications</h2>')
    if first > 0:
        y = z[(first + 21):]
        return(y[:y.find('<h2>')])
    else:
        return('')

def extract_publication_link(string):
    x = extract_publication_text(string)
    return(find_links_in_non_html_string(x))

def link_import(fileslug = 'S01-139A_t-cell-regulatory-genes-associated'):
    x = paper_page_request(fileslug)
    return(extract_publication_link(x['result']['body']))
    # return(extract_publication_link(x['result']['body']).replace('"', ''))
    
# def abstract_extract_nature(url = "https://www.nature.com/articles/s41566-018-0217-1"):
#     response = requests.request("GET", url)
#     response = response.text
#     response = response[13+response.find('Abstract'):response.find('Access options')]
#     response = re.sub('<[^>]*>|\\n|<h[0-9].*', '', response)
#     return(re.sub('\s*$', '', response))    

# def abstract_extract_ncbi(url = 'http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=pubmed&dopt=Abstract&list_uids=15585880'):
#     response = requests.request("GET", url)
#     response = response.text
#     response = re.search('<h3>Abstract.*class=\"aux\"', response)
#     response = response.group(0)[12:]
#     return(re.sub('<[^>]*>|\\n|<div class="aux', '', response))