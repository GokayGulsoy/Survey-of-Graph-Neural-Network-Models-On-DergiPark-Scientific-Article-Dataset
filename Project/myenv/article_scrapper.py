from bs4 import BeautifulSoup
import pandas as pd
import requests
import string 

def generate_dataset():
    journal_archive_list = [("https://dergipark.org.tr/tr/pub/bbmd/archive", 29, "Computer Sciences"), ("https://dergipark.org.tr/tr/pub/bbd/archive", 9, "Computer Sciences"), ("https://dergipark.org.tr/tr/pub/asbi/archive", 7, "Social Sciences"), 
                            ("https://dergipark.org.tr/tr/pub/eujhs/archive", 11, "Health Sciences"), ("https://dergipark.org.tr/tr/pub/aucevrebilim/archive", 21, "Environmental Sciences"), ("https://dergipark.org.tr/tr/pub/ucbad/archive", 20, "Environmental Sciences"), 
                            ("https://dergipark.org.tr/tr/pub/jssr/archive", 15, "Sports Sciences")]
    
     # create dataframe to represent corpus
    article_df = pd.DataFrame(columns=["authors", "turkish_title", "turkish_keywords", "turkish_abstract", "english_title", "english_keywords", "english_abstract", "citations", "category"])
    for (journal_url, scrap_count, article_category) in journal_archive_list:
       article_df = create_article_corpus(article_df, journal_url, scrap_count, article_category)
        

    print(article_df.head())
    # saving created article dataframe as a corpus to article_corpus.csv file
    article_df.to_csv("article_corpus.csv", sep=",", index=False)


def create_article_corpus(article_df, journal_url, scrap_count, article_category):
    journal_archive = requests.get(journal_url)
    scrapper = BeautifulSoup(journal_archive.content, "html.parser")

    table_with_volume_and_number_info = scrapper.find("table", {"class": "table"})
    volume_and_number_anchors = table_with_volume_and_number_info.select("[href]")
    
    """
        at the end of the iteration, all the archive
        will have been scraped, but limitting number 
        can be set to indicate how many volumes are 
        going to be scrapped
    """
    for count, volume_and_number_anchor in enumerate(volume_and_number_anchors):
        if count < scrap_count:
            volume_and_number_url = volume_and_number_anchor["href"]
            
            if not (volume_and_number_url.startswith("https")):
                volume_and_number_url = "https:" + volume_and_number_url
                
            article_df = scrap_specific_volume_and_number(volume_and_number_url, article_df, article_category)   
        else:
            break
    
    return article_df
 


def scrap_specific_volume_and_number(volume_and_number_url,article_df, article_category):
    # scrap all articles in a specific volume and specific number
    article_listing_page = requests.get(volume_and_number_url)  
    scrapper = BeautifulSoup(article_listing_page.content, "html.parser")
    
    # https://dergipark.org.tr/tr/pub/bbmd/issue/88381/1484477
    # find all the anchor tags and extract their
    # link from href attribute to scrap each article
    article_anchors = scrapper.find_all("a", {"class": "card-title article-title"})
    count_of_articles = 0
    for article_anchor in article_anchors:
        article_url = article_anchor["href"]
        
        if not (article_url.startswith("https")):
            article_url = "https:" + article_url
        
        print(article_url)
        count_of_articles += 1
        en_tr_article_dict = scrap_article(article_url, article_category)
        # if any section of any article is missing 
        # then do not add that article to dataframe
        if (en_tr_article_dict == "skip"):
            continue
        
        # update the dataframe with new article data
        new_artice_df = pd.DataFrame([en_tr_article_dict])
        article_df = pd.concat([article_df, new_artice_df], ignore_index=True)
    
    print("Number of articles scrapped: ", count_of_articles)     
    print(article_df.head())
    return article_df
        

def scrap_article(article_url, article_category):
    article_page_req = requests.get(article_url)
    scrapper = BeautifulSoup(article_page_req.content, "html.parser")

    # scrap both english and turkish version as a dictionary
    turkish_dict = scrap_turkish_version(scrapper, article_category)
    english_dict = scrap_english_version(scrapper, article_category)
    
    if not (turkish_dict == "skip" or english_dict == "skip"):
        # combine two dictionaries into single
        en_tr_article_dict = {**turkish_dict, **english_dict}
        print(en_tr_article_dict)
        
        return en_tr_article_dict
    
    else:
        return "skip"
    

def scrap_english_version(scrapper, article_category):
     en_tab = scrapper.find("a", attrs={"href": "#article_en"})
     
     # if EN tab is not available do not add this article to dataframe
     if (en_tab is None): 
        return "skip"
    
     en_tab_content = scrapper.find("div", {"id": "article_en"})

     english_article_dict = get_article_dict(en_tab_content, "en", article_category) 
     return english_article_dict
     


def scrap_turkish_version(scrapper, article_category):
    tr_tab = scrapper.find("a", attrs={"href": "#article_tr"})
     
    # if TR tab is not available do not add this article to dataframe
    if (tr_tab is None): 
        return "skip"
    
    
    tr_tab_content = scrapper.find("div", {"id": "article_tr"})
    turkish_tab_content = get_article_dict(tr_tab_content, "tr", article_category)

    return turkish_tab_content



def create_article_entries(content, lang, article_category):
    # getting the paragraph containing author information
    paragraph_with_author_info = content.find(attrs={"class": "article-authors"})
    if (paragraph_with_author_info is None):
        return "skip"
    
    author_names = []
    ## ACQUIRING AUTHOR NAMES
    # iterate through all children and select anchor
    # tags containing author names and obtain their 
    # navigable strings
    for child in paragraph_with_author_info.children:
        if (child.name == "a"):
            author_names.append(child.get_text().strip())

    print(author_names)

    ## ACQUIRING ARTICLE TITLE
    header_with_title_info = content.find(attrs={"class": "article-title"})
    if (header_with_title_info is None):
        return "skip"
    
    
    article_title = header_with_title_info.get_text().strip()

    print(article_title)

    ## ACQUIRING ARTICLE KEYWORDS
    div_with_keyword_info = content.find(attrs={"class": "article-keywords data-section"})
    if (div_with_keyword_info is None): 
        return "skip" 
    
    keywords = []
    # iterate through all children and select anchor
    # tags containing keywords and obtain their 
    # navigable strings
    for child in div_with_keyword_info.children:
        if (child.name == "p"):
            for child_of_paragraph in child.children:
                if (child_of_paragraph.name == "a"):
                    keyword = remove_punctuations(child_of_paragraph.get_text())
                    keywords.append(keyword)

    if keywords[0] == "-" or keywords == []:
        return "skip"

    print(keywords)

    ## ACQUIRING ARTICLE ABSTRACT
    div_with_abstract_info = content.find(attrs={"class": "article-abstract data-section"})

    # iterate through all children and select 
    # paragraph tag then obtain it's navigable string
    abstract = ""
    for child in div_with_abstract_info.children:
        if (child.name == "p"):
            abstract = child.get_text()
    
    
    if ((abstract.strip() == "-") or (abstract.strip() == "...") or (abstract == "")):
       return  "skip"

    print("\n", abstract)

    ## ACQUIRING ARTICLE CITATIONS
    div_with_citation_info = content.find(attrs={"class": "article-citations data-section"})
    if (div_with_citation_info is None):
        return "skip"

    citations = []
    ul_with_citation_info = div_with_citation_info.find("ul")
    for child in ul_with_citation_info.children:
        if (child.name == "li"):
            citations.append(child.get_text())

    if (citations[0] == "-" or citations == []):
        return "skip"    

    print("\n",citations)
    
    if (lang == "tr"):
       return {"authors": author_names, "turkish_title": article_title, "turkish_keywords": keywords, "turkish_abstract": abstract, "category": article_category}
    
    else: # as author names, citations, and category are same for both language versions (tr and en), we don't need to repeat them
          # when creating article dictionary 
        return {"english_title": article_title, "english_keywords": keywords, "english_abstract": abstract, "citations": citations}
    
    
    
def get_article_dict(content, lang, article_category):
     article_dict = create_article_entries(content, lang, article_category)
     
     return article_dict
     


# function to remove all punctuations from a keyword
def remove_punctuations(keyword):
    no_punct = "".join(char for char in keyword if char not in string.punctuation)

    return no_punct

# call generate dataset function
# to create .csv file containing all article information
generate_dataset()