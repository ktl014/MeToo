def freq_dict(file_list,noun_type='all',most_common_no=100):
    '''Returns dictionary of string to number, i.e., noun to number of occurences, for tweets stored in files in specified file list.
    Can show proper or common nouns selectively.
    Default number of most frequent names/nouns = 100'''
    assert (noun_type=='all' or noun_type=='proper'), 'Only \'proper\' and \'all\' options allowed'
    import string
    import re
    import pandas as pd
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    ret_list = []
    for file_name in file_list:
        df = pd.read_csv(file_name)
        tweet_texts = df['text']
        for tweet in tweet_texts:
            orig_tweet = tweet
            #CLEAN THE TWEET
            tweet = re.sub(r"â€|Ã©|™|¦|œ|", "", tweet)
            tweet = ' '.join(word for word in tweet.split(' ') if not (re.match(r"http",word) or re.match(r".*\..*/",word) or re.match(r"#",word) or re.match(r"\d+",word) or (len(word)==1)))
            tweet = tweet.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
            tweet = [w for w in tweet.split(' ') if not w in stop_words]
            tweet = ' '.join(w for w in tweet)
            if noun_type=='proper':
                prop_noun_string = pos_tag(tweet.split())
                proper_nouns = [word for word,pos in prop_noun_string if pos == 'NNP']
                fake_proper_nouns = ['Women','Im','Thank','Please','Sexual','New','Movement','My','How','No','NOT','Men','Me','RT','THE','Hey','Year','Dont','TIME','â\x81','YOU','Are','Do','Join','Harassment','U','Her','News','Judge','Too','Rape','Times','State','ALL','Remember','March','Ive','Just','Assault','Time','IT','Let','Read','TO','Has','Up','Be','So','Person','THIS','Did','Twitter','NO','Media','R','Stop','Cold','Where','Watch','Dear','Baby','Street','Police','Today','South','Against','Outside','Heard','Justice','Check','Great','Or','All','Fake','Ill','Ad','Listen','Green','IN','Act','Christmas','Singer','Well','Who','ON','Day','NOW','Wow','Get','University','Man','TheRestlessQuil','NBC']
                ret_list.extend([word for word in proper_nouns if not word in fake_proper_nouns])
            else:
                tokens = word_tokenize(tweet.lower())
                is_noun = lambda pos: pos[:2] == 'NN'
                nouns = [word for (word, pos) in nltk.pos_tag(tokens) if is_noun(pos)]
                #ret_list.extend([word for word in nouns]) #FOR ACTUAL NOUNS
                #FOR ROOT WORD OF NOUN ONLY
                stemmed = [porter.stem(word) for word in nouns]
                ret_list.extend([word for word in stemmed])
    c = Counter(ret_list)
    return (dict(c.most_common(most_common_no)))
