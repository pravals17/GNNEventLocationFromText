#utils file
def haversineDistance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def binary_search(arr, low, high, x): 
    """
    Implementation of a binary search
    """
    # Check base case 
    if high >= low:          
        mid = (high + low) // 2
        if arr[mid] == x: 
            return True 
  
        # If element is smaller than mid, then it can only 
        elif arr[mid] > x: 
            return binary_search(arr, low, mid - 1, x) 
  
        # Else the element can only be present in right subarray 
        else: 
            return binary_search(arr, mid + 1, high, x) 
  
    else: 
        # Element is not present in the array 
        return False
    
# Funciton for SANS i.e., NER for India   
def has_loc_suffix(word):
    """
    Identify if a given word has a suffix. SANS helper function.
    """
    #suffix = ['abad', 'adri', 'bagh', 'bakkam', 'vakkam', 'bandar', 'basti', 'ganj', 'gangj','gaon','garh','giri','gudi','gunta','gutta', 'ghar', 'halli', 'kere', 'keri', 'konda', 'nagar', 'nath','palli', 'pur','puram','pura', 'sandra']
    suffix = ['pur', 'adi', 'iya', 'gaon', 'orest', 'ani', 'patti', 'palle', 'khurd', 'purwa', 'dih', 'chak', 'minor', 'garh', 'singh', 'uru', 'palem', 'ain', 'ganj', 'anga', 'and', 'padu', 'uzurg', 'utary', 'pet', 'attu', 'ane', 'angi', 'kh.', 'bk.'] #most common suffixes (top 30) obtained from suffix extractor
    for suf in suffix:
        if word.lower().endswith(suf) and word.lower() != suf:
            return(True)
    return(False)

def has_loc_domainwords(word, postagprev):
    """
    Identify if a given word has a domain word. SANS helper function.
    """
    domainwords = ['nagar','colony','street','road','hill','river','temple','village','sector', 'district', 'taluk', 'town', 'mutt', 'fort', 'masjid', 'church']
    for entry in domainwords:
        if word.lower() == entry:
            if postagprev in ['NNP','NNPS']:
                return(True)
    return(False)

def has_prep(word, postagnext):
    """
    Identify if a given word has a suffix. SANS helper function.
    """
    preps = ['near', 'via', 'in', 'from', 'between', 'at', 'versus', 'like', 'towards', 'of', 'toward', 'across'] # Place name prepositions with location likelihood scores greater than 0.1
    for prep in preps:
        if word.lower() == prep:
            if postagnext in ['NNP','NNPS']:
                return(True)
    return(False)

def get_wordshape(word):
    """
    Identify the shape of a given word. SANS helper function.
    """
    shape1 = re.sub('[A-Z]', 'X',word)
    shape2 = re.sub('[a-z]', 'x', shape1)
    return re.sub('[0-9]', 'd', shape2)

def is_in_gazetteer(word, postag, placeNames):
    """
    Identify if a given word is in the gazetteer. SANS helper function.
    """
    if postag in ['NNP', 'NNPS'] and word in placeNames:
        return True
    return False

def word2features(sent, i,placeNames):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.shape':get_wordshape(word),
        'hassuffix:':has_loc_suffix(word),
        'is_in_gazetteer:':is_in_gazetteer(word, postag,placeNames),
        'wordallcap': len([x for x in word if x.isupper()])==len(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:wordallcap': len([x for x in word1 if x.isupper()])==len(word1),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:prep': has_prep(word, postag1),
            '+1:hasdomain':has_loc_domainwords(word1,postag),
            '+1:wordallcap': len([x for x in word1 if x.isupper()])==len(word1),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent,placeNames):
    return [word2features(sent, i,placeNames) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


def getLocationsGazPOSandSAPR(reports, placeNames):
    """
    Get all the locations mentioned in text using Gazetteer and POS
    """
    preps = ['near', 'via', 'in', 'from', 'between', 'at', 'versus', 'like', 'towards', 'of', 'toward', 'across']
    suffix = ['pur', 'adi', 'iya', 'gaon', 'orest', 'ani', 'patti', 'palle', 'khurd', 'purwa', 'dih', 'chak', 'minor', 'garh', 'singh', 'uru', 'palem', 'ain', 'ganj', 'anga', 'and', 'padu', 'uzurg', 'utary', 'pet', 'attu', 'ane', 'angi', 'kh.', 'bk.']

    tokenizedContent = word_tokenize(reports)
    tokenized_tags = pos_tag(tokenizedContent)
    locationgaz = []
    locationsapr = []
    for j in range(0,len(tokenized_tags)):
        if tokenized_tags[j][1] in ['NNP', 'NNPS']:
            if binary_search(placeNames, 0, len(placeNames) -1 , tokenized_tags[j][0]) == True:
                locationgaz.append(tokenized_tags[j][0].title())
            else:
                
                if tokenized_tags[j-1][0].lower() in preps: 
                    place = tokenized_tags[j][0].title()
                    k = j + 1
                    while k < len(tokenized_tags): #Get all the NNPs before a prep. They may represent a multi-word place names
                        if tokenized_tags[k][1] == 'NNP' or tokenized_tags[k][1] == 'NNPS':
                            place = place + ' ' + tokenized_tags[k][0].title()
                            k = k + 1
                        else: 
                            break
                    locationsapr.append(place)
                
                for suff in suffix:
                    if tokenized_tags[j][0].lower().endswith(suff):
                        locationsapr.append(tokenized_tags[j][0].title())
    
    return(locationgaz, locationsapr)

def get_locs_NER(doc, NER, placeNames):
    """Get the locations present in the documents"""
    output = get_NER_5WNER(sent_tokenize(doc), NER,placeNames) #sentence tokenize the document and pass it as parameter to get_NER_5WNER to get the NER tags
    locations = concat_placenames(output)
    locations = [plc.title().strip() for plc in locations if plc in doc]
    
    list_set = set(locations) 
    # convert the set to the list 
    unique_locs = (list(list_set))
    return unique_locs

def concat_placenames(original_tags):
    """
    Combine names of the locations if the locations consists of two words eg. New Delhi
    """
    locations = []
    l = len(original_tags)
    i=0;
    # Iterate over the tagged words.
    while i<l:
        #print(i)
        e,t = original_tags[i]
        # If it's a location, then check the next 3 words.
        if t == 'LOCATION':
            j = 1
            s = e
            # Verify the tags for the next 3 words.
            while i+j<len(original_tags):
                # If the next words are also locations, then concatenate them to make a longer string. This is useful for place names with multiple words. e.g., New Delhi
                if original_tags[i+j][1] == 'LOCATION':
                    s = s+" "+original_tags[i+j][0]
                    j+=1
                else:
                    break
            i = i+j
            # Save the locations to a locations list
            locations+=[s]
        else:
            i=i+1
        #print(locations)
    return locations


def get_NER_5WNER(doc, NER,placeNames):
    """
    Use SANS to identify location names in text.
    """
    input_ner = []
    tags = []
    for sent in doc:
        text_tokens = pos_tag(word_tokenize(sent))
        input_ner.append(sent2features(text_tokens,placeNames)) #create input to the 5WNER tagger from the text file
        tags.extend(word_tokenize(sent))
    output_ner = (NER.predict(input_ner))
    ner_list = [item for sublist in output_ner for item in sublist] #convert the list of list i.e. output_ner which contains all the NER tags for each sentence of the report
    ner_tags = [(w,t) for w,t in zip(tags, ner_list)] #give output same as that of Stanford NER [(word, NER tag)]
    return(ner_tags)

def getNodeFeatures(names, report, arrangedconames, where, what,modelBert, tokenizerBert):
    "get a 2d matrix of the shape [Num Coplaces, num of possible footprints of a toponym]"
    
    allNodeFeatures = []
    
    for name in names:

        features = []
        input_ids = torch.tensor(tokenizerBert.encode(name)).unsqueeze(0)  # Batch size 1
        outputs = modelBert(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        features.extend(last_hidden_states[:,0,:].squeeze().tolist()) #get the CLS (contextual embeddings from BERT)
        placeemb = last_hidden_states[:,0,:] #.squeeze().tolist()
        allNodeFeatures.append(np.array(features))
    allNodeFeatures = np.asarray(allNodeFeatures)

    return(torch.tensor(allNodeFeatures, dtype=torch.float))

def getCoToponyms(report, NER, placeNames, df_gaz):
    """
    Extract all the gazetteer place names that cooccur with a given non-gazetteer place name in the corpus.
    """
    sansplc = []
    
    sansplc  = sansplc + get_locs_NER(report, NER, placeNames)
        
    cooccuringplaces_lats = []
    cooccuringplaces_longs = []
    cooccuringplaces_latlong = []
    allplaces = []
    
    indices = {plc:report.title().index(plc) for plc in sansplc}
    sortedPlaces = sorted(indices,key=lambda x:indices[x])

    #Get the geocoordinates of the cooccuring places
    for place in sortedPlaces:
        try:
            index = [df_gaz.index[df_gaz['Placename'] == re.sub('[^A-Za-z0-9(),.\']+', ' ', place).title()][0]]
            lats = [df_gaz['Lat'][k] for k in index]
            longs = [df_gaz['Long'][k] for k in index]
            for i in range(0, len(lats)):
                allplaces.append(place)
                cooccuringplaces_lats.append(lats[i])
                cooccuringplaces_longs.append(longs[i])
                cooccuringplaces_latlong.append([lats[i],longs[i]])
        except:
            allplaces.append(place)
            cooccuringplaces_lats.append(0)
            cooccuringplaces_longs.append(0)
            cooccuringplaces_latlong.append([0,0])
            continue
    for i in range(0, len(cooccuringplaces_lats)):
        if cooccuringplaces_lats[i] == 0:
            if len(cooccuringplaces_lats) < 1:
                cooccuringplaces_lats[i] = 20.79
                cooccuringplaces_longs[i] = 79.58
            else:
                cooccuringplaces_lats[i] = sum(cooccuringplaces_lats)/len(cooccuringplaces_lats)
                cooccuringplaces_longs[i] = sum(cooccuringplaces_longs)/len(cooccuringplaces_longs)
                    
    return (allplaces, cooccuringplaces_lats, cooccuringplaces_longs)
    
def createDataObjects(df, SANS, modelBert, tokenizerBert, country, state, district, subdistrict, placename):
    """
    Extract place names from the documents and create graphs using the place names.
    """
    data_list = []
    for i,row in tqdm(df.iterrows()):  
        print(i)
        y=[]
        lats = []
        longs = []
        report  = row['Title'] +'. ' + row['Content']
        conames,colats, colongs = getCoToponyms(row['Title'] +'. ' + row['Content'], SANS, placeNames, df_gaz)
        
        arrangedconames = {'country': [], 'state':[], 'district': [], 'subdistrict':[], 'placename':[], 'nongaz':[]}
        for i in range(0,len(conames)):
            if binary_search(state, 0, len(state) - 1, conames[i].lower()) == True:
                arrangedconames['state'].append(conames[i])
            elif binary_search(district, 0, len(district) - 1, conames[i].lower()) == True:
                arrangedconames['district'].append(conames[i])
            elif binary_search(subdistrict, 0, len(subdistrict) - 1, conames[i].lower()) == True:
                arrangedconames['subdistrict'].append(conames[i])
            elif binary_search(placename, 0, len(placename) - 1, conames[i].lower()) == True:
                arrangedconames['placename'].append(conames[i])
            else:
                arrangedconames['nongaz'].append(conames[i])


        for i in range(0,len(conames)):
            if conames[i].strip().lower() == str(row['Where']).split(', ')[0].strip().lower(): #when ro['where'] is empty in excel file, it is read as float variable. So, str is used to convert it to string and supress error for strip() on a float variable.
                y.append(1)
                lats.append(colats[i])
                longs.append(colongs[i])
            else:
                y.append(0)
                if i == len(conames)-1 and len(lats) == 0:
                    lats.append(0)
                    longs.append(0)
        y = torch.tensor(y).type(torch.LongTensor)
        lats = torch.tensor(lats).type(torch.LongTensor)
        longs = torch.tensor(longs).type(torch.LongTensor)
        G = nx.Graph()
        i = 0
        while i <= len(conames)-2:
            dist = abs(1 - (haversineDistance(colongs[i], colats[i], colongs[i+1], colats[i+1])/3200))
            edgeweight = dist
            G.add_edge(conames[i], conames[i+1], weight=edgeweight)
            i = i + 1
        if len(conames) == 1:
            G.add_node(conames[0])

        edgeList = from_networkx(G).edge_index
        if len(G.edges) > 0:
            edgeFeature = from_networkx(G).weight
        else:
            edgeFeature = torch.empty((0), dtype=torch.float32)
                    
        data = Data(edge_index=edgeList, 
                    x = getNodeFeatures(list(nx.nodes(G)), report, arrangedconames, str(row['Where']).strip(), str(row['What']).strip(), modelBert, tokenizerBert),
                    lats = lats, longs=longs, colats = colats, 
                    colongs=colongs, y=y) #x=getNodeFeatures(list(nx.nodes(G)), report, arrangedconames, str(row['Where']).strip(), str(row['What']).strip(), modelBert, tokenizerBert),
        data.edge_attr = edgeFeature
        data_list.append(data)
    return data_list