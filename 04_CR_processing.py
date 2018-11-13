import re
import nltk.data
from gensim.models import word2vec

###############################################################################
#Get the texts
###############################################################################

file_name1 =  'C:\\Users\\Enrico\\Desktop\\Projet Innovation\\EXAMPLE_CR_1.txt'
file_name2 =  'C:\\Users\\Enrico\\Desktop\\Projet Innovation\\EXAMPLE_CR_2.txt'
f = open(file_name1,'r')
text1 = f.read()
f.close()

f = open(file_name2,'r')
text2 = f.read()
f.close()
#print(text)

f = open('French stopwords.txt','r')
stopwords = f.readlines()
stopwords = [ w[:-1] for w in stopwords]


tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

###############################################################################
#Parse the reviews
###############################################################################
    
def text_to_wordlist( raw_review , remove_stopwords=False):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    TAG_RE = re.compile(r'<[^>]+>')
    review_text = TAG_RE.sub('', raw_review)
    
    #review_text = BeautifulSoup(raw_review, "lxml").get_text() 
    #print('review text',review_text)
    #
    # 2. Remove non-letters        
    review_text = re.sub(":", " ", review_text) 
    review_text = re.sub(" - ", " ", review_text)
    review_text = re.sub(r'\d{2}.\d{2}.\d{2,4}', " ", review_text) #I get rid of dates
    review_text = re.sub('[.]', " ", review_text)
   
    #review_text = re.sub(".\n", "", review_text)
    #print('lett+num only',lett_and_numb)
    #
    # 3. Convert to lower case, split into individual words
    words = review_text.split()    
    
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords)
        words = [w for w in words if not w in stops]
    
    #MANUAL REMOVAL
    i=0
    while i<len(words)-3 and words[i]!='HOSPITALISATION' and words[i+3]!='DU': #take away the header
        i+=1
    if i<len(words)-3:
        words=words[i+9:]
    
    i=0
    count=0
    n=len(words)
    while i<n-count:
        if len(words[i])==1 and re.match('[^0-9]',words[i]): #remove 1-chars except numbers
            words.pop(i)
            i-=1
            count+=1
        elif re.match('_{2,}',words[i]): #cut out the underscore lines
            words.pop(i)
            i-=1
            count+=1
        if (words[i]=='Signé' and words[i+1]=='électroniquement'): #rip off the end
            words=words[:i]
            break
        i+=1
    #END OF MANUAL REMOVAL    
     
    
    # 5. Return a list of words
    return(words)



# Define a function to split a review into parsed sentences
def text_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Tokenize to split the paragraph into sentences
    raw_sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',review)
    #raw_sentences = tokenizer.tokenize(review.strip()) #EQUIVALENT
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( text_to_wordlist( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
sentences = []  # Initialize an empty list of sentences

sentences+=text_to_sentences(text1,tokenizer,True)
sentences+=text_to_sentences(text2,tokenizer,True)


###############################################################################
#Word2Vec training
###############################################################################

# Set values for various parameters
num_features = 10    # Word vector dimensionality                      
min_word_count = 1    # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

#model.save(os.path.join('C:/Users/Enrico/Desktop/Projet Innovation/', "imdb_w2v.cpkt"))
model.wv.save_word2vec_format("DITEP_w2v.txt", fvocab=None, binary=False)
#model.save_word2vec_tsv_format()
# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
#model_name = "300features_40minwords_10context"
#model.save(model_name)
