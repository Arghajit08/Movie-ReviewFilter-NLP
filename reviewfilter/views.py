from django.http import HttpResponse
from django.shortcuts import render
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
def home(request):
    return render(request,"index.html")
def result(request):
    X_train=["This was an awesome movie",
        "Great movie!I liked it",
        "Happy Ending! awesome acting by the hero",
        "loved it! truly great",
        "bad not upto the mark",
        "could have been better",
        "surely a disappointing movie"]
    y_train=[1,1,1,1,0,0,0]
    review=request.GET['name']
    X_test=[]
    X_test.append(review)
    
    tokenizer=RegexpTokenizer(r'\w+')
    en_stopwords=set(stopwords.words('english'))
    ps=PorterStemmer()

    def getCleanedText(text):
        text=text.lower()

        tokens=tokenizer.tokenize(text)

        new_tokens=[token for token in tokens if token not in en_stopwords]

        stemmed_tokens= [ps.stem(tokens) for tokens in new_tokens]

        clean_text=" ".join(stemmed_tokens)

        return clean_text
    
    X_clean = [getCleanedText(i) for i in X_train]
    Xt_clean = [getCleanedText(i) for i in X_test]

    cv = CountVectorizer(ngram_range=(1,2))

    X_vec=cv.fit_transform(X_clean).toarray()

    Xt_vec=cv.transform(Xt_clean).toarray()

    from sklearn.naive_bayes import MultinomialNB

    nn=MultinomialNB()

    nn.fit(X_vec,y_train)

    res=nn.predict(Xt_vec)
    print(res)

    if res==[0]:
        return render(request,"result.html",{'result':"This is a negative review"})
    
    else:
        return render(request,"result.html",{'result':"This is a positive review"})

    