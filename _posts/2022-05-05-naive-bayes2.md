---
layout: post
title: Mutinomial Naive Bayes Classifier from scratch(Part 2)
subtitle: 
date: 2022-05-03
tags: [machine learning, deep learning, courses, coursera, Amharic]
comments: true
use_math: true

---




<center><h2>Vectorized Implementation of Multinomial Naive Bayes Classifier</h2></center>
<p markdown="1" style="text-align: justify;">In [Part 1](https://amaneth.github.io/2022/05/03/naive-bayes.html) we have discussed the Multinomial Naive Bayes Classifier. Here we'll see the implementation of it.
It's usual to do it using for loops, here I've done it in a vectorized way. It looks smart :).</p>
<h4>Preprocessing</h4>
<p style="text-align: justify;">The first step is preprocessing. Preprocessing is about cleaning the document and tokenizing it. First, we remove punctuations and stop words. We then lowercase it and split it into words. Optionally we can also do lemmatization. Lemmatization is reducing a word to its root word. For example words love, loved, and loving will be lemmatized to one common word love.  In addition to this, we add each word to the vocabulary. The vocabulary contains unique words of the whole document with some id assigned. </p>

```python
  def lemmantize(self, word):
    stemmer = SnowballStemmer(language = 'english')
    lemmantizer = WordNetLemmatizer()
    stemized= stemmer.stem(word,) 
    lemmantized= lemmantizer.lemmatize(stemized,)
    return lemmantized

  def addto_vocublary(self,word):
    self.vocublary[word]=self.id
    self.id+=1

  def tokenize(self, documents, train=True, lemmantize=False):
    tokenized_docs=[]
    for document in documents:
      tokenized_doc=[]
      words =re.sub('[^A-Za-z ]+', '', document).split()
      for word in words:
        word =word.lower()
        if word not in self.stop_words:
          if lemmantize:
            word=lemmantize(word)
          if train:
            if word not in self.vocublary:
              self.addto_vocublary(word)
          tokenized_doc.append(word)  
      tokenized_docs.append(tokenized_doc)
    return tokenized_docs 
``` 
<br/>
<h4>Word frequency</h4>
<p style="text-align: justify;">As you remember from part-1 to calculate the prior and the likelihood probabilities we need to have the frequency of each word in each category. Let's put this in a matrix of shape the number of words by the number of classes.
</p>
```python
  def count_words(self, documents, labels):
    counts = np.zeros((len(self.vocublary), self.num_classes))
    for document, label in zip(documents,labels):
      for word in document:
        if word in self.vocublary.keys():
          counts[self.vocublary[word]][label] += 1
     
    return counts
```
<br/>
<h4>Prior probability</h4>
<p style="text-align: justify;">The prior probability is calculated as the number of documents in the category divided by the total number of documents. Here is how I did it.</p>
```python
  def prior_probablity(self):
    prior_probabilities = np.zeros(self.num_classes)
    total_documents = len(self.labels)
    for cat in range(self.num_classes):
      prior_probabilities[cat] = self.labels.value_counts()[cat]/total_documents
    self.prior = prior_probabilities
```
<br/>
<h4>Likelihood probability</h4>
<p style="text-align: justify;">The likelihood probability is to divide the frequency of the word in a category by the total number of words in the category. We already have the frequency matrix(counts) but there could be zeroes in the count matrix. This can create a big problem. For example, If we have a message "You've won a lottery!" and assume the word 'won' occurred 1000 times and the word 'lottery' occurred 0 times in spam messages, when we do the likelihood probability of each word and multiply them, the probability of the message will go to zero. This is not right. To solve this we add some constant value k(usually 1) to each word frequency. What we are doing here is Laplacian smoothing. <br/>
After adding the Laplacian smoothing value k to the Count matrix(the operation is broadcasting) we divide it by the category count. The country count is the vector of the number of words in each category. The count matrix is $\bbox[yellow]{num\_words\ x\ num\_classes}$ and the category count is $\bbox[yellow]{1\ x\ num\_classes}$. That means dividing the count by the category count is possible by broadcasting operation.   </p>
```python
  def likelihood_probablity(self, counts):
    vocabulary_size = len(self.vocublary)
    word_probablities=np.zeros((vocabulary_size, self.num_classes))
    category_count=np.sum(counts,axis=0)

    counts = counts+self.k
    category_count= category_count+ self.k*vocabulary_size
    word_probablities=counts/category_count
    self.likelihood=word_probablities
```
<br/>
<h4>Training</h4>
<p style="text-align: justify;">In the training we don't do anything different, we just call the functions we have defined earlier. Training in the multinomial naive byes is all about preparing the bag of words(the vocabulary), the likelihood, and prior probabilities.</p>
```python
  def fit(self, X, y):
    #fit the the train data to the model
    self.documents = self.tokenize(X, train=True)
    classes=np.unique(y)
    self.num_classes=len(classes)
    self.labels = y.map(lambda x: np.where(classes==x)[0][0])
  
    counts = self.count_words(self.documents, self.labels)
    self.prior_probablity()
    self.likelihood_probablity(counts)
```
<br/>
<h4>Prediction</h4>
<p style="text-align: justify;">The prediction is just to multiply the likelihood probabilities of the words in the document to be predicted by their prior probabilities. Here again, I used a vectorized implementation instead of for loops.<br/>
 Before going to the prediction we need to transform each word in the document to be predicted to their id in the vocabulary so that the already trained 'model' understands it. The transform function below returns a matrix of shape $\bbox[yellow]{len\_documents\ x\ len\_vocublary}$. The matrix is matrix of zeros and ones which indicates if the word in the documents is in the vocabulary. </p>
```python
  def transform(self, documents):
    sparse= np.zeros((len(documents), len(self.vocublary)))
    documents= self.tokenize(documents, train=False)
    for i, document in enumerate(documents):
      for word in document:
        if word in self.vocublary.keys():
          sparse[i][self.vocublary[word]]=1
    return sparse
```
<br/>
<p style="text-align: justify;">We add a new axis to the transformed matrix and the likelihood probabilities matrix so that the shape of the transformed matrix becomes $\bbox[yellow]{len\_documents\ x\ len\_vocabulary\ x\ 1}$ and the shape of the likelihood matrix becomes $\bbox[yellow]{1\ x\ len\_vocabulary\ x num\_classes}$. These two matrices are broadcastable, we can multipy them and get a matrix of shape $\bbox[yellow]{len\_documents\ x\ len\_vocabulary\ x\ num\_classes}$. We sum the resulted matrix across axis=1(the length of the vocabulary) and get a matrix of shape $\bbox[yellow]{len\_documents\ x\ num\_classes}$ which represents the likelihood probability of each document in each class. We add this to the prior probability matrix(again a broadcasting operation) and results in a matrix of shape $\bbox[yellow]{len\_documents\ x\ num\_classes}$ which is the probability of each document in being each calss. Finally, we take argmax of this and we get the prediction class of each document. I made the prediction to be done in a batch, this is to be memory efficient, unless the operation will fail if the size of the matrices is bigger.</p>
```python
  def predict(self, X, batch_size=500):
    #returns the prediction of categories for a test document
    batches = len(X)//batch_size
    if (len(X)//batch_size)!=0:
      batches=batches+1
    predictions=[]
    for batch in range(batches):
      X_new=X[batch*batch_size:(batch+1)*(batch_size)]
      X_new= self.transform(X_new)
      log_probablity= X_new[..., np.newaxis]*np.log(self.likelihood[np.newaxis,...])
      cat_log_probablity= log_probablity.sum(axis=1)

      category_probablity = np.log(self.prior) + cat_log_probablity
      prediction=np.argmax(category_probablity,axis=1)
      predictions.extend(prediction)
    return predictions
```
<br/>
<p style="text-align: justify;"> Finally, we can test the accuracy by calculating the percentage of correct predictions.
</p>
```python
  def accuracy(self, y, yPred):
    #returns the accuracy of the model in percentage
    accuracy=(np.sum(y == yPred)/len(y))*100
    return f"{accuracy:.2f}%"
```
<br/>
If you to see the full code check my github [here.](https://github.com/amaneth/Multinomial-Naive-Bayes)