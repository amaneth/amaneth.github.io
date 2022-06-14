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

<script src="https://gist.github.com/amaneth/7928ee5e65500653626fa484beeae516.js"></script>
<br/>
<h4>Word frequency</h4>
<p style="text-align: justify;">As you remember from part-1 to calculate the prior and the likelihood probabilities we need to have the frequency of each word in each category. Let's put this in a matrix of shape the number of words by the number of classes.
</p>
<script src="https://gist.github.com/amaneth/b6563ddc731cb674003854706d000e2e.js"></script>
<br/>
<h4>Prior probability</h4>
<p style="text-align: justify;">The prior probability is calculated as the number of documents in the category divided by the total number of documents. Here is how I did it.</p>
<script src="https://gist.github.com/amaneth/b34d7c0ddb99223e7fce0f6994a7ae98.js"></script>
<br/>
<h4>Likelihood probability</h4>
<p style="text-align: justify;">The likelihood probability is to divide the frequency of the word in a category by the total number of words in the category. We already have the frequency matrix(counts) but there could be zeroes in the count matrix. This can create a big problem. For example, If we have a message "You've won a lottery!" and assume the word 'won' occurred 1000 times and the word 'lottery' occurred 0 times in spam messages, when we do the likelihood probability of each word and multiply them, the probability of the message will go to zero. This is not right. To solve this we add some constant value k(usually 1) to each word frequency. What we are doing here is Laplacian smoothing. <br/>
After adding the Laplacian smoothing value k to the Count matrix(the operation is broadcasting) we divide it by the category count. The country count is the vector of the number of words in each category. The count matrix is $\bbox[yellow]{num\_words\ x\ num\_classes}$ and the category count is $\bbox[yellow]{1\ x\ num\_classes}$. That means dividing the count by the category count is possible by broadcasting operation.   </p>

<script src="https://gist.github.com/amaneth/0ba6546a6a35e8803228c1fedf59dd0a.js"></script>
<br/>
<h4>Training</h4>
<p style="text-align: justify;">In the training we don't do anything different, we just call the functions we have defined earlier. Training in the multinomial naive byes is all about preparing the bag of words(the vocabulary), the likelihood, and prior probabilities.</p>
<script src="https://gist.github.com/amaneth/6876c515c1aa2d3383a03476cdff98d7.js"></script>
<br/>
<h4>Prediction</h4>
<p style="text-align: justify;">The prediction is just to multiply the likelihood probabilities of the words in the document to be predicted by their prior probabilities. Here again, I used a vectorized implementation instead of for loops.<br/>
 Before going to the prediction we need to transform each word in the document to be predicted to their id in the vocabulary so that the already trained 'model' understands it. The transform function below returns a matrix of shape $\bbox[yellow]{len\_documents\ x\ len\_vocublary}$. The matrix is matrix of zeros and ones which indicates if the word in the documents is in the vocabulary. </p>
<script src="https://gist.github.com/amaneth/16fda5d1e48d4be41e354cef3fd7557b.js"></script>
<br/>
<p style="text-align: justify;">We add a new axis to the transformed matrix and the likelihood probabilities matrix so that the shape of the transformed matrix becomes $\bbox[yellow]{len\_documents\ x\ len\_vocabulary\ x\ 1}$ and the shape of the likelihood matrix becomes $\bbox[yellow]{1\ x\ len\_vocabulary\ x num\_classes}$. These two matrices are broadcastable, we can multipy them and get a matrix of shape $\bbox[yellow]{len\_documents\ x\ len\_vocabulary\ x\ num\_classes}$. We sum the resulted matrix across axis=1(the length of the vocabulary) and get a matrix of shape $\bbox[yellow]{len\_documents\ x\ num\_classes}$ which represents the likelihood probability of each document in each class. We add this to the prior probability matrix(again a broadcasting operation) and results in a matrix of shape $\bbox[yellow]{len\_documents\ x\ num\_classes}$ which is the probability of each document in being each calss. Finally, we take argmax of this and we get the prediction class of each document. I made the prediction to be done in a batch, this is to be memory efficient, unless the operation will fail if the size of the matrices is bigger.</p>
<script src="https://gist.github.com/amaneth/196de323efbb39a8223ecf8d8089868a.js"></script>
<br/>
<p style="text-align: justify;"> Finally, we can test the accuracy by calculating the percentage of correct predictions.
</p>
<script src="https://gist.github.com/amaneth/8cd29509b8eace58e7dbe183e294bb5a.js"></script>
<br/>
If you to see the full code check my github [here.](https://github.com/amaneth/Multinomial-Naive-Bayes)