
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from pyvi import ViTokenizer
import tensorflow as tf
import gensim
import numpy as np
import pickle
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app=Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


import gensim.models.keyedvectors as word2vec
model_embedding = word2vec.KeyedVectors.load('./wordgensim.model')

word_labels = []
max_seq = 100
embedding_size = 128

for word in model_embedding.vocab.keys():
    word_labels.append(word)

def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    lencmt = len(words)
    
    so_lan_du_1_cau = 0

    for i in range(max_seq):
        indexword = i % lencmt

        so_lan_du_1_cau = i//lencmt
        if (max_seq - so_lan_du_1_cau*lencmt < lencmt):
            break
        if(words[indexword] in word_labels):
            matrix[i] = model_embedding[words[indexword]]

    matrix = np.array(matrix)
    return matrix

def process_special_word(text):
    file = open('not.txt', 'r', encoding="utf8")
    not_lst = file.read().split('\n')
    file.close()
    for khong in not_lst:
      new_text = ''
      text_lst = text.split()
      i= 0
      if khong in text_lst:
          while i <= len(text_lst) - 1:
              word = text_lst[i]
              if  word == khong:
                  next_idx = i+1
                  if next_idx <= len(text_lst) -1:
                      word = word +'_'+ text_lst[next_idx]
                  i= next_idx + 1
              else:
                  i = i+1
              new_text = new_text + word + ' '
      else:
          new_text = text
      text=new_text
    return new_text.strip()


def ValuePredictor(comment):

    comment = process_special_word(comment)
    comment = ''.join( c for c in comment if  c not in '?:!/;#,1234567890.' )

    comment = gensim.utils.simple_preprocess(comment)
    comment = ' '.join(comment)
    comment = ViTokenizer.tokenize(comment)

    model_sentiment = tf.keras.models.load_model("sentiment_model.h5")
    maxtrix_embedding = np.expand_dims(comment_embedding(comment), axis=0)
    maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)
    result = model_sentiment.predict(maxtrix_embedding)
    return result

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        input_ = request.form.to_dict()
        
        result = ValuePredictor(input_["cmt"])
        result = float(result)
        result = np.round(result,2)
        max_index = np.round(result)
        if max_index == 1: 
            prediction = "Tích cực"            
       
        if max_index == 0: 
            prediction = "Tiêu cực"

        return render_template("result.html",prediction=prediction,c=input_["cmt"],p=result)

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=9999, debug=True)