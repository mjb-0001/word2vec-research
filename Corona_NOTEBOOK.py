#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
@author: Mitch
"""
import collections
import math
import os
import random
import zipfile
import numpy as np
import io

from six.moves import urllib
from six.moves import xrange

import argparse
import os
import sys
from tempfile import gettempdir


# USE FOR TENSORFLOW 1.0
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


# USE FOR TENSORFLOW 2.0
# import tensorflow.compat.v1 as tf
# from tensorboard.plugins import projector


# In[7]:


tf.disable_v2_behavior()
tf.compat.v1.reset_default_graph()


# In[8]:


vocabulary_size = 70000


# In[9]:


file = open('../all-text-05-29-2020-unicode-removed.txt', 'r', encoding='utf-8')
# words = tf.compat.as_str( file.read()  ).split()
text = file.read()
file.close()


# In[10]:



import nltk
from nltk.corpus import stopwords
words_tokenized = nltk.word_tokenize(text)
words_filtered = [w for w in words_tokenized if w.isalnum() ]

stop_words = set(stopwords.words('english')) 
words = [w for w in words_filtered if not w in stop_words] 


# In[11]:



# print('Data size', len(words), '\n')

# ''' at this point words looks like this
#  words = ["anarchish", "originated", "as", "a", "term", "of", "abuse", "first", "used", ...]

# '''


# In[12]:



# this is a frequency count
count = []
count.extend(collections.Counter(words).most_common(vocabulary_size -1 ))
count.append(['UNK', -1])

dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)


# In[13]:



data = list()
unk_count = 0

for word in words:
    if word in dictionary:
        index = dictionary[word]
    else:
        index == 0   # dictionary['UNK']
        unk_count += 1
    data.append(index)


for i in range(10):
    print(words[i], ' ----> ', data[i])

print('\n')

reverse_dictionary = dict(zip( dictionary.values(), dictionary.keys()   )  )
print('Most common words ', '-------> ', count[:5], '\n')
print('Sample data ', data[:10],' ', [reverse_dictionary[i]  for i in data[:10]  ])


# In[14]:



data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return(batch, labels)


print('generate batch does this')
print('given the data like this in data')

print('sample data ', data[:10], [reverse_dictionary[i] for i in data[:10]])

print('it generates pairs of words like this')

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], ' ---> ', labels[i,0], reverse_dictionary[labels[i,0]])

print('this is the batch\n', batch)
print('these are the labels\n', labels)


# # Initialization variables

# In[15]:


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.


valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


# In[16]:



def inference(train_inputs_x, vocabulary_size, embedding_size):

    # embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # embeddings = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    with tf.name_scope('embeddings'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed_x = tf.nn.embedding_lookup(embeddings, train_inputs_x)


    nce_weights = tf.Variable(  tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))   )
    nce_biases = tf.Variable(  tf.zeros([vocabulary_size])  )
    result = tf.matmul(embed_x, tf.transpose(nce_weights)) + nce_biases


    return(embed_x, result, nce_weights, nce_biases, embeddings)


# In[17]:



def nce_loss( nce_weights, nce_biases, train_labels_y, embed_x, num_sampled, vocabulary_size, result):

    loss = tf.reduce_mean(
          tf.nn.nce_loss(
                       weights=nce_weights,         # [vocab_size, embed_size]
                       biases=nce_biases,           # [embed_size]
                       labels=train_labels_y,       # [bs, 1]
                       inputs=embed_x,              # [bs, embed_size]
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    #print('HERE!!!!!!!!!!!!!!!!!!!!!!!!!!! TEST')
    # loss_sum = tf.compat.v1.summary.scalar('loss', loss)

    return( loss )


# In[18]:



def train(cost):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cost)
    return(optimizer)


# In[19]:



def validation(embeddings, valid_dataset):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup( normalized_embeddings, valid_dataset)

    similarity = tf.matmul(  valid_embeddings, normalized_embeddings, transpose_b=True )
    return(similarity)


# In[20]:



train_inputs_x = tf.placeholder(tf.int32, shape=[batch_size], name='train_inputs')
train_labels_y = tf.placeholder(tf.int32, shape=[batch_size, 1], name='train_labels')

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


# In[21]:



embed_x, result, nce_weights, nce_biases, embeddings = inference(train_inputs_x, vocabulary_size, embedding_size)

loss = nce_loss(nce_weights, nce_biases, train_labels_y, embed_x, num_sampled, vocabulary_size, result)
#print('HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!')

train_op = train(loss)

similarity = validation(embed_x, valid_dataset)


init = tf.global_variables_initializer()

# Create a saver.
# saver = tf.train.Saver()
tf.compat.v1.summary.scalar('loss', loss)
merged = tf.compat.v1.summary.merge_all()
saver = tf.compat.v1.train.Saver()


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'corona_logs'),
    help='The log directory for TensorBoard summaries.')
flags, unused_flags = parser.parse_known_args()

log_dir = flags.log_dir


# In[ ]:





# In[23]:


from tqdm import tqdm

num_steps= 100001
sess = tf.Session()
sess.run(init)
average_loss = 0


with tf.compat.v1.summary.FileWriter(log_dir, sess.graph):

    writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)


    # Main loop
    for step in tqdm(xrange(num_steps)):
        # batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs_x: batch_inputs, train_labels_y: batch_labels}

        # Define metadata variable.
        run_metadata = tf.RunMetadata()


        train_o, loss_val, summary = sess.run([train_op, loss, merged], feed_dict={train_inputs_x: batch_inputs, train_labels_y: batch_labels}, run_metadata=run_metadata)
        average_loss += loss_val

        writer.add_summary(summary, step)


        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)


        if step % valid_window == 0 or step == 1:
            if step > 1:
                average_loss /= valid_window
            #print("Step " + str(step) + ", Average Loss= " + "{:.4f}".format(average_loss))
            average_loss = 0


    determiners = ['the', 'a', 'an', 'this', 'that', 'these', 'those',
           'my', 'your', 'his', 'her', 'its', 'our', 'their'
           'one', 'ten', 'thirty',
           'all', 'both', 'half', 'either', 'neither', 'each', 'every',
           'other', 'another',
           'such', 'what', 'rather', 'quite',

           ]



    symptoms = ['fever', 'chills', 'shaking', 'headache', 'sore',
        'pain', 'taste', 'smell', 'bluish', 'pressure',
        'fatigue','cough', 'appetite', 'mucus', 'phlegm'
        'nausea', 'vomiting', 'diarrhea', 'tiredness'

        ]

    # https://www.goodrx.com/blog/coronavirus-treatments-on-the-way/
    possible_treatments = ['remdesivir', 'hydroxychloroquine', 'chloroquine', 'azithromycin',
                   'convalescent', 'actemra', 'tocilizumab', 'calquence',
                   'acalabrutinib', 'xeljanz', 'tofacitinib', 'jakafi', 'ruxolitinib',
                   'olumiant', 'baricitinib', 'kineret', 'anakinra', 'mavrilimumab',
                   'kevzara', 'sarilumab', 'kaletra', 'lopinavir', 'ritonavir',
                   'tamiflu', 'oseltamivir', 'avigan', 'favipiravir', 'colcrys', 'colchicine',
                   'ivermectin', 'oxygen',

                    ]

    # https://www.webmd.com/lung/coronavirus-complications#2-9
    complications = ['cardiomyopathy', 'pneumonia', 'septic', 'sepsis',
            'strep', 'staph', 'rhabdomyolysis', 'arrhythmias'

            ]

    complications2 = ['ARDS','DIC', 'CAC',
             ]



    positive_words = np.genfromtxt(r'C:/Users/strik/Desktop/05-30-2020/pos_word_list_no_dups.csv', dtype='U', usecols=0, delimiter=',')
    negative_words = np.genfromtxt(r'C:/Users/strik/Desktop/05-30-2020/neg_word_list_no_dups.csv', dtype='U', usecols=0, delimiter=',')

    # chems and disease
    chem_words = np.genfromtxt(r"C:\Users\strik\Desktop\06-17-2020\05-30-2020\all_found_chemicals_no_dupes.csv", dtype='U', delimiter=',')
    disease_words = np.genfromtxt(r"C:\Users\strik\Desktop\06-17-2020\05-30-2020\all_found_diseases_no_dupes.csv", dtype='U', delimiter=',')

    import time
    from tqdm import tqdm


    start_time = time.time()

    total_count_array = np.empty([70000,3], dtype='<U50')



    with open(os.path.join(log_dir, 'metadata.tsv'), 'w+') as file:
        new_count = np.asarray(count)
        file.write('Word'+'\t'+'Count'+'\t'+'Category'+ '\t'+ 'Index'  +'\n')

        # Initialize total_count_array with data from new_count
        for i in range(len(total_count_array)):
            total_count_array[i,0] = new_count[i,0] # default row 0 values
            total_count_array[i,1] = new_count[i,1] # default row 1 values
            total_count_array[i,2] = 'nothing' # default row 2 values
            # total_count_array[i,3] = 'nothing'


        # find determiner words
        for i in tqdm(range(len(new_count))):
            for word in determiners:
                if new_count[i,0].lower() == word:
                    total_count_array[i,2] = 'determiner'

        # find symptom words (from cdc)
        for i in tqdm(range(len(new_count))):
            for word in symptoms:
                if new_count[i,0].lower() == word or word in new_count[i,0].lower():
                    total_count_array[i,2] = 'symptom'

        """
        # find possible_treatment words
        for i in range(len(new_count)):
        for word in possible_treatments:
            if new_count[i,0].lower() == word or word in new_count[i,0].lower():
                total_count_array[i,2] = 'possible_treatments'

        # find complication words
        for i in range(len(new_count)):
        for word in complications:
            if new_count[i,0].lower() == word or word in new_count[i,0].lower():
                total_count_array[i,2] = 'complications'

        # find complication2 words **NEEDS WORK**
        for i in range(len(new_count)):
        for word in complications2:
            if new_count[i,0] == word or word in new_count[i,0]:
                total_count_array[i,2] = 'complications'
        """ 

        # find CHEMS/possible_treatments
        for i in tqdm(range(len(new_count))):
            for word in possible_treatments:
                if new_count[i,0].lower() == word or word in new_count[i,0].lower():
                    total_count_array[i,2] = 'possible_treatments'    

        # find diseases
        for i in tqdm(range(len(new_count))):
            for word in possible_treatments:
                if new_count[i,0].lower() == word or word in new_count[i,0].lower():
                    total_count_array[i,2] = 'disease'    



        # find POSITIVE words 
        for i in tqdm(range(len(new_count))):
            for word in positive_words:
                if new_count[i,0] == word or word in new_count[i,0].lower():
                    total_count_array[i,2] = 'positive'

        # find NEGATIVE words 
        for i in tqdm(range(len(new_count))):
            for word in negative_words:
                if new_count[i,0] == word or word in new_count[i,0].lower():
                    total_count_array[i,2] = 'negative'

        # [total_count_array[i,2] = 'negative' for i in range(len(new_count)) for word in negative_words if new_count[i,0] == word or word in new_count[i,0]]

        # Write metadata matrix to file
        for i in tqdm(range(len(new_count))):
            file.write(new_count[i,0]+'\t'+new_count[i,1]+   '\t' + total_count_array[i,2] + '\t' + str(i) + '\t' +'\n')
        print(f'total time took {(time.time() - start_time) / 60} minutes')
    saver.save(sess, os.path.join(log_dir, 'model.ckpt'))


        # Create a configuration for visualizing embeddings with the labels in
        # TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

    writer.close()


        #WRITE VECTOR FILES 

        # vec_test = sess.run(embeddings)

        # np.savetxt('vector-file-05-30-2020.csv', vec_test, delimiter=',')
        # np.save('vector-file-05-30-2020.npy', vec_test)
        # np.savetxt('vector-file-05-30-2020.tsv', vec_test, delimiter='\t', fmt='%.6f')

        # np.savetxt('TEST-TEST-TEST-TEST.tsv', vec_test, delimiter='\t', fmt='%.6f')




# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




