import numpy as np
import pandas as pd
import keras as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, Dropout, Dense, Dot, Concatenate, BatchNormalization, Multiply
from keras.optimizers import adam_v2

from keras.utils.vis_utils import plot_model


def load_data():
    '''
    load in alldata.tsv. 
    then trasfer all category into a [1, n] sequence. n = number of categories
    We did not use name tag in learning so i removed it
    return: 
        alldata:  all the data after processed
    '''
    # load in alldata.tsv
    names = ['user_id', 'age', 'gender', 'movie_id', 'name',
             'year', 'genre1', 'genre2', 'genre3', 'rating']
    alldata = pd.read_csv(
        "Movie Recommendation Lab/allData.tsv", header=0, sep='\t', names=names)
    alldata = alldata.drop(axis=1, columns='name')

    movie_id_to_new_id = dict()
    user_id_to_new_id = dict()
    genre_to_id = dict()
    genre_to_id[np.nan] = 1
    genre_id = 1
    movie_id = 1
    user_id = 1
    # change spares user_id and movie_id in to a sequence from 1 to number of movies and users in
    for index, row in alldata.iterrows():
        if movie_id_to_new_id.get(row['movie_id']) is None:
            movie_id_to_new_id[row['movie_id']] = movie_id
            alldata.at[index, 'movie_id'] = movie_id
            movie_id += 1
        else:
            alldata.at[index, 'movie_id'] = movie_id_to_new_id.get(
                row['movie_id'])

        if user_id_to_new_id.get(row['user_id']) is None:
            user_id_to_new_id[row['user_id']] = user_id
            alldata.at[index, 'user_id'] = user_id
            user_id += 1
        else:
            alldata.at[index, 'user_id'] = user_id_to_new_id.get(
                row['user_id'])

        if genre_to_id.get(row['genre1']) is None:
            genre_to_id[row['genre1']] = genre_id
            alldata.at[index, 'genre1'] = genre_id
            genre_id += 1
        else:
            alldata.at[index, 'genre1'] = genre_to_id.get(row['genre1'])

        if genre_to_id.get(row['genre2']) is None:
            genre_to_id[row['genre2']] = genre_id
            alldata.at[index, 'genre2'] = genre_id
            genre_id += 1

        else:
            alldata.at[index, 'genre2'] = genre_to_id.get(row['genre2'])

        if genre_to_id.get(row['genre3']) is None:
            genre_to_id[row['genre3']] = genre_id
            alldata.at[index, 'genre3'] = genre_id
            genre_id += 1
        else:
            alldata.at[index, 'genre3'] = genre_to_id.get(row['genre3'])

        if row['gender'] == 'M':
            alldata.at[index, 'gender'] = 1
        else:
            alldata.at[index, 'gender'] = 2
    
    return alldata


class WideDeepModel():
    
    def __init__(self):
        
        pass
    
    def build_model(self, dataset):
        '''
        build the model, save it
        input:  
            dataset: the whole dataset, not train or test only
        
        '''

        num_users = int(max(dataset['user_id'].unique()))
        num_movies = int(max(dataset['movie_id'].unique()))
        num_genre1 = int(max(dataset['genre1'].unique()))
        num_genre2 = int(max(dataset['genre2'].unique()))
        num_genre3 = int(max(dataset['genre3'].unique()))
        num_gender = 2

        #input and embedding

        id_dim = 6
        movie_id_input = Input(shape=[1], name='movie-id-input')
        movie_id_embedding = Embedding(
            num_movies + 1, id_dim, name='movie-id-embedding')(movie_id_input)
        movie_id_vec = Flatten(name='movie-flatten')(movie_id_embedding)

        user_id_input = Input(shape=[1], name='user-id-input')
        user_id_embedding = Embedding(
            num_users + 1, id_dim, name='user-id-embedding')(user_id_input)
        user_id_vec = Flatten(name='user-flatten')(user_id_embedding)

        genre_dim = 6
        genre1_input = Input(shape=[1], name='gener1')
        genre1_embedding = Embedding(
            num_genre1 + 1, genre_dim, name='gener1-embedding')(genre1_input)
        genre1_vec = Flatten(name='gener1_flatten')(genre1_embedding)

        genre2_input = Input(shape=[1], name='gener2')
        genre2_embedding = Embedding(
            num_genre2 + 1, genre_dim, name='gener2-embedding')(genre2_input)
        genre2_vec = Flatten(name='gener2_flatten')(genre2_embedding)

        genre3_input = Input(shape=[1], name='gener3')
        genre3_embedding = Embedding(
            num_genre3 + 1, genre_dim, name='gener3-embedding')(genre3_input)
        genre3_vec = Flatten(name='gener3_flatten')(genre3_embedding)

        gender_dim = 6
        gender_input = Input(shape=[1], name='gender')
        gender_embedding = Embedding(
            num_gender+1, gender_dim, name='gender-embedding')(gender_input)
        gender_vec = Flatten(name='gender_flatten')(gender_embedding)

        age_input = Input(shape=[1], name='age_input')
        year_input = Input(shape=[1], name='year_input')

        # deep neural network
        DN_input_list = [movie_id_vec, user_id_vec, genre1_vec,
                         genre2_vec, genre3_vec, gender_vec, age_input, year_input]
        concatenate = Concatenate(axis=-1)(DN_input_list)
        con_bn = BatchNormalization(name='con_batch-norm')(concatenate)
        fc_1 = Dense(24, name='fc-1', activation='relu')(con_bn)
        fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
        fc_1_dropout = Dropout(0.2, name='fc-1-dropout')(fc_1_bn)
        fc_2 = Dense(12, name='fc-2', activation='relu')(fc_1_dropout)
        fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
        # wide network
        wide_list = [genre1_input, genre2_input, genre3_input,
                     gender_input, movie_id_input, user_id_input, ]
        doct_product = Multiply()(wide_list)
        doct_product_bn = BatchNormalization(
            name='doct_product_batch-norm-2')(doct_product)
        # use logistic to concatenate 2 model
        input_list = [movie_id_input, user_id_input, genre1_input,
                      genre2_input, genre3_input, gender_input, age_input, year_input]
        concatenate2 = Concatenate(axis=1)([fc_2_bn, doct_product_bn])
        bn = BatchNormalization(name='batch-norm-4')(concatenate2)
        logistic = Dense(1, name='logistic', activation='sigmoid')(bn)

        model = Model(input_list, logistic)
        model.compile(optimizer=adam_v2.Adam(lr=0.1), loss='mean_squared_error')
        
        return model
    
class Matrix_Factorization():
    
    def __init__(self,ids_dim = 10):
        self.latent_dim = ids_dim
    
    def build_model(self,dataset):
        '''
        
        '''
        num_users = int(max(dataset['user_id'].unique()))
        num_movies = int(max(dataset['movie_id'].unique()))
        latent_dim = self.latent_dim
        movie_input = Input(shape=[1],name='movie-input')
        movie_embedding = Embedding(num_movies + 1, latent_dim, name='movie-embedding')(movie_input)
        movie_vec = Flatten(name='movie-flatten')(movie_embedding)
        user_input = Input(shape=[1],name='user-input')
        user_embedding = Embedding(num_users + 1, latent_dim, name='user-embedding')(user_input)
        user_vec = Flatten(name='user-flatten')(user_embedding)
        prod = Dot(axes=1)([movie_vec, user_vec])
        model = Model([movie_input,user_input], prod)
        model.compile('adam', 'mean_squared_error')
        return model

class Neural_Network_MF():
    def __init__(self,ids_dim = 10):
        self.latent_dim = ids_dim
    
    def build_model(self,dataset):
        num_users = max(dataset['user_id'].unique())
        num_movies = max(dataset['movie_id'].unique())
        latent_dim = self.latent_dim
        movie_input = Input(shape=[1],name='movie-input')
        movie_embedding = Embedding(num_movies + 1, latent_dim, name='movie-embedding')(movie_input)
        movie_vec = Flatten(name='movie-flatten')(movie_embedding)
        user_input = Input(shape=[1],name='user-input')
        user_embedding = Embedding(num_users + 1, latent_dim, name='user-embedding')(user_input)
        user_vec = Flatten(name='user-flatten')(user_embedding)
        concat = Dot(axes=[1,1])([movie_vec, user_vec])
        fc_1 = Dense(20, name='fc-1', activation='relu')(concat)
        fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
        fc_1_dropout = Dropout(0.5, name='fc-1-dropout')(fc_1_bn)
        fc_2 = Dense(10, name='fc-2', activation='relu')(fc_1_dropout)
        fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
        fc_2_dropout = Dropout(0.5, name='fc-2-dropout')(fc_2_bn)
        fc_3 = Dense(1, name='fc-3', activation='relu')(fc_2_dropout)
        model = Model([movie_input,user_input], fc_3)
        model.compile(optimizer=adam_v2.Adam(lr=0.1), loss='mean_squared_error')
        return model

def test_WDM():
    dataset = load_data()
    dataset = dataset.astype('float64')
    # turn the rating into [0,1] so logistic way can work
    dataset['rating'] = dataset['rating'].map(lambda x: x/5)
    a=WideDeepModel()
    model = a.build_model(dataset)
    train, test = train_test_split(dataset, test_size=0.2)
    train_inputs = [train.movie_id,train.user_id,train.genre1, train.genre2,train.genre3,train.gender, train.age,train.year]
    test_inputs = [test.movie_id,test.user_id ,test.genre1, test.genre2,test.genre3,test.gender, test.age,test.year]
    history = model.fit(train_inputs, train.rating, epochs=30)
    y_hat = np.round(model.predict(test_inputs), decimals=2)
    y_true = test.rating
    y1 = [i * 5 for i in y_true]
    y2= [i * 5 for i in y_hat]
    print('mean_absolute_error = ',mean_absolute_error(y1, y2))
    
def test_NNMF():
    dataset = load_data()
    a=Neural_Network_MF()
    model = a.build_model(dataset)
    train, test = train_test_split(dataset, test_size=0.2)
    train_inputs = [train.movie_id,train.user_id]
    test_inputs = [test.movie_id,test.user_id]
    history = model.fit(train_inputs, train.rating, epochs=10)
    y_hat = np.round(model.predict(test_inputs), decimals=2)
    y_true = test.rating
    print('mean_absolute_error = ',mean_absolute_error(y_true, y_hat))

def test_MF():
    dataset = load_data()
    pra = [1,11,24]
    for i in pra :
        a=Matrix_Factorization(id_dim=i)
        model = a.build_model(dataset)
        train, test = train_test_split(dataset, test_size=0.2)
        train_inputs = [train.movie_id,train.user_id]
        test_inputs = [test.movie_id,test.user_id]
        history = model.fit(train_inputs, train.rating, epochs=10)
        y_true = test.rating
        print('mean_absolute_error = ',mean_absolute_error(y_true, y_hat))
    
    
    


def main():
    np.random.seed(100)
    test_MF()
    test_NNMF()
    test_WDM()
    
    

    
    pd.Series(history.history['loss']).plot(logy=True)
    plt.xlabel("Epoch")
    plt.ylabel("Train Error")
    plt.show()
    
    
    

if __name__ == '__main__':
    main()
