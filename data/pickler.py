import pickle


# pip install pandas==1.4.3
def pickle_klines(path= "./database/klines_indicators_dict.pickle",data=None):
    w = ['rb' if data is None else 'wb'][0]
    with open(f'{path}', w) as handle:
        if data is None:
            data = pickle.load(handle)
            return data
        else: 
            pickle.dump(data, handle)