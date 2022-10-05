import pickle

def pickle_this(data=None, pickle_name="df",path="./data/"):
    w = ['rb' if data is None else 'wb'][0]
    with open(f'{path}{pickle_name}.pickle',w) as handle:
        if data is None:
            data = pickle.load(handle)
            return data
        else:
            pickle.dump(data,handle)