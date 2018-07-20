# import aleGraph
import pandas as pd
# import aleGraphWiki
import doubleGraph
import contextDoubleGraph
if __name__ == "__main__":
    c_train = pd.read_csv('data/c_train2.csv')
    nc_train = pd.read_csv('data/nc_train2.csv')
    c_test = pd.read_csv('data/c_test2.csv')
    nc_test = pd.read_csv('data/nc_test2.csv')

    #need to join since batter test will generate the splits by itself
    c_train = pd.concat([c_train, c_test],sort=False)
    c_train.index = range(len(c_train.index))
    c_train = c_train.loc[:, ['post_text', 'post_title']]
    nc_train = pd.concat([nc_train, nc_test],sort=False)
    nc_train.index = range(len(nc_train.index))
    nc_train = nc_train.loc[:, [ 'post_text', 'post_title']]
    c_train = c_train.drop_duplicates()
    nc_train = nc_train.drop_duplicates()

    #preprocess data, result is a series
    c_train = c_train.apply(doubleGraph.preprocess_text, axis=1)
    nc_train = nc_train.apply(doubleGraph.preprocess_text,axis=1)

    contextDoubleGraph.batter_test_args(c_train[:100], nc_train[:100], 51, 52, 2, 3, 4, 1, 32, 33, 1, 0, 5, 1, 10)


