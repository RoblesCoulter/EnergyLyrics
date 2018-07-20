from sklearn.metrics import homogeneity_score, completeness_score
import pandas as pd

def eval_syn_cluster(df_word2cluster, version=''):
    df_syn_cluster = pd.read_pickle('df_syn_olp_cluster{}.pkl'.format(version))
    if version == '' and clean_index : df_word2cluster.word = df_word2cluster.word.apply(lambda x: x[2:-1])
    val_vocab = list(df_syn_cluster[df_syn_cluster[0] >= 0].index)
    y_pred = list(df_word2cluster[df_word2cluster.word.isin(val_vocab)].cluster)
    y_true = list(df_syn_cluster[df_syn_cluster[0] >= 0][0].apply(int))
    return {'homogeneity_score':homogeneity_score(labels_pred=y_pred, labels_true=y_true),
            'completeness_score':completeness_score(labels_pred=y_pred, labels_true=y_true)}

def eval_inter_syn_cluster(df_word2cluster, version='', clean_index=True):
    df_syn_cluster = pd.read_pickle('df_syn_olp_cluster{}.pkl'.format(version))
    if version == '' and clean_index : df_word2cluster.word = df_word2cluster.word.apply(lambda x: x[2:-1])
    val_vocab = set(df_syn_cluster[df_syn_cluster[0] >= 0].index).intersection(set(df_word2cluster.word))
    y_pred = list(df_word2cluster[df_word2cluster.word.isin(val_vocab)].cluster)
    y_true = list(df_syn_cluster[df_syn_cluster.index.isin(val_vocab)][df_syn_cluster[0] >= 0][0].apply(int))
    return {'homogeneity_score':homogeneity_score(labels_pred=y_pred, labels_true=y_true),
            'completeness_score':completeness_score(labels_pred=y_pred, labels_true=y_true)}

def eval_emo_vocab_cluster(df_word2cluster):
    df_syn_cluster = pd.read_pickle('../Data/emo_vocab_610.pkl')
    val_vocab = set(df_syn_cluster.word).intersection(set(df_word2cluster.word))
    y_pred = list(df_word2cluster[df_word2cluster.word.isin(val_vocab)].drop_duplicates().cluster)
    y_true = list(df_syn_cluster[df_syn_cluster.word.isin(val_vocab)].drop_duplicates().cluster)
    return {'homogeneity_score':homogeneity_score(labels_pred=y_pred, labels_true=y_true),
            'completeness_score':completeness_score(labels_pred=y_pred, labels_true=y_true)}