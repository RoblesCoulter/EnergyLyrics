import re
import nltk
from copy import deepcopy
import networkx as nx
import pandas as pd
import multiprocessing as mp
import multiprocessing
import math
import time
import csv
import random
from time import sleep
import os.path

def preprocess_text(posts):
    text = str(posts['post_title'])+' . '+ str(posts['post_text'])+' .'
    text =  re.sub('tl[;]?dr','',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+[0-9]+[s]?[ /\(,)]*f[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+[0-9]+[s]?[ /\(,)]*m[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+f[ /\(,)]*[0-9]+[s]?[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[ \(\[]+m[ /\(,)]*[0-9]+[s]?[ \]\)]+',' ',text,flags=re.IGNORECASE)
    text = re.sub('[0-9]+','NUM',text,flags=re.IGNORECASE)
    text = re.sub('u/[^\s]+','AT_USER',text,flags=re.IGNORECASE)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text,flags=re.IGNORECASE)  #Convert www.* or https?://* to <url>
    text = text.split("[.]?\n[\* \[\(/]*[eE]dit")[0]
    text = text.split("[.]?\n[\* \[\(/]*EDIT")[0]
    text = text.split("[.]?\n[\* \[\(/]*big edit")[0]
    text = text.split("[.]?\n[\* \[\(/]*important edit")[0]
    text = text.split("[.]?\n[\* \[\(/]*[uU]pdate")[0]
    text = text.split("[.]?\n[\* \[\(/]*UPDATE")[0]
    text = text.split("[.]?\n[\* \[\(/]*big update")[0]
    text = text.split("[.]?\n[\* \[\(/]*important update")[0]
    text = text.split("[.]?\nfor an update")[0]
    text = text.replace('\r', '')
    return text

def bigram_frequency(posts):
    # corpus = posts.str.cat(sep='. ')
    fdist = nltk.FreqDist()
    for corpus in posts:
        tokens = nltk.word_tokenize(corpus)
        bigrm = nltk.bigrams(tokens)
        fdist.update(bigrm)
    #normalize
    max_freq = fdist.get(fdist.max())
    return {k:v/max_freq for k,v in fdist.items()}

def get_freq(df):
    return bigram_frequency(df)

def aggregate_freq(arg_c_freq,arg_nc_freq):
    res_freq = deepcopy(arg_c_freq)
    for k,nc_v in arg_nc_freq.items():
        if res_freq.get(k) is not None:
            c_v = res_freq.get(k)
            if c_v <= nc_v:
                res_freq.pop(k)   #not conflict has higher frequency so we dont delete the edge
            else:
                res_freq[k]= c_v - nc_v
    full_c_freq = deepcopy(res_freq)
    for k,c_v in full_c_freq.items():
        if c_v == 1:
            res_freq.pop(k)
    return res_freq

def create_graph(fdist,thresh):
    fdist = {k: v for k, v in fdist.items() if v>thresh}
    nodes = set()
    edges = []
    for k,v in fdist.items():
        lst = list(k)
        [nodes.add(word) for word in lst]
        lst.append(v)
        edges.append(lst)
    DG=nx.DiGraph()
    DG.add_nodes_from(nodes)
    DG.add_weighted_edges_from(edges)
    return DG

def typify_graph(args_c_graph,eigen=0.03,cluster=0.32):
    res_c_graph = args_c_graph
    eigen_c = nx.eigenvector_centrality(res_c_graph)
    eigen_df = pd.DataFrame.from_dict(eigen_c, orient='index')
    nodes = list(res_c_graph)
    nodes_to_delete = pd.DataFrame(nodes,columns=['node'])
    eigen_df.columns = ['eigen_centrality']
    eigen_df['node'] = eigen_df.index
    context_words = eigen_df[eigen_df['eigen_centrality']>=eigen]
    c_u_graph = res_c_graph.to_undirected()
    clustering_score = nx.clustering(c_u_graph)
    cluster_df = pd.DataFrame.from_dict(clustering_score, orient='index')
    cluster_df.columns = ['clustering_score']
    cluster_df['node'] = cluster_df.index
    conflict_words = cluster_df[cluster_df['clustering_score']>=cluster]
    both_words = context_words.merge(conflict_words,how='inner',on=['node'])
    both_words = both_words.loc[:,['node']]
    context_words = context_words.merge(both_words,how='left', on=['node'],indicator=True)
    context_words = context_words[context_words['_merge'] == 'left_only']
    conflict_words = conflict_words.merge(both_words,how='left', on=['node'],indicator=True)
    conflict_words = conflict_words[conflict_words['_merge'] == 'left_only']
    context_words['type'] = 1
    conflict_words['type'] = 2
    both_words['type']=3
    print(len(context_words))
    print(len(conflict_words))
    context_words = context_words.loc[:,['node','type']]
    conflict_words = conflict_words.loc[:,['node','type']]
    words = pd.concat([context_words,conflict_words],ignore_index=True)
    words = pd.concat([words,both_words],ignore_index=True)
    #delete nodes that are not context words or conflict words
    nodes_to_delete = nodes_to_delete.merge(words,how='left',on=['node'],indicator=True)
    nodes_to_delete = nodes_to_delete[nodes_to_delete['_merge']=='left_only']
    for num,name in nodes_to_delete.iterrows():
            res_c_graph.remove_node(name['node'])
    words.index = words['node']
    words = words.loc[:,['type']]
    words = words.to_dict()
    words = words['type']
    nx.set_node_attributes(res_c_graph,words,'type')
    return res_c_graph

def generate_patterns(c_graph,num_clusters):
    patterns = []
    for node in c_graph.nodes():
        current_type = c_graph.node[node]['type']
        for suc in c_graph.successors(node):
            suc_type = c_graph.node[suc]['type']
            if (((current_type == 2) & (suc_type != 2)) | ((current_type != 2) & (suc_type == 2))):
                patterns.append((node if current_type != 2 else '.+') + ' ' + (suc if suc_type != 2 else '.+'))
            if (((current_type == 1) & (suc_type == 3)) | ((current_type == 3) & (suc_type == 1))):
                patterns.append((node if current_type != 3 else '.+') + ' ' + (suc if suc_type != 3 else '.+'))
            if ((current_type == 3) & (suc_type == 3)):
                patterns.append(node + ' .+')
                patterns.append('.+ ' + suc)

            for suc2 in c_graph.successors(suc):
                suc2_type = c_graph.node[suc2]['type']
                mis_tri = True
                if (((current_type == 2) & (suc_type != 2) & (suc2_type != 2)) | (
                        (current_type != 2) & (suc_type == 2) & (suc2_type != 2)) | (
                        (current_type != 2) & (suc_type != 2) & (suc2_type == 2))):
                    patterns.append(
                        (node if current_type != 2 else '.+') + ' ' + (suc if suc_type != 2 else '.+') + ' ' + (
                        suc2 if suc2_type != 2 else '.+'))
                    mis_tri = False
                if (mis_tri) & (((current_type == 1) & (suc_type == 1) & (suc2_type == 3)) | (
                        (current_type == 1) & (suc_type == 3) & (suc2_type == 1)) | (
                        (current_type == 3) & (suc_type == 1) & (suc2_type == 1))):
                    patterns.append(
                        (node if current_type != 3 else '.+') + ' ' + (suc if suc_type != 3 else '.+') + ' ' + (
                        suc2 if suc2_type != 3 else '.+'))
                    mis_tri = False
                if (mis_tri) & ((current_type == 1) & (suc_type == 3) & (suc2_type == 3)):
                    patterns.append(node + ' .+ ' + suc2)
                    patterns.append(node + ' ' + suc + ' .+')
                    mis_tri = False
                if (mis_tri) & ((current_type == 3) & (suc_type == 1) & (suc2_type == 3)):
                    patterns.append('.+ ' + suc + suc2)
                    patterns.append(node + ' ' + suc + ' .+')
                    mis_tri = False
                if (mis_tri) & ((current_type == 3) & (suc_type == 3) & (suc2_type == 1)):
                    patterns.append('.+ ' + suc + suc2)
                    patterns.append(node + ' .+ ' + suc2)
                    mis_tri = False
                if (mis_tri) & ((current_type == 3) & (suc_type == 3) & (suc2_type == 3)):
                    patterns.append('.+ ' + suc + suc2)
                    patterns.append(node + ' .+ ' + suc2)
                    patterns.append(node + ' ' + suc + ' .+')
                    mis_tri = False
    patterns = list(set(patterns))
    #now we need to EXPLODE THEM have to find the wildcard to substitute
    patterns = pd.DataFrame([[pat,re.sub('[.][+]','.+C'+str(new_pat),pat)] for pat in patterns for new_pat in range(num_clusters+1)], columns=['og_pattern','pattern'])
    return patterns

def build_graph_and_model(conflict,not_conflict,t,eigen,cluster,current_k,pat_deg_thres,clusters):
    #check if current pattern combination has been done
    name_conflict = 'conflict_ale_'+str(t)+'_'+str(eigen)+'_'+str(cluster)+'_'+str(current_k)
    name_not_conflict = 'not_conflict_ale_'+str(t)+'_'+str(eigen)+'_'+str(cluster)+'_'+str(current_k)
    file_exists = os.path.isfile('patterns/'+name_conflict +'.csv')
    if not file_exists:
        #get bigram normalized frequencies to build the graph
        print('normalizing bigram frequencies...')
        c_freq = get_freq(conflict)
        nc_freq = get_freq(not_conflict)
        print('aggregating frequencies... ')
        #aggregate frequencies to build conflict graph
        print('building conflict graph... ')
        c_graph = create_graph(aggregate_freq(c_freq, nc_freq),thresh=t)
        #get eigen vector and clustering score and filter c_graph
        print('getting important nodes... ')
        print(len(c_graph.nodes()))
        c_graph = typify_graph(c_graph,eigen,cluster)
        #bootstrap patterns
        print('bootstrapping patterns... ')
        pats = generate_patterns(c_graph,clusters.cluster.max())
        print('total patterns bootstrapped:',len(pats))
        start = time.time()
        pattern_features_new(conflict,pats,name_conflict,name_conflict,clusters)
        print('-------------',time.time() - start,'seconds ------------')
        # get bigram normalized frequencies to build the graph
        print('aggregating frequencies... ')
        # aggregate frequencies to build conflict graph
        print('building conflict graph... ')
        nc_graph = create_graph(aggregate_freq(nc_freq, c_freq), thresh=t)
        # get eigen vector and clustering score and filter c_graph
        print('getting important nodes... ')
        print(len(nc_graph.nodes()))
        nc_graph = typify_graph(nc_graph, eigen, cluster)
        # bootstrap patterns
        print('bootstrapping patterns... ')
        pats = generate_patterns(nc_graph,clusters.cluster.max())
        print('total patterns bootstrapped:', len(pats))
        pattern_features_new(not_conflict,pats,name_not_conflict,name_not_conflict,clusters)
    else:
        print('file already exists for this set!....')
    print('done getting features for patterns... ')
    return build_models(name_conflict, name_not_conflict, pat_deg_thres, conflict, not_conflict)

def prepare_for_features(texts):
    new_texts = texts.apply(lambda x:nltk.word_tokenize(x))
    return new_texts

def threadize(og_df,pats,cw,name,clusters):
    cores = multiprocessing.cpu_count()
    total = len(og_df)
    inc = math.ceil(total/cores)
    processes = [mp.Process(target=pattern_feature_thread, args=(og_df.loc[i*inc:(i*inc)+inc],pats,cw,name+'_t'+str(i),name,clusters)) for i in range(cores)]
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()

def pattern_feature_thread(posts,patterns,context_words,name,sec,clusters):
    total = len(posts)
    count = 0
    inc = 100
    thres = inc
    pat_counts = pd.DataFrame()
    tot_div = pd.DataFrame()
    print('start...', name, sec)
    for doc in posts:
        doc_pats,my_diversity = generate_post_patterns(clean_post(doc,context_words),clusters,True,True)
        doc_pats['doc_freq'] = 1
        pat_counts = pd.concat([pat_counts,doc_pats],sort=False)
        pat_counts = pat_counts.groupby('pattern').agg({'p_freq': 'sum', 'doc_freq': 'sum'})
        pat_counts.reset_index(level=0, inplace=True)
        tot_div = pd.concat([tot_div,my_diversity],sort=False)
        tot_div = tot_div.drop_duplicates()
        count += 1
        if count > thres:
            print('done with ',count,'out of ',total,'of ',name)
            thres += inc
    pat_counts = patterns.merge(pat_counts, on='pattern', how='inner')
    tot_div = patterns.merge(tot_div,on='pattern',how='inner')
    pat_counts.to_csv('pat_thread/'+name+'_pat.csv',index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
    tot_div.to_csv('pat_thread/' + name + '_div.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    done = pd.DataFrame([name])
    done['sec']=sec
    done.to_csv('controls/threads_pat_done.csv',index=False,encoding='utf-8',mode='a',header=False)

def pattern_features_new(df,patterns,name,sec,clusters):
    context_words = get_words_pat(patterns)
    pre_posts = prepare_for_features(df)
    threadize(pre_posts,patterns,context_words,name,clusters)
    pat_counts,tot_div = unify_t(name)
    pat_counts = pat_counts.groupby('pattern').agg({'p_freq': 'sum', 'doc_freq': 'sum'})
    pat_counts.reset_index(level=0, inplace=True)
    tot_div = tot_div.drop_duplicates()
    patterns = patterns.merge(pat_counts,on='pattern',how='inner')
    tot_div = tot_div.groupby('pattern').count()
    tot_div.reset_index(level=0, inplace=True)
    patterns = patterns.merge(tot_div,on='pattern',how='left')
    patterns.to_csv('patterns/'+name+'.csv',index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
    done = pd.DataFrame([name])
    done['sec']=sec
    done.to_csv('controls/threads_done.csv',index=False,encoding='utf-8',mode='a',header=False)

def unify(name):
    result = pd.DataFrame()
    control = pd.read_csv('controls/threads_done.csv')
    control = control[control['sec']==name]
    for num,file in control.iterrows():
        cur_file = pd.read_csv('patterns/'+file['name']+'.csv')
        result = pd.concat([result,cur_file],sort=False)
    return result

def unify_t(name):
    pats = pd.DataFrame()
    divs = pd.DataFrame()
    control = pd.read_csv('controls/threads_pat_done.csv')
    control = control[control['sec']==name]
    for num,file in control.iterrows():
        pat = pd.read_csv('pat_thread/'+file['name']+'_pat.csv')
        pats = pd.concat([pats,pat],sort=False)
        div = pd.read_csv('pat_thread/'+file['name']+'_div.csv')
        divs = pd.concat([divs,div],sort=False)
    return pats,divs

def c_score(pats, docs):
    if pats['doc_freq'] == 0:
        return 0
    freq = math.log(pats['p_freq'] + 1)
    doc_freq = math.log(1 + (docs / pats['doc_freq']))
    diversity = math.log(pats['word'] + 1)

    return (freq * doc_freq * diversity)

def build_models(name_conflict,name_not_conflict,pat_deg_thres,conflict,not_conflict):
    print('starting building model...')
    c_degrees = unify(name_conflict)
    nc_degrees = unify(name_not_conflict)
    c_degrees = c_degrees.drop_duplicates()
    nc_degrees = nc_degrees.drop_duplicates()
    c_degrees['degree'] = c_degrees.apply(lambda x: c_score(x, len(conflict)), axis=1)
    nc_degrees['degree'] = nc_degrees.apply(lambda x: c_score(x, len(not_conflict)), axis=1)

    print('filtering those under pre-defined threshold...')
    c_degrees = c_degrees[c_degrees['p_freq'] > 0]
    nc_degrees = nc_degrees[nc_degrees['p_freq'] > 0]

    c_degrees = c_degrees[c_degrees['degree'] >= pat_deg_thres]
    nc_degrees = nc_degrees[nc_degrees['degree'] >= pat_deg_thres]
    if len(c_degrees)==0:
        return -1

    # #trimming to balance patterns
    max_size = min([len(c_degrees),len(nc_degrees)])
    c_degrees = c_degrees.sort_values(by=['degree'],ascending=False)
    nc_degrees = nc_degrees.sort_values(by=['degree'], ascending=False)
    c_degrees = c_degrees.iloc[:max_size-1]
    nc_degrees = nc_degrees.iloc[:max_size-1]

    pat_dif = 15  #how much difference between the two models are we willing to accept?
    while True:
        c_min =c_degrees['degree'].min()
        nc_min = nc_degrees['degree'].min()
        if (c_min <= (nc_min+pat_dif)) and (c_min >= (nc_min-pat_dif)):
            break
        else:
            c_degrees = c_degrees.iloc[:len(c_degrees) - 2]
            nc_degrees = nc_degrees.iloc[:len(nc_degrees) - 2]
        if len(c_degrees)<500:
            break

    print('building ranks...')
    c_degrees = c_degrees.drop_duplicates()
    nc_degrees = nc_degrees.drop_duplicates()
    c_degrees = c_degrees.sort_values(by=['degree'],ascending=True)
    nc_degrees = nc_degrees.sort_values(by=['degree'], ascending=True)
    c_degrees['rank'] = range(len(c_degrees))
    c_degrees['rank'] = c_degrees['rank'] + 1
    nc_degrees['rank'] = range(len(nc_degrees))
    nc_degrees['rank'] = nc_degrees['rank'] + 1

    new_name = name_conflict + '_' +str(pat_deg_thres)
    c_degrees = c_degrees.loc[:,['og_pattern_x','pattern', 'p_freq', 'doc_freq', 'word','degree', 'rank']]
    nc_degrees = nc_degrees.loc[:,['og_pattern_x','pattern', 'p_freq', 'doc_freq', 'word','degree', 'rank']]
    c_degrees.columns = ['og_pattern','pattern','freq','doc_freq','div','degree','rank']
    nc_degrees.columns = ['og_pattern','pattern','freq','doc_freq','div','degree','rank']
    c_degrees.to_csv('data/c_model_'+new_name+'.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    nc_degrees.to_csv('data/nc_model_'+new_name+'.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    print('models built:','data/c_model_'+new_name+'.csv','data/nc_model_'+new_name+'.csv')
    return new_name

def threadize_class(og_df,name,label,clusters):
    done = pd.read_csv('controls/class_threads_done.csv')
    new_name = name + '_t0_' + label
    if len(done[done['name']==new_name])==0:
        cores = multiprocessing.cpu_count()
        total = len(og_df)
        inc = math.ceil(total/cores)
        if inc*cores > total:
            cores = math.ceil(total/inc)
        processes = [mp.Process(target=classify_w_model, args=(og_df[i*inc:(i*inc)+inc],name+'_t'+str(i),name,label,clusters)) for i in range(cores)]
        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()

def classify_w_model(posts, name, sec,label,clusters):
    print('starting...', name, sec,label)
    c_model = pd.read_csv('data/c_model_'+sec+'.csv')
    nc_model = pd.read_csv('data/nc_model_'+sec+'.csv')
    pre_posts = prepare_for_features(posts)
    f_result = pd.DataFrame(columns=['c_score', 'nc_score','class', 'post_size'])
    count = 0
    inc = 20
    top = inc
    c_model_words = get_words(c_model)
    nc_model_words = get_words(nc_model)
    for post in pre_posts:
        result = generate_post_patterns(clean_post_cluster(post, c_model_words,clusters),clusters)
        c_score = 0
        if len(result)>0:
            c_result = result.merge(c_model, how='inner', on=['pattern'])
            if len(c_result) > 0:
                c_result = c_result.apply(lambda x_res: x_res['p_freq'] * x_res['rank'], axis=1)
                c_score = c_result.sum()

        result = generate_post_patterns(clean_post_cluster(post, nc_model_words,clusters),clusters)
        nc_score = 0
        if len(result)>0:
            nc_result = result.merge(nc_model, how='inner', on=['pattern'])
            if len(nc_result)>0:
                nc_result = nc_result.apply(lambda x_res: x_res['p_freq'] * x_res['rank'], axis=1)
                nc_score = nc_result.sum()
        res_class = 'c' if c_score >= nc_score else 'nc'
        f_result = f_result.append(
            {'c_score': c_score,'nc_score':nc_score, 'class': res_class, 'post_size': len(post)}, ignore_index=True)
        count += 1
        if count > top:
            print('done with ', count, 'total', len(posts))
            top += inc
    f_result.index = range(len(f_result.index))
    f_result['label'] = label
    name_to_save = name + '_'+label
    f_result.to_csv('classification/' + name_to_save + '.csv', index=False, encoding='utf-8',
                                    quoting=csv.QUOTE_NONNUMERIC)
    done = pd.DataFrame([name_to_save])
    done['sec'] = sec
    done.to_csv('controls/class_threads_done.csv', index=False, encoding='utf-8', mode='a', header=False)

def unify_class(name):
    result = pd.DataFrame()
    control = pd.read_csv('controls/class_threads_done.csv')
    control = control[control['sec']==name]
    for num,file in control.iterrows():
        cur_file = pd.read_csv('classification/'+file['name']+'.csv')
        result = pd.concat([result,cur_file])
    return result

def classify(c_test,nc_test,name,clusters):
    threadize_class(c_test, name, 'c',clusters)
    threadize_class(nc_test, name, 'nc',clusters)
    res_c = unify_class(name)
    res_c['correct'] = res_c.apply(lambda row: 1 if row['class'] == row['label'] else 0, axis=1)
    pre_c = res_c[res_c['label']=='c']
    pre_tf = res_c[res_c['class']=='c']
    accuracy = res_c['correct'].sum()/len(res_c)
    precision = 0
    if len(pre_tf)>0:
        precision = pre_c['correct'].sum()/len(pre_tf)
    recall = 0
    if len(pre_c)>0:
        recall = pre_c['correct'].sum()/len(pre_c)
    F1 = 0
    if (precision+recall)>0:
        F1 = 2*((precision*recall)/(precision+recall))
    result = pd.DataFrame([name],columns=['name'])
    result['accuracy']=accuracy
    result['precision']=precision
    result['recall']=recall
    result['F1']=F1
    return result

def batter_test_args(c_train,nc_train,t_min,t_max,t_inc,e_min,e_max,e_inc,c_min,c_max,c_inc,p_min,p_max,p_inc,cv,c=500):
    done = pd.read_csv('classification/batter_results.csv')
    clusters = pd.read_csv('clusters/'+str(c)+'.csv')
    restart = False
    first = False
    mode = 'a'
    if restart:
        first = True
        mode = 'w'
    kfold = cv
    bucket_size = math.ceil(len(c_train)/kfold)
    for t in range(t_min,t_max,t_inc):
        for e in range(e_min,e_max,e_inc):
            for c in range(c_min,c_max,c_inc):
                for p in range(p_min,p_max,p_inc):
                    for k in range(kfold):
                        name = 'conflict_ale_' + str(t/100000) + '_' + str(e / 100) + '_' + str(c / 100) + '_' + str(
                            k) + '_' + str(p)
                        if (len(done[done['name'] == name]) == 0):
                            print(k*bucket_size,(k+1)*bucket_size)
                            c_test = c_train.loc[k*bucket_size:(k+1)*bucket_size]
                            nc_test = nc_train.loc[k*bucket_size:(k+1)*bucket_size]
                            if k > 0:
                                c_train_now = pd.concat([c_train.loc[:k*bucket_size-1],c_train.loc[(k+1)*bucket_size+1:]],sort=False)
                                nc_train_now = pd.concat([nc_train.loc[:k*bucket_size-1],nc_train.loc[(k+1)*bucket_size+1:]],sort=False)
                            else:
                                c_train_now = c_train.loc[(k+1)*bucket_size+1:]
                                nc_train_now = nc_train.loc[(k+1)*bucket_size+1:]
                            name = build_graph_and_model(c_train_now,nc_train_now,t/100000,e/100,c/100,k,p,clusters)
                            r = classify(c_test,nc_test,name,clusters)
                            r['t']=t/100000
                            r['e']=e/100
                            r['c']=c/100
                            r['k']=k
                            r['p']=p
                            r.to_csv('classification/batter_results.csv', index=False, encoding='utf-8', mode=mode,header=first)
                            first = False
                            mode = 'a'
                        else:
                            print('the current combination has already been done:', t, e, c, p, k)

def clean_post(post,words):
    res_post = deepcopy(post)
    for i in range(len(post)):
        if res_post[i] not in words:
            res_post[i] = '.+' + res_post[i]
    return res_post

def clean_post_cluster(post,words,clusters):
    res_post = deepcopy(post)
    for i in range(len(post)):
        if res_post[i] not in words:
            res_post[i] = get_cluster_name(res_post[i],clusters)
    return res_post


def get_words(model):
    new_texts = model['og_pattern'].apply(lambda x: nltk.word_tokenize(x))
    new_texts = list(new_texts)
    new_texts = [item for sublist in new_texts for item in sublist]
    new_texts = list(set(new_texts))
    new_texts.remove('.+')
    return new_texts

def get_words_pat(pats):
    new_texts = [nltk.word_tokenize(pat) for pat in pats.og_pattern]
    new_texts = [item for sublist in new_texts for item in sublist]
    new_texts = list(set(new_texts))
    new_texts.remove('.+')
    return new_texts

def add_to_count_dict(pat,my_dict):
    my_dict[pat] = my_dict.get(pat, 0) + 1
    return my_dict

def add_diversity_word(pat,word,my_dict):
    my_dict[pat] = my_dict.get(pat, []) + [word]
    return my_dict

def generate_post_patterns(post,clusters,div_res=False,fix_wildcards=False):
    my_pats = {}
    my_div = {}
    for i in range(len(post) - 1):
        if ~post[i].startswith('.+') and ~post[i + 1].startswith('.+'):  #none of the words are wildcards
            my_pats = add_to_count_dict(get_cluster_name(post[i],clusters) +' ' + post[i + 1],my_pats)
            my_pats = add_to_count_dict(post[i] + ' ' + get_cluster_name(post[i+1],clusters),my_pats)
            my_div = add_diversity_word(get_cluster_name(post[i],clusters)+ ' ' + post[i + 1], post[i],my_div)
            my_div = add_diversity_word(post[i] + ' ' + get_cluster_name(post[i+1],clusters), post[i + 1],my_div)
        else:
            if fix_wildcards:
                if post[i].startswith('.+') and ~post[i+1].startswith('.+'):
                    my_pats = add_to_count_dict(get_cluster_name(post[i].replace('.+',''),clusters) + ' ' + post[i + 1],my_pats)
                    my_div = add_diversity_word(get_cluster_name(post[i].replace('.+',''),clusters) + ' ' + post[i + 1], post[i], my_div)
                elif ~post[i].startswith('.+') and post[i+1].startswith('.+'):
                    my_pats = add_to_count_dict(post[i] + ' ' + get_cluster_name(post[i+1].replace('.+',''),clusters), my_pats)
                    my_div = add_diversity_word(post[i] + ' ' + get_cluster_name(post[i+1].replace('.+',''),clusters),post[i+1], my_div)
            else:
                if post[i].startswith('.+') and ~post[i+1].startswith('.+'):
                    my_pats = add_to_count_dict(post[i] + ' ' + post[i + 1],my_pats)
                    my_div = add_diversity_word(post[i] + ' ' + post[i + 1], post[i], my_div)
                elif ~post[i].startswith('.+') and post[i+1].startswith('.+'):
                    my_pats = add_to_count_dict(post[i] + ' ' + post[i + 1], my_pats)
                    my_div = add_diversity_word(post[i] + ' ' + post[i + 1], post[i+1], my_div)
        if i < len(post) - 2:
            if ~post[i].startswith('.+') and ~post[i + 1].startswith('.+') and ~post[i + 2].startswith('.+'):
                my_pats = add_to_count_dict(get_cluster_name(post[i].replace('.+',''),clusters) + ' ' + post[i + 1] + ' ' + post[i + 2],my_pats)
                my_pats = add_to_count_dict(post[i] + ' ' + get_cluster_name(post[i+1].replace('.+',''),clusters) + ' ' + post[i + 2],my_pats)
                my_pats = add_to_count_dict(post[i] + ' ' + post[i + 1] + ' ' + get_cluster_name(post[i+2].replace('.+',''),clusters),my_pats)
                my_div = add_diversity_word(get_cluster_name(post[i].replace('.+',''),clusters) + ' ' + post[i + 1] + ' ' + post[i + 2], post[i], my_div)
                my_div = add_diversity_word(post[i] + ' ' + get_cluster_name(post[i+1].replace('.+',''),clusters) + ' ' + post[i + 2], post[i + 1], my_div)
                my_div = add_diversity_word(post[i] + ' ' + post[i + 1] + ' ' + get_cluster_name(post[i+2].replace('.+',''),clusters), post[i + 2], my_div)
            else:
                # we need to find the wildcard agh
                if fix_wildcards:
                    if post[i].startswith('.+') and ~post[i+1].startswith('.+') and  ~post[i+2].startswith('.+'):
                        my_pats = add_to_count_dict(get_cluster_name(post[i].replace('.+',''),clusters)+ ' ' + post[i + 1] + ' ' + post[i + 2], my_pats)
                        my_div = add_diversity_word(get_cluster_name(post[i].replace('.+',''),clusters)+ ' ' + post[i + 1] + ' ' + post[i + 2], post[i], my_div)
                    elif ~post[i].startswith('.+') and post[i+1].startswith('.+') and  ~post[i+2].startswith('.+'):
                        my_pats = add_to_count_dict(post[i] + ' ' + get_cluster_name(post[i+1].replace('.+',''),clusters)+ ' ' + post[i + 2], my_pats)
                        my_div = add_diversity_word(post[i] + ' ' + get_cluster_name(post[i+1].replace('.+',''),clusters)+ ' ' + post[i + 2], post[i + 1], my_div)
                    elif ~post[i].startswith('.+') and ~post[i+1].startswith('.+') and  post[i+2].startswith('.+'):
                        my_pats = add_to_count_dict(post[i] + ' ' + post[i + 1] + ' ' + get_cluster_name(post[i+2].replace('.+',''),clusters), my_pats)
                        my_div = add_diversity_word(post[i] + ' ' + post[i + 1] + ' ' + get_cluster_name(post[i+2].replace('.+',''),clusters), post[i + 2], my_div)
                else:
                    if post[i].startswith('.+') and ~post[i + 1].startswith('.+') and ~post[i + 2].startswith('.+'):
                        my_pats = add_to_count_dict(post[i] + ' ' + post[i + 1] + ' ' + post[i + 2], my_pats)
                        my_div = add_diversity_word(post[i] + ' ' + post[i + 1] + ' ' + post[i + 2], post[i], my_div)
                    elif ~post[i].startswith('.+') and post[i + 1].startswith('.+') and ~post[i + 2].startswith('.+'):
                        my_pats = add_to_count_dict(post[i] + ' ' + post[i + 1] + ' ' + post[i + 2], my_pats)
                        my_div = add_diversity_word(post[i] + ' ' + post[i + 1] + ' ' + post[i + 2], post[i + 1], my_div)
                    elif ~post[i].startswith('.+') and ~post[i + 1].startswith('.+') and post[i + 2].startswith('.+'):
                        my_pats = add_to_count_dict(post[i] + ' ' + post[i + 1] + ' ' + post[i + 2], my_pats)
                        my_div = add_diversity_word(post[i] + ' ' + post[i + 1] + ' ' + post[i + 2], post[i + 2], my_div)
    my_pats_df = pd.DataFrame.from_dict(my_pats,orient='index',columns=['p_freq'])
    my_pats_df['pattern'] = my_pats_df.index
    my_div_df = pd.DataFrame([[v, j] for v in my_div for j in my_div[v]], columns=['pattern', 'word'])
    if div_res:
        return my_pats_df,my_div_df
    return my_pats_df

def get_cluster_name(word,clusters):
    c = clusters[clusters['word']==word]
    if len(c)>0:
        return '.+C'+str(c.cluster.iloc[0])
    else:
        return '.+C'+str(clusters.cluster.max()+1)