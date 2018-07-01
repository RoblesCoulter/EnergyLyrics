from ngrams import weightedBigrams
import operator
import codecs
import sys
import os


def minusNetworks(subjPath, objPath, dir_path = './carlos_network', modes = [2,3]):
    print("Processing ",subjPath)

    # Get edges for the subjective graph
    edges1 = weightedBigrams(subjPath,modes[0])
    '''print "subjective"
    for edge in edges1:
        print "%s %f"%(edge,edges1[edge])'''

    # Get edges for the objective graph
    edges2 = weightedBigrams(objPath,modes[1])
    '''print "objective"
    for edge in edges2:
        print "%s %f"%(edge,edges2[edge])'''

    diffTh = 0.0;
    edges = {}
    droped = {}
    for edge in edges1:
        if edge in edges2:
            value = edges1[edge] - edges2[edge] 
            if(value >= diffTh):
                edges[edge] = value
            else:
                #print "Drop this shared edge %s: %f"%(edge,value)
                droped[edge] = value
        else:
            if edges1[edge] >= diffTh:
                edges[edge] = edges1[edge]
            else:
                #print "Drop this unique edge %s: %f"%(edge,edges1[edge])
                droped[edge] = edges1[edge]
        
    sorted_edges = sorted(edges.items(), key=operator.itemgetter(1), reverse=True)
    out = codecs.open(os.path.join(dir_path, "subjective"),"w", "utf-8-sig")
    words = {}
    netEdges = {}
    countWords = 0
    for (edge, value) in sorted_edges:

        print("%s\t%f"%(edge.encode('utf-8'),value))
        tokens = edge.split(" ")
        if tokens[0] not in words:
            countWords = countWords + 1
            words[tokens[0]] = countWords
        if tokens[1] not in words:
            countWords = countWords + 1
            words[tokens[1]] = countWords
        netEdge = str(words[tokens[0]]) + " " + str(words[tokens[1]])
        netEdges[netEdge] = value

        out.write("%s\t%f\n"%(edge,value))
    out.close

    out = codecs.open(os.path.join(dir_path,"dropped"),"w", "utf-8-sig")
    sorted_droped = sorted(droped.items(), key=operator.itemgetter(1))
    for (edge, value) in sorted_droped:
        print("%s\t%f"%(edge.encode('utf-8'),value))
        out.write("%s\t%f\n"%(edge,value))
    out.close

    saveNetwork(words,netEdges, dir_path)

#def processTweets(path):


#def tokenize:

def saveNetwork(words, edges, dir_path = './carlos_network'):
    out = codecs.open(os.path.join(dir_path,"minused.net"),"w", "utf-8-sig")
    out2 = codecs.open(os.path.join(dir_path,"minused.vertices"),"w", "utf-8-sig")
    out3 = codecs.open(os.path.join(dir_path,"minused.edges"),"w", "utf-8-sig")
    #Write the vertices names
    out.write("*Vertices "+ str(len(words)) + "\n")
            
    sorted_words = sorted(words.items(), key=operator.itemgetter(1))
    for (word, value) in sorted_words:
        line = str(value) + " \"" + word + "\" 0.0 0.0 0.0";

        out.write(line+"\n");
        out2.write(str(value) + " " + word +"\n")
    #Write the Edges
    out.write("*Arcs \n");
    sorted_edges = sorted(edges.items(), key=operator.itemgetter(1), reverse=True)
    for (edge, value) in sorted_edges:
           
        out.write(edge + " " + str(value)+"\n");
        out3.write(edge + " " + str(value)+"\n")
    out.close();
    out2.close()
    out3.close()


def replaceWordByTag(word):

    # If HT
    if word[0] == '#':
        word = "<H>";
        

    #If URL
    if "http" in word or "https" in word:
        word = "<U>";
        

    #If mini URL
    if len(word) > 3 and "co/" in word[0, 3]:
        word = "<MU>";


    #If user mention
    if word[0] == '@':
        word = "<U>";
        

    return word;

def findEmotionWords(pathEdges, pathNodes, pathPWs, pathEigen):
    out = codecs.open("network/hws_t_bi.csv","w", "utf-8-sig")

    f1 = codecs.open(pathEdges,"r", "utf-8")
    f2 = codecs.open(pathNodes,"r", "utf-8")
    f3 = codecs.open(pathPWs,"r", "utf-8")
    #Add eigen
    f4 = codecs.open(pathEigen,"r","utf-8")

    #Read the edges
    edges = []
    for line in f1:
        edges.append(line)

    #Read the nodes
    nodes = {}
    for line in f2:
        #print line
        tokens = line.split(" ")
        nodes[tokens[0]] = tokens[1].strip()

    #Add Read eigen
    eigen = {}
    for line in f4:
        tokens = line.split("\t")
        eigen[tokens[0]] = tokens[1].strip()

    for line in f3:
        tokens = line.split("\t")
        index = tokens[0]
        word = tokens[1]
        print(line)
        for edge in edges:
            tokens = edge.split(" ")
            index1 = tokens[0]
            index2 = tokens[1]
            value = tokens[2].strip()
            if index1 in nodes and index2 in nodes:
#                if index1 == index or index2 == index:
                if index1 == index:
                    #Add index2 eigen check
#                    if float(value) > 0.001 and float(eigen[index2]) > 0.3:
                    if float(value) > 0.0001:
                        print( word+": "+nodes[index1]+nodes[index2]+" - "+value)
                        out.write(nodes[index1]+nodes[index2]+"\t"+value+"\n")
                elif index2 == index:
#                    if float(value) > 0.001 and float(eigen[index1]) > 0.3:
                    if float(value) > 0.0001:
                        print( word+": "+nodes[index1]+nodes[index2]+" - "+value)
                        out.write(nodes[index1]+nodes[index2]+"\t"+value+"\n")
    out.close()

#findEmotionWords("network/dropped","network/pws")
#findEmotionWords("network/minused.edges","network/minused.vertices","network/hws")
#findEmotionWords("network/minused.edges","network/minused.vertices","network/hws","network/eigen_p")

#minusNetworks("Murmur/total/all", "Twitter/News/all")
#minusNetworks("network/emo","network/all")

# dir_path = './carlos_network'
# minusNetworks("./datasets/emo/Train/train_merged_8emo.tsv","./datasets/news", dir_path, modes=[2,3])

dir_path = './luis_network'

minusNetworks("../data/IEMOCAP_sentences_text","../data/news", dir_path, modes=[2,3])
