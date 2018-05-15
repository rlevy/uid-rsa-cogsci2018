import lm,math,itertools,numpy,numpy.random,copy, multiprocessing, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_seed',type=int,default=0)
parser.add_argument('--num_seeds',type=int,default=100)
parser.add_argument('--num_processes',type=int,default=21)

def score_string(lm,s,k,c):
    """k: UID exponent; c: string cost"""
    return lm.score_string_UID(s,k) + c*len(s)

def reweight_string_set(lm,strings,total_mass,k,c,alpha):
    """The RSA S1 speaker model on the alternative utterance set consisting of the strings in the string set, all for the same meaning.  Reallocate total_mass among them"""
    scores = [0] * len(strings)
    for i in range(len(strings)):
        scores[i] = math.exp(- alpha * score_string(lm,strings[i],k,c))
    z = sum(scores)
    return [total_mass*x/z for x in scores]
    

def strings():
    A = ["a","A"]
    B = ["b","B"]
    C = ["c","C"]
    T = ["t",""]
    EOS = "$"
    strings = []
    pairs = []
    for a12 in list(itertools.product(A,A)):
        for c12 in list(itertools.product(C,C)):
            strings.append("".join(list(a12)+list(c12)+[EOS]))
            for b12 in list(itertools.product(B,B)):
                thispair = []
                for t in T:
                    thispair.append("".join(list(a12)+[t]+list(b12)+list(c12)+[EOS]))
                pairs.append(thispair)
    return((strings,pairs))

def weight_strings(strings,pairs,B_prob=0.5,t_prob=0.5,random_state=numpy.random):
    """Initialize weights (probabilities) for a set of strings & string pairs"""
    ws = random_state.dirichlet([1.0]*len(strings))
    weighted_strings = zip(strings,[(1-B_prob)*w for w in ws])
    ws = random_state.dirichlet([1.0]*len(pairs))
    weighted_string_pairs = [((s1,t_prob*B_prob*w),(s2,(1-t_prob)*B_prob*w))  for ((s1,s2),w) in zip(pairs,ws)]
    return (weighted_strings,weighted_string_pairs)


def S1(lm,weighted_string_pairs,k,c,alpha):
    """Reweight the weighted string pairs according to reweight_string_set() for each pair"""
    string_pairs = [(s1,s2) for ((s1,_w1),(s2,_w2)) in weighted_string_pairs]
    string_pair_masses = [w1+w2 for ((_s1,w1),(_s2,w2)) in weighted_string_pairs]
    new_pair_weights = [reweight_string_set(lm,p,m,k,c,alpha) for (p,m) in zip(string_pairs,string_pair_masses)]
    return [((s1,w1),(s2,w2)) for ((w1,w2),((s1,_old_w1),(s2,_old_w2))) in zip(new_pair_weights,weighted_string_pairs)]

def learn_lm(weighted_strings,weighted_string_pairs):
    result = lm.LM()
    tmp = []
    for ((s1,w1),(s2,w2)) in weighted_string_pairs:
        tmp = tmp + [(s1,w1),(s2,w2)]
    result.learn_lm(weighted_strings + tmp)
    return result

def learn_lm_ignoring_that(weighted_strings,weighted_string_pairs):
    wsp = []
    for i in range(len(weighted_string_pairs)):
        p = weighted_string_pairs[i]
        wsp.append( ((p[0][0][0:2]+p[0][0][3:len(p[0][0])],p[0][1]),p[1]) )
    return learn_lm(weighted_strings,wsp)
        
def compare_old_new_weights(weighted_string_pairs,reweighted_string_pairs):
    """Sanity check function for testing program behavior"""
    for i in range(len(weighted_string_pairs)):
        print "#\n#"
        print "Old: ", weighted_string_pairs[i][0]
        print "New: ", reweighted_string_pairs[i][0]
        print "#"
        print "Old: ", weighted_string_pairs[i][1]
        print "New: ", reweighted_string_pairs[i][1]

def record_nextword_prob_and_that_use(lm_no_t,weighted_string_pairs):
    """This function will not generalize well, as written -- it is very specific to the length-2 context, second-in-pair-is-t-omitted case"""
    nextword_probs = []
    that_probs = []
    for x in weighted_string_pairs:
        nextword_prob = lm_no_t.condprob(x[1][0][0:2],x[1][0][2])
        nextword_probs.append(nextword_prob)
        that_prob = x[0][1] / (x[0][1]+x[1][1])
        that_probs.append(that_prob)        
    return (nextword_probs,that_probs)

def overall_that_rate(weighted_string_pairs):
    Z = 0.0
    q = 0.0
    for p in weighted_string_pairs:
        q += p[0][1]
        Z += p[0][1] + p[1][1]
    return q/Z

def find_fixed_point(strings,pairs,k,c,alpha,tol,seed):
    random_state = numpy.random.RandomState(seed)
    B_prob = random_state.beta(1,1)
    t_prob = random_state.beta(1,1)    
    (wstrings,wpairs) = weight_strings(strings,pairs,B_prob,t_prob)
    lm0 = learn_lm(wstrings,wpairs)
    lm_no_t0 = learn_lm_ignoring_that(wstrings,wpairs)
    np = record_nextword_prob_and_that_use(lm_no_t0,wpairs)
    old_r = numpy.corrcoef(np[0],np[1])[0,1]
    old_that_rate = overall_that_rate(wpairs)
    numgenerations = 0
    while numgenerations < 100:
        numgenerations += 1
        wpairs1 = S1(lm0,wpairs,k,c,alpha)
        #compare_old_new_weights(wpairs,wpairs1)
        lm1 = learn_lm(wstrings,wpairs1)
        lm_no_t1 = learn_lm_ignoring_that(wstrings,wpairs1)
        np1 = record_nextword_prob_and_that_use(lm_no_t1,wpairs1)    
        r = numpy.corrcoef(np1[0],np1[1])[0,1]
        that_rate = overall_that_rate(wpairs1)
        if that_rate == 0.0 or that_rate == 1.0:
            break
        if abs(r- old_r) < tol and abs(that_rate - old_that_rate) < tol:
        #if abs(that_rate - old_that_rate) < tol:
            break
        wpairs = wpairs1
        lm0 = lm1
        old_r = r
        old_that_rate = that_rate
    return (B_prob,t_prob,numgenerations,r,that_rate)


if __name__ == "__main__":
    args = parser.parse_args()
    tol = 0.001
    (strings,pairs) = strings()
    pool = multiprocessing.Pool(processes=args.num_processes)
    print "Tolerance\tk\tc\tB_prob\tt_prob\tseed\tgenerations\tthatrate\tr"
    for seed in range(args.start_seed,args.start_seed+args.num_seeds):
        for k in [float(x)/20.0 for x in range(20,41)]:
            results = [(pool.apply(find_fixed_point,(strings,pairs,k,c,1.0,1e-04,seed)),c) for c in [float(x)/10.0 for x in range(0,21)]]
            for ((B_prob,t_prob,numgenerations,r,that_rate),c) in results:
                print "\t".join([str(x) for x in [tol,k,c,B_prob,t_prob,seed,numgenerations,that_rate,r]])
            sys.stdout.flush()
