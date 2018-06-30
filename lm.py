import sys, re, collections, copy, math

class LM:

    def __init__(self,logbydefault=False):
        self.condprobs = None
        self.logcondprobs = None
        self.logbydefault = logbydefault

    def learn_lm(self,weightedstrings):
        events = {}
        for (s,w) in weightedstrings:
            for i in range(len(s)):
                context = s[0:i]
                event = s[i]
                if not context in events:
                    events[context] = collections.Counter()
                events[context][event] += w
        #print events
        for context in events:
            z = float(sum(events[context].values()))
            for event in events[context].keys():
                ##if z==0.0 and float(events[context][event]) != 0.0:
                ##    print "#Something's wrong: context count
                #if z==0.0:
                #    print "Found zero-weight support for context ",context," when event ",event," has weight, ", float(events[context][event])
                #    if float(events[context][event]):
                if z==0.0:
                    events[context][event] = 0.0 # a bit inelegant
                else:
                    events[context][event] = float(events[context][event])/z
        #print events
        self.condprobs = copy.deepcopy(events)
        for context in events:
            for event in events[context].keys():
                if events[context][event]==0.0:
                    events[context][event] = -float('inf')
                else:
                    events[context][event] = math.log(events[context][event])
        #print events
        self.logcondprobs = events

    def condprob(self,context,sym,reportlog=None):
        if reportlog==True or (reportlog==None and self.logbydefault):
            return self.logcondprobs[context][sym]
        else:
            return self.condprobs[context][sym]

    def score_string_UID(self,s,k):
        """High scores are bad"""
        result = 0
        for i in range(len(s)):
            p = self.condprob(s[0:i],s[i],reportlog=False)
            if p==0:
                return float('inf')
            else:
                result += (-math.log(p))**k
        return result
        
if __name__ == "__main__":
    lm = LM()
    lm.learn_lm([("ab$",1.0),("ac$",3.0)])
    print lm.score_string_UID("ab$",1)
    print lm.score_string_UID("ac$",1)
    
