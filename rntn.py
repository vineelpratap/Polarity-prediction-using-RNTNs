import numpy as np
import collections
np.seterr(over='raise',under='raise')


def softmax(v):
	e = np.exp(v)
	e = e/np.sum(e)
	return e

def fprime(x):
	der = np.ones(x.shape) - x**2
	return der

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)
        
        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. V, W, Ws, b, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
           
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

       	self.L,self.V,self.W,self.b,self.Ws,self.bs = self.stack

        # Zero gradients
        self.V[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata: 
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    def forwardProp(self,node,correct = [], guess= [], test = False):
        cost  =  total = 0.0 # cost should be a running number and total is the total examples we have seen used in accuracy reporting later
        ################
        # TODO: Implement the recursive forwardProp function
        #  - you should update node.probs, node.hActs1, node.fprop, and cost
        #  - node: your current node in the parse tree
        #  - correct: this is a running list of truth labels
        #  - guess: this is a running list of guess that our model makes
        #     (we will use both correct and guess to make our confusion matrix)
        ################
        if node.isLeaf:
			node.hActs1 = self.L[:,node.word]	
        else:
			if not node.left.fprop:
				c, t = self.forwardProp(node.left, correct, guess)
				cost, total = cost+c,total+t
			if not node.right.fprop:
				c, t = self.forwardProp(node.right, correct, guess)
				cost, total = cost+c,total+t
			children = np.concatenate((node.left.hActs1, node.right.hActs1))
			node.hActs1 = np.tanh(children.T.dot(self.V).dot(children) + self.W.dot(children) + self.b)
		
        node.probs = softmax(self.Ws.dot(node.hActs1) + self.bs)
        cost = cost -np.log(node.probs[node.label])
        correct.append(node.label)
        guess.append(np.argmax(node.probs))
        node.fprop = True

        return cost, total + 1

    def backProp(self,node,error=None):
        # Clear nodes
        node.fprop = False

        ################
        # TODO: Implement the recursive backProp function
        #  - you should update self.dWs, self.dbs, self.dW, self.db, and self.dL[node.word] accordingly
        #  - node: your current node in the parse tree
        #  - error: error that has been passed down from a previous iteration
        ################
	
		#derivative wrt Ws and bs 
        true = [0]*len(node.probs)
        true[node.label] = 1
        deltasm = node.probs - true
        self.dWs += np.outer(deltasm,node.hActs1)		
        self.dbs += deltasm

		#deltas described in the paper 
        deltas = self.Ws.T.dot(deltasm)*fprime(node.hActs1) #delta due to softmax
        deltacom = deltas
        print deltacom.shape
        if error: deltacom = error + deltas #add the backproped delta to it

        if node.isLeaf: 
            self.dL[node.word] += deltacom #eq. to node with input 1 and weight L[ind]
        else:
            children = np.concatenate((node.left.hActs1, node.right.hActs1))	
            print children.shape
            self.dW += np.outer(deltacom, children)
            self.db += deltacom
			
            tomult = (deltacom, children, children)
            self.dV += reduce(np.multiply, np.ix_(*tomult))
			
            deltadown = self.W.T.dot(deltacom) + deltacom.dot(self.V+self.V.T).dot(children)
            self.backProp(node.left, deltadown[:self.wvecDim])	
            self.backProp(node.right, deltadown[self.wvecDim:])	
        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)
        err1 = 0.0
        count = 0.0

        print "Checking dW... (might take a while)"
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






