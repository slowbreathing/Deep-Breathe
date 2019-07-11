

class ExecutionContext(object):

    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if ExecutionContext.__instance == None:
            ExecutionContext()
        return ExecutionContext.__instance
    def __init__(self):
        """ Virtually private constructor. """
        if ExecutionContext.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            ExecutionContext.__instance = self
        self.layers=[]
        self.layeyhashdict={}
        self.curr_layer=None
    def register(self,layer):
        """
        Keeps track of sequence of operations
        and runs compute_gradient in reverse order
        """
        if(hash(layer) in self.layeyhashdict):
            """"""
        else:
            self.layeyhashdict[hash(layer)]=layer
            self.layers.append(layer)
            layerlen=len(self.layers)
            if (layerlen>1):
                prevlayer=self.layers[layerlen-2]
                prevlayer.prev=layer
                layer.next=prevlayer

    def get_layer(self,index):
        return self.layers[index]

    def get_prev_layer(self):

        layerlen=len(self.layers)
        if(layerlen != 0 ):
            if(self.curr_layer is None):
                return self.layers[-1]
            else:
                if(layerlen >= 2):
                    return self.layers[-2]
                else:
                    return None
        else:
            return None

    def compute_gradients(self):
        totalgrad=[]
        #print("len(self.layers):",len(self.layers),type(self.layers[0]))
        #lrs=zip(self.layers)
        for i in range(len(self.layers)):
            l=self.layers[i]
            #print("compute_gradient():",type(l),l.name)
            #if isinstance(l,FFlayer):
                #print(l.layer.kernel_shape)
        #[print("l.name:",l.name) for l in lrs]
        for i,item in enumerate(reversed(self.layers)):
            #print("item.yhat:",item.yhat)
            grad=item.compute_gradient()
            #print("compute_gradient():",type(item),item.name)

            if(grad is not None):
                if(item.name is None):
                    name=i
                else:
                    name=item.name
                totalgrad.append((name,grad))


        """
        list of tuple of name, gradient
        where the gradient is a dict of cell name and its gradients as shown
        [(layer,grad)]
        ||------------|----------|----------------|
        ||            |  Y       |((dWy,y)(dBy,b))|
        ||            |----------|----------------|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||   Decoder  |     fw_cell       |--------------------|------------------------------------------------------------------|
        ||            |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |     bw_cell       |--------------------|------------------------------------------------------------------|
        ||            |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||------------|-------------------|--------------------|------------------------------------------------------------------|
        ||------------|-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |     fw_cell       |--------------------|------------------------------------------------------------------|
        ||  Encoder   |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |-------------------|--------------------|------------------------------------------------------------------|
        ||            |                   |    position(0)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||            |     bw_cell       |--------------------|------------------------------------------------------------------|
        ||            |                   |--------------------|------------------------------------------------------------------|
        ||            |                   |    position(1)     |((dwi,wi)(dwc,wc)(dwf,wf)(dwo,wo)(dbi,bi)(dbc,bc)(dbf,bf)(dbo,bo))|
        ||------------|-------------------|--------------------|------------------------------------------------------------------|
        """
        return totalgrad

    def current_layer(self,c_layer):
        self.curr_layer=c_layer

    def get_current_layer(self):
        return self.curr_layer

    def clean_current(self):
        self.curr_layer=None

    def clean(self):
        self.clean_current()
        del self.layers[:]
        self.layeyhashdict.clear()

