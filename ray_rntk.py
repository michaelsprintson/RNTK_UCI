import ray
import numpy as np
import jax
import symjax
import symjax.tensor as T

@ray.remote
def diag_func(actor, rntkobj, tiprime, ti, max_dim_1, max_dim_2, n):
    while ((tiprime <= max_dim_1) & (ti <= max_dim_2)):
        print("inner iteration - ", ti, tiprime) #// DIM 1 IS T PRIME, DIM 2 IS T
        actor.add_message.remote(f"inner iteration - {ti} - {tiprime}")
        if ((ti > 0) & (tiprime > 0)):
            # S, D = actor.VT(actor.get_lambda_val.remote(0, ti-1, tiprime-1))
            # actor.set_lambda_val.remote(0, ti, tiprime, actor.sw ** 2 * S + actor.su ** 2 * X + actor.sb ** 2)
            # actor.set_phi_val.remote(0, ti, tiprime, actor.get_lambda_val.remote(0, ti, tiprime) + actor.sw ** 2 * actor.get_phi_val.remote(0, ti - 1, tiprime - 1) * D)
            # actor.set_lambda_val.remote(0, ti, tiprime, actor.get_lambda_val.remote(0, ti-1, tiprime-1))
            # actor.set_phi_val.remote(0, ti, tiprime, actor.get_lambda_val.remote(0, ti, tiprime))
            break
        else:
            # test = rntkobj.sh ** 2 * rntkobj.sw ** 2 + (rntkobj.su ** 2)+ rntkobj.sb ** 2
            test = rntkobj.sh ** 2 * rntkobj.sw ** 2 * T.eye(n, n) + (rntkobj.su ** 2) + rntkobj.sb ** 2
            print("inner iteration test value - ", test)
            actor.add_message.remote(f"inner iteration test value - {test}")
            # actor.add_message.remote(f"inner iteration eye test value - {T.eye(10, 10)}")
            # test = rntkobj.sh ** 2 * rntkobj.sw ** 2 + (rntkobj.su ** 2)+ rntkobj.sb ** 2
            
            
            # phimatrix[0,ti,tiprime] = lambdamatrix[0,ti,tiprime] = T.expand_dims(test, axis = 0) # line 2, alg 1
            # phimatrix[0,ti,tiprime] = lambdamatrix[0,ti,tiprime] = test # line 2, alg 1
            # actor.set_phi_val.remote(0, ti, tiprime, test)
            actor.set_lambda_val.remote(0, ti, tiprime, test)
            # actor.set_lambda_val.remote(0, ti, tiprime, rntkobj.sh ** 2 * rntkobj.sw ** 2 + (rntkobj.su ** 2)+ rntkobj.sb ** 2)
            actor.set_phi_val.remote(0, ti, tiprime, 2)

        tiprime+=1
        ti+=1

@ray.remote
class RNTKActor(object):
    def __init__(self, dim_1, dim_2, dic):
        self.lambdamatrix = np.zeros([1, dim_1 + 1, dim_2 + 1],dtype = object)
        self.phimatrix = np.zeros([1, dim_1 + 1, dim_2 + 1],dtype = object)
        self.logs = []
    
    def set_lambda_val(self, l_idx, t_idx, tp_idx, set_value):
        # self.lambdamatrix[l_idx, t_idx, tp_idx] = 3
        self.lambdamatrix[l_idx, t_idx, tp_idx] = set_value
    
    def set_phi_val(self, l_idx, t_idx, tp_idx, set_value):
        # self.phimatrix[l_idx, t_idx, tp_idx] = 2
        self.phimatrix[l_idx, t_idx, tp_idx] = set_value
    
    def get_lambda_val(self, l_idx, t_idx, tp_idx):
        return self.lambdamatrix[l_idx, t_idx, tp_idx]
    
    def get_phi_val(self, l_idx, t_idx, tp_idx):
        return self.phimatrix[l_idx, t_idx, tp_idx]

    def get_lambda(self):
        return self.lambdamatrix
    
    def get_phi(self):
        return self.phimatrix

    def add_message(self, message):
        self.logs.append(message)
    
    def get_and_clear_messages(self):
        logs = self.logs
        # self.logs = []
        return logs

class RNTKTEST():

    def __init__(self, dic):
        self.sw = 1
        self.su = 1
        self.sb = 1
        self.sh = 1
        self.L = 1
        self.Lf = 0
        self.sv = 1
        self.N = int(dic["n_patrons1="])
        self.length = int(dic["n_entradas="])
        
    def VT(self, M):
        A = T.diag(M)  # GP_old is in R^{n*n} having the output gp kernel
        # of all pairs of data in the data set
        B = A * A[:, None]
        C = T.sqrt(B)  # in R^{n*n}
        D = M / C  # this is lamblda in ReLU analyrucal formula
        E = T.clip(D, -1, 1)  # clipping E between -1 and 1 for numerical stability.
        F = (1 / (2 * np.pi)) * (E * (np.pi - T.arccos(E)) + T.sqrt(1 - E ** 2)) * C
        G = (np.pi - T.arccos(E)) / (2 * np.pi)
        return F,G