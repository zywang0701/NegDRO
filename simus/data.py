import numpy as np

class StructuralCausalModelSimu1:
    
    def __init__(self, p):
        self.p = p
    
    def sample(self, n, mode=2, seed=None):
        """
        Generates synthetic data for causal inference simulations.
        
        Parameters:
        - n: Number of samples per environment.
        
        Returns:
        - x_list: List of predictor matrices for each environment.
        - y_list: List of response vectors for each environment.
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate the heterogeneous noises
        n_env = 4
        x_list, y_list = [], []
        for e in range(n_env):
            X, Y = [], []
            if mode == 1:
                H = np.random.randn(n)
                alpha_e = 0.5 + 0.2 * e
                noise_y = np.random.randn(n) + 0.5 * H
                base_mat = np.random.randn(n, self.p)
                delta_mat = np.zeros((n, self.p))
                if e == 1:
                    delta_mat[:,:5] = 3 * np.random.randn(n, 5)
                if e == 2:
                    delta_mat[:,:5] = np.array([1, 2, -1, -2, 1])
                if e == 3:
                    delta_mat[:,:5] = np.random.uniform(low=-1, high=1, size=(n, 5))
                delta_mat[:, 5:] = 0.5 * e * np.random.randn(n, self.p - 5)
                delta_mat[:, 0] += alpha_e * H
            
            if mode == 2:
                noise_y = np.random.randn(n)
                base_mat = np.random.randn(n, self.p)
                delta_mat = np.zeros((n, self.p))
                if e == 1:
                    delta_mat[:,:5] = 3 * np.random.randn(n, 5)
                if e == 2:
                    delta_mat[:,:5] = np.array([1, 2, -1, -2, 1])
                if e == 3:
                    delta_mat[:,:5] = np.random.uniform(low=-1, high=1, size=(n, 5))
                delta_mat[:, 5:] = 0.5 * e * np.random.randn(n, self.p - 5)
            
            noise_x = base_mat + delta_mat
            
            for i in range(n):
            
                X1_val = noise_x[i, 0]
                X2_val = X1_val + noise_x[i, 1]
                X3_val = X2_val + noise_x[i, 2]
                Y_val = 0.5 * X1_val - 0.5 * X3_val + noise_y[i]
                X4_val = 0.5 * Y_val + noise_x[i, 3]
                X5_val = - 0.5 * Y_val + noise_x[i, 4]
                Xjunk_val = noise_x[i, 5:]
                
                # Combine all predictors into one vector
                X_row = np.concatenate((np.array([X1_val, X2_val, X3_val, X4_val, X5_val]), Xjunk_val))
                X.append(X_row)
                Y.append(Y_val)
            X, Y = np.array(X), np.array(Y)
            x_list.append(X)
            y_list.append(Y)
            
        return x_list, y_list
    
    def test(self, n, sigma0_list, seed=None):
        """
        Generates synthetic data for evaluation.
        
        Parameters:
        - n: Number of samples in each test environment.
        - sigma0_list: List of noise levels for each test environment.
        
        Returns:
        - x0_list: List of predictor matrices for each test environment.
        - y0_list: List of response vectors for each test environment.
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate the heterogeneous noises
        x0_list, y0_list = [], []
        for e in range(len(sigma0_list)):
            X, Y = [], []
            noise_y = np.random.randn(n)
            base_mat = np.random.randn(n, self.p)
            delta_mat = np.zeros((n, self.p))
            delta_mat[:,0:5] = sigma0_list[e] * np.random.randn(n, 5)
            
            noise_x = base_mat + delta_mat
            
            for i in range(n):
            
                X1_val = noise_x[i, 0]
                X2_val = X1_val + noise_x[i, 1]
                X3_val = X2_val + noise_x[i, 2]
                Y_val = X1_val + X3_val + noise_y[i]
                X4_val = Y_val + noise_x[i, 3]
                X5_val = - Y_val + noise_x[i, 4]
                Xjunk_val = noise_x[i, 5:]
                
                # Combine all predictors into one vector
                X_row = np.concatenate((np.array([X1_val, X2_val, X3_val, X4_val, X5_val]), Xjunk_val))
                X.append(X_row)
                Y.append(Y_val)
            X, Y = np.array(X), np.array(Y)
            x0_list.append(X)
            y0_list.append(Y)
            
        return x0_list, y0_list


class StructuralCausalModelEg2:
    
    def __init__(self, p):
        self.p = 4
    
    def sample(self, n, mode=1, seed=None):
        """
        Generates synthetic data for Limited Interventions.
        
        Parameters:
        - n: Number of samples per environment.
        - mode: Mode of the simulation. model=1, indicates limited interventions;
                model=2, indicates weak interventions;
                model=3, indicates full interventions.
        
        Returns:
        - x_list: List of predictor matrices for each environment.
        - y_list: List of response vectors for each environment.
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate the heterogeneous noises
        n_env = 2
        x_list, y_list = [], []
        for e in range(n_env):
            X, Y = [], []
            noise_y = np.random.randn(n)
            base_mat = np.random.randn(n, self.p)
            delta_mat = np.zeros((n, self.p))
            if e == 1:
                delta_mat[:,2] = 2 * np.random.randn(n)
                if mode == 1:
                    delta_mat[:,[0,1,3]] = 0.
                if mode == 2:
                    delta_mat[:,[0,1,3]] = np.random.randn(n, 3) * 0.1 #1 / np.sqrt(n)
                if mode == 3:
                    delta_mat[:,[0,1,3]] = np.random.randn(n, 3) * 0.4
            
            noise_x = base_mat + delta_mat
            
            for i in range(n):
            
                X1_val = noise_x[i, 0]
                X2_val = X1_val + noise_x[i, 1]
                Y_val = 2 * X2_val + noise_y[i]
                X3_val = 0.5 * X1_val + 0.5 * Y_val + noise_x[i, 2]
                X4_val = noise_x[i, 3]
                
                # Combine all predictors into one vector
                X_row = np.array([X1_val, X2_val, X3_val, X4_val])
                X.append(X_row)
                Y.append(Y_val)
            X, Y = np.array(X), np.array(Y)
            x_list.append(X)
            y_list.append(Y)
            
        return x_list, y_list