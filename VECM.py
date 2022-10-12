import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.vector_ar.vecm import VECM
import pandas as pd
from tqdm import tqdm
import seaborn as sns


np.random.seed(1)



def simulate_series(size, phi, alpha, gamma):
    delta_X = [np.array([0, 0])]
    X = [np.array([0, 0])]
    
    noise_array = np.random.normal(size=(2,size))

    
    for i in range(1, size):
        delta_x = gamma@(alpha.T@X[i-1])+phi@delta_X[i-1]+noise_array[:,i]
        delta_X.append(delta_x)
        X.append(X[i-1]+delta_X[i])
            
    return np.array(X)


#%% Building VECM(1)


def vecm(df:pd.DataFrame, ar_lags:int=1):
    df_x_tilde = pd.DataFrame()
    targets = []
    model = sm.OLS(endog=df.iloc[:,0], exog=df.iloc[:,1])
    
    res1 = model.fit()
    
    alpha = np.array([1, -res1.params[1]])
    z = df.dot(alpha).rename('z')
    
    for col in df:
        targets.append(str(col)+"_lag_0")
        for ar_lag in range(0, ar_lags+1):
            df_x_tilde[str(col)+"_lag_"+str(ar_lag)] = df[col].diff(1).shift(ar_lag)
    
    df_x_tilde = df_x_tilde.merge(z.shift(1), left_index=True, right_index=True)
    df_x_tilde = df_x_tilde.dropna()
    
    model1 = sm.OLS(endog=df_x_tilde[targets[0]], exog=df_x_tilde[['0_lag_1','1_lag_1','z']])
    res1 = model1.fit()
    #print(res1.summary())
    
    model2 = sm.OLS(endog=df_x_tilde[targets[1]], exog=df_x_tilde[['0_lag_1','1_lag_1','z']])
    res2 = model2.fit()
    #print(res2.summary())
    
    summary ={
        "phi_1_1": res1.params[0],
        "phi_1_2": res1.params[1],
        "gamma_0": res1.params[2],
        "phi_2_1": res2.params[0],
        "phi_2_2": res2.params[1],
        "gamma_1": res2.params[2],
        }
    
    return df_x_tilde, alpha,z, targets, summary
#%% Simulation


size = 2500
n_sim = 200
phi = np.array([[0.2, -0.1],
                [0, -0.25]])

alpha = np.array([[1],
                  [-1]])


gammas = [
    np.array([[0],
                    [0.3]]),
    np.array([[0],
                    [0.03]]),
    np.array([[-0.25],
                    [0.1]]),
    np.array([[-0],
                    [0]])
    
    ]


phi_1_1 = []
phi_1_2 = []
phi_2_1 = []
phi_2_2 = []
gamma_0 = []
gamma_1 = []

#%%

gamma=gammas[0]
X = simulate_series(size, phi, alpha, gamma)


# model = VECM(endog=X, k_ar_diff=1)
# result = model.fit()
# print(result.summary())

df_X = pd.DataFrame(X)
dfm, a, z, targets, summary= vecm(df_X)
phi_1_1.append(summary['phi_1_1'])
phi_1_2.append(summary['phi_1_2'])
phi_2_1.append(summary['phi_2_1'])
phi_2_2.append(summary['phi_2_2'])
gamma_0.append(summary['gamma_0'])
gamma_1.append(summary['gamma_1'])


sns.distplot(gamma_0)
plt.title(f'Distribution of gamma_0, real gamma_0={gamma[0][0]:,.2f}')
plt.show()

sns.distplot(gamma_1)
plt.title(f'Distribution of gamma_1, real gamma_1={gamma[1][0]:,.2f}')
plt.show()
plt.close()






#%%

for gamma in gammas:
    phi_1_1 = []
    phi_1_2 = []
    phi_2_1 = []
    phi_2_2 = []
    gamma_0 = []
    gamma_1 = []
    for i in tqdm(range(n_sim)):
        
        X = simulate_series(size, phi, alpha, gamma)
        
        
        # model = VECM(endog=X, k_ar_diff=1)
        # result = model.fit()
        # print(result.summary())
        
        
        df_X = pd.DataFrame(X)
        dfm, a, z, targets, summary= vecm(df_X)
        phi_1_1.append(summary['phi_1_1'])
        phi_1_2.append(summary['phi_1_2'])
        phi_2_1.append(summary['phi_2_1'])
        phi_2_2.append(summary['phi_2_2'])
        gamma_0.append(summary['gamma_0'])
        gamma_1.append(summary['gamma_1'])
    
        
    sns.distplot(gamma_0)
    plt.title(f'Distribution of gamma_0, real gamma_0={gamma[0][0]:,.2f}')
    plt.show()
    
    sns.distplot(gamma_1)
    plt.title(f'Distribution of gamma_1, real gamma_1={gamma[1][0]:,.2f}')
    plt.show()
    plt.close()



#%%

from statsmodels.tsa.vector_ar.var_model import VAR, VARProcess, VARResults
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order

varX = VAR( df_X.diff().dropna())
print('\t', varX.select_order(maxlags=10).summary(), '\n')
varX_pstar =  varX.select_order().bic
print('BIC lag order: ', varX_pstar,'\n\n')


model = VAR( df_X.diff().dropna())
res = model.fit(maxlags=2)
res.summary()


#%% Building VECM(1)

def var(df:pd.DataFrame, ar_lags:int=1):
    df_x_tilde = pd.DataFrame()
    targets = []
    model = sm.OLS(endog=df.iloc[:,0], exog=df.iloc[:,1])
    
    res1 = model.fit()
    
    
    for col in df:
        targets.append(str(col)+"_lag_0")
        for ar_lag in range(0, ar_lags+1):
            df_x_tilde[str(col)+"_lag_"+str(ar_lag)] = df[col].diff(1).shift(ar_lag)
    
    df_x_tilde = df_x_tilde.dropna()
    
    regressors = [col for col in df_x_tilde.columns if 'lag_0' not in col]
    
    model1 = sm.OLS(endog=df_x_tilde[targets[0]], exog=sm.tools.add_constant(df_x_tilde[regressors]))
    res1 = model1.fit()
    print(res1.summary())
    
    model2 = sm.OLS(endog=df_x_tilde[targets[1]], exog=sm.tools.add_constant(df_x_tilde[regressors]))
    res2 = model2.fit()
    print(res2.summary())
    
    summary ={
        "phi_1_1": res1.params[0],
        "phi_1_2": res1.params[1],
        "gamma_0": res1.params[2],
        "phi_2_1": res2.params[0],
        "phi_2_2": res2.params[1],
        "gamma_1": res2.params[2],
        }
    
    total_bic = res1.bic + res2.bic
    
    return summary, total_bic


def optimize_var_bic(df, max_lag):

   scores = {} 
   for lag in range(1,max_lag):
       _, tbic = var(df, ar_lags=lag)
       scores[lag] = tbic
   return scores


scores = optimize_var_bic(df_X, 10)
var(df_X, ar_lags=2)
