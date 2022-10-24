import numpy as np
from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

############################
# community dataset
##############################

# Read the data
attrib = read_csv('attributes.csv', delim_whitespace = True)
data = read_csv('communities.data', names = attrib['attributes'])

# remove non-predictive features
data = data.drop(columns=['state','county', 'community','communityname', 'fold'], axis=1)

# remove missing values 
data = data.replace('?', np.nan)
data = data.dropna()


# Standardize features by removing the mean and scaling to unit variance
X = data.iloc[:, 0:100].values
y = data.iloc[:, 100].values
Y =y.astype(float)
sc = StandardScaler()
Xst = sc.fit_transform(X)

# pca for dim reduction
c = 30
pca = PCA(n_components = c)
Xpca = pca.fit_transform(Xst)

# turn data in numpy array 
X = np.asarray(X).astype(float)
Y = np.asarray(Y).astype(float)
Xst = np.asarray(Xst).astype(float)
Xpca = np.asarray(Xpca).astype(float)
# np.savez('communities.npz', X, y, Xst, Xpca)

# append y to the final column of X, Xst, Xpca
D = np.column_stack((X, Y)).astype(float)
Dst = np.column_stack((Xst, Y)).astype(float)
Dpca = np.column_stack((Xpca, Y)).astype(float)

# save the above into  csv files
np.savetxt('communities.csv', D, delimiter=',')
np.savetxt('communities_st.csv', Dst, delimiter=',')
np.savetxt('communities_pca.csv', Dpca, delimiter=',')




############################
#  superconductivity dataset
##############################
data = read_csv('superconductive.csv')
df = data.loc[np.random.choice(data.index, 100, replace=False)]
# remove missing values 
df = df.replace('?', np.nan)
df = df.dropna()


# Standardize features by removing the mean and scaling to unit variance
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
Y = y.astype(float)
sc = StandardScaler()
Xst = sc.fit_transform(X)

# turn data in numpy array 
X = np.asarray(X).astype(float)
Y = np.asarray(Y).astype(float)
Xst = np.asarray(Xst).astype(float)

# append y to the final column of X, Xst, Xpca
D = np.column_stack((X, Y)).astype(float)
Dst = np.column_stack((Xst, Y)).astype(float)

# save the above into  csv files
np.savetxt('super.csv', D, delimiter=',')
np.savetxt('super_st.csv', Dst, delimiter=',')