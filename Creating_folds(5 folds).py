#Creating folds (5 folds)
from sklearn.model_selection import StratifiedKFold
folds = pd.read_csv('./category.csv')
"""
-----------------------
| index | name | CATE |
-----------------------
| 0     | 1.jpg | 0   |
-----------------------
| 1     | 2.jpg | 1   |
-----------------------"""

#folds = train_df
kf = StratifiedKFold(n_splits = 5, shuffle = True) # n_splits = 5 (spliting the data into 5 folds)

train_labels = folds['CATE'].values  #getting all the values of 'CATE' column
for fold, (train_index, valid_index) in enumerate(kf.split(folds.values, train_labels)):
    folds.loc[valid_index, 'fold'] = int(fold)
folds['fold'] = folds['fold'].astype(int)

folds.head()
folds.to_csv('./data_folds.csv', index = False)