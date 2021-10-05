import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt

data = pd.read_csv("./creditcard.csv")
# Primera lección aprendida, windows puede no escribir las extensiones y el nombre es diferentes

# Contamos cuantas operaciones son verdaderas >>> 0, y cuantas son falsas >>>> 1
count_classes = pd.value_counts(data['Class'], sort = False)
# Método válido para clasificar cuando hay pocas categorias 
# count_classes es una serie

count_classes.plot (kind='bar')
plt.title ("Operaciones fraudulentas sobre no fraudulentas")
plt.xlabel ("Fraudulentas")
plt.ylabel ("Frecuencia")

# Pueden aplicarse formulas sobre las columnas de un Data Frame
data['logAmount'] = np.log(data['Amount']+1)
# Para después dibujar un histograma
data['logAmount'].sort_values().plot.hist()

from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape (-1,1))
data = data.drop (['Time', 'Amount','logAmount'], axis = 1);

# Separamos los datos en dos arrays, uno con las variables X y otro con las variables y
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
len(y[y.Class ==1]);

# Resampling --> Para casos en los que el sistema está desbalanceado como éste

#  - Undersampling --> Eliminar casos del tipo mayoritario
#  - Oversampling -->  Replicar de forma sintética los casos minoritarios
#  - SMOTE --> Una técnica combinación de las dos anteriores 

# Contamos el número de casos de fraude que existen ¡¡
number_records_fraud = len (data[data.Class==1])
# Y extraemmos los índices donde están los casos de fraude y los de no fraude
fraud_indices = np.array (data[data.Class==1].index)
normal_indices = np.array (data[data.Class==0].index)

# Obtenemos de forma aleatoria un número de indices de no fraude, igual al de fraude
random_normal_indices = np.random.choice (normal_indices, number_records_fraud, replace = False )
# Unimos en un solo array los indices de fraude con los de no fraude escogidos aleatoriamente
under_sample_indices = np.concatenate ([fraud_indices, random_normal_indices])

# Ahora escogemos los valores de dichos indices
under_sample_data = data.iloc[under_sample_indices,:]
# Separamos la X de la y de nuevo
X_undersample = under_sample_data.iloc [:, under_sample_data.columns != 'Class'];
y_undersample = under_sample_data.iloc [:, under_sample_data.columns == 'Class'];

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = 0.3, random_state = 0)
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split (X_undersample,y_undersample, test_size = 0.3, random_state = 0)

from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier

MLPC = MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000)
MLPC.fit(X_train_under, y_train_under)
y_pred = MLPC.predict(X_test)
# Obtenemos valores de recall
recall_acc = recall_score (y_test,y_pred)
X_prova = np.array([-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919,0])
X_prova = np.reshape(X_prova, (1, -1))
print(MLPC.predict(X_prova))