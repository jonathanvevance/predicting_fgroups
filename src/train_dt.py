
from data.dataset import reactionDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

BATCH_SIZE = 512
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EPOCHS = 100000
REPRESENTATION = 'chembl'


# ----- load datasets
train_dataset = reactionDataset('train', representation = REPRESENTATION)
val_dataset = reactionDataset('val', representation = REPRESENTATION)
test_dataset = reactionDataset('test', representation = REPRESENTATION)

X_train, Y_train = train_dataset.X, train_dataset.Y
X_val, Y_val = val_dataset.X, val_dataset.Y
X_test, Y_test = test_dataset.X, test_dataset.Y


print(X_train.shape)
exit()
# ----- load model
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train, Y_train)

# ----- 
y_pred = classifier.predict(X_test)


print(accuracy_score(Y_test, y_pred))




