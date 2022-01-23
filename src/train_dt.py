
from data.dataset import reactionDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

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

# ----- load model
classifier = DecisionTreeClassifier(class_weight = 'balanced', random_state=0)
classifier.fit(X_train, Y_train)

# ----- 
y_pred = classifier.predict(X_test)


print("AUROC:", roc_auc_score(Y_test, y_pred))
print("Detail:", precision_recall_fscore_support(Y_test, y_pred))
print("Accuracy:",accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))




