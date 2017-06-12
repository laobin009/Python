from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image
from scipy.misc import imread
import matplotlib.pyplot as plt
import pydotplus
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


dot_data = tree.export_graphviz(clf, out_file=None,
                                 feature_names=iris.feature_names,
                                 class_names=iris.target_names,
                                 filled=True, rounded=True,
                                 special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
#create output
graph.write_pdf("output\iris.pdf")

graph.write_jpg("output\iris.jpg")

#show image
img = imread('output\iris.jpg')
plt.imshow(img)
plt.show()
