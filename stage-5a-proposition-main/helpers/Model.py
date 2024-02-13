from typing import Literal

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay, f1_score, recall_score, precision_score, accuracy_score

from umap import UMAP
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment as linear_assignment

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

import numpy as np
import pandas as pd

from kneed import KneeLocator

import matplotlib.pyplot as plt
import seaborn as sns

from . import Corpus, CDBSCAN
from .Terms import CoreConceptValue


Reducers    = Literal["none", "pca", "umap"]
Clusterers  = Literal["dbscan", "c-dbscan"]
Classifiers = Literal["svm", "ann"]

class Model():
    def __init__(
        self,
        corpus      : Corpus,
        min_points  : int,
        dimensions  : int,
        reducer     : Reducers      = "umap",
        classifier  : Classifiers   = None,
        clusterer   : Clusterers    = None
    ) -> None:
        """
        Initializes the Model object with given parameters.
        
        Parameters:
            - corpus: Corpus object representing the data corpus.
            - min_points: Integer representing the minimum number of points for clustering.
            - dimensions: Integer representing the number of dimensions for dimensionality reduction.
            - reducer: String representing the type of dimensionality reduction method to be used. Default is 'umap'.
            - classifier: String representing the type of classifier to be used. Default is None.
            - clusterer: String representing the type of clusterer to be used. Default is None.

        Returns:
            None.
        """
        self.corpus         : Corpus        = corpus
        self.core_concept_indices =  np.array([self.corpus.vocabulary.get_index(cc) for cc in self.corpus.vocabulary.core_concepts if cc != "other"])

        self.min_points     : int           = min_points
        self.epsilon        : float         = None
        self.dimensions     : int           = dimensions
        self.embedding      : list          = None
        self.__do_clustering   : bool = (clusterer != None)

        self.reducer_type = reducer
        self.classifier_type = classifier
        self.clusterer_type = clusterer

        self.reducer    = None
        self.classifier = None
        self.clusterer  = None

        self.y_pred = None

        self.embeddings = []
        self.predictions = []

        self.silhouette_scores = []
        self.davies_bouldin_scores = []
        self.calinski_harabasz_scores = []
        
        CoreConceptValue.reset()

        self.__check_for_init_errors(reducer, classifier, clusterer)


    def iterate(self, X, y, nb_iter = 5, verbose = 0):
        """
        Iteratively fits and transforms the data.
        
        Parameters:
            - X: Input data.
            - y: Target values.
            - nb_iter: Number of iterations to run. Default is 5.
            - verbose: Verbosity level. Default is 0.

        Returns:
            - y_pred: Predicted values after the last iteration.
        """
        self.y_pred = np.copy(y)
        self.__set_reducer()
        for i in range(nb_iter):
            is_last_iter = i == (nb_iter - 1)
            self.y_pred = self.fit_transform(X, self.y_pred, is_last_iter)
            if self.clusterer_type == "c-dbscan":
                self.corpus.update_vocabulary(self.y_pred)
            if verbose == 1:
                if self.clusterer_type != None :
                    self.__store_clustering_result(X, self.y_pred)
                self.__show_embedding()
                self.__show_prediction(self.y_pred)

        if verbose == 1 :
            if self.clusterer_type != None :
                self.__show_clustering_result(nb_iter)

        self.__show_classification()

        return self.y_pred


    def fit_transform(self, X, y, last_iter = False):
        """
        Fits the data and returns the transformed data.
        
        Parameters:
            - X: Input data.
            - y: Target values.
            - last_iter: Boolean indicating if it's the last iteration.

        Returns:
            Transformed data after fitting.
        """
        self.fit(X, y)
        return self.transform(X, y, last_iter)


    def fit(self, X, y):
        """
        Fits the data based on the specified dimensionality reduction, clustering, and classification methods.
        
        Parameters:
            - X: Input data.
            - y: Target values.

        Returns:
            None.
        """
        self.__reduce_dimensions(X, self.y_pred)
        self.__get_epsilon_value(self.embedding)
        self.__set_classifier()
        self.__set_clusterer()


    def transform(self, X, y, last_iter = False):
        """
        Transforms the data based on the specified clustering or classification method.
        
        Parameters:
            - X: Input data.
            - y: Target values.
            - last_iter: Boolean indicating if it's the last iteration.

        Returns:
            Transformed data.
        """
        # self.__filter_relevant_terms()
        if self.__do_clustering :
            return self.__cluster(self.embedding, y, last_iter)
        
        return self.__classify(self.embedding, y)

    def __get_epsilon_value(self, X):
        """
        Calculates the epsilon value for DBSCAN clustering.
        
        Parameters:
            - X: Input data.

        Returns:
            self: The instance of the Model class.
        """
        n_neighbors = 2 * self.dimensions

        nbrs = NearestNeighbors(n_neighbors= n_neighbors + 1).fit(X)
        
        distances, indices = nbrs.kneighbors(X)
        
        sort_neigh_dist = np.sort(distances, axis = 0)
        # print(sort_neigh_dist.shape)
        k_dist = sort_neigh_dist[:, n_neighbors]
        # print(k_dist.shape)
        kneedle = KneeLocator(x = range(1, len(distances)+1), y = k_dist, S = 1.0, 
                        curve = "concave", direction = "increasing", online=True)
        
        self.epsilon = kneedle.knee_y
        # print(kneedle.knee)
        # print(kneedle.elbow)
        # kneedle.plot_knee()
        # plt.show()
        return self


    def __reduce_dimensions(self, X, y):
        """
        Reduces the dimensionality of the data based on the specified method.
        
        Parameters:
            - X: Input data.
            - y: Target values.

        Returns:
            None.
        """
        if self.reducer_type == "none":
            self.embedding = X
            return

        self.embedding = self.reducer.fit_transform(X, y)


    def __filter_relevant_terms(self):
        """
        Filters relevant terms from the corpus.

        Returns:
            None.
        """
        # Assign terms to class "other"
        outliers_scores = LocalOutlierFactor(contamination=0.05).fit_predict(self.embedding)
        outlying_digits = np.array(self.corpus.vocabulary.terms)[outliers_scores == -1]
        # print(outlying_digits.shape)
        # print(outlying_digits)


    def __classify(self, X, y):
        """
        Classifies the data based on the specified classifier.
        
        Parameters:
            - X: Input data.
            - y: Target values.

        Returns:
            Predicted labels for the input data.
        """
        X_train = X[y >= 0]
        y_train = y[y >= 0]
        self.classifier.fit(X_train, y_train)
        return self.classifier.predict(X)


    def __cluster(self, X, y, last_iter = False):
        """
        Clusters the data based on the specified clusterer.
        
        Parameters:
            - X: Input data.
            - y: Target values.
            - last_iter: Boolean indicating if it's the last iteration.

        Returns:
            Clustered labels for the input data.
        """
        # Assign terms to every other classes
        if type(self.clusterer) == CDBSCAN:
            return self.clusterer.fit_predict(X, self.corpus.vocabulary, self.core_concept_indices, last_iter)
        
        return self.clusterer.fit_predict(X)


    def __set_reducer(self):
        """
        Sets the dimensionality reduction method.

        Returns:
            None.
        """
        if self.reducer_type == "pca":
            self.reducer = PCA(n_components = self.dimensions)
            return
        
        if self.reducer_type == "umap":
            self.reducer = UMAP(n_components = self.dimensions, n_neighbors=15, min_dist=0.0, random_state=42)
            return


    def __set_classifier(self):
        """
        Sets the classifier.

        Returns:
            None.
        """
        if self.classifier_type == "svm":
            self.classifier = BaggingClassifier(base_estimator= SVC(), n_estimators=1, random_state=42)
            return
        
        if self.classifier_type == "ann":
            self.classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=42)
            return
        
    
    def __set_clusterer(self):
        """
        Sets the clusterer.

        Returns:
            None.
        """
        if self.clusterer_type == "dbscan":
            self.clusterer = DBSCAN(eps=self.epsilon * 0.5, min_samples=self.min_points)
            return
        
        if self.clusterer_type == "c-dbscan":
            self.clusterer = CDBSCAN(epsilon=self.epsilon, min_points=self.min_points)
            return

    
    def __check_for_init_errors(self, reducer : Reducers, classifier : Classifiers, clusterer : Clusterers):
        """
        Checks for initialization errors.
        
        Parameters:
            - reducer: String representing the type of dimensionality reduction method to be used.
            - classifier: String representing the type of classifier to be used.
            - clusterer: String representing the type of clusterer to be used.

        Returns:
            None.
        """
        if reducer == None :
            raise Exception("No dimension reduction model has been passed.")
        
        if clusterer == None and classifier == None :
            raise Exception("No prediction model has been passed.")

        if clusterer != None and classifier != None :
            raise Warning("A clusterer and a classifier model have been passed at the same time. Only the clusterer will be taken into account.")


    def __store_clustering_result(self, X, predicted_labels):
        """
        Stores the results of clustering.
        
        Parameters:
            - X: Input data.
            - predicted_labels: Labels predicted by the clustering algorithm.

        Returns:
            None.
        """

        # data = np.array([point.position for point in self.points])
        data = X
        silhouette = silhouette_score(data, predicted_labels)
        davies_bouldin = davies_bouldin_score(data, predicted_labels)
        calinski_harabasz = calinski_harabasz_score(data, predicted_labels)

        # Append metrics to the lists
        self.silhouette_scores.append(silhouette)
        self.davies_bouldin_scores.append(davies_bouldin)
        self.calinski_harabasz_scores.append(calinski_harabasz)

    def __show_clustering_result(self, num_iterations):
        """
        Displays the results of clustering.
        
        Parameters:
            - num_iterations: The number of iterations performed by the clustering algorithm.

        Returns:
            None.
        """
        # Create plots
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.lineplot(x=range(1, num_iterations+1), y=self.silhouette_scores)
        plt.title("Silhouette Score over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Silhouette Score")

        plt.subplot(1, 3, 2)
        sns.lineplot(x=range(1, num_iterations+1), y=self.davies_bouldin_scores)
        plt.title("Davies-Bouldin Score over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Davies-Bouldin Score")

        plt.subplot(1, 3, 3)
        sns.lineplot(x=range(1, num_iterations+1), y=self.calinski_harabasz_scores)
        plt.title("Calinski-Harabasz Score over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Calinski-Harabasz Score")

        plt.tight_layout()
        plt.show()

        print("Silhouette : " + str(self.silhouette_scores[-1]))
        print("Davies Bouldin : " + str(self.davies_bouldin_scores[-1]))
        print("Calinski Harabasz : " + str(self.calinski_harabasz_scores[-1]))


    def __show_classification(self):
        """
        Displays the results of classification.

        Returns:
            None.
        """
        y_pred = {}
        y_true = {}

        term_column = self.corpus.colums[0]
        label_column = self.corpus.colums[1]

        for term in set(self.corpus.ontology[term_column].to_list()).intersection(set(self.corpus.vocabulary.terms)):
            y_pred[term] = self.y_pred[self.corpus.vocabulary.get_index(term)]
            y_true[term] = self.corpus.vocabulary.get_term_from_string(self.corpus.ontology[self.corpus.ontology[term_column] == term][label_column].values[0].lower()).get_core_concept()
        
        y_pred_list = []
        y_true_list = []
        for term in y_pred:
            if y_pred[term] == -1:
                continue
            y_pred_list.append(y_pred[term])
            y_true_list.append(y_true[term])

        # normalize pred = precision, normalize true = recall
        normalize = "true"
        conf_matrix = confusion_matrix(y_true_list, y_pred_list, normalize=normalize)

        if self.clusterer_type == "dbscan":
            conf_matrix = self.__reorder_confusion_matrix(conf_matrix)

        axis_labels = self.corpus.vocabulary.core_concepts
        sns.heatmap(conf_matrix, xticklabels=axis_labels, yticklabels=axis_labels).set(title='Confusion matrix ' + "(recall)" if normalize == "true" else "(precision)")
        plt.show()

        self.accuracy = accuracy_score(y_true_list, y_pred_list)
        average = "weighted"
        # average = "macro"
        self.precision = precision_score(y_true_list, y_pred_list, average=average)
        self.recall = recall_score(y_true_list, y_pred_list, average=average)
        self.f1 = f1_score(y_true_list, y_pred_list, average=average)

        print("Accuracy : " + str(self.accuracy))
        print("Precision : " + str(self.precision))
        print("Recall : " + str(self.recall))
        print("f1 : " + str(self.f1))


        y_pred_terms = [self.corpus.vocabulary.core_concepts[cc_index] for cc_index in y_pred_list]
        y_true_terms = [self.corpus.vocabulary.core_concepts[cc_index] for cc_index in y_true_list]
        concat = np.array([y_pred_terms, y_true_terms]).T

        df = pd.DataFrame(concat, columns=["Prediction", "Ground truth"])
        print(df.head())

        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x="Ground truth", hue="Prediction", multiple="stack", shrink=0.75)
        plt.show()

    def __reorder_confusion_matrix(self, matrix):
        """
        Reorders the confusion matrix.
        
        Parameters:
            - matrix: The confusion matrix to be reordered.

        Returns:
            The reordered confusion matrix.
        """
        def _make_cost_m(cm):
            s = np.max(cm)
            return (- cm + s)

        indexes = linear_assignment(_make_cost_m(matrix))
        js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
        return matrix[:, js]


    def __show_embedding(self):
        """
        Displays the embeddings.

        Returns:
            None.
        """
        plt.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            s=0.1,
            color=(0.5, 0.5, 0.5),
        )
        plt.scatter(
            self.embedding[self.core_concept_indices, 0],
            self.embedding[self.core_concept_indices, 1],
            s=5,
        )
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the extracted corpus (fine-tuned)', fontsize=12)
        plt.xlim([-3, 9])
        plt.ylim([-7, 3])
        plt.show()

    
    def __show_prediction(self, y_pred):
        """
        Displays the predictions.
        
        Parameters:
            - y_pred: Predicted labels.

        Returns:
            None.
        """
        clustered = (y_pred >= 0)
        plt.scatter(self.embedding[~clustered, 0],
                    self.embedding[~clustered, 1],
                    color=(0.5, 0.5, 0.5),
                    s=1,
                    alpha=0.5)
        plt.scatter(self.embedding[clustered, 0],
                    self.embedding[clustered, 1],
                    c=y_pred[clustered],
                    s=1,
                    cmap='Spectral')
        plt.scatter(
            self.embedding[self.core_concept_indices, 0],
            self.embedding[self.core_concept_indices, 1],
            color=(0, 0, 0),
            s=5,
        )
        plt.xlim([-3, 9])
        plt.ylim([-7, 3])
        plt.show()
