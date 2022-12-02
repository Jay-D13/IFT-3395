import numpy as np
# from matplotlib import pyplot as plt
banknote = np.genfromtxt('data_banknote_authentication.txt',delimiter=',')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

class Q1:

    def feature_means(self, banknote):
        return np.mean(banknote[:,:4], axis=0)
            
    def covariance_matrix(self, banknote):
        return np.cov(banknote[:,:4], rowvar=False)

    def feature_means_class_1(self, banknote):
        label_1=banknote[banknote[:,4] == 1]
        return np.mean(label_1[:,:4], axis=0)

    def covariance_matrix_class_1(self, banknote):
        labels_1=banknote[banknote[:,4] == 1]
        return np.cov(labels_1[:,:4], rowvar=False)
    
def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
        # Initialization of the count matrix and the predicted classes array
        counts = np.zeros(self.n_classes)
        classes_pred = np.zeros(test_data.shape[0])
        
        # For each test datapoint
        for (i, point) in enumerate(test_data):

            # i is the row index
            # point is the i'th row

            # Find the distances to each training set point
            d=minkowski_mat(point, self.train_inputs)
            
            # Go through the training set to find the neighbors of the current point (ex)
            rayon=self.h
            indices_nn=np.array([k for k in range(len(d)) if d[k]<=rayon])
            
            if len(indices_nn)==0:
                classes_pred[i]=draw_rand_label(point,self.label_list)
            else:
                # Calculate the number of neighbors belonging to each class and write them in counts[i,:]
                for j in range(self.n_classes):
                    counts[j] = sum([1 for k in indices_nn if int(self.train_labels[k])==int(self.label_list[j])])
                
                # From the counts matrix, define classes_pred[i] (don't forget that classes are labeled from 1 to n)
                classes_pred[i]=np.argmax(counts)
                
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma
        
    def kernel(self, xi, x):
        # print("xi: ",xi)
        # print("x: ",x)
        d=x.shape[0]
        dist=(np.abs(xi - x)**2).sum(axis=0)**(1.0/2)
        a = 1/(((2*np.pi)**(d/2))*(self.sigma**d))
        b = np.exp((-1/2)*(dist/(self.sigma**2)))
        return a*b

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
        counts = np.zeros((test_data.shape[0], self.n_classes))
        classes_pred = np.zeros(test_data.shape[0])
        
        # For each test datapoint
        for (i, point) in enumerate(test_data):
            for j in range(self.n_classes):
                counts[i,j] = sum([self.kernel(self.train_inputs[k],point) for k in range(self.train_inputs.shape[0]) if int(self.train_labels[k])==int(self.label_list[j])])
                    
            classes_pred[i]=np.argmax(counts[i,:])
                
        return classes_pred


def split_dataset(banknote):
    entrainement,validation,test=[],[],[]

    for i in range(banknote.shape[0]):
        rest=i%5
        if rest in [0,1,2]:
            entrainement.append(banknote[i])
        if rest==3:
            validation.append(banknote[i]) 
        if rest==4:
            test.append(banknote[i])
    
    return (np.array(entrainement), np.array(validation), np.array(test))

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val


    def hard_parzen(self, h):
        parzen = HardParzen(h)
        parzen.train(self.x_train, self.y_train)
        prediction_x = parzen.compute_predictions(self.x_val)
        
        missclassified = [1 for (i, classe) in enumerate(prediction_x) if classe != self.y_val[i]]
        
        return (sum(missclassified) / len(self.y_val))
        # confmat = conf_matrix(prediction_x, self.y_val)
        # return 1.0 - (float(np.sum(np.diag(confmat))) / float(np.sum(confmat)))

    def soft_parzen(self, sigma):
        parzen = SoftRBFParzen(sigma)
        parzen.train(self.x_train, self.y_train)
        prediction_x = parzen.compute_predictions(self.x_val)
        
        missclassified = [1 for (i, classe) in enumerate(prediction_x) if classe != self.y_val[i]]
        
        return (sum(missclassified) / len(self.y_val))


def get_test_errors(banknote):
    train, validation, test = split_dataset(banknote)
    test_error = ErrorRate(train[:,0:-1], train[:,-1], validation[:,0:-1], validation[:,-1])

    para = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    
    test_errors_hp,test_errors_sp=[],[]
    for p in para:
        test_errors_hp.append(test_error.hard_parzen(p))
        test_errors_sp.append(test_error.soft_parzen(p))
    
    h_star = para[np.argmin(test_errors_hp)]
    sigma_star = para[np.argmin(test_errors_sp)]
    
    test_error = ErrorRate(train[:,0:-1], train[:,-1], test[:,0:-1], test[:,-1])
    
    return np.array([test_error.hard_parzen(h_star), test_error.soft_parzen(sigma_star)])


def random_projections(X, A):
    pass

# if __name__=="__main__":
#     banknotes = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
    # get_test_errors(banknotes)
    
    # train, validation, test = split_dataset(banknote)
    # test_error = ErrorRate(train[:,0:-1], train[:,-1], validation[:,0:-1], validation[:,-1])

    # para = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    
    # test_errors_hp,test_errors_sp=[],[]
    # for p in para:
    #     test_errors_hp.append(test_error.hard_parzen(p))
    #     test_errors_sp.append(test_error.soft_parzen(p))
    
    # plt.plot(para,test_errors_hp,label="Error rates hard parzen")
    # plt.plot(para,test_errors_sp,label="Error rates soft parzen")
    # plt.title("Error rates graph")
    # plt.xlabel("Parametres")
    # plt.ylabel("Error de validation")
    # plt.legend()
    # plt.savefig("plot.png")
    
    # Both start high then low then high cause bigger radius
    # soft parzen better cause can reach 0
    
    #plus sigma petit, on caclule toujours le kernel de tous les points
    #alors que hard parzen, plus le rayon est petit, alors il y a moins de points voisins sur lequel boucler
    
    