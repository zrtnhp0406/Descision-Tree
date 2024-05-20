from __future__ import print_function
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        """
        Khởi tạo một node trên cây quyết định
        ids                          Vị trí của node trong tập dữ liệu
        entropy                      Giá trị entropy ứng với nút đó
        depth                        Khoảng cách từ nút gốc đến nút này
        children                     Tập các nút con của nút này
        split_attribute              Thuộc tính đã chọn để tách nút, nút này không phải lá
        order                        Thứ tự giá trị của split attribute trong các nút con
        label                        Nhãn tương ứng với nút này, nếu nút này là lá
        """
        self.ids = ids         
        self.entropy = entropy 
        self.depth = depth   
        self.split_attribute = None
        self.children = children
        self.order = None   
        self.label = None      


    def set_properties(self, split_attribute, order):
        """
        Gán thuộc tính được chọn để tách nút cũng như là thứ tự giá trị của nó
        """
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        """
        Set nhãn cho nút
        """
        self.label = label

class DecisionTreeCHAID(object):
    def __init__(self, max_depth= 10, min_samples_split = 2, min_gain = 1e-4):
        """
        Khởi tạo cây với các ràng buộc cho điều kiện dừng gồm max_depth, min_samples_split,min_gain
        ngoài ra còn có các thông tin sau:
        
        Root        Gốc của cây
        Ntrain      Số lượng dữ liệu huấn luyện
        """

        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.Ntrain = 0

    def fit(self, data, target):
        """
        Hàm chính cho việc huấn luyện mô hình, xây dựng cây  dựa trên tập huấn luyện là data và nhãn tương ứng là target      
        """
        self.Ntrain = data.count()[0] #Đếm số lượng phần tử trong tập data
        self.data = data #Lưu trữ lại tập data đầu vào
        self.attributes = list(data)
        self.target = target # Lưu trữ tập data kết quả
        self.labels = target.unique() #lọc ra các nhãn  của tập

        ids = range(self.Ntrain)
        #Khởi tạo một node gốc
        self.root = TreeNode(ids = ids, entropy = self._chi_square(ids), depth = 0)
        queue = [self.root]
        #Tiến hành xây dựng cây
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth :
                node.children = self._split(node)
                if not node.children: #đây là node lá
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)

    def _chi_square(self, ids):
        """
        Hàm này tính chi square cho nút có vị trí ids
        """
        if len(ids) == 0:
            return 0
        ids = [i + 1 for i in ids]  # panda series index starts from 1
        observed_freq = np.array(self.target[ids].value_counts())
        expected_freq = np.array(self.target.value_counts()) / len(self.target)
        chi_square=0
        for i in range(0,len(observed_freq)):
            chi_square += math.sqrt((observed_freq[i] - expected_freq[i]) ** 2 / expected_freq[i])
        return chi_square

    def _set_label(self, node):
        """
         Tìm nhãn của node nếu nó là node lá
         Bằng cách chọn ra nhãn chiếm phần lớn trong dữ liệu kết quả
        """
        target_ids = [i + 1 for i in node.ids]
        node.set_label(self.target[target_ids].mode()[0]) # Nhãn có tần số xuất hiện lớn nhất
    def _split(self, node):
        """
        Hàm tìm thuộc tính phù hợp nhất để phát triển(tách ra) tại node này.
        Trong này, gồm có tính gain information cũng như tìm thuộc tính tốt nhất để phát triển tại node này
        """
        ids = node.ids
        best_chi_square = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]

        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1:
                continue 

        splits = []
        for val in values:
            sub_ids = sub_data.index[sub_data[att] == val].tolist()
            splits.append([sub_id - 1 for sub_id in sub_ids])

        if min(map(len, splits)) < self.min_samples_split:
            continue

        # Hàm tính chi square cho từng thuộc tính
        chi_square = 0
        for split in splits:
            chi_square += self._chi_square(split)
            if chi_square > best_chi_square:
                best_chi_square = chi_square
                best_splits = splits
                best_attribute = att
                order = values

        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split, entropy=self._chi_square(split), depth=node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        """
        Hàm dự đoán tập dữ liệu mới 
        """
        npoints = new_data.count()[0]
        labels = [None]*npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]
            # Bắt đầu từ node gốc và đi xuống dọc theo các con đường đúng với điều kiện đến khi tới node lá
            #trả về nhãn của node
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label

        return labels

if __name__ == "__main__":
    """
    Hàm main chính, nơi đọc, xử lý dữ liệu thô, call hàm xây dựng tree, thực hiện dự đoán dữ liệu và chấm điểm mô hình.
    """
    df = pd.read_csv('data_car.csv', index_col = 0, parse_dates = True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.20, random_state=1412)
    
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_train.index = range(1, len(X_train) + 1)
    y_train.index = range(1, len(y_train) + 1)

    
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    X_test.index = range(1, len(X_test) + 1)
    y_test.index = range(1, len(y_test) + 1)


    tree = DecisionTreeCHAID(max_depth = 3, min_samples_split = 2)
    tree.fit(X_train, y_train)
    print(accuracy_score(y_test, tree.predict(X_test)))
    print(f1_score(y_test, tree.predict(X_test), average='weighted'))