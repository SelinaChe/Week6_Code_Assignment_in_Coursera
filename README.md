# Week6-Code-Assignment 
## for Machine_Learning_Foundations:A_Case_Study_Approach
### 1.What’s the least common category in the training data?
<code>groupby_label = image_train.groupby(key_columns='label', operations={'total_count':graphlab.aggregate.COUNT()})
<code>groupby_label.sort('total_count', ascending=True)</code>

### 2.Of the images below, which is the nearest ‘cat’ labeled image in the training data to the the first image in the test data (image_test[0:1])?
<code>image_test[0:1]['image'].show()</code>
<code>image_train_cat = image_train.filter_by(['cat'],'label')</code>
<code>knn_model_cat = graphlab.nearest_neighbors.create(image_train_cat,features=['deep_features'],
                                             label='id')</code>
<code>cat_neighbors = get_images_from_ids(knn_model_cat.query(image_test[0:1],k=1))</code>
<code>cat_neighbors['image'].show()</code> 

### 3.Of the images below, which is the nearest ‘dog’ labeled image in the training data to the the first image in the test data (image_test[0:1])?
<code>image_test[0:1]['image'].show()</code>
<code>image_train_dog = image_train.filter_by(['dog'],'label')</code>
<code>knn_model_dog = graphlab.nearest_neighbors.create(image_train_dog,features=['deep_features'],
                                             label='id')</code>
<code>dog_neighbors = get_images_from_ids(knn_model_dog.query(image_test[0:1],k=1))</code>
<code>dog_neighbors['image'].show()</code>

### 4.For the first image in the test data, in what range is the mean distance between this image and its 5 nearest neighbors that were labeled ‘cat’ in the training data?
<code>knn_model_cat.query(image_test[0:1],k=5)['distance'].mean()</code>

### 5.For the first image in the test data, in what range is the mean distance between this image and its 5 nearest neighbors that were labeled ‘dog’ in the training data?
<code>knn_model_dog.query(image_test[0:1],k=5)['distance'].mean()</code>

### 6.On average, is the first image in the test data closer to its 5 nearest neighbors in the ‘cat’ data or in the ‘dog’ data?
#### cat

### 7.In what range is the accuracy of the 1-nearest neighbor classifier at classifying ‘dog’ images from the test set?
<code>knn_classifier_model_dog = graphlab.nearest_neighbor_classifier.create(image_train,
                                                                       features=['deep_features'],
                                                                       target='label'
                                                                       )</code>
<code>image_test_dog = image_test.filter_by(['dog'],'label')</code>
<code>knn_classifier_model_dog.evaluate(image_test_dog,max_neighbors=1)</code>
