# Trained classifier

`dino_vitb8_point_classifier.pt` is the checkpoint used for every result in the paper. It was trained by step 2 and selected at the epoch with the highest validation accuracy (epoch 10, 0.674). It is released under CC BY 4.0.

The checkpoint is a PyTorch dictionary with the classifier weights (`classifier`), the eight class IDs in output order (`class_ids`), their pointcoral names (`class_names`), and the training mean and standard deviation of the 768 DINO features (`x_mean`, `x_std`). Load it with `densemap.classifier.load_classifier_checkpoint` and apply it with step 3.

`class_index_map.csv` maps output indices to class IDs and `training_history.csv` holds the loss and accuracy of every epoch.
