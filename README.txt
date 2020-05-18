I constructed a decision tree to classify butterflies and birds. I’m given a
2D dataset. I constructed a decision tree (max depth is three, excluding the leaf nodes) using information gain. And draw the decision
Boundaries. For each decision boundary, I’ll provide
1) depth/level of the decision node,
2) which axis is selected and the corresponding boundary value such as x=2.33, and
3) the information gain obtained at that point.

Results :
level = 1 direction = Root Y=4.007067137809185 infogain = 0.31597083060209075
level = 2 direction = Root-left X=3.0 infogain = 0.07458222599042463
level = 2 direction = Root-right Predict : bird
level = 3 direction = Root-left-left Predict : bird
level = 3 direction = Root-left-right X=7.1828621908127195 infogain = 0.11300426979851874
leaf direction = Root-left-right-left Predict : butterfly
leaf direction = Root-left-right-right Predict : bird
