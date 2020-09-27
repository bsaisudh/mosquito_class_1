# Resullts

## Run 1

Comments:
- Vgg 16
- resized to 340*480
- center crop 244

Results:
```text
==================================================
Testing the model : ./Weights/vgg16_aug_pt.pth
----- Confusion Matrx (%) -----
[[100.     0.     0.     0.     0.     0.  ]
 [  0.   100.     0.     0.     0.     0.  ]
 [  0.     0.   100.     0.     0.     0.  ]
 [  0.     0.     6.67  86.67   6.67   0.  ]
 [  0.     0.     0.     0.    93.33   6.67]
 [  0.     0.     0.     0.    20.    80.  ]]
-------------------------------
Correct 84 of 90.
Test Accuracy : 93.33%
Elapsed time : 5.64 [sec]
==================================================
```

## Run 2

Comments:
- Vgg 16
- Applied augmentation
- resized to 244

```text
==================================================
Testing the model : ./Weights/vgg16_aug_pt_run2.pth
----- Confusion Matrx (%) -----
[[100.     0.     0.     0.     0.     0.  ]
 [  0.   100.     0.     0.     0.     0.  ]
 [  0.     0.    86.67  13.33   0.     0.  ]
 [  6.67   0.     0.    86.67   6.67   0.  ]
 [  0.     0.     0.     0.    60.    40.  ]
 [  0.     0.     0.     0.     6.67  93.33]]
-------------------------------
Correct 79 of 90.
Test Accuracy : 87.78%
Elapsed time : 2.46 [sec]
==================================================
```
