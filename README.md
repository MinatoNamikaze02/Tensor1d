### 1-Dimensional Tensors in C

Attempting to replicate the tensors logic in C. Since this does not have autograd, this is basically a numpy 1d array haha. Good exercise though. Will extend to Nd tensors as well. 

```
t = Tensor1d.range_init(20)
print(t[3])
t[-1] = 100.0
print(t[-1]) 
print(t)
print(t[5:15:2])
print(t[5:15:2][2:7])
t = t + 10.0 
print(t)
t2 = Tensor1d.range_init(20)
print(t2)
t3 = t + t2
t4 = (t + (Tensor1d.range_init(20) + 250))[2:10]
print(t4)
```

```
3.0
100.0
Tensor (size: 20): [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 100.00]
Tensor (size: 5): [5.00, 7.00, 9.00, 11.00, 13.00]
Tensor (size: 3): [9.00, 11.00, 13.00]
Tensor (size: 20): [10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00, 21.00, 22.00, 23.00, 24.00, 25.00, 26.00, 27.00, 28.00, 110.00]
Tensor (size: 20): [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00]
Tensor (size: 8): [264.00, 266.00, 268.00, 270.00, 272.00, 274.00, 276.00, 278.00]
```

Cheers.