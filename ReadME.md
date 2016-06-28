```
rm log/tmp.csv
for i in {1..10}; do
    for subject in {1..9}; do
        python train.py $subject 1 --method linear
        for session in {2..4}; do
            python accuracy.py $subject $session --method linear --tmp linear
        done
    done
done
```

```
for i in {1..9}; do
    python train.py 1 1
    for session in {2..4}; do
        python accuracy.py 1 $session --tmp debug$i
    done
done
```

## Linear SVM

```
rm log/linearSVM.csv
for i in {1..10}; do
    for subject in {1..9}; do
        python train.py $subject 1 --method linear
        for session in {2..4}; do
            python accuracy.py $subject $session --method linear --tmp linear-euclidean-1
        done
    done
done
```




##SWLDA
```
rm log/swlda.csv
for subject in {1..9}; do
    python train.py $subject 1 --method swlda --average 1
    for session in {2..4}; do
        python accuracy.py $subject $session --method swlda --tmp swlda --average 1
    done
done
```
