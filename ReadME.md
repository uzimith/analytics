## Linear SVM for cosine undersampling

```
time for far in {0..400}; do
    for subject in {1..9}; do
        python train.py $subject 1 --method linear --undersampling-far $far
        for session in {2..4}; do
            python accuracy.py $subject $session --method linear --tmp linear-cosine-2
        done
    done
    echo "\n" >> log/linear-cosine-2.csv
done
```

### Linear SVM
```
for subject in {1..9}; do
    python train.py $subject 1 --method linear --undersampling-far 0
    for session in {2..4}; do
        python accuracy.py $subject $session --method linear --tmp kensuke-linearsvc-maxabsscaler
    done
done
```

### SWLDA
```
for subject in {1..9}; do
    python train.py $subject 1 --method swlda --no-undersampling --average 1
    for session in {2..4}; do
        python accuracy.py $subject $session --method swlda --tmp kensuke-swlda-average1 --average 1
    done
done
```
### SWLDA AVERAGE 5
```
for subject in {1..9}; do
    python train.py $subject 1 --method swlda --no-undersampling --average 5
    for session in {2..4}; do
        python accuracy.py $subject $session --method swlda --tmp kensuke-swlda-average5 --average 5
    done
done
```

### SVM
```
for subject in {1..9}; do
    python train.py $subject 1 --method svm --undersampling-far 0
    for session in {2..4}; do
        python accuracy.py $subject $session --method svm --tmp kensuke-rbfsvm
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

## plot

```
python plot.py --undersampling --undersampling-far 400 --type all 1 1
```


## Kodama
### LinearSVM
```
for subject in {1,3,4,5,6,7,8,9,10}; do
    python train.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject 0 --method linear --undersampling-far 0
    for session in {1..5}; do
        python accuracy.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject $session --method linear --tmp kodama-linear
    done
done
```
### RBFSVM
```
for subject in {1,3,4,5,6,7,8,9,10}; do
    python train.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject 0 --method svm --undersampling-far 0
    for session in {1..5}; do
        python accuracy.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject $session --method svm --tmp kodama-svm-025
    done
done
```

### SWLDA
```
for subject in {1,3,4,5,6,7,8,9,10}; do
    python train.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject 0 --method linear --undersampling-far 0
    for session in {1..5}; do
        python accuracy.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject $session --method linear --tmp kodama-swlda
    done
done
```


```
for far in {0..300}; do
    for subject in {1,3,4,5,6,7,8,9,10}; do
        python train.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject 0 --method linear --undersampling-far $far --modelname kodama
        for session in {1..5}; do
            python accuracy.py --filename ../mat_files_kodama/subject%d_section%d.mat --repeat 10 $subject $session --method linear --tmp kodama-linear-cosine --modelname kodama
        done
    done
    echo "\n" >> log/kodama-linear-cosine.csv
done
```

# kodama's data
```
for subject in {1,3,4,5,6,7,8,9,10}; do
    python train.py --kodama --repeat 10 $subject 0 --method svm
    for session in {1..5}; do
        python accuracy.py --kodama --repeat 10 $subject $session --method svm --tmp kodamamat-rbfsvm-absmax
    done
done
```
# LibSVM
```
for subject in {1,3,4,5,6,7,8,9,10}; do
    python train.py --kodama --repeat 10 $subject 0 --method libsvm
    for session in {1..5}; do
        python accuracy.py --kodama --repeat 10 $subject $session --method libsvm --tmp kodamamat_libsvm
    done
    echo "\n" >> log/kodamamat_libsvm.csv
done
echo "\n" >> log/kodamamat_libsvm.csv
```
### Linear SVM
```
for subject in {1,3,4,5,6,7,8,9,10}; do
    python train.py $subject 0 --method linear --undersampling-far 0 --kodama --repeat 10
    for session in {1..5}; do
        python accuracy.py $subject $session --method linear --tmp kodama-linearsvm --kodama --repeat 10
    done
    echo "\n" >> log/kodama-linearsvm.csv
done
echo "\n" >> log/kodama-linearsvm.csv
```
