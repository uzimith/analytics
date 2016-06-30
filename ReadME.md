#Linear SVM
```
method=linear
filename=kensuke-linearsvm
touch log/$filename.csv
for factor in {1..25}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1..9}; do
        python train.py $subject 1 --method $method --decimate $factor
        if [ $? = 0 ]; then
            for session in {2..4}; do
                python accuracy.py $subject $session --method $method --tmp $filename --decimate $factor
            done
        fi
    done
    echo "\n" >> log/$filename.csv
done
```

#Stepwise Linear SVM
```
method=swlinearsvm
filename=kensuke-swlinearsvm
touch log/$filename.csv
for factor in {1..25}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1..9}; do
        python train.py $subject 1 --method $method --decimate $factor
        if [ $? = 0 ]; then
            for session in {2..4}; do
                python accuracy.py $subject $session --method $method --tmp $filename --decimate $factor
            done
        fi
    done
    echo "\n" >> log/$filename.csv
done
```

## SWLDA

```
method=swlda
filename=kensuke-stepwiseswlda
touch log/$filename.csv
for factor in {1..25}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1..9}; do
        python train.py $subject 1 --method $method --decimate $factor
        if [ $? = 0 ]; then
            for session in {2..4}; do
                python accuracy.py $subject $session --method $method --tmp $filename --decimate $factor
            done
        fi
    done
    echo "\n" >> log/$filename.csv
done
```

```
method=swlda
filename=kensuke-stepwiseswlda
touch log/$filename.csv
for factor in {1..25}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1..9}; do
        python train.py $subject 1 --method $method --decimate $factor
        if [ $? = 0 ]; then
            for session in {2..4}; do
                python accuracy.py $subject $session --method $method --tmp $filename --decimate $factor
            done
        fi
    done
    echo "\n" >> log/$filename.csv
done
```

# Kodama

```
method=svm
filename=kodama-rbfsvm-factor-maxabs
touch log/$filename.csv
for factor in {1..5}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1,3,4,5,6,7,8,9,10}; do
        python train.py $subject 0 --method $method --decimate $factor --kodama
        if [ $? = 0 ]; then
            for session in {1..5}; do
                python accuracy.py $subject $session --method $method --tmp $filename --decimate $factor --kodama
            done
        fi
    done
    echo "\n" >> log/$filename.csv
done
```
