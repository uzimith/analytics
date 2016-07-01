# Trial 2 -- 2 answer
# Linear SVM
```
method=linear
filename=kensuke-linearsvm-trial2
touch log/$filename.csv
for subject in {1..9}; do
    python train.py $subject 1 --method $method --decimate 25 --modelname trial2
    if [ $? = 0 ]; then
        for session in {2..4}; do
            python accuracy.py $subject $session --method $method --tmp $filename --decimate 25 --modelname trial2 --repeat 2 --skip 1 --problem 2
        done
    fi
done
```

# SWLDA
```
method=swlda
filename=kensuke-swlda-trial2
touch log/$filename.csv
for subject in {1..9}; do
    python train.py $subject 1 --method $method --decimate 25 --modelname trial2
    if [ $? = 0 ]; then
        for session in {2..4}; do
            python accuracy.py $subject $session --method $method --tmp $filename --decimate 25 --modelname trial2 --repeat 2 --skip 1 --problem 2
        done
    fi
done
```

# Trial 5 -- 1 answer

#Linear SVM with other mat
```
method=swlda
filename=kensuke-swlda-300~800
touch log/$filename.csv
for factor in {1..45}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1..9}; do
        python train.py $subject 1 --method $method --decimate $factor --filename ../mat/512hz4555_300-800/sub%s_sec%d.mat
        if [ $? = 0 ]; then
            for session in {2..4}; do
                python accuracy.py $subject $session --method $method --tmp $filename --decimate $factor --filename ../mat/512hz4555_300-800/sub%s_sec%d.mat
            done
        fi
    done
    echo "\n" >> log/$filename.csv
done
```

#Linear SVM
```
method=linear
filename=kensuke-linearsvm-euclid
touch log/$filename.csv
for factor in {1..45}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1..9}; do
        python train.py $subject 1 --method $method --decimate $factor --undersampling-far 180 --undersampling-method euclidean
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
filename=kensuke-swlinearsvm-euclid
touch log/$filename.csv
for factor in {1..45}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1..9}; do
        python train.py $subject 1 --method $method --decimate $factor --undersampling-far 180 --undersampling-method euclidean
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
filename=kensuke-stepwiseswlda-euclid
touch log/$filename.csv
for factor in {1..45}; do
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

# Plot
```
python plot.py 1 1 --decimate 1 --filename ../mat/512hz4555_300-800/sub%s_sec%d.mat
```

# Kodama

```
method=swlda
filename=kodama-swlda
touch log/$filename.csv
for factor in {15..25}; do
    echo -n $factor, >> log/$filename.csv
    for subject in {1,3,4,5,6,7,8,9,10}; do
        python train.py $subject 0 --method $method --decimate $factor --kodama --repeat 10 --modelname kodama
        if [ $? = 0 ]; then
            for session in {1..5}; do
                python accuracy.py $subject $session --method $method --tmp $filename --decimate $factor --kodama --repeat 10 --modelname kodama
            done
        fi
    done
    echo "" >> log/$filename.csv
done
```
