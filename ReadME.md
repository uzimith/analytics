```
rm log/tmp.csv
for subject in {1..9}; do
    python train.py $subject 1
    for session in {2..4}; do
        python accuracy.py $subject $session
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
