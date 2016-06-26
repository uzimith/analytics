```
rm log/tmp.csv
for subject in {1..9}; do
    python train.py $subject 1
    for session in {2..4}; do
        python accuracy.py $subject $session
    done
done
```

