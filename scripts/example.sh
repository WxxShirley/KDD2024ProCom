DEVICE=cuda:0

for DATASET in facebook amazon dblp lj twitter ; do 
    python -u run.py --dataset=$DATASET --threshold=0.2  --from_scratch=0 >logs/$DATASET.log ;
done 


# shot experiment
for DATASET in lj amazon dblp ; do 
   for SHOT in 5 20 25 50 100 ; do 
       python -u run.py --device=$DEVICE --dataset=$DATASET --num_shot=$SHOT  --from_scratch=0   >logs/$DATASET+$SHOT.log  ;
    done
done 
