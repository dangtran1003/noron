Run 5 times, index from 0 to 4:
{
    python train_gpu.py --cross_val_index={{index}} 
    => run for 1 hour
    run 10 times: 
    {
        python train_gpu.py --cross_val_index={{index}} --schedule=evaluate
        => note the result

        python train_gpu.py --cross_val_index={{index}} --is_test=True --schedule=evaluate
        => note the result 
    }
}
