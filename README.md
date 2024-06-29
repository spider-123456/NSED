# Neighborhood Structure Enhancement And Denoising Method For Multi-Behavior Recommendation.

## This is the implementation of NSED on python=3. 8, pytorch=1.13.

**Getting started:**

first please check that the environment contains the packages in requirements.txt.

 You need to install torch-cluster, torch-scatter, torch-sparse before you install torch_geometric, this link should help you.[pytorch-geometric.com/whl/](https://pytorch-geometric.com/whl/) And you can see the [Documentation](https://github.com/rusty1s/pytorch_scatter) when installing torch_geometric.

You can refer to https://github.com/MingshiYan/CRGCN?tab=readme-ov-file in processing data to deal with data sets and preparation. Let's take the Beibei dataset as an example.

**Before you run the main.py , you are supposed build a folder name: check_point to store the model.pth**.

## Log files and weight files are provided for better results of compound line experiments.
### We released the Beibei dataset for better experiments. Additional data sets can be obtained from here ： https://drive.google.com/drive/folders/1PCmlXPGR-rgfbK7e9Ia5c073-DE9xi9x 
Run command: Some other parameters you can view in the code.
```
python main.py --device=cuda:3 --ssl_reg=3e-5 --ssl_weight=1e-5  --inter_reg=1e-5 --lr=0.001 --data_name=taobao --task_weight=1,1,1 --layers=2,3,3
```
**how to get the best result**.
```
#taobao best result
python main.py --device=cuda:3 --ssl_reg=3e-5 --ssl_weight=1e-5  --inter_reg=1e-5 --lr=0.001 --data_name=taobao --task_weight=1,1,1 --layers=2,3,3

#tmall best result
python main.py --device=cuda:3 --ssl_reg=3e-5 --ssl_weight=1e-5  --inter_reg=1e-5 --lr=0.001 --data_name=tmall --task_weight=2,1,1,2 --layers=2,2,2,3

#beibei best result
python main.py --device=cuda:1 --ssl_reg=3e-5 --ssl_weight=1e-5  --inter_reg=1e-5 --lr=0.001 --data_name=beibei --task_weight=0.14286,0.42857,0.42857 --layers=2,3,3
```



## Model 


![Fig3](https://github.com/spider-123456/NSED/assets/73099091/1fe87770-82f6-4c45-9448-11c564c9a347)


### Denoise Module architecture diagram

![Fig4](https://github.com/spider-123456/NSED/assets/73099091/d5e46ecc-fd03-4457-9238-d9b42fbce1f7)


### Code architecture

We have shown the code architecture to facilitate replicating the work of this article.

```
D:\CODE\NSED
|   README.md
|   gcn_conv.py
|   main.py
|   metrics.py
|   model_cascade.py
|   utils.py
|   trainer.py
|   data_set.py
|   requirements.txt
\---Data
+---Tmall
|   |       cart.txt
|   |       cart_dict.txt
|   |       click.txt
|   |       click_dict.txt
|   |       collect.txt
|   |       collect_dict.txt
|   |       count.txt
|   |       data_process.py
|   |       test.txt
|   |       test_dict.txt
|   |       trn_click
|   |       test.py
|   |       buy.txt
|   |       buy_dict.txt
|   |       train.txt
|   |       all.txt
|   |       all_dict.txt  
\---check_point  
\---log
    +---beibei
    |       beibei_enb_64_2024-03-18 14_46_34.log
    |       Beibei_enb_64_2024-06-01 09_53_53.log
    |       
    +---Tmall
    |       Tmall_enb_64_2024-04-16 12_41_37.log
    |       Tmall_enb_64_2024-04-16 12_42_37.log

    |       
    \---taobao
            taobao_enb_64_2024-04-18 15_06_32.log
            taobao_enb_64_2024-04-18 15_06_49.log
```

